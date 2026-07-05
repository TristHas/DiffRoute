# Agent findings — DiffRoute GEOGloWS routing & GB200 kernel work

> **Machine context — this file was produced on the RIKEN R-CCS "rikyu" AI4S machine.**
> This repo is shared across machines, so treat every hardware/path/version fact below as
> **rikyu-specific** and re-verify before relying on it elsewhere.
>
> - GPU: **NVIDIA GB200** (Blackwell, compute capability `sm_100a`, 184 GiB) — single device `cuda:0`.
> - `torch 2.12.1+cu130` (CUDA runtime 13.0), `triton 3.7.1`, Python 3.14.6 (conda-forge).
> - Env with all deps: **`/home/ea0126/miniconda3-aarch64/envs/rivers/bin/python`**
>   (the base `python` on PATH has no torch). Sibling checkouts live under
>   `/home/ea0126/workspace/rivers/`: `DiffRoute/`, `DiffHydro/`, `xtensor/`, `external_tools/AKO4ALL/`.

## 1. Running the full GEOGloWS simulation from random runoff

- Script: **`DiffHydro/examples/benchmark_full_geoglows_random_runoff.py`**. It builds the
  full clustered GEOGloWS graph and routes uniform-random reach runoff (no pixel runoff /
  interpolation / retrospective discharge needed).
- Data present on rikyu: **`DiffHydro/examples/data/geoglows/configs/`** — all **125 VPU**
  folders, each with `rapid_connect.csv`, `riv_bas_id.csv`, `k.csv`, `x.csv` (verified complete, 370 MB).
  The `305_daily_sparse_runoff.feather` / `305_interp_weight.feather` in `.../input/` are for the
  *RAPID IO* example (VPU 305 only), **not** for this full-random benchmark.
- Full run: **125 VPUs, 6,838,900 reaches, 729 clusters, 29,220 daily steps**, routed in
  **~18–19 s** forward (inference). Baseline result: `DiffHydro/examples/benchmark_results/full_geoglows_random_runoff_benchmark.json`.
- Run it: `cd DiffHydro/examples && <rivers python> benchmark_full_geoglows_random_runoff.py`.
  Smoke test: add `--vpu-limit 3 --time-steps 256`.

## 2. Setup-phase cache (implemented)

The ~276 s "setup" is almost entirely the **pure-Python networkx clustering** in
`read_multiple_rapid_graphs` (`define_schedule`); CSV reads are cheap. There was no
serialization. A **lean, reclustering-free cache** is now implemented:

- `diffroute.io.save_graph_cache(graph, path)` / `load_graph_cache(path)` (re-exported from
  `diffhydro.io`). Backed by `RivTreeCluster._cache_state()` / `._from_cache()` in
  `diffroute/structs/riv_graphs.py`.
- Stores **only tensors + light metadata** routing actually reads (`edges`, `path_cumsum`,
  `params`, per-node reach ids, `prefix_jump_rounds`, `include_index_diag`, `irf_fn`, transfer
  tables). The networkx graphs and pandas objects are **not** stored (they're unused at route time).
- Size: **~26 bytes/node → ~170 MiB** for the full 6.84 M graph (vs ~1.2 GiB for a naive
  `torch.save(graph)`). Reload is **~0.03 s vs ~276 s rebuild** and routes bit-for-bit within
  TF32 atomic noise (~2e-6 rel).
- Wired into the benchmark: `--cache PATH` (build+write if absent, else load). Verified
  setup 14.55 s → 0.045 s on reload.

## 3. Hardware utilization — does it use the tensor cores? (Partly.)

Verified by compiling the kernels and reading the emitted PTX (no `ncu`/`nsys` on rikyu).

- **Triton block-sparse conv (the main compute)** uses `tl.dot`, which defaults to TF32 in
  Triton 3.7.1. Compiled forward PTX contains
  `mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32` on `sm_100a` → **tensor cores are used, in TF32.**
- **But**: (a) it's the **legacy `mma.sync`** warp-MMA, *not* Blackwell 5th-gen `tcgen05.mma`/TMEM
  nor Hopper `wgmma` — on `sm_100` this runs well below tensor-core peak; (b) `block_size=16`
  forces the **smallest** `m16n8k8` tile, and the conv is memory/latency-bound, so tensor-core
  FLOP peak is largely irrelevant until arithmetic intensity is raised; (c) TF32 silently
  truncates FP32 inputs (10-bit mantissa) — implicit accuracy trade-off. `input_precision="ieee"`
  would be bit-exact but drop tensor cores; `"tf32x3"` is more accurate and still tensor-core.
- **Torch side**: `torch.backends.cuda.matmul.allow_tf32 = False`, `float32_matmul_precision =
  "highest"` → any torch FP32 matmul runs **without** tensor cores. Hot path is mostly custom
  Triton; the only torch op is `F.conv1d` in `agg/temporal_sampler.py` (plain FP32).

## 4. Forward-time breakdown (VPU 305, 15,562 nodes, 8192 steps, block_size=16)

| substep | median | share |
|---|---:|---:|
| **convolution.forward** | **42.1 ms** | **~73%** |
| aggregation.forward | 14.3 ms | ~25% |
| blockize.forward | 1.5 ms | ~3% |
| full.forward | 57.8 ms | 100% |

→ The **block-sparse conv is the dominant forward cost.** Measure with
`DiffRoute/benchmarks/rapid_io_benchmark.py` (`--correctness` for the triton-vs-torch guard).

### Forward optimization status — pure parameterization is saturated (measured 2026-07)

Exploratory measure-only sweeps on the 305 forward conv (8192 steps) found **no free win**
from launch/parameterization knobs — the ~42 ms is algorithmic, not config-bound:

- `num_warps` (hardcoded 8): sweeping {4,8,16} — 8 is best; 16 is ~1.8× slower.
- `num_stages`: no effect (the tap loop is data-dependent; Triton can't pipeline it).
- forward time-tile `BLOCK_SIZE_N` (=`block_n`, 256 on GB200): {128,256,512,1024} — 256 optimal;
  best alternative (128/warps=4) is only 1.02×.
- `NZB_BLOCK_SIZE`: already tuned to 1 in iter 14.
- **`block_size`**: 16 is optimal. 32→conv 60 ms, 64→conv 76 ms, and block-value memory
  **doubles per step** (846→1708→3416 MiB) — the graph is so sparse that 16×16 blocks are
  mostly empty, so larger blocks just compute more zeros. (`block_size<16` is invalid: `tl.dot`
  needs contraction ≥16.)

**Implication:** further forward speedup needs an *algorithmic/kernel restructure* (Track 2/3),
not tuning. The conv is ~866k tiny programs (27k nonzero blocks × 32 time tiles), each a
16×16 block × 32 taps with TF32 `mma.sync` + atomic output — latency/occupancy bound.
Candidate directions (all require keeping the single autograd fwd+bwd path intact): fuse the
32-tap loop to raise arithmetic intensity, reduce program count / atomics, or a different
sparse-conv formulation (e.g. im2col-per-block, FFT). These are non-trivial and trade against
code clarity — scope with the maintainer before committing to one.

## 5. GPU-kernel optimization skill

**`external_tools/AKO4ALL/`** (skill `ako4all`): agentic profile→modify→bench→log→commit loop
for GPU kernels, with correctness checks and `ITERATIONS.md` logging. This branch already follows
its protocol (`DiffRoute/ITERATIONS.md`, `[iter N]` commits). It is **not** registered as a
Claude Code slash command — drive it by reading `AKO4ALL/SKILL.md` and following it. Its scaffold
subprocesses `python`, so point it at the `rivers` env bin.

## 6. Continuation after rate-limit recovery (2026-07-05)

User constraint for the continuation: **do not modify the lean torch/router/aggregator code**;
optimize only low-level implementation details. This is also recorded in `DiffRoute/HINTS.md`.

Committed low-level sparse-packing iterations:

- **Iter 19** (`f93bade`): In CUDA dense-key `BlockSparseKernel.from_coo`, replaced generic
  `index_put` with `index_copy_` because each COO row maps to one unique flattened block-value row.
  RAPID-305 conversion microbenchmark: **~1.4387 ms → ~1.0254 ms** median; standard T=500
  `substep.blockize.forward`: **1.4591 ms → 1.0645 ms** median.
- **Iter 20** (`2960b46`): Build `block_col_order` / `block_col_offsets` directly from the dense
  presence map, avoiding generic sort/bincount metadata construction in the dense-key path.
  RAPID-305 conversion microbenchmark: **~1.0403 ms → ~0.9613 ms** median. Standard T=500
  blockize improved only slightly: **1.0645 ms → 1.0491 ms** median.
- **Iter 21** (`1470dc5`): Under `torch.inference_mode()` / no grad, skip backward-only column
  metadata entirely and return empty metadata tensors. Grad-enabled conversion still builds the
  metadata for optimized `dX` backward. RAPID-305 no-grad blockize median: **~0.9050 ms →
  ~0.6662 ms**. Full 3-VPU GEOGloWS smoke passed.

Full 125-VPU GEOGloWS validation after iter 21:

- Built `/tmp/full_geoglows_graph_cache.pt` (169 MiB). Uncached build+write setup:
  **274.294 s**.
- Reused cache setup: **0.366 s**. This is a graph/setup cache only; it does **not** cache routing
  kernels, aggregation outputs, block-sparse values, sparse conversion outputs, runoff, discharge,
  or any forward-computation artifact.
- Full random-runoff-plus-routing timing: recovered baseline **18.659 s**; iter 21 cached-pass
  timing **15.023 s**. Random generation was **~0.473 s**, so estimated routing-only is
  **18.186 s → 14.550 s**.

Remaining optimization directions identified:

- **Main remaining forward cost is still sparse convolution**, not blockization. Prior sweeps found
  `block_size=16`, `BLOCK_SIZE_N=256`, `NZB_BLOCK_SIZE=1`, `num_warps=8` already best among simple
  knobs on GB200.
- Further conv speedup likely needs a real kernel/formulation change: reduce atomic output traffic,
  group work by output row block/time tile, increase arithmetic intensity across the 32 taps, or try
  an im2col/GEMM-like or FFT-style formulation for this sparse pattern.
- The current Triton forward uses TF32 tensor cores through legacy `mma.sync`, not Blackwell
  `tcgen05`/TMEM. A deeper Blackwell-specific rewrite could explore larger/taller tiles or CUDA
  kernels using newer primitives, but that is higher-risk and should be scoped separately.
- Full setup remains dominated by NetworkX clustering when no graph cache is available. Optimizing
  that would be a setup/data-structure project, not a low-level forward-kernel optimization.
