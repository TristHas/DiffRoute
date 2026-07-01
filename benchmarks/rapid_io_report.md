# RAPID IO DiffRoute Benchmark Report

This benchmark targets the workload used by
`DiffHydro/examples/1. RAPID IO.ipynb` on a single NVIDIA GB200.

## AKO4ALL Setup And Takeaways

AKO4ALL was cloned to:

```text
/home/ea0126/workspace/rivers/external_tools/AKO4ALL
```

and symlinked as a local Codex skill:

```text
/home/ea0126/.codex/skills/ako4all
```

AKO4ALL is primarily a workflow skill, not a runtime dependency. The useful
pieces for DiffRoute are:

- resolve the optimization target, reference, inputs, and benchmark mode before
  editing kernels;
- keep a correctness gate separate from timing;
- profile first, optimize the measured bottleneck, and keep iteration results;
- use `ncu` when available, but continue analytically when it is not;
- avoid reward-hacking or benchmark-only shortcuts.

Its built-in KernelBench harness is oriented around a single `Model.forward`.
DiffRoute's target is a multi-step graph/router computation with saved RAPID
data, so `rapid_io_benchmark.py` is a custom harness following the same loop.

## Benchmark Procedure

Environment:

```bash
source /home/ea0126/miniconda3-aarch64/etc/profile.d/conda.sh
conda activate rivers
```

Correctness gate:

```bash
python benchmarks/rapid_io_benchmark.py \
  --correctness \
  --correct-time-steps 16 \
  --atol 2e-2 \
  --conv-impl auto
```

Full RAPID IO profile:

```bash
python benchmarks/rapid_io_benchmark.py \
  --conv-impl auto \
  --trials 3 \
  --backward-trials 3 \
  --warmup 1 \
  --backward-warmup 1
```

Aggregation forward breakdown:

```bash
python benchmarks/rapid_io_aggregation_forward_breakdown.py \
  --trials 15 \
  --warmup 3
```

The benchmark reports full forward, full backward wrt runoff `X`, full backward
wrt routing parameters, and the substeps:

- IRF aggregation;
- sparse-to-block-sparse conversion;
- block-sparse convolution;
- one-sided convolution backward wrt `X`;
- one-sided convolution backward wrt block-sparse weights `W`.

After `module load nvhpc`, Nsight Compute and Nsight Systems are available.
Nsight Compute kernel counters are blocked by GPU counter permissions
(`ERR_NVGPUCTRPERM`), so the analysis below uses CUDA-event timings, Nsight
Systems kernel attribution, and code-level attribution.

## Workload

Final benchmark metadata:

| Field | Value |
|---|---:|
| GPU | NVIDIA GB200 |
| CUDA capability | 10.0 |
| PyTorch / CUDA | 2.12.1+cu130 / 13.0 |
| Runoff shape | `[1, 15562, 29220]` |
| Routing params shape | `[15562, 2]` |
| RAPID paths | 1,653,119 |
| Block-sparse blocks | 27,088 |
| Block values shape | `[27088, 16, 16, 32]` |
| `max_delay`, `dt` | 32, 1/24 |

## Results

Initial baseline was the original Triton implementation with one
`BLOCK_SIZE_N=64` and an autograd backward that computed both `dX` and `dW`
even when only one gradient was requested.

The intermediate state added one-sided backward and GB200 forward/`dW` tile
specialization. The final state additionally uses:

- a column-CSR, no-atomic sparse-conv `dX` kernel;
- a `B == 1` time-reduction sparse-conv `dW` kernel;
- fused aggregation post-processing backward for ReLU, triangular downsampling,
  flip, and normalization;
- saved pointer-jump tables for prefix-sum backward instead of one launch per
  graph depth level;
- hardware auto dispatch on GB200:
  - forward `BLOCK_SIZE_N=256`;
  - `dX` `BLOCK_SIZE_N=256`;
  - `dW` `BLOCK_SIZE_N=64`.

| Measurement | Initial ms | Pre-backward-opt ms | Final ms | Speedup vs initial |
|---|---:|---:|---:|---:|
| Full forward | 303.19 | 203.21 | 200.69 | 1.51x |
| Full backward `dL/dX` | 1712.97 | 799.98 | 169.33 | 10.12x |
| Full backward `dL/dparams` | 1902.61 | 854.87 | 512.36 | 3.71x |
| Aggregation forward | 25.57 | 25.62 | 22.51 | 1.14x |
| Blockize forward | 1.55 | 1.57 | 1.96 | 0.79x |
| Convolution forward | 276.37 | 176.39 | 176.36 | 1.57x |
| Aggregation backward `dL/dparams` | 190.94 | 191.43 | 34.12 | 5.60x |
| Blockize backward | 0.31 | 0.31 | 0.31 | 1.00x |
| Convolution backward `dL/dX` | 1713.92 | 801.01 | 170.35 | 10.06x |
| Convolution backward `dL/dW` | 1713.93 | 666.10 | 486.33 | 3.52x |

Isolated kernel benchmarks on the RAPID shape:

| Kernel path | Baseline ms | Final ms | Speedup |
|---|---:|---:|---:|
| `dX` sparse convolution | 799.16 | 168.79 | 4.73x |
| `dW` sparse convolution | 663.87 | 484.66 | 1.37x |

Short-sequence (`T=500`) result after aggregation optimization:

| Measurement | Before aggregation opt ms | Final ms | Speedup |
|---|---:|---:|---:|
| Full backward `dL/dparams` | 200.60 | 43.58 | 4.60x |
| Aggregation backward `dL/dparams` | 190.25 | 34.25 | 5.55x |
| Convolution backward `dL/dW` | 11.07 | 11.07 | 1.00x |

Aggregation forward breakdown on the RAPID shape:

| Stage | CUDA median ms |
|---|---:|
| Aggregator forward total | 17.78 |
| Aggregate IRF before sampler | 13.17 |
| IRF generation | 0.43 |
| RFFT | 0.06 |
| Stable log transform | 0.13 |
| Prefix sum | 0.89 |
| Closure sub | 2.57 |
| Complex exp | 1.70 |
| IFFT | 7.51 |
| Temporal sampler total | 4.63 |
| Temporal sampler ReLU | 1.45 |
| Temporal sampler triangular downsample | 2.60 |
| Temporal sampler flip | 0.14 |
| Temporal sampler sum | 0.31 |
| Temporal sampler normalize | 0.18 |

The aggregation-forward timing also checked prefix-sum alternatives for
attribution. The current autograd prefix path measured `0.887 ms` median; a
version without saved jump tables measured `0.757 ms`; a fixed-round version
without the per-round `.all()` check measured `0.314 ms`. This attribution was
not committed as a production change.

## Analysis

The first major inefficiency was structural: `BlockSparseConv1dFn.backward`
always launched both backward kernels. As a result:

- `dL/dX` paid for both `dx` and `dvalues`;
- `dL/dparams` paid for both `dx` and `dvalues`, even though only `dvalues`
  can flow into aggregation and routing parameters.

Using `ctx.needs_input_grad` removed that wasted path. This is why the
intermediate full backward measurements improved by more than 2x without
changing numerical semantics.

The second improvement was tile-size specialization. GB200 is faster with wider
time tiles for forward, while backward needs path-specific settings:

- forward is best at `BLOCK_SIZE_N=256`;
- the CSR `dX` kernel is best at `BLOCK_SIZE_N_DX=256`;
- the time-reduction `dW` kernel is best at `BLOCK_SIZE_N_DW=64`.

The final dispatcher uses separate tile sizes per path, avoiding a single
compromise tile.

The main backward optimization was changing ownership.

For `dX`, the RAPID COO blocks are row-grouped for forward, but nearly
ungrouped by input column: 27,024 current-order column runs for 27,088 blocks.
Each input block receives 27.84 block contributors on average and up to 96.
The final `dX` kernel uses a CSR view by input block, owns
`dx[c_block, time_tile]`, accumulates all row-block contributors locally, and
stores once. This removes the scattered atomic-add pattern.

For `dW`, the original kernel atomically accumulated each weight block/tap over
115 time tiles. The final `B == 1` kernel owns one sparse block and one kernel
tap, reduces over time internally, and stores the gradient once. It is a
smaller win than `dX` because each program carries a 16x16 accumulator through
a long time loop, but it removes the time-tile atomic accumulation.

Aggregation backward was optimized in two places. First, the post-processing
backward now fuses normalization, flip, triangular downsample backward, and
ReLU masking in a Triton kernel. Nsight Systems had shown the previous cuDNN
triangular-downsample input-gradient as the largest single aggregation-backward
kernel. Second, `prefix_sum` backward now saves the forward pointer-jump tables
and reverses those rounds. This replaces one launch per RAPID graph depth level
with one launch per pointer-jump round.

Current remaining bottlenecks:

- full forward is now almost entirely sparse convolution;
- full `dL/dX` is no longer imbalanced relative to forward (`169 ms` vs
  `201 ms`);
- full `dL/dparams` is now sparse-conv `dW` dominated: about `486 ms` of
  `dW` convolution plus about `34 ms` aggregation backward.

Aggregation forward is small for the full RAPID workload. Aggregation backward
is still meaningful for short sequences, but after the pointer-jump change it
is no longer the dominant long-sequence cost. For short sequences, aggregation
and fixed overheads are now in the same range as convolution.

## Dispatch Architecture

The minimal dispatch layer is `diffroute/backend.py`.

It returns a `ConvConfig` with:

- implementation: `auto`, `triton`, or `torch`;
- forward time tile;
- `dX` time tile;
- `dW` time tile;
- optional block-channel tile override.

`BlockSparseCausalConv` uses this selector at call time. `LTIRouter` and
`LTIStagedRouter` expose the same knobs for explicit overrides.

Environment overrides:

```text
DIFFROUTE_CONV_IMPL
DIFFROUTE_BLOCK_M
DIFFROUTE_BLOCK_N
DIFFROUTE_BLOCK_N_DX
DIFFROUTE_BLOCK_N_DW
```

Current defaults are intentionally conservative except where measured on GB200:

- GB200/B200/SM100+: forward 256, `dX` 256, `dW` 64;
- H100/H200/SM90: forward 128, `dX` 64, `dW` 128;
- A100/SM80: 64 for all paths.
