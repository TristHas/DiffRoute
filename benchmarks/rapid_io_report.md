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

The benchmark reports full forward, full backward wrt runoff `X`, full backward
wrt routing parameters, and the substeps:

- IRF aggregation;
- sparse-to-block-sparse conversion;
- block-sparse convolution;
- one-sided convolution backward wrt `X`;
- one-sided convolution backward wrt block-sparse weights `W`.

Nsight Compute and Nsight Systems were not installed on this host, so the
analysis below uses CUDA-event timings and code-level attribution.

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

Initial baseline was the current Triton implementation with a single
`BLOCK_SIZE_N=64` and an autograd backward that computed both `dX` and `dW`
even when only one gradient was requested.

Final result uses:

- one-sided sparse-conv backward using `ctx.needs_input_grad`;
- hardware auto dispatch on GB200:
  - forward `BLOCK_SIZE_N=256`;
  - `dX` `BLOCK_SIZE_N=64`;
  - `dW` `BLOCK_SIZE_N=256`.

| Measurement | Initial ms | Final ms | Speedup |
|---|---:|---:|---:|
| Full forward | 303.19 | 203.21 | 1.49x |
| Full backward `dL/dX` | 1712.97 | 799.98 | 2.14x |
| Full backward `dL/dparams` | 1902.61 | 854.87 | 2.23x |
| Aggregation forward | 25.57 | 25.62 | 1.00x |
| Blockize forward | 1.55 | 1.57 | 0.99x |
| Convolution forward | 276.37 | 176.39 | 1.57x |
| Aggregation backward `dL/dparams` | 190.94 | 191.43 | 1.00x |
| Blockize backward | 0.31 | 0.31 | 1.00x |
| Convolution backward `dL/dX` | 1713.92 | 801.01 | 2.14x |
| Convolution backward `dL/dW` | 1713.93 | 666.10 | 2.57x |

## Analysis

The original major inefficiency was structural: `BlockSparseConv1dFn.backward`
always launched both backward kernels. As a result:

- `dL/dX` paid for both `dx` and `dvalues`;
- `dL/dparams` paid for both `dx` and `dvalues`, even though only `dvalues`
  can flow into aggregation and routing parameters.

Using `ctx.needs_input_grad` removes that wasted path. This is why both full
backward measurements improve by more than 2x without changing numerical
semantics.

The second improvement is tile-size specialization. GB200 is faster with wider
time tiles for forward and `dW`, but not for `dX`:

- `BLOCK_SIZE_N=64`: good `dX`, slower forward and `dW`;
- `BLOCK_SIZE_N=128`: faster forward and `dW`, mildly slower `dX`;
- `BLOCK_SIZE_N=256`: best forward and `dW`, much slower `dX`.

The final dispatcher uses separate tile sizes per path, avoiding a single
compromise tile.

Current remaining bottlenecks:

- full forward is now almost entirely sparse convolution;
- full `dL/dX` is sparse-conv `dX`;
- full `dL/dparams` is sparse-conv `dW` plus about 191 ms of aggregation
  backward.

Aggregation forward is small for this workload. Aggregation backward is still
meaningful for training routing parameters, but it is no longer the dominant
cost.

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

Current defaults are intentionally conservative:

- GB200/B200/SM100+: forward 256, `dX` 64, `dW` 256;
- H100/H200/SM90: forward 128, `dX` 64, `dW` 128;
- A100/SM80: 64 for all paths.

Future backends can fit into the same selector without a large framework:
add an implementation label, provide a callable with the same tensor contract,
and add one hardware rule or environment override.
