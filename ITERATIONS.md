# DiffRoute RAPID IO Kernel Optimization Iterations

## Iteration 1 - Column-grouped dX atomics

- Target: `block_sparse_conv_1d_bwd_dx_kernel`
- Hypothesis: RAPID block indices are row-grouped for forward, but almost ungrouped by input block for `dL/dX`. Sorting by input block and reducing same-column contributions inside each Triton program should reduce atomic traffic.
- Change: Added a `BlockSparseKernel.block_col_order` permutation and a `block_sparse_conv_1d_bwd_dx_col_grouped_kernel` that consumes that order while keeping the original values tensor layout.
- Correctness: `python benchmarks/rapid_io_conv_kernel_benchmark.py --target dx --correctness --correct-time-steps 16` passed with `max_abs=0.015270233154296875`, `atol=0.02`.
- Runtime: `python benchmarks/rapid_io_conv_kernel_benchmark.py --target dx --trials 5 --warmup 2` reported `mean_ms=767.6581909179688`, `min_ms=767.6215209960938`.
- Baseline comparison: committed isolated `dX` baseline was `mean_ms=799.156`, so this is `3.9%` faster.
- Notes: The win is real but modest, which means atomics are not the only cost; the kernel still performs the same dot work and now gathers values through a column order.

## Iteration 2 - CSR dX without atomics

- Target: `block_sparse_conv_1d_bwd_dx_kernel`
- Hypothesis: A program should own one input block and one `dx` time tile, then loop over that input block's upstream contributors. This changes the time tile from upstream `dy` time to output `dx` time and removes all `dx` atomics.
- Change: Added `BlockSparseKernel.block_col_offsets` and `block_sparse_conv_1d_bwd_dx_col_csr_kernel`. The new kernel uses `block_col_order + block_col_offsets` to accumulate each `dx[c_block, time_tile]` directly and writes it once.
- Correctness: `python benchmarks/rapid_io_conv_kernel_benchmark.py --target dx --correctness --correct-time-steps 16` passed with `max_abs=0.015386581420898438`, `atol=0.02`.
- Runtime: `python benchmarks/rapid_io_conv_kernel_benchmark.py --target dx --trials 5 --warmup 2` reported `mean_ms=275.5728698730469`, `min_ms=275.4349670410156`.
- Baseline comparison: committed isolated `dX` baseline was `mean_ms=799.156`, so this is `2.90x` faster and reduces latency by `65.5%`.
- Notes: This validates that the dominant `dX` bottleneck was the atomics/scattered ownership pattern, not the unavoidable matrix multiply count alone.

## Iteration 3 - GB200 CSR dX tile tuning

- Target: `block_sparse_conv_1d_bwd_dx_col_csr_kernel`
- Hypothesis: The old `BLOCK_N_DX=64` default was tuned around the atomic kernel. The CSR kernel owns one `dx` tile and benefits from larger time tiles.
- Sweep: `BLOCK_N_DX=32` gave `542.045 ms`, `128` gave `187.171 ms`, `256` gave `168.717 ms`, and `512` gave `189.361 ms` in 3-trial probes.
- Change: Updated the GB200/B200 backend default from `BLOCK_N_DX=64` to `BLOCK_N_DX=256`.
- Correctness: `python benchmarks/rapid_io_conv_kernel_benchmark.py --target dx --correctness --correct-time-steps 16` passed with `max_abs=0.015386581420898438`, `atol=0.02`.
- Runtime: `python benchmarks/rapid_io_conv_kernel_benchmark.py --target dx --trials 5 --warmup 2` reported `mean_ms=168.7881652832031`, `min_ms=168.66015625`.
- Baseline comparison: committed isolated `dX` baseline was `mean_ms=799.156`, so the current default is `4.73x` faster and reduces latency by `78.9%`.
- Notes: A100/H100 defaults were intentionally left unchanged because this sweep was measured only on GB200.
