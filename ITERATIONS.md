# DiffRoute RAPID IO Kernel Optimization Iterations

## Iteration 1 - Column-grouped dX atomics

- Target: `block_sparse_conv_1d_bwd_dx_kernel`
- Hypothesis: RAPID block indices are row-grouped for forward, but almost ungrouped by input block for `dL/dX`. Sorting by input block and reducing same-column contributions inside each Triton program should reduce atomic traffic.
- Change: Added a `BlockSparseKernel.block_col_order` permutation and a `block_sparse_conv_1d_bwd_dx_col_grouped_kernel` that consumes that order while keeping the original values tensor layout.
- Correctness: `python benchmarks/rapid_io_conv_kernel_benchmark.py --target dx --correctness --correct-time-steps 16` passed with `max_abs=0.015270233154296875`, `atol=0.02`.
- Runtime: `python benchmarks/rapid_io_conv_kernel_benchmark.py --target dx --trials 5 --warmup 2` reported `mean_ms=767.6581909179688`, `min_ms=767.6215209960938`.
- Baseline comparison: committed isolated `dX` baseline was `mean_ms=799.156`, so this is `3.9%` faster.
- Notes: The win is real but modest, which means atomics are not the only cost; the kernel still performs the same dot work and now gathers values through a column order.
