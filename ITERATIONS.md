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

## Iteration 4 - dW time reduction without atomics

- Target: `block_sparse_conv_1d_bwd_dvalues_kernel`
- Hypothesis: The weight-gradient kernel atomically accumulates each block/tap across time tiles. For the RAPID workload `B=1`, one program can own one sparse block and one kernel tap, reduce over all time tiles internally, and store the result once.
- Change: Added `block_sparse_conv_1d_bwd_dvalues_time_reduce_kernel` for `B == 1`. The previous atomic kernel remains the fallback for larger batches. The benchmark harness now treats optional `dX` metadata as optional for local weight-gradient block views.
- Correctness: `python benchmarks/rapid_io_conv_kernel_benchmark.py --target dw --correctness --correct-time-steps 16` passed with `max_abs=1.1313386494293809e-07`, `atol=0.02`.
- Runtime: `python benchmarks/rapid_io_conv_kernel_benchmark.py --target dw --trials 5 --warmup 2` reported `mean_ms=534.3837768554688`, `min_ms=534.3588256835938`.
- Baseline comparison: committed isolated `dW` baseline was `mean_ms=663.871`, so this is `1.24x` faster and reduces latency by `19.5%`.
- Notes: The improvement is smaller than `dX` because each program now performs a long time reduction with a 16x16 accumulator.

## Iteration 5 - GB200 dW time-reduction tile tuning

- Target: `block_sparse_conv_1d_bwd_dvalues_time_reduce_kernel`
- Hypothesis: The new time-reduction kernel has a different optimum than the old atomic kernel. Smaller `BLOCK_N_DW` can improve occupancy and reduce accumulator pressure even though it increases the number of internal time chunks.
- Sweep: `BLOCK_N_DW=32` gave `516.653 ms`, `64` gave `484.675 ms`, `128` gave `491.929 ms`, `256` gave `534.384 ms`, and `512` gave `748.623 ms`.
- Change: Updated the GB200/B200 backend default from `BLOCK_N_DW=256` to `BLOCK_N_DW=64`.
- Correctness: `python benchmarks/rapid_io_conv_kernel_benchmark.py --target dw --correctness --correct-time-steps 16` passed with `max_abs=1.1313386494293809e-07`, `atol=0.02`.
- Runtime: `python benchmarks/rapid_io_conv_kernel_benchmark.py --target dw --trials 5 --warmup 2` reported `mean_ms=484.6630493164063`, `min_ms=484.6455993652344`.
- Baseline comparison: committed isolated `dW` baseline was `mean_ms=663.871`, so the current default is `1.37x` faster and reduces latency by `27.0%`.
- Notes: This setting is GB200-specific until the same sweep is run on A100/H100.

## Iteration 6 - Fused aggregation postprocess backward

- Target: aggregation backward through ReLU, triangular downsampling, flip, and normalization.
- Hypothesis: Nsight Systems showed the largest aggregation-backward kernel was cuDNN grouped convolution input-gradient from the triangular downsampler, about 64 ms. A custom Triton backward can fuse normalization, flip, triangular downsample backward, and ReLU masking and avoid the cuDNN grouped-conv path.
- Change: Added `_DownTriReluFlipNormalize` in `diffroute/agg/temporal_sampler.py` and routed CUDA avg-mode aggregation post-processing through it. Non-CUDA and non-avg modes keep the existing PyTorch sequence.
- Correctness: Existing full correctness gate passed with `python benchmarks/rapid_io_benchmark.py --correctness --correct-time-steps 16 --atol 2e-2`. An explicit reference comparison against the old PyTorch post-processing sequence matched forward exactly and gave random-upstream parameter-gradient RMS error about `1.05e-4`.
- Runtime: `T=500` full benchmark reported `substep.aggregation.backward.dL_dparams mean_ms=132.30342610677084`, median `124.12153625488281`; focused 7-trial aggregation timing reported mean `143.90956115722656`, median `140.2414093017578`, min `140.13885498046875`.
- Baseline comparison: previous `T=500` aggregation backward was about `190.248 ms`, so the focused median is `1.36x` faster and the best full-benchmark median is about `1.51x` faster.
- Notes: This optimizes the dominant post-processing kernel but leaves closure/prefix and FFT backward costs in place.

## Iteration 7 - Prefix-sum backward via saved pointer jumps

- Target: `prefix_sum` backward inside frequency-domain aggregation.
- Hypothesis: Prefix forward already uses pointer jumping, but backward pushed gradients one graph edge per launch and needed about 357 launches for RAPID depth. Saving the forward jump tables and reversing those rounds should reduce the backward to the pointer-jump depth, about 9 launches.
- Change: `PrefixSum.forward` now saves the per-round jump tables. `PrefixSum.backward` walks those tables in reverse, cloning the self contribution and using the existing Triton push kernel to propagate one jump distance per round.
- Correctness: Exact small-graph checks against a PyTorch reference passed with backward max error below `8e-6`. The full RAPID correctness gate passed with `python benchmarks/rapid_io_benchmark.py --correctness --correct-time-steps 16 --atol 2e-2`.
- Runtime: focused 7-trial aggregation backward timing reported mean `47.76411383492606`, median `44.222496032714844`, min `44.10697555541992`. `T=500` full benchmark reported `full.backward.dL_dparams mean_ms=43.57938130696615`, median `37.97795104980469`, and `substep.aggregation.backward.dL_dparams mean_ms=34.24970626831055`, median `26.134239196777344`.
- Baseline comparison: before aggregation optimization, `T=500` full `dL/dparams` was about `200.596 ms`; after this iteration it is about `37.978 ms` median, a `5.28x` full-path speedup for short sequences.
- Notes: The remaining aggregation backward cost is now split among closure backward, FFTs, and tensor elementwise/copy overhead rather than the previous depth-level prefix propagation.
