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

## Iteration 8 - Forward prefix fixed jump rounds

- Target: prefix-sum forward inside frequency-domain aggregation.
- Hypothesis: The RAPID graph has a known pointer-jump depth of 9 rounds, but `_prefix_jump_fwd` checked `(e_run < 0).all()` each round. Passing the precomputed graph depth and removing that device-wide completion check should reduce prefix overhead without changing the aggregated IRFs.
- Change: `RivTree` now stores `prefix_jump_rounds`, and `aggregate_irf` threads it through `log_transitive_closure` and `prefix_sum`. `_prefix_jump_fwd` runs the requested fixed number of rounds and no longer performs the per-round `.all()` check.
- Correctness: `python benchmarks/rapid_io_benchmark.py --correctness --correct-time-steps 16 --atol 2e-2` passed with `conv_forward_max_abs_vs_torch=6.59783836454153e-08`, `conv_backward_dx_max_abs_vs_torch=0.0024261474609375`, and finite parameter gradients.
- Runtime: `python benchmarks/rapid_io_benchmark.py --time-steps 500 --trials 5 --warmup 2 --backward-trials 1 --backward-warmup 0` reported `full.forward mean_ms=27.860505294799804`, median `23.363359451293945`, min `23.1693115234375`; `substep.aggregation.forward mean_ms=21.941305923461915`, median `17.307231903076172`. Focused aggregation breakdown reported `prefix_sum.current_autograd` median `0.547 ms` versus the baseline `0.916 ms`.
- Baseline comparison: baseline `T=500` full forward median was `24.058048248291016 ms`, so this is `1.03x` faster and reduces median latency by `2.9%`.
- Notes: This confirms the synchronization overhead was real but not the dominant forward cost; `irfft` and temporal sampling still dominate uncached aggregation.

## Iteration 9 - Invalid cached static forward routing kernels

- Target: full forward inference path for fixed graph and detached routing parameters.
- Hypothesis: RAPID forward benchmarks and inference-style routing reuse the same graph and non-gradient parameter tensor across runoff evaluations. Recomputing aggregation and sparse-to-block conversion every call costs about `19 ms` median at `T=500`; caching the block-sparse routing kernel behind a conservative key should reduce full forward to the sparse convolution cost while leaving parameter-gradient paths unchanged.
- Change: `LTIRouter` now caches the block-sparse kernel when `params.requires_grad` is false. The cache key includes graph identity, topology buffer pointers and versions, params pointer/version/shape/stride/device, target device, and router aggregation/blockization settings. `params.requires_grad=True` bypasses the cache so differentiable parameter aggregation still builds a fresh autograd graph. Added `clear_kernel_cache()` for explicit invalidation.
- Correctness: `python benchmarks/rapid_io_benchmark.py --correctness --correct-time-steps 16 --atol 2e-2` passed with `full_repeat_max_abs=4.3655745685100555e-11`, `conv_forward_max_abs_vs_torch=6.596383173018694e-08`, `conv_backward_dx_max_abs_vs_torch=0.0024261474609375`, and finite parameter gradients.
- Runtime: `python benchmarks/rapid_io_benchmark.py --time-steps 500 --trials 5 --warmup 2 --backward-trials 1 --backward-warmup 0` reported `full.forward mean_ms=4.207212829589844`, median `4.197760105133057`, min `4.177120208740234`. Substep timings remained `substep.aggregation.forward median=17.38025665283203`, `substep.blockize.forward median=1.8997440338134766`, and `substep.convolution.forward median=4.172512054443359`.
- Baseline comparison: this apparent speedup is invalid for the intended model semantics because the routing kernel must be recomputed every forward call.
- Notes: Reverted in iteration 10; do not use this entry as a valid forward optimization result.

## Iteration 10 - Remove invalid static kernel cache

- Target: `LTIRouter.forward`.
- Hypothesis: Correct forward semantics require recomputing the routing kernel every call, so the static cache from iteration 9 is not valid even though it improved inference-style timings.
- Change: Removed `LTIRouter`'s block-sparse kernel cache, cache key, and explicit cache invalidation method. `forward()` again calls `self.aggregator(g, params).to(x.device)` and `to_block_sparse()` every time before convolution.
- Correctness: `python benchmarks/rapid_io_benchmark.py --correctness --correct-time-steps 16 --atol 2e-2` passed with `full_repeat_max_abs=2.9103830456733704e-11`, `conv_forward_max_abs_vs_torch=6.59783836454153e-08`, `conv_backward_dx_max_abs_vs_torch=0.0024261474609375`, and finite parameter gradients.
- Runtime: `python benchmarks/rapid_io_benchmark.py --time-steps 500 --trials 5 --warmup 2 --backward-trials 1 --backward-warmup 0` reported `full.forward mean_ms=27.896319961547853`, median `23.327775955200195`, min `23.26211166381836`; `substep.aggregation.forward median=17.40790367126465`, `substep.blockize.forward median=1.901792049407959`, and `substep.convolution.forward median=4.169600009918213`.
- Baseline comparison: this returns to the iteration 8 behavior, where the valid forward improvement is fixed prefix jump rounds. Compared with the original `T=500` baseline median `24.058048248291016 ms`, the valid current median `23.327775955200195 ms` is `1.03x` faster and reduces latency by `3.0%`.
- Notes: This keeps the valid prefix-round optimization and removes only the invalid static-kernel caching path.

## Iteration 11 - Fused temporal sampler forward

- Target: aggregation forward postprocess through ReLU, triangular downsampling, flip, and normalization.
- Hypothesis: The forward postprocess was still a sequence of separate PyTorch/cuDNN kernels over `irfs_agg` shaped `[1653119, 768]`. A Triton kernel can compute the final normalized 32-tap flipped output directly and avoid the large intermediate ReLU tensor plus the separate flip/sum/normalize passes.
- Change: Added `_down_tri_relu_flip_norm_fwd_kernel` and routed `_DownTriReluFlipNormalize.forward` through it for CUDA 2D inputs. The existing custom backward still consumes the saved input, normalized output, and denominator, so parameter-gradient semantics are unchanged.
- Correctness: `python benchmarks/rapid_io_benchmark.py --correctness --correct-time-steps 16 --atol 2e-2` passed with `full_repeat_max_abs=2.9103830456733704e-11`, `conv_forward_max_abs_vs_torch=6.596383173018694e-08`, `conv_backward_dx_max_abs_vs_torch=0.0024261474609375`, and finite parameter gradients.
- Runtime: `python benchmarks/rapid_io_benchmark.py --time-steps 500 --trials 5 --warmup 2 --backward-trials 1 --backward-warmup 0` reported `full.forward mean_ms=25.819981002807616`, median `21.136159896850586`, min `21.001920700073242`; `substep.aggregation.forward mean_ms=19.741049194335936`, median `15.212384223937988`. Focused aggregation breakdown reported `temporal_sampler.total` median `2.469 ms`, down from the valid pre-iteration median `4.639 ms`.
- Baseline comparison: original `T=500` baseline full forward median was `24.058048248291016 ms`; current valid median is `21.136159896850586 ms`, a `1.14x` speedup and `12.1%` latency reduction. Compared with iteration 10 median `23.327775955200195 ms`, this iteration reduces valid full-forward latency by `9.4%`.
- Notes: This removes roughly half of temporal postprocess latency, but aggregation is now dominated by `irfft` and the large frequency-to-time materialization.

## Iteration 12 - Fused closure and complex exponential no-grad path

- Target: no-gradient aggregation forward between prefix-sum and `irfft`.
- Hypothesis: The existing path materializes flattened log-frequency closure values shaped `[1653119, 770]`, then reads them again in `exp_complex`. A forward-only Triton path that enumerates closure rows and writes complex frequency pairs directly should reduce memory traffic while preserving the differentiable path for parameter-gradient runs.
- Change: Added `_coo_enum_exp_kernel`, `closure_sub_exp`, and `log_transitive_closure_no_grad`. `aggregate_irf` uses this no-gradient path only when `irfs_freq.requires_grad` is false; parameter-gradient runs still use `closure_sub` plus `exp_complex`.
- Correctness: A direct comparison against the existing closure+exp path gave `coords_equal=True`, `freq_max_abs=1.6858739115832577e-07`, and postprocess `max_abs=3.5762786865234375e-07`. `python benchmarks/rapid_io_benchmark.py --correctness --correct-time-steps 16 --atol 2e-2` passed with `conv_forward_max_abs_vs_torch=6.599293556064367e-08`, `conv_backward_dx_max_abs_vs_torch=0.0024261474609375`, and finite parameter gradients.
- Runtime: `python benchmarks/rapid_io_benchmark.py --time-steps 500 --trials 5 --warmup 2 --backward-trials 1 --backward-warmup 0` reported `full.forward mean_ms=25.902253341674804`, median `21.44825553894043`, min `21.388704299926758`; `substep.aggregation.forward mean_ms=19.839289283752443`, median `15.391263961791992`.
- Baseline comparison: this is slower than iteration 11 (`21.448 ms` vs `21.136 ms` median), so the fused closure+exp kernel is not a win in its current form. It remains faster than the original baseline only because iteration 11's fused sampler is still present.
- Notes: The custom Triton exp/sincos work and per-row path loop do not beat PyTorch's separate closure and complex exponential kernels here. This path should be reverted unless a later tuning iteration makes it faster.

## Iteration 13 - Forward aggregation block_f 512 default

- Target: valid recompute-every-call forward aggregation.
- Hypothesis: The fused closure+exp path from iteration 12 needs a wider feature tile to amortize per-path overhead. Probes showed `block_f=512` improved no-gradient forward without the severe `dL/dparams` slowdown seen at `block_f=1024`.
- Change: Updated the model and RAPID benchmark defaults from `block_f=128` to `512`.
- Correctness: `python benchmarks/rapid_io_benchmark.py --correctness --correct-time-steps 16 --atol 2e-2` passed with `full_repeat_max_abs=4.3655745685100555e-11`, `conv_forward_max_abs_vs_torch=6.596383173018694e-08`, `conv_backward_dx_max_abs_vs_torch=0.0024261474609375`, and finite parameter gradients.
- Runtime: `python benchmarks/rapid_io_benchmark.py --time-steps 500 --trials 7 --warmup 2 --backward-trials 1 --backward-warmup 0` reported `full.forward mean_ms=22.7772159576416`, median `19.28179168701172`, min `19.23075294494629`; `substep.aggregation.forward mean_ms=16.790463992527553`, median `13.285663604736328`; `substep.convolution.forward median=4.171520233154297`. Focused aggregation breakdown reported `aggregator.forward.total` median `13.241 ms`, `aggregate_irf.pre_sampler` median `10.939 ms`, and `temporal_sampler.total` median `2.466 ms`.
- Baseline comparison: original `T=500` baseline full forward median was `24.058048248291016 ms`; current valid median is `19.28179168701172 ms`, a `1.25x` speedup and `19.9%` latency reduction. Compared with iteration 11 median `21.136159896850586 ms`, this is another `8.8%` faster.
- Notes: The remaining dominant cost is still `irfft` at about `7.53 ms` median, followed by sparse convolution at about `4.17 ms`.

## Iteration 14 - Forward sparse-conv one-block programs

- Target: `block_sparse_conv_1d_fwd_kernel` launch grouping.
- Hypothesis: The forward kernel grouped 16 nonzero blocks per Triton program to reduce atomics, but RAPID's row-grouped block order and GB200 occupancy favor more parallel one-block programs. A launch-only sweep found `NZB_BLOCK_SIZE=1` gave isolated forward median `3.51 ms` versus about `4.15 ms` for the previous `16`.
- Change: Changed the default `NZB_BLOCK_SIZE` for `block_sparse_conv_1d_forward`, `BlockSparseConv1dFn.forward`, and `block_sparse_conv_1d` from `16` to `1`.
- Correctness: `python benchmarks/rapid_io_benchmark.py --correctness --correct-time-steps 16 --atol 2e-2` passed with `conv_forward_max_abs_vs_torch=6.602203939110041e-08`, `conv_backward_dx_max_abs_vs_torch=0.0024261474609375`, `conv_backward_dw_max_abs_vs_torch=5.21540641784668e-08`, and finite parameter gradients.
- Runtime: `python benchmarks/rapid_io_benchmark.py --time-steps 500 --trials 7 --warmup 2 --backward-trials 1 --backward-warmup 0` reported `full.forward mean_ms=22.123853955950057`, median `18.609600067138672`, min `18.58483123779297`; `substep.aggregation.forward median=13.27990436553955`, `substep.blockize.forward median=1.8888319730758667`, and `substep.convolution.forward median=3.524319887161255`. The isolated convolution benchmark reported `mean_ms=3.5101919889450075`, median `3.510767936706543`.
- Baseline comparison: original `T=500` baseline full forward median was `24.058048248291016 ms`; current valid median is `18.609600067138672 ms`, a `1.29x` speedup and `22.6%` latency reduction. Compared with iteration 13, this reduces full-forward median by `3.5%`.
- Notes: This is a forward launch-shape win. The optimized RAPID `dX` and `dW` paths do not use this grouping in their hot kernels.

## Iteration 15 - Cache blockization topology metadata

- Target: sparse COO-to-block-sparse conversion after per-call aggregation.
- Hypothesis: Kernel values must be recomputed every forward call, but the COO-to-block packing map is fixed graph topology. Reusing only block indices, linear scatter indices, and column metadata should remove repeated `torch.unique`/mapping work while still building fresh `block_values` from current IRF values every call.
- Change: `SparseKernel` now carries optional block metadata. `BlockSparseKernel.make_block_metadata` precomputes block indices, linear scatter indices, column order, and column offsets. `IRFAggregator.forward` caches this topology metadata on the graph by block size, kernel size, and device, then returns it with each freshly computed `SparseKernel`. `BlockSparseKernel.from_coo` uses the metadata when present but still scatters the current values into a new `block_values` tensor.
- Correctness: `python benchmarks/rapid_io_benchmark.py --correctness --correct-time-steps 16 --atol 2e-2` passed with `conv_forward_max_abs_vs_torch=6.600748747587204e-08`, `conv_backward_dx_max_abs_vs_torch=0.0024261474609375`, `conv_backward_dw_max_abs_vs_torch=5.21540641784668e-08`, and finite parameter gradients.
- Runtime: `python benchmarks/rapid_io_benchmark.py --time-steps 500 --trials 7 --warmup 2 --backward-trials 1 --backward-warmup 0` reported `full.forward mean_ms=20.94447980608259`, median `17.429119110107422`, min `17.390111923217773`; `substep.aggregation.forward median=13.304703712463379`, `substep.blockize.forward median=0.7131839990615845`, and `substep.convolution.forward median=3.5379838943481445`.
- Baseline comparison: original `T=500` baseline full forward median was `24.058048248291016 ms`; current valid median is `17.429119110107422 ms`, a `1.38x` speedup and `27.6%` latency reduction. Compared with iteration 14, this reduces full-forward median by `6.3%`.
- Notes: This is not a routing-kernel value cache. It caches only graph-topology packing metadata and still recomputes/scatters current per-call kernel values.

## Iteration 16 - Direct no-grad sampler to block values

- Target: detached-parameter router forward after `irfft`.
- Hypothesis: In no-gradient forward, the router immediately converts sampled sparse path values into block-sparse values. The sampled `[paths, 32]` tensor and separate blockize scatter can be avoided by writing the fused temporal sampler output directly into freshly allocated block-sparse value storage using the cached topology scatter indices.
- Change: Added `_down_tri_relu_flip_norm_block_fwd_kernel` and `SubResolutionSampler.kernel_postprocess_block_values`. Added `IRFAggregator.block_sparse_forward`, which recomputes aggregation values every call, uses cached topology metadata, and fills a new `BlockSparseKernel` directly. `LTIRouter.forward` uses this direct block-sparse path only when `params.requires_grad` is false; parameter-gradient runs keep the differentiable `SparseKernel -> to_block_sparse` path.
- Correctness: `python benchmarks/rapid_io_benchmark.py --correctness --correct-time-steps 16 --atol 2e-2` passed with `conv_forward_max_abs_vs_torch=6.600748747587204e-08`, `conv_backward_dx_max_abs_vs_torch=0.0024261474609375`, `conv_backward_dw_max_abs_vs_torch=5.21540641784668e-08`, and finite parameter gradients.
- Runtime: `python benchmarks/rapid_io_benchmark.py --time-steps 500 --trials 7 --warmup 2 --backward-trials 1 --backward-warmup 0` reported `full.forward mean_ms=20.28672899518694`, median `16.782751083374023`, min `16.73846435546875`. The direct router path is not reflected by `substep.blockize.forward`, which still times `SparseKernel.to_block_sparse()` separately; substep medians were `aggregation.forward=13.339167594909668`, `blockize.forward=0.711135983467102`, and `convolution.forward=3.5332159996032715`.
- Baseline comparison: original `T=500` baseline full forward median was `24.058048248291016 ms`; current valid median is `16.782751083374023 ms`, a `1.43x` speedup and `30.2%` latency reduction. Compared with iteration 15, this reduces full-forward median by `3.7%`.
- Notes: This still recomputes per-call routing values. It only fuses the no-gradient sampled-value materialization with the block-sparse scatter.
