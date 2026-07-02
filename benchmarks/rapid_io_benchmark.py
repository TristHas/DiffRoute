#!/usr/bin/env python
"""Benchmark DiffRoute on the DiffHydro RAPID IO example workload.

The script prepares the VPU 305 RAPID graph and reach-aligned runoff used by
DiffHydro's `examples/1. RAPID IO.ipynb`, then measures:

* full router forward
* full router backward wrt runoff X
* full router backward wrt routing parameters
* aggregation, blockization, and sparse-convolution substeps
* sparse-convolution backward wrt X and wrt block-sparse weights

It is intentionally independent of notebooks so optimized kernels can be
checked and timed from CI, shell scripts, or an optimization loop.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import statistics
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable

import pandas as pd
import torch
import xtensor as xt

import diffhydro as dh
from diffroute import LTIRouter
from diffroute.backend import select_conv_config
from diffroute.conv.block_sparse_conv import conv1d_block_sparse
from diffroute.io import read_rapid_graph
from diffroute.structs import BlockSparseKernel, SparseKernel


@dataclass
class Timing:
    name: str
    mean_ms: float
    median_ms: float
    min_ms: float
    max_ms: float
    std_ms: float
    trials: int
    peak_allocated_gib: float
    peak_reserved_gib: float


def _stats(name: str, times: list[float], device: torch.device) -> Timing:
    peak_alloc = torch.cuda.max_memory_allocated(device) / 1024**3
    peak_reserved = torch.cuda.max_memory_reserved(device) / 1024**3
    return Timing(
        name=name,
        mean_ms=statistics.mean(times),
        median_ms=statistics.median(times),
        min_ms=min(times),
        max_ms=max(times),
        std_ms=statistics.stdev(times) if len(times) > 1 else 0.0,
        trials=len(times),
        peak_allocated_gib=peak_alloc,
        peak_reserved_gib=peak_reserved,
    )


def time_cuda(
    name: str,
    fn: Callable[[], object],
    *,
    device: torch.device,
    warmup: int,
    trials: int,
) -> Timing:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize(device)
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)

    times: list[float] = []
    for _ in range(trials):
        torch.cuda.synchronize(device)
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        fn()
        end.record()
        torch.cuda.synchronize(device)
        times.append(start.elapsed_time(end))
    return _stats(name, times, device)


def time_host(
    name: str,
    fn: Callable[[], object],
    *,
    device: torch.device,
    warmup: int,
    trials: int,
) -> Timing:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize(device)
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)

    times: list[float] = []
    for _ in range(trials):
        torch.cuda.synchronize(device)
        start = time.perf_counter()
        fn()
        torch.cuda.synchronize(device)
        times.append((time.perf_counter() - start) * 1000.0)
    return _stats(name, times, device)


def default_examples_dir() -> Path:
    return Path(__file__).resolve().parents[2] / "DiffHydro" / "examples"


def ensure_example_data(examples_dir: Path) -> None:
    """Download the two small RAPID IO inputs if the notebook already did not."""
    sys.path.insert(0, str(examples_dir))
    from utils.download import ZENODO_RECORD, download_rapid_config, download_zenodo_file

    root = examples_dir / "data"
    runoff_path = root / "geoglows" / "input" / "305_daily_sparse_runoff.feather"
    interp_path = root / "geoglows" / "input" / "305_interp_weight.feather"

    if not runoff_path.exists():
        download_zenodo_file(
            f"https://zenodo.org/records/{ZENODO_RECORD}/files/305_daily_sparse_runoff.feather",
            str(runoff_path),
        )
    if not interp_path.exists():
        download_zenodo_file(
            f"https://zenodo.org/records/{ZENODO_RECORD}/files/305_interp_weight.feather",
            str(interp_path),
        )
    download_rapid_config(root, 305)


def load_problem(args: argparse.Namespace, device: torch.device):
    examples_dir = args.examples_dir.resolve()
    if args.download:
        ensure_example_data(examples_dir)

    root = examples_dir / "data"
    rapid_path = root / "geoglows" / "rapid_config" / "305" / "configs" / "305"
    runoff_path = root / "geoglows" / "input" / "305_daily_sparse_runoff.feather"
    interp_path = root / "geoglows" / "input" / "305_interp_weight.feather"

    missing = [p for p in (rapid_path, runoff_path, interp_path) if not p.exists()]
    if missing:
        paths = "\n".join(f"  - {p}" for p in missing)
        raise FileNotFoundError(f"Missing RAPID IO inputs:\n{paths}\nRun with --download first.")

    graph = read_rapid_graph(rapid_path).to(device)

    pixel_runoff = xt.read_feather(runoff_path, dims=("time", "spatial"))
    if args.time_steps is not None:
        pixel_runoff = pixel_runoff.isel(time=slice(0, args.time_steps))
    pixel_runoff = (
        pixel_runoff.expand_dims("batch")
        .transpose("batch", "spatial", "time")
        .to(device)
        / (3600.0 * 24.0)
    )

    interp_df = pd.read_feather(interp_path)
    if "river_id" in interp_df.columns:
        interp_df = interp_df.set_index("river_id")

    interpolator = dh.CatchmentInterpolator(graph, pixel_runoff, interp_df).to(device)
    runoff = interpolator(pixel_runoff).values.contiguous()
    return graph, runoff


def make_router(args: argparse.Namespace) -> LTIRouter:
    router = LTIRouter(
        max_delay=args.max_delay,
        dt=args.dt,
        block_size=args.block_size,
        block_f=args.block_f,
        conv_imp=args.conv_impl,
        block_n=args.block_n,
        block_n_dx=args.block_n_dx,
        block_n_dw=args.block_n_dw,
    )
    return router


def build_kernel(router: LTIRouter, graph, params: torch.Tensor):
    sparse = router.aggregator(graph, params)
    block = sparse.to_block_sparse(router.block_size)
    return sparse, block


def profile(args: argparse.Namespace) -> dict:
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    torch.cuda.set_device(device)
    graph, runoff = load_problem(args, device)
    router = make_router(args).to(device)
    timer = time_host if args.host_time else time_cuda

    params = graph.params.detach().contiguous()
    sparse, block = build_kernel(router, graph, params)
    block_values = block.block_values.detach().contiguous()
    block_indices = block.block_indices.detach().contiguous()

    # Compile and stabilize the main Triton kernels before measuring substeps.
    router(runoff, graph, params)
    router.conv(runoff, block)
    torch.cuda.synchronize(device)

    results: list[Timing] = []
    results.append(
        timer(
            "full.forward",
            lambda: router(runoff, graph, params),
            device=device,
            warmup=args.warmup,
            trials=args.trials,
        )
    )

    x_req = runoff.detach().clone().requires_grad_(True)
    y_dx = router(x_req, graph, params)
    loss_dx = y_dx.sum()
    results.append(
        timer(
            "full.backward.dL_dX",
            lambda: torch.autograd.grad(loss_dx, x_req, retain_graph=True)[0],
            device=device,
            warmup=args.backward_warmup,
            trials=args.backward_trials,
        )
    )

    p_req = params.detach().clone().requires_grad_(True)
    y_dp = router(runoff.detach(), graph, p_req)
    loss_dp = y_dp.sum()
    results.append(
        timer(
            "full.backward.dL_dparams",
            lambda: torch.autograd.grad(loss_dp, p_req, retain_graph=True)[0],
            device=device,
            warmup=args.backward_warmup,
            trials=args.backward_trials,
        )
    )

    results.append(
        timer(
            "substep.aggregation.forward",
            lambda: router.aggregator(graph, params),
            device=device,
            warmup=args.warmup,
            trials=args.trials,
        )
    )
    results.append(
        timer(
            "substep.blockize.forward",
            lambda: sparse.to_block_sparse(router.block_size),
            device=device,
            warmup=args.warmup,
            trials=args.trials,
        )
    )
    results.append(
        timer(
            "substep.convolution.forward",
            lambda: router.conv(runoff, block),
            device=device,
            warmup=args.warmup,
            trials=args.trials,
        )
    )

    p_agg = params.detach().clone().requires_grad_(True)
    sparse_for_grad = router.aggregator(graph, p_agg)
    agg_loss = sparse_for_grad.vals.sum()
    results.append(
        timer(
            "substep.aggregation.backward.dL_dparams",
            lambda: torch.autograd.grad(agg_loss, p_agg, retain_graph=True)[0],
            device=device,
            warmup=args.backward_warmup,
            trials=args.backward_trials,
        )
    )

    vals_for_block = sparse.vals.detach().clone().requires_grad_(True)
    sparse_for_block = SparseKernel(sparse.coords.detach(), vals_for_block, sparse.size)
    block_for_grad = sparse_for_block.to_block_sparse(router.block_size)
    blockize_loss = block_for_grad.block_values.sum()
    results.append(
        timer(
            "substep.blockize.backward.dL_dSparseVals",
            lambda: torch.autograd.grad(blockize_loss, vals_for_block, retain_graph=True)[0],
            device=device,
            warmup=args.backward_warmup,
            trials=args.backward_trials,
        )
    )

    x_conv = runoff.detach().clone().requires_grad_(True)
    y_conv_dx = router.conv(x_conv, block)
    conv_dx_loss = y_conv_dx.sum()
    results.append(
        timer(
            "substep.convolution.backward.dL_dX",
            lambda: torch.autograd.grad(conv_dx_loss, x_conv, retain_graph=True)[0],
            device=device,
            warmup=args.backward_warmup,
            trials=args.backward_trials,
        )
    )

    vals_conv = block_values.detach().clone().requires_grad_(True)
    y_conv_dw = block_sparse_call(
        runoff.detach(),
        block_indices,
        vals_conv,
        block.size,
        block.block_size,
        args.block_n,
        args.conv_impl,
        args.block_n_dx,
        args.block_n_dw,
        block.block_col_order,
        block.block_col_offsets,
    )
    conv_dw_loss = y_conv_dw.sum()
    results.append(
        timer(
            "substep.convolution.backward.dL_dW",
            lambda: torch.autograd.grad(conv_dw_loss, vals_conv, retain_graph=True)[0],
            device=device,
            warmup=args.backward_warmup,
            trials=args.backward_trials,
        )
    )

    metadata = {
        "device": torch.cuda.get_device_name(device),
        "device_capability": torch.cuda.get_device_capability(device),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "nodes": len(graph),
        "paths": int(graph.path_cumsum[-1].item()),
        "runoff_shape": list(runoff.shape),
        "params_shape": list(params.shape),
        "sparse_kernel_values": int(sparse.vals.shape[0]),
        "block_sparse_blocks": int(block.block_indices.shape[0]),
        "block_values_shape": list(block.block_values.shape),
        "max_delay": args.max_delay,
        "dt": args.dt,
        "block_size": args.block_size,
        "block_n": args.block_n,
        "block_n_dx": args.block_n_dx,
        "block_n_dw": args.block_n_dw,
        "block_f": args.block_f,
        "conv_impl": args.conv_impl,
    }
    payload = {"metadata": metadata, "timings": [asdict(x) for x in results]}
    write_outputs(args, payload)
    return payload


def block_sparse_call(
    x: torch.Tensor,
    block_indices: torch.Tensor,
    block_values: torch.Tensor,
    kernel_size,
    block_size: int,
    block_n: int,
    conv_impl: str,
    block_n_dx: int | None = None,
    block_n_dw: int | None = None,
    dx_block_order: torch.Tensor | None = None,
    dx_block_col_offsets: torch.Tensor | None = None,
) -> torch.Tensor:
    config = select_conv_config(
        device=x.device,
        impl=conv_impl,
        block_n=block_n,
        block_n_dx=block_n_dx,
        block_n_dw=block_n_dw,
    )
    if config.impl == "triton":
        from diffroute.ops.conv import block_sparse_conv_1d

        return block_sparse_conv_1d(
            x,
            block_indices,
            block_values,
            kernel_size,
            block_size,
            config.block_n,
            BLOCK_SIZE_N_DX=config.block_n_dx,
            BLOCK_SIZE_N_DVALUES=config.block_n_dw,
            DX_BLOCK_ORDER=dx_block_order,
            DX_BLOCK_COL_OFFSETS=dx_block_col_offsets,
        )
    cols = block_indices[:, 1]
    rows = block_indices[:, 0]
    return conv1d_block_sparse(x, block_values, cols, rows)


def max_abs_rel(a: torch.Tensor, b: torch.Tensor) -> tuple[float, float]:
    diff = (a - b).abs()
    max_abs = float(diff.max().item())
    denom = b.abs().clamp_min(1e-8)
    max_rel = float((diff / denom).max().item())
    return max_abs, max_rel


def correctness(args: argparse.Namespace) -> dict:
    """Correctness checks for current and future optimized implementations.

    The default compares Triton sparse convolution with the pure-PyTorch block
    sparse fallback on the RAPID-derived graph and a short time slice. Full
    router determinism and gradient finiteness are also checked.
    """
    args.time_steps = args.correct_time_steps
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    torch.cuda.set_device(device)
    graph, runoff = load_problem(args, device)
    router = make_router(args).to(device)
    params = graph.params.detach().contiguous()
    sparse, block = build_kernel(router, graph, params)

    with torch.no_grad():
        out_a = router(runoff, graph, params)
        out_b = router(runoff, graph, params)
    full_abs, full_rel = max_abs_rel(out_a, out_b)

    with torch.no_grad():
        y_triton = block_sparse_call(
            runoff,
            block.block_indices,
            block.block_values,
            block.size,
            block.block_size,
            args.block_n,
            "triton",
            args.block_n_dx,
            args.block_n_dw,
            block.block_col_order,
            block.block_col_offsets,
        )
        y_torch = block_sparse_call(
            runoff,
            block.block_indices,
            block.block_values,
            block.size,
            block.block_size,
            args.block_n,
            "torch",
            args.block_n_dx,
            args.block_n_dw,
        )
    conv_abs, conv_rel = max_abs_rel(y_triton, y_torch)

    x_t = runoff.detach().clone().requires_grad_(True)
    vals_t = block.block_values.detach().clone().requires_grad_(True)
    y_t = block_sparse_call(
        x_t,
        block.block_indices,
        vals_t,
        block.size,
        block.block_size,
        args.block_n,
        "triton",
        args.block_n_dx,
        args.block_n_dw,
        block.block_col_order,
        block.block_col_offsets,
    )
    dx_t, dw_t = torch.autograd.grad(y_t.sum(), (x_t, vals_t))

    x_r = runoff.detach().clone().requires_grad_(True)
    vals_r = block.block_values.detach().clone().requires_grad_(True)
    y_r = block_sparse_call(
        x_r,
        block.block_indices,
        vals_r,
        block.size,
        block.block_size,
        args.block_n,
        "torch",
        args.block_n_dx,
        args.block_n_dw,
    )
    dx_r, dw_r = torch.autograd.grad(y_r.sum(), (x_r, vals_r))
    dx_abs, dx_rel = max_abs_rel(dx_t, dx_r)
    dw_abs, dw_rel = max_abs_rel(dw_t, dw_r)

    p_req = params.detach().clone().requires_grad_(True)
    out = router(runoff.detach(), graph, p_req)
    dp = torch.autograd.grad(out.sum(), p_req)[0]
    dp_finite = bool(torch.isfinite(dp).all().item())

    checks = {
        "full_repeat_max_abs": full_abs,
        "full_repeat_max_rel": full_rel,
        "conv_forward_max_abs_vs_torch": conv_abs,
        "conv_forward_max_rel_vs_torch": conv_rel,
        "conv_backward_dx_max_abs_vs_torch": dx_abs,
        "conv_backward_dx_max_rel_vs_torch": dx_rel,
        "conv_backward_dw_max_abs_vs_torch": dw_abs,
        "conv_backward_dw_max_rel_vs_torch": dw_rel,
        "full_dparams_finite": dp_finite,
        "passed": (
            full_abs <= args.atol
            and conv_abs <= args.atol
            and dx_abs <= args.atol
            and dw_abs <= args.atol
            and dp_finite
        ),
        "atol": args.atol,
        "correct_time_steps": args.correct_time_steps,
    }
    payload = {
        "metadata": {
            "device": torch.cuda.get_device_name(device),
            "runoff_shape": list(runoff.shape),
            "nodes": len(graph),
            "paths": int(graph.path_cumsum[-1].item()),
            "block_values_shape": list(block.block_values.shape),
        },
        "correctness": checks,
    }
    write_outputs(args, payload, suffix="correctness")
    if not checks["passed"]:
        raise SystemExit(json.dumps(payload, indent=2))
    return payload


def write_outputs(args: argparse.Namespace, payload: dict, suffix: str = "profile") -> None:
    if args.output_dir is None:
        return
    args.output_dir.mkdir(parents=True, exist_ok=True)
    out_path = args.output_dir / f"rapid_io_{suffix}.json"
    out_path.write_text(json.dumps(payload, indent=2))


def print_payload(payload: dict) -> None:
    print(json.dumps(payload, indent=2))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--examples-dir", type=Path, default=default_examples_dir())
    parser.add_argument("--download", action="store_true", help="Download RAPID IO inputs if missing.")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--time-steps", type=int, default=None, help="Optional leading time slice.")
    parser.add_argument("--max-delay", type=int, default=32)
    parser.add_argument("--dt", type=float, default=1 / 24)
    parser.add_argument("--block-size", type=int, default=16)
    parser.add_argument("--block-n", type=int, default=None)
    parser.add_argument("--block-n-dx", type=int, default=None)
    parser.add_argument("--block-n-dw", type=int, default=None)
    parser.add_argument("--block-f", type=int, default=512)
    parser.add_argument("--conv-impl", choices=("auto", "triton", "torch"), default="auto")
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--trials", type=int, default=5)
    parser.add_argument("--backward-warmup", type=int, default=1)
    parser.add_argument("--backward-trials", type=int, default=3)
    parser.add_argument("--host-time", action="store_true", help="Use synchronized host wall time.")
    parser.add_argument("--output-dir", type=Path, default=Path("benchmarks/results"))
    parser.add_argument("--correctness", action="store_true", help="Run correctness checks instead of profile.")
    parser.add_argument("--correct-time-steps", type=int, default=64)
    parser.add_argument("--atol", type=float, default=1e-3)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("This benchmark requires CUDA.")
    payload = correctness(args) if args.correctness else profile(args)
    print_payload(payload)


if __name__ == "__main__":
    main()
