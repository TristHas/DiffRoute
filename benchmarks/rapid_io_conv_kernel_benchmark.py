#!/usr/bin/env python
"""Isolated RAPID IO block-sparse convolution kernel benchmark.

This is the tight-loop harness for optimizing the DiffRoute sparse convolution
kernels. It uses the RAPID IO workload shape but measures only one kernel path
at a time after all graph/runoff/kernel preparation is complete.
"""

from __future__ import annotations

import argparse
import json
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
from diffroute.ops.conv import block_sparse_conv_1d


@dataclass
class Timing:
    target: str
    mean_ms: float
    median_ms: float
    min_ms: float
    max_ms: float
    std_ms: float
    trials: int


def default_examples_dir() -> Path:
    return Path(__file__).resolve().parents[2] / "DiffHydro" / "examples"


def ensure_example_data(examples_dir: Path) -> None:
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
        raise FileNotFoundError(
            "Missing RAPID IO inputs:\n"
            + "\n".join(f"  - {p}" for p in missing)
            + "\nRun with --download first."
        )

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

    router = LTIRouter(
        max_delay=args.max_delay,
        dt=args.dt,
        block_size=args.block_size,
        block_f=args.block_f,
        conv_imp="triton",
        block_n=args.block_n,
        block_n_dx=args.block_n_dx,
        block_n_dw=args.block_n_dw,
    ).to(device)
    params = graph.params.detach().contiguous()
    sparse = router.aggregator(graph, params)
    block = sparse.to_block_sparse(router.block_size)
    return runoff, block


def triton_conv(x, block, config):
    return block_sparse_conv_1d(
        x,
        block.block_indices,
        block.block_values,
        block.size,
        block.block_size,
        config.block_n,
        BLOCK_SIZE_N_DX=config.block_n_dx,
        BLOCK_SIZE_N_DVALUES=config.block_n_dw,
        DX_BLOCK_ORDER=block.block_col_order,
    )


def torch_conv(x, block):
    return conv1d_block_sparse(
        x,
        block.block_values,
        block.block_indices[:, 1],
        block.block_indices[:, 0],
    )


def make_target(args: argparse.Namespace, runoff, block, config):
    torch.manual_seed(args.seed)
    if args.target == "fwd":
        x = runoff.detach()

        def fn():
            return triton_conv(x, block, config)

        return fn

    if args.target == "dx":
        x = runoff.detach().clone().requires_grad_(True)
        y = triton_conv(x, block, config)
        dy = torch.randn_like(y)

        def fn():
            return torch.autograd.grad(y, x, dy, retain_graph=True)[0]

        return fn

    if args.target == "dw":
        vals = block.block_values.detach().clone().requires_grad_(True)

        class BlockView:
            pass

        local_block = BlockView()
        local_block.block_indices = block.block_indices
        local_block.block_values = vals
        local_block.size = block.size
        local_block.block_size = block.block_size
        y = triton_conv(runoff.detach(), local_block, config)
        dy = torch.randn_like(y)

        def fn():
            return torch.autograd.grad(y, vals, dy, retain_graph=True)[0]

        return fn

    raise ValueError(args.target)


def time_cuda(name: str, fn: Callable[[], object], device, warmup: int, trials: int) -> Timing:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize(device)

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
    return Timing(
        target=name,
        mean_ms=statistics.mean(times),
        median_ms=statistics.median(times),
        min_ms=min(times),
        max_ms=max(times),
        std_ms=statistics.stdev(times) if len(times) > 1 else 0.0,
        trials=len(times),
    )


def diff_stats(a: torch.Tensor, b: torch.Tensor) -> dict[str, float]:
    diff = (a - b).abs()
    rel = diff / b.abs().clamp_min(1e-8)
    return {
        "max_abs": float(diff.max().item()),
        "mean_abs": float(diff.mean().item()),
        "rms_abs": float(torch.sqrt(torch.mean(diff * diff)).item()),
        "max_rel": float(rel.max().item()),
        "mean_rel": float(rel.mean().item()),
    }


def correctness(args: argparse.Namespace) -> dict:
    args.time_steps = args.correct_time_steps
    device = torch.device(args.device)
    torch.cuda.set_device(device)
    runoff, block = load_problem(args, device)
    config = select_conv_config(
        device=device,
        impl="triton",
        block_n=args.block_n,
        block_n_dx=args.block_n_dx,
        block_n_dw=args.block_n_dw,
    )
    torch.manual_seed(args.seed)

    if args.target == "fwd":
        with torch.no_grad():
            got = triton_conv(runoff, block, config)
            expected = torch_conv(runoff, block)
        stats = diff_stats(got, expected)
    elif args.target == "dx":
        x = runoff.detach().clone().requires_grad_(True)
        y = triton_conv(x, block, config)
        dy = torch.randn_like(y)
        got = torch.autograd.grad(y, x, dy)[0]

        x_ref = runoff.detach().clone().requires_grad_(True)
        y_ref = torch_conv(x_ref, block)
        expected = torch.autograd.grad(y_ref, x_ref, dy)[0]
        stats = diff_stats(got, expected)
    else:
        vals = block.block_values.detach().clone().requires_grad_(True)
        class BlockView:
            pass
        local_block = BlockView()
        local_block.block_indices = block.block_indices
        local_block.block_values = vals
        local_block.size = block.size
        local_block.block_size = block.block_size
        y = triton_conv(runoff.detach(), local_block, config)
        dy = torch.randn_like(y)
        got = torch.autograd.grad(y, vals, dy)[0]

        vals_ref = block.block_values.detach().clone().requires_grad_(True)
        ref_block = BlockView()
        ref_block.block_indices = block.block_indices
        ref_block.block_values = vals_ref
        ref_block.size = block.size
        ref_block.block_size = block.block_size
        y_ref = torch_conv(runoff.detach(), ref_block)
        expected = torch.autograd.grad(y_ref, vals_ref, dy)[0]
        stats = diff_stats(got, expected)

    return {
        "target": args.target,
        **stats,
        "atol": args.atol,
        "passed": stats["max_abs"] <= args.atol,
        "runoff_shape": list(runoff.shape),
        "block_values_shape": list(block.block_values.shape),
        "config": asdict(config),
    }


def run(args: argparse.Namespace) -> dict:
    device = torch.device(args.device)
    torch.cuda.set_device(device)
    runoff, block = load_problem(args, device)
    config = select_conv_config(
        device=device,
        impl="triton",
        block_n=args.block_n,
        block_n_dx=args.block_n_dx,
        block_n_dw=args.block_n_dw,
    )
    fn = make_target(args, runoff, block, config)

    # Compile before timing.
    fn()
    torch.cuda.synchronize(device)

    timing = time_cuda(args.target, fn, device, args.warmup, args.trials)
    return {
        "device": torch.cuda.get_device_name(device),
        "runoff_shape": list(runoff.shape),
        "block_values_shape": list(block.block_values.shape),
        "config": asdict(config),
        "timing": asdict(timing),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target", choices=("fwd", "dx", "dw"), default="dx")
    parser.add_argument("--examples-dir", type=Path, default=default_examples_dir())
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--time-steps", type=int, default=None)
    parser.add_argument("--max-delay", type=int, default=32)
    parser.add_argument("--dt", type=float, default=1 / 24)
    parser.add_argument("--block-size", type=int, default=16)
    parser.add_argument("--block-f", type=int, default=128)
    parser.add_argument("--block-n", type=int, default=None)
    parser.add_argument("--block-n-dx", type=int, default=None)
    parser.add_argument("--block-n-dw", type=int, default=None)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--trials", type=int, default=10)
    parser.add_argument("--correctness", action="store_true")
    parser.add_argument("--correct-time-steps", type=int, default=32)
    parser.add_argument("--atol", type=float, default=2e-2)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    payload = correctness(args) if args.correctness else run(args)
    print(json.dumps(payload, indent=2))
    if args.correctness and not payload["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
