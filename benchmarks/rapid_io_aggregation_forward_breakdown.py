#!/usr/bin/env python
"""Break down RAPID IO aggregation forward latency by internal stage."""

from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable

import torch
import torch.nn.functional as F
import triton

from diffroute import LTIRouter
from diffroute.agg.kernel_aggregator import aggregate_irf
from diffroute.io import read_rapid_graph
from diffroute.irfs import IRF_FN
from diffroute.ops.closure_sub import closure_sub
from diffroute.ops.prefix_sum import _prefix_jump_fwd, _prefix_jump_kernel, prefix_sum
from diffroute.ops.transitive_closure import exp_complex, stable_log_flattened


@dataclass
class Timing:
    name: str
    cuda_mean_ms: float
    cuda_median_ms: float
    cuda_min_ms: float
    cuda_max_ms: float
    cuda_std_ms: float
    host_mean_ms: float
    host_median_ms: float
    host_min_ms: float
    host_max_ms: float
    host_std_ms: float
    trials: int
    peak_allocated_gib: float
    peak_reserved_gib: float


def default_examples_dir() -> Path:
    return Path(__file__).resolve().parents[2] / "DiffHydro" / "examples"


def ensure_example_data(examples_dir: Path) -> None:
    sys.path.insert(0, str(examples_dir))
    from utils.download import download_rapid_config

    root = examples_dir / "data"
    download_rapid_config(root, 305)


def load_graph(examples_dir: Path, device: torch.device, download: bool):
    if download:
        ensure_example_data(examples_dir)
    rapid_path = (
        examples_dir
        / "data"
        / "geoglows"
        / "rapid_config"
        / "305"
        / "configs"
        / "305"
    )
    if not rapid_path.exists():
        raise FileNotFoundError(f"Missing RAPID graph at {rapid_path}; rerun with --download.")
    return read_rapid_graph(rapid_path).to(device)


def _stats(name: str, cuda_times: list[float], host_times: list[float], device: torch.device) -> Timing:
    peak_alloc = torch.cuda.max_memory_allocated(device) / 1024**3
    peak_reserved = torch.cuda.max_memory_reserved(device) / 1024**3
    return Timing(
        name=name,
        cuda_mean_ms=statistics.mean(cuda_times),
        cuda_median_ms=statistics.median(cuda_times),
        cuda_min_ms=min(cuda_times),
        cuda_max_ms=max(cuda_times),
        cuda_std_ms=statistics.stdev(cuda_times) if len(cuda_times) > 1 else 0.0,
        host_mean_ms=statistics.mean(host_times),
        host_median_ms=statistics.median(host_times),
        host_min_ms=min(host_times),
        host_max_ms=max(host_times),
        host_std_ms=statistics.stdev(host_times) if len(host_times) > 1 else 0.0,
        trials=len(cuda_times),
        peak_allocated_gib=peak_alloc,
        peak_reserved_gib=peak_reserved,
    )


def time_stage(
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

    cuda_times: list[float] = []
    host_times: list[float] = []
    for _ in range(trials):
        torch.cuda.synchronize(device)
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        host_start = time.perf_counter()
        start.record()
        fn()
        end.record()
        torch.cuda.synchronize(device)
        host_times.append((time.perf_counter() - host_start) * 1000.0)
        cuda_times.append(start.elapsed_time(end))
    return _stats(name, cuda_times, host_times, device)


def prefix_jump_fixed_rounds(
    irf: torch.Tensor,
    edges: torch.Tensor,
    *,
    rounds: int,
    block_f: int,
):
    irf = irf.contiguous()
    edges = edges.contiguous()
    n, f = irf.shape
    buf0 = irf.clone()
    buf1 = torch.empty_like(buf0)
    e_run = edges.clone()
    e_snap = torch.empty_like(e_run)
    grid = (n,)
    with torch.cuda.device(irf.device):
        for _ in range(rounds):
            e_snap.copy_(e_run)
            _prefix_jump_kernel[grid](buf0, buf1, e_run, e_snap, n, f, BLOCK_F=block_f)
            buf0, buf1 = buf1, buf0
    return buf0


def pointer_jump_rounds(edges: torch.Tensor) -> int:
    e = edges.detach().cpu()
    rounds = 0
    while bool((e >= 0).any()):
        valid = e >= 0
        next_e = torch.full_like(e, -1)
        next_e[valid] = e[e[valid]]
        e = next_e
        rounds += 1
    return rounds


def profile(args: argparse.Namespace) -> dict:
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    torch.cuda.set_device(device)
    graph = load_graph(args.examples_dir.resolve(), device, args.download)
    router = LTIRouter(
        max_delay=args.max_delay,
        dt=args.dt,
        sampling_mode=args.sampling_mode,
        block_size=args.block_size,
        block_f=args.block_f,
    ).to(device)

    params = graph.params.detach().contiguous()
    irf_fn = IRF_FN[graph.irf_fn]
    time_window_expanded = args.max_delay * int(1 / args.dt)

    irfs = irf_fn(params, time_window=args.max_delay, dt=args.dt).squeeze().contiguous()
    irfs_freq = torch.fft.rfft(irfs, n=time_window_expanded, dim=-1)
    amp = torch.abs(irfs_freq)
    log_amp = torch.log(amp.clamp_min(args.log_epsilon))
    angle = torch.angle(irfs_freq)
    log_irfs_freq = stable_log_flattened(irfs_freq)
    prefix = prefix_sum(log_irfs_freq, graph.edges, args.block_f)
    coords, log_vals = closure_sub(
        prefix,
        graph.edges,
        graph.path_cumsum,
        graph.include_index_diag,
        args.block_f,
    )
    freq_agg = exp_complex(log_vals)
    irfs_agg = torch.fft.irfft(freq_agg, n=time_window_expanded, dim=-1)
    relu_irfs_agg = torch.relu(irfs_agg)
    down = F.conv1d(
        relu_irfs_agg.unsqueeze(1),
        router.aggregator.sampler.tri_kernel,
        stride=router.aggregator.sampler.factor,
        padding=router.aggregator.sampler.factor - 1,
    ).squeeze(1)
    flipped = down.flip(-1)
    denom = flipped.sum(-1, keepdim=True)
    torch.cuda.synchronize(device)

    jump_rounds = pointer_jump_rounds(graph.edges)
    max_rounds = math.ceil(math.log2(max(1, len(graph))))
    prefix_ref = _prefix_jump_fwd(log_irfs_freq, graph.edges, args.block_f, return_jumps=False)
    prefix_fixed = prefix_jump_fixed_rounds(
        log_irfs_freq,
        graph.edges,
        rounds=jump_rounds,
        block_f=args.block_f,
    )
    prefix_fixed_max_abs = float((prefix_ref - prefix_fixed).abs().max().item())

    timings: list[Timing] = []
    add = timings.append
    add(time_stage(
        "aggregator.forward.total",
        lambda: router.aggregator(graph, params),
        device=device,
        warmup=args.warmup,
        trials=args.trials,
    ))
    add(time_stage(
        "aggregate_irf.pre_sampler",
        lambda: aggregate_irf(
            params,
            irf_fn=irf_fn,
            edges=graph.edges,
            path_cumsum=graph.path_cumsum,
            dt=args.dt,
            time_window=args.max_delay,
            include_index_diag=graph.include_index_diag,
            block_f=args.block_f,
        ),
        device=device,
        warmup=args.warmup,
        trials=args.trials,
    ))
    add(time_stage(
        "irf.generate",
        lambda: irf_fn(params, time_window=args.max_delay, dt=args.dt).squeeze(),
        device=device,
        warmup=args.warmup,
        trials=args.trials,
    ))
    add(time_stage(
        "fft.rfft",
        lambda: torch.fft.rfft(irfs, n=time_window_expanded, dim=-1),
        device=device,
        warmup=args.warmup,
        trials=args.trials,
    ))
    add(time_stage(
        "log_stable.total",
        lambda: stable_log_flattened(irfs_freq),
        device=device,
        warmup=args.warmup,
        trials=args.trials,
    ))
    add(time_stage(
        "log_stable.abs",
        lambda: torch.abs(irfs_freq),
        device=device,
        warmup=args.warmup,
        trials=args.trials,
    ))
    add(time_stage(
        "log_stable.clamp_log",
        lambda: torch.log(amp.clamp_min(args.log_epsilon)),
        device=device,
        warmup=args.warmup,
        trials=args.trials,
    ))
    add(time_stage(
        "log_stable.angle",
        lambda: torch.angle(irfs_freq),
        device=device,
        warmup=args.warmup,
        trials=args.trials,
    ))
    add(time_stage(
        "log_stable.stack_contiguous_view",
        lambda: torch.stack([log_amp, angle], dim=-1).contiguous().view(amp.shape[0], -1),
        device=device,
        warmup=args.warmup,
        trials=args.trials,
    ))
    add(time_stage(
        "prefix_sum.current_autograd",
        lambda: prefix_sum(log_irfs_freq, graph.edges, args.block_f),
        device=device,
        warmup=args.warmup,
        trials=args.trials,
    ))
    add(time_stage(
        "prefix_sum.no_saved_jumps",
        lambda: _prefix_jump_fwd(log_irfs_freq, graph.edges, args.block_f, return_jumps=False),
        device=device,
        warmup=args.warmup,
        trials=args.trials,
    ))
    add(time_stage(
        "prefix_sum.fixed_rounds_no_all_check",
        lambda: prefix_jump_fixed_rounds(
            log_irfs_freq,
            graph.edges,
            rounds=jump_rounds,
            block_f=args.block_f,
        ),
        device=device,
        warmup=args.warmup,
        trials=args.trials,
    ))
    add(time_stage(
        "prefix_sum.max_rounds_no_all_check",
        lambda: prefix_jump_fixed_rounds(
            log_irfs_freq,
            graph.edges,
            rounds=max_rounds,
            block_f=args.block_f,
        ),
        device=device,
        warmup=args.warmup,
        trials=args.trials,
    ))
    add(time_stage(
        "closure_sub.total",
        lambda: closure_sub(
            prefix,
            graph.edges,
            graph.path_cumsum,
            graph.include_index_diag,
            args.block_f,
        ),
        device=device,
        warmup=args.warmup,
        trials=args.trials,
    ))
    add(time_stage(
        "exp_complex.total",
        lambda: exp_complex(log_vals),
        device=device,
        warmup=args.warmup,
        trials=args.trials,
    ))
    add(time_stage(
        "ifft.irfft",
        lambda: torch.fft.irfft(freq_agg, n=time_window_expanded, dim=-1),
        device=device,
        warmup=args.warmup,
        trials=args.trials,
    ))
    add(time_stage(
        "temporal_sampler.total",
        lambda: router.aggregator.sampler.kernel_postprocess(irfs_agg),
        device=device,
        warmup=args.warmup,
        trials=args.trials,
    ))
    add(time_stage(
        "temporal_sampler.relu",
        lambda: torch.relu(irfs_agg),
        device=device,
        warmup=args.warmup,
        trials=args.trials,
    ))
    add(time_stage(
        "temporal_sampler.tri_conv_downsample",
        lambda: F.conv1d(
            relu_irfs_agg.unsqueeze(1),
            router.aggregator.sampler.tri_kernel,
            stride=router.aggregator.sampler.factor,
            padding=router.aggregator.sampler.factor - 1,
        ).squeeze(1),
        device=device,
        warmup=args.warmup,
        trials=args.trials,
    ))
    add(time_stage(
        "temporal_sampler.flip",
        lambda: down.flip(-1),
        device=device,
        warmup=args.warmup,
        trials=args.trials,
    ))
    add(time_stage(
        "temporal_sampler.sum",
        lambda: flipped.sum(-1, keepdim=True),
        device=device,
        warmup=args.warmup,
        trials=args.trials,
    ))
    add(time_stage(
        "temporal_sampler.normalize",
        lambda: flipped / denom,
        device=device,
        warmup=args.warmup,
        trials=args.trials,
    ))

    metadata = {
        "device": torch.cuda.get_device_name(device),
        "device_capability": torch.cuda.get_device_capability(device),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "nodes": len(graph),
        "paths": int(graph.path_cumsum[-1].item()),
        "irf_fn": graph.irf_fn,
        "params_shape": list(params.shape),
        "irfs_shape": list(irfs.shape),
        "irfs_freq_shape": list(irfs_freq.shape),
        "log_irfs_freq_shape": list(log_irfs_freq.shape),
        "prefix_shape": list(prefix.shape),
        "closure_vals_shape": list(log_vals.shape),
        "irfs_agg_shape": list(irfs_agg.shape),
        "sampler_factor": router.aggregator.sampler.factor,
        "max_delay": args.max_delay,
        "dt": args.dt,
        "block_f": args.block_f,
        "jump_rounds": jump_rounds,
        "max_rounds": max_rounds,
        "prefix_fixed_max_abs_vs_no_saved": prefix_fixed_max_abs,
    }
    payload = {"metadata": metadata, "timings": [asdict(x) for x in timings]}
    if args.output_dir is not None:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        (args.output_dir / "rapid_io_aggregation_forward_breakdown.json").write_text(
            json.dumps(payload, indent=2)
        )
    return payload


def print_summary(payload: dict) -> None:
    print(json.dumps(payload["metadata"], indent=2))
    print()
    print(f"{'stage':45s} {'cuda_mean':>10s} {'cuda_med':>10s} {'host_mean':>10s} {'host_med':>10s}")
    for row in payload["timings"]:
        print(
            f"{row['name']:45s} "
            f"{row['cuda_mean_ms']:10.3f} "
            f"{row['cuda_median_ms']:10.3f} "
            f"{row['host_mean_ms']:10.3f} "
            f"{row['host_median_ms']:10.3f}"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--examples-dir", type=Path, default=default_examples_dir())
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-delay", type=int, default=32)
    parser.add_argument("--dt", type=float, default=1 / 24)
    parser.add_argument("--sampling-mode", default="avg")
    parser.add_argument("--block-size", type=int, default=16)
    parser.add_argument("--block-f", type=int, default=128)
    parser.add_argument("--log-epsilon", type=float, default=1e-8)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--trials", type=int, default=7)
    parser.add_argument("--output-dir", type=Path, default=Path("benchmarks/results/aggregation_forward_breakdown"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("This benchmark requires CUDA.")
    payload = profile(args)
    print_summary(payload)


if __name__ == "__main__":
    main()
