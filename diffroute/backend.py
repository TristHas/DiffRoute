"""Small hardware-aware kernel configuration helpers."""

from __future__ import annotations

from dataclasses import dataclass
import os

import torch


@dataclass(frozen=True)
class ConvConfig:
    impl: str
    block_m: int | None
    block_n: int
    block_n_dx: int
    block_n_dw: int


def _env_int(name: str) -> int | None:
    value = os.environ.get(name)
    if value is None or value == "":
        return None
    return int(value)


def _env_str(name: str) -> str | None:
    value = os.environ.get(name)
    return value if value else None


def _cuda_props(device: torch.device | str | int | None):
    if not torch.cuda.is_available():
        return None
    if device is None:
        device = torch.cuda.current_device()
    device = torch.device(device)
    if device.type != "cuda":
        return None
    return torch.cuda.get_device_properties(device)


def select_conv_config(
    *,
    device: torch.device | str | int | None = None,
    impl: str | None = "auto",
    block_m: int | None = None,
    block_n: int | None = None,
    block_n_dx: int | None = None,
    block_n_dw: int | None = None,
) -> ConvConfig:
    """Select a sparse-convolution implementation and tile sizes.

    Explicit arguments win first, then environment variables, then conservative
    hardware defaults. Environment overrides:

    * ``DIFFROUTE_CONV_IMPL``
    * ``DIFFROUTE_BLOCK_M``
    * ``DIFFROUTE_BLOCK_N``
    * ``DIFFROUTE_BLOCK_N_DX``
    * ``DIFFROUTE_BLOCK_N_DW``
    """
    env_impl = _env_str("DIFFROUTE_CONV_IMPL")
    impl = env_impl or impl or "auto"

    props = _cuda_props(device)
    if impl == "auto":
        impl = "triton" if props is not None else "torch"

    default_n = 64
    default_n_dx = 64
    default_n_dw = 64
    if props is not None:
        name = props.name.upper()
        major = props.major
        if "GB200" in name or "B200" in name or major >= 10:
            default_n = 256
            default_n_dx = 64
            default_n_dw = 256
        elif "H100" in name or "H200" in name or major == 9:
            default_n = 128
            default_n_dx = 64
            default_n_dw = 128
        elif "A100" in name or major == 8:
            default_n = 64
            default_n_dx = 64
            default_n_dw = 64

    block_m = _env_int("DIFFROUTE_BLOCK_M") if block_m is None else block_m
    block_n = _env_int("DIFFROUTE_BLOCK_N") if block_n is None else block_n
    block_n_dx = _env_int("DIFFROUTE_BLOCK_N_DX") if block_n_dx is None else block_n_dx
    block_n_dw = _env_int("DIFFROUTE_BLOCK_N_DW") if block_n_dw is None else block_n_dw

    return ConvConfig(
        impl=impl,
        block_m=block_m,
        block_n=block_n or default_n,
        block_n_dx=block_n_dx or default_n_dx,
        block_n_dw=block_n_dw or default_n_dw,
    )
