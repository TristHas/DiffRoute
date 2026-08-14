"""Routing beyond a small kernel window (Task D, part b).

Two things can go wrong once the tap count K gets large (hourly resolution:
K ~ 768, vs daily's K ~ 32):

1. `BlockSparseKernel`'s triton conv kernels compute address offsets like
   `nzb * (K * block_size**2)` for the weight-block index. Triton wraps
   int32 silently rather than trapping (verified directly below), and that
   product exceeds 2^31 once a kernel has a few thousand nonzero blocks at
   K=768 -- reachable by an ordinary scattered output selection (a gauge
   set, an arbitrary batch), not just a pathological one. Before the fix
   this produced silently-wrong finite output as often as it crashed,
   depending on where the wrapped (negative) offset happened to land in the
   process's address space.
2. Even once addressing is safe, block storage itself doesn't scale: it
   pads the (dest, src) path set out to whole blocks, which is wasteful at
   the low fill a scattered selection produces (a few percent is common),
   and at K=768 that padding can reach tens of GiB for a kernel whose real
   COO data is a few hundred MB. `BlockSparseCausalConv` dispatches to a
   frequency-domain COO convolution (`freq_conv_coo`) instead once the
   projected block storage crosses a threshold, invisibly to the caller.
"""
import torch
import torch.nn.functional as F
import pytest

from diffroute.structs import SparseKernel
from diffroute.ops import block_sparse_conv_1d, freq_conv_coo
from diffroute.conv import BlockSparseCausalConv

DEVICE = "cuda:0"


def test_triton_int32_wraps_silently():
    """Ground truth for why this fix is address-arithmetic, not a value fix:
    confirm the runtime actually wraps int32 in the exact multiply pattern
    the conv/closure kernels use, rather than trapping or promoting."""
    import triton
    import triton.language as tl

    @triton.jit
    def _probe(inp_ptr, out_ptr, KS: tl.constexpr, M: tl.constexpr):
        c = tl.load(inp_ptr)
        tl.store(out_ptr, (c * (KS * M * M)).to(tl.int64))

    nzb = 18583  # a real deep-path VPU 209 batch's block count, see commit log
    inp = torch.tensor([nzb], dtype=torch.int32, device=DEVICE)
    out = torch.zeros(1, dtype=torch.int64, device=DEVICE)
    _probe[(1,)](inp, out, KS=768, M=16)
    expected = nzb * 768 * 16 * 16
    assert expected > 2**31, "test setup: this offset should exceed int32 range"
    assert out.item() != expected, (
        "Triton stopped wrapping int32 multiplication -- if this now holds, "
        "the int64 casts in conv_temp_1D_triton.py/closure_sub.py may no "
        "longer be load-bearing, but they remain correct and should stay.")


def test_conv_survives_and_matches_past_the_int32_threshold():
    """N diagonal (dest, src) blocks, each independent (no shared rows or
    columns), so the FIRST M blocks' output rows are provably unaffected by
    the other N-M: the reference is exact, not approximate, whatever N is.
    N=11,200 pushes the weight-block offset (nzb * K * 16 * 16) to ~1.02x
    2^31 at K=768 -- past the threshold in
    diffroute/ops/conv/conv_temp_1D_triton.py that used to wrap and either
    crash or silently corrupt the output.
    """
    N, K, T, B = 11200, 768, 16, 16
    idx = torch.arange(N, dtype=torch.int64) * B
    coords = torch.stack([idx, idx], dim=1).to(DEVICE)
    torch.manual_seed(0)
    vals = torch.rand(N, K, device=DEVICE)
    C = N * B
    kernel = SparseKernel(coords, vals, (C, C, K))
    bs = kernel.to_block_sparse(B)
    weight_offset = (bs.block_indices.shape[0] - 1) * B * B * K
    assert weight_offset > 2**31, "test setup: should cross the int32 threshold"

    x = torch.rand(1, C, T, generator=torch.Generator().manual_seed(1)).to(DEVICE)
    y_full = block_sparse_conv_1d(x, bs.block_indices, bs.block_values, bs.size, B, 64)
    assert torch.isfinite(y_full).all()

    M = 100
    sub = SparseKernel(coords[:M], vals[:M], (C, C, K)).to_block_sparse(B)
    y_safe = block_sparse_conv_1d(x, sub.block_indices, sub.block_values, sub.size, B, 64)

    rows = M * B
    assert torch.equal(y_full[:, :rows], y_safe[:, :rows])


def _naive_causal_conv(x, coords, vals, kernel_shape):
    """y[b,r,t] = sum_i sum_k vals[i,k] * x[b, coords[i,1], t+k-(K-1)] -- the
    same formula the triton kernel implements (conv_temp_1D_triton.py's
    t_in indexing), computed with plain indexing so it shares no code with
    either fast path being tested against it."""
    B_, C_in, T = x.shape
    C_out, _, K = kernel_shape
    y = x.new_zeros(B_, C_out, T)
    xp = F.pad(x, (K - 1, 0))
    for i in range(coords.shape[0]):
        r, c = int(coords[i, 0]), int(coords[i, 1])
        for k in range(K):
            y[:, r, :] += vals[i, k] * xp[:, c, k:k + T]
    return y


@pytest.mark.parametrize("K,C_in,C_out,T,n_path,chunk", [
    (5, 4, 3, 20, 8, 20),        # single chunk
    (17, 6, 5, 50, 15, 16),      # chunk shorter than K
    (768, 10, 8, 300, 25, 64),   # hourly-scale K, chunk << K
])
def test_freq_conv_matches_naive_reference(K, C_in, C_out, T, n_path, chunk):
    torch.manual_seed(0)
    x = torch.rand(2, C_in, T, device=DEVICE, dtype=torch.float64)
    coords = torch.stack([torch.randint(0, C_out, (n_path,)),
                          torch.randint(0, C_in, (n_path,))], dim=1).to(DEVICE)
    vals = torch.rand(n_path, K, device=DEVICE, dtype=torch.float64)

    y_ref = _naive_causal_conv(x, coords, vals, (C_out, C_in, K))
    y_freq = freq_conv_coo(x.float(), coords, vals.float(), (C_out, C_in, K), chunk=chunk).double()

    rel = (y_ref - y_freq).abs().max() / y_ref.abs().max().clamp_min(1e-12)
    assert rel < 1e-3


def test_freq_conv_gradient_matches_naive_reference():
    """Compared against the naive reference's own autograd (both are analytical
    gradients through a differentiable graph), not a finite difference:
    freq_conv_coo computes internally in fp32 (matching the triton kernel's
    accumulate-in-fp32 convention), so a finite-difference check against fp64
    would mostly measure that rounding gap rather than a real discrepancy."""
    torch.manual_seed(0)
    C_in, C_out, K, T, n_path = 4, 3, 5, 12, 6
    coords = torch.stack([torch.randint(0, C_out, (n_path,)),
                          torch.randint(0, C_in, (n_path,))], dim=1).to(DEVICE)

    x64 = torch.rand(1, C_in, T, device=DEVICE, dtype=torch.float64, requires_grad=True)
    v64 = torch.rand(n_path, K, device=DEVICE, dtype=torch.float64, requires_grad=True)
    _naive_causal_conv(x64, coords, v64, (C_out, C_in, K)).sum().backward()

    x32 = x64.detach().float().requires_grad_(True)
    v32 = v64.detach().float().requires_grad_(True)
    freq_conv_coo(x32, coords, v32, (C_out, C_in, K), chunk=12).sum().backward()

    rel_x = (x32.grad.double() - x64.grad).abs().max() / x64.grad.abs().max().clamp_min(1e-12)
    rel_v = (v32.grad.double() - v64.grad).abs().max() / v64.grad.abs().max().clamp_min(1e-12)
    assert max(rel_x.item(), rel_v.item()) < 1e-3


def test_dispatch_routes_by_projected_block_size_and_agrees():
    """BlockSparseCausalConv routes a SparseKernel to freq_conv_coo once the
    projected block storage exceeds freq_threshold_bytes, and to the
    existing triton block-sparse path otherwise -- forcing the threshold to
    each side (rather than needing gigabytes of data to cross the real
    default) isolates the dispatch decision itself. Same kernel, same
    input: the two algorithms (FFT vs direct time-domain accumulation)
    must still agree, at ordinary fp32 cross-method tolerance rather than
    bitwise -- see conv_freq's docstring.
    """
    torch.manual_seed(0)
    C_in, C_out, K, T, n_path = 40, 30, 128, 64, 200
    coords = torch.stack([torch.randint(0, C_out, (n_path,)),
                          torch.randint(0, C_in, (n_path,))], dim=1).to(DEVICE)
    coords = torch.unique(coords, dim=0)
    vals = torch.rand(coords.shape[0], K, device=DEVICE)
    kernel = SparseKernel(coords, vals, (C_out, C_in, K))
    x = torch.rand(1, C_in, T, device=DEVICE)

    conv_freq = BlockSparseCausalConv(freq_threshold_bytes=1)
    conv_block = BlockSparseCausalConv(freq_threshold_bytes=2**40)
    y_freq = conv_freq(x, kernel)
    y_block = conv_block(x, kernel)

    assert torch.isfinite(y_freq).all() and torch.isfinite(y_block).all()
    rel = (y_freq - y_block).abs().max() / y_block.abs().max().clamp_min(1e-12)
    assert rel < 1e-2
