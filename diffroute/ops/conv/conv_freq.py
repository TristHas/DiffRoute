import torch
import torch.nn.functional as F


def freq_conv_coo(x: torch.Tensor,
                  coords: torch.Tensor,
                  vals: torch.Tensor,
                  kernel_shape,
                  chunk: int = 1024) -> torch.Tensor:
    """Causal convolution evaluated directly on a COO routing kernel.

    `BlockSparseKernel` pads the (dest, src) path set out to whole
    `block_size x block_size` tiles -- cheap when paths cluster spatially,
    wasteful when they don't: a narrow or scattered output selection at a
    large tap count K can pad a few-percent-fill COO kernel out to tens of
    GiB of mostly-zero block storage (see `BlockSparseCausalConv`'s
    dispatch). This evaluates exactly the N_path paths that exist instead,
    via FFT: memory scales with N_path and `chunk`, never with K or with how
    the paths are distributed across (dest, src) space.

    T is handled in fixed-size overlap-add chunks so memory is independent
    of sequence length too -- the alternative (one FFT over the whole
    series, one frequency-domain buffer of shape [N_path, T]) reintroduces
    exactly the memory blowup this function exists to avoid, just moved from
    block-storage into time.

    All operations here (rfft/index_select/complex-mul/index_add/irfft) are
    ordinary differentiable torch ops, so no custom autograd.Function is
    needed -- gradients w.r.t. both `x` and `vals` fall out of autograd.

    Args:
        x: [B, C_in, T] input.
        coords: [N_path, 2] int, (dest_row, src_col) -- same convention as
            `SparseKernel`: dest in 0..C_out-1, src in 0..C_in-1.
        vals: [N_path, K] float, the same tap layout `BlockSparseKernel`
            consumes (vals[..., -1] is lag 0 -- see `IRFAggregator`'s
            `.flip(-1)`, which this function's internal `h = vals.flip(-1)`
            undoes to recover the causal impulse response).
        kernel_shape: (C_out, C_in, K).
        chunk: overlap-add input chunk length, in time steps.

    Returns:
        [B, C_out, T] causal convolution output, `x`'s dtype.
    """
    B, C_in, T = x.shape
    C_out, C_in_decl, K = kernel_shape
    assert C_in == C_in_decl, "Input channel mismatch with kernel_shape."

    dst, src = coords[:, 0].long(), coords[:, 1].long()
    h = vals.float().flip(-1)                           # lag 0 first; fp32 accumulate
    n_fft = 1
    while n_fft < chunk + K - 1:
        n_fft *= 2
    H = torch.fft.rfft(h, n=n_fft, dim=-1)               # [N_path, F], built once

    out_dtype = x.dtype
    x = x.float()
    out_len = T + K - 1
    out = x.new_zeros(B, C_out, out_len)
    for start in range(0, T, chunk):
        x_chunk = x[..., start:start + chunk]
        if x_chunk.shape[-1] < chunk:
            x_chunk = F.pad(x_chunk, (0, chunk - x_chunk.shape[-1]))
        X = torch.fft.rfft(x_chunk, n=n_fft, dim=-1)                  # [B, C_in, F]
        contrib = X.index_select(1, src) * H.unsqueeze(0)             # [B, N_path, F]
        Y = contrib.new_zeros(B, C_out, H.shape[-1]).index_add(1, dst, contrib)
        y_chunk = torch.fft.irfft(Y, n=n_fft, dim=-1)                 # [B, C_out, n_fft]
        # y_chunk's support beyond (this chunk's real length + K - 1) is exactly
        # zero (the linear-conv length bound), and that bound never exceeds
        # out's remaining room -- see conv_freq design notes -- so clipping to
        # whichever is shorter never drops a nonzero sample.
        keep = min(n_fft, out_len - start)
        out = out + F.pad(y_chunk[..., :keep], (start, out_len - start - keep))
    return out[..., :T].to(out_dtype)
