import torch
import torch.nn.functional as F
from .conv_temp_1D_triton import (
    block_sparse_conv_1d_fwd_kernel,
    block_sparse_conv_1d_bwd_dx_kernel,
    block_sparse_conv_1d_bwd_dvalues_kernel
)

def pad_block_permute(x: torch.Tensor, block_size: int):
    """
    x: [B, C, T]
    Returns:
      x_blk:   [B, n_blocks, T, block_size]
      orig_C:  original channel count
      padded_C: padded channel count (multiple of block_size)
      n_blocks: number of channel blocks
    """
    B, C, T = x.shape
    n_blocks = (C + block_size - 1) // block_size
    padded_C = n_blocks * block_size
    if padded_C != C: x = F.pad(x, (0, 0, 0, padded_C - C))
    x_blk = x.view(B, n_blocks, block_size, T).permute(0, 1, 3, 2).contiguous()
    return x_blk, C, padded_C, n_blocks

def unpermute_unpad_block(x_blk: torch.Tensor, orig_C: int):
    """
    x_blk: [B, n_blocks, T, block_size]
    Returns: [B, orig_C, T]
    """
    B, n_blocks, T, block_size = x_blk.shape
    x = x_blk.permute(0, 1, 3, 2).contiguous().view(B, n_blocks * block_size, T)
    return x[:, :orig_C, :]

# ---------------------------------------------------------------------------
# Forward
# ---------------------------------------------------------------------------
def block_sparse_conv_1d_forward(
        x: torch.Tensor,
        coo_block_coords: torch.Tensor,     # [N_NONZERO_BLOCKS, 2] (r_block, c_block)
        values: torch.Tensor,               # [N_NONZERO_BLOCKS, M_out, M_in, K]
        kernel_shape,                       # tuple (C_out, C_in, K)
        BLOCK_SIZE_M: int,
        BLOCK_SIZE_N: int,
        NZB_BLOCK_SIZE: int = 16
    ):
    """
    Block-sparse causal 1D convolution:
      y[b, :, t] = sum_{(r,c) in blocks} sum_{k=0..K-1} W_{(r,c)}[:, :, k] * x[b, :, t-k]
    Only for those (r,c) given in coo_block_coords (block-sparse pattern).

    Args:
      x: [B, C_in, T]
      coo_block_coords: [Nnz, 2] block indices (r_block, c_block)
      values: [Nnz, M_out, M_in, K] (per nonzero block)
      kernel_shape: (C_out, C_in, K)
    Returns:
      y: [B, C_out, T]
    """
    B, C_in, T = x.shape
    C_out, C_in_decl, K = kernel_shape
    assert C_in == C_in_decl, "Input channel mismatch with kernel_shape."

    # Permute weights to [Nnz, K, M_out, M_in] once for kernels
    values_perm = values.permute(0, 3, 1, 2).contiguous()
    N_NONZERO_BLOCKS = coo_block_coords.shape[0]
    assert values_perm.shape[1] == K

    # Allocate output
    y = torch.zeros(B, C_out, T, device=x.device, dtype=x.dtype)

    # Block/pad input and output (separately)
    x_blk,  orig_C_in,  padded_C_in,  n_in_blocks  = pad_block_permute(x, BLOCK_SIZE_M)
    y_blk,  orig_C_out, padded_C_out, n_out_blocks = pad_block_permute(y, BLOCK_SIZE_M)

    # Launch parameters
    n_time_tiles = (T + BLOCK_SIZE_N - 1) // BLOCK_SIZE_N
    grid = (
        (N_NONZERO_BLOCKS + NZB_BLOCK_SIZE - 1) // NZB_BLOCK_SIZE,  # along NZB groups
        n_time_tiles,                                               # time tiles
        B                                                           # batch
    )

    with torch.cuda.device(x.device):
        block_sparse_conv_1d_fwd_kernel[grid](
            x_blk,
            coo_block_coords.int(),
            values_perm,
            y_blk,
            B,
            n_in_blocks,
            n_out_blocks,
            T,
            BLOCK_SIZE_M,
            BLOCK_SIZE_N,
            K,
            N_NONZERO_BLOCKS,
            NZB_BLOCK_SIZE,
            num_warps=8
        )

    # Unblock / unpad output
    y = unpermute_unpad_block(y_blk, C_out)
    return y

# ---------------------------------------------------------------------------
# Backward
# ---------------------------------------------------------------------------

def block_sparse_conv_1d_backward(
    dy: torch.Tensor,              # [B, C_out, T]
    x: torch.Tensor,               # [B, C_in,  T]
    coo_block_coords: torch.Tensor,# [Nnz, 2]
    values: torch.Tensor,          # [Nnz, M_out, M_in, K]
    BLOCK_SIZE_M: int,
    BLOCK_SIZE_N: int,
    NZB_BLOCK_SIZE: int = 16
):
    """
    Backward pass:
      Given upstream grad dy, compute dx and dvalues.

    Returns:
      dx:       [B, C_in,  T]
      dvalues:  [Nnz, M_out, M_in, K] (same layout as input 'values')
    """
    B, C_in, T = x.shape
    Bdy, C_out, Tdy = dy.shape
    assert Bdy == B and Tdy == T, "Shape mismatch dy vs x."

    N_NONZERO_BLOCKS = coo_block_coords.shape[0]
    K = values.shape[-1]

    # Block/pad independently
    x_blk,  orig_C_in,  padded_C_in,  n_in_blocks  = pad_block_permute(x,  BLOCK_SIZE_M)
    dy_blk, orig_C_out, padded_C_out, n_out_blocks = pad_block_permute(dy, BLOCK_SIZE_M)

    # Prepare dx (blocked) & permuted weights
    dx_blk = torch.zeros_like(x_blk)
    values_perm = values.permute(0, 3, 1, 2).contiguous()  # [Nnz, K, M_out, M_in]

    n_time_tiles = (T + BLOCK_SIZE_N - 1) // BLOCK_SIZE_N
    grid_dx = (
        (N_NONZERO_BLOCKS + NZB_BLOCK_SIZE - 1) // NZB_BLOCK_SIZE,
        n_time_tiles,
        B
    )

    with torch.cuda.device(x.device):
        block_sparse_conv_1d_bwd_dx_kernel[grid_dx](
            dy_blk,
            dx_blk,
            coo_block_coords.int(),
            values_perm,
            B,
            n_in_blocks,
            n_out_blocks,
            T,
            BLOCK_SIZE_M,
            BLOCK_SIZE_N,
            K,
            N_NONZERO_BLOCKS,
            NZB_BLOCK_SIZE,
            num_warps=8
        )

    dx = unpermute_unpad_block(dx_blk, C_in)

    # dvalues accumulation (perm shape)
    dvalues_perm = torch.zeros_like(values_perm)

    grid_dv = (
        (N_NONZERO_BLOCKS + NZB_BLOCK_SIZE - 1) // NZB_BLOCK_SIZE,
        n_time_tiles,
        B
    )

    with torch.cuda.device(x.device):
        block_sparse_conv_1d_bwd_dvalues_kernel[grid_dv](
            x_blk,
            dy_blk,
            coo_block_coords.int(),
            dvalues_perm,
            B,
            n_in_blocks,
            n_out_blocks,
            T,
            BLOCK_SIZE_M,
            BLOCK_SIZE_N,
            K,
            N_NONZERO_BLOCKS,
            NZB_BLOCK_SIZE,
            num_warps=4
        )

    # Return dvalues to original layout [Nnz, M_out, M_in, K]
    dvalues = dvalues_perm.permute(0, 2, 3, 1).contiguous()
    return dx, dvalues


class BlockSparseConv1dFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, coo_block_coords, values, kernel_shape,
                BLOCK_SIZE_M, BLOCK_SIZE_N, NZB_BLOCK_SIZE=16):
        y = block_sparse_conv_1d_forward(
            x, coo_block_coords, values, kernel_shape,
            BLOCK_SIZE_M, BLOCK_SIZE_N, NZB_BLOCK_SIZE
        )
        ctx.save_for_backward(x, coo_block_coords, values)
        ctx.kernel_shape = kernel_shape
        ctx.BLOCK_SIZE_M = BLOCK_SIZE_M
        ctx.BLOCK_SIZE_N = BLOCK_SIZE_N
        ctx.NZB_BLOCK_SIZE = NZB_BLOCK_SIZE
        return y

    @staticmethod
    def backward(ctx, dy):
        x, coo_block_coords, values = ctx.saved_tensors
        dx, dvalues = block_sparse_conv_1d_backward(
            dy, x, coo_block_coords, values,
            ctx.BLOCK_SIZE_M, ctx.BLOCK_SIZE_N, ctx.NZB_BLOCK_SIZE
        )
        # None for non-tensor args
        return dx, None, dvalues, None, None, None, None

def block_sparse_conv_1d(x, coo_block_coords, values, kernel_shape,
                         BLOCK_SIZE_M, BLOCK_SIZE_N, NZB_BLOCK_SIZE=16):
    return BlockSparseConv1dFn.apply(
        x, coo_block_coords, values, kernel_shape,
        BLOCK_SIZE_M, BLOCK_SIZE_N, NZB_BLOCK_SIZE
    )