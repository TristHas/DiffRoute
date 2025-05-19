import torch
import torch.nn.functional as F
from .triton_kernels import (block_sparse_conv_1d_bwd_dx_kernel, 
                             block_sparse_conv_1d_bwd_dvalues_kernel, 
                             block_sparse_conv_1d_fwd_kernel)

def block_sparse_conv_1d_forward(
    x, 
    coo_block_coords, 
    values,                  # originally shape [N, M, M, K], but we will permute
    BLOCK_SIZE_M, 
    BLOCK_SIZE_N, 
    NZB_BLOCK_SIZE=16
):
    """
    Performs a block-sparse causal 1D convolution with permuted weights:
      values -> shape [N_NONZERO_BLOCKS, KERNEL_SIZE, BLOCK_SIZE_M, BLOCK_SIZE_M]
    Args:
      x: Tensor of shape [B, C, T]
      coo_block_coords: Tensor of shape [N_NONZERO_BLOCKS, 2] => (r_block, c_block)
      values: Tensor of shape [N_NONZERO_BLOCKS, BLOCK_SIZE_M, BLOCK_SIZE_M, KERNEL_SIZE]
              => permuted to [N_NONZERO_BLOCKS, KERNEL_SIZE, BLOCK_SIZE_M, BLOCK_SIZE_M]
      BLOCK_SIZE_M: int
      BLOCK_SIZE_N: int
      NZB_BLOCK_SIZE: int
    Returns:
      y: Tensor of shape [B, C, T]
    """
    B, n_channels, n_time_steps = x.shape
    # The last dimension is KERNEL_SIZE
    # but after permutation we want shape [N, KERNEL_SIZE, M, M]
    # so let's do that once:
    values_perm = values.permute(0, 3, 1, 2).contiguous()
    KERNEL_SIZE = values_perm.shape[1]
    N_NONZERO_BLOCKS = coo_block_coords.shape[0]
    # ----------------------------------------------------------------
    # Pad channels if needed to be multiple-of-BLOCK_SIZE_M
    # ----------------------------------------------------------------
    n_blocks = (n_channels + BLOCK_SIZE_M - 1) // BLOCK_SIZE_M
    new_channels = n_blocks * BLOCK_SIZE_M
    if new_channels != n_channels: x = F.pad(x, (0, 0, 0, new_channels - n_channels))
    # Reshape/permute input to [B, n_blocks, T, BLOCK_SIZE_M]
    B, padded_channels, n_time_steps = x.shape
    x_reshaped = x.view(B, n_blocks, BLOCK_SIZE_M, n_time_steps).permute(0, 1, 3, 2).contiguous()
    # Prepare an output buffer in the same shape
    y_perm = torch.zeros_like(x_reshaped)
    # Compute how many tiles in time dimension
    n_time_tiles = (n_time_steps + BLOCK_SIZE_N - 1) // BLOCK_SIZE_N
    # Triton launch grid
    grid = (
        (N_NONZERO_BLOCKS + NZB_BLOCK_SIZE - 1) // NZB_BLOCK_SIZE,  # tile along nonzero blocks
        n_time_tiles,                                              # tile along time
        B                                                          # tile along batch
    )

    # Launch
    with torch.cuda.device(x.device):
        block_sparse_conv_1d_fwd_kernel[grid](
            x_reshaped, 
            coo_block_coords.int(), 
            values_perm, 
            y_perm,
            B,
            n_blocks,
            n_time_steps,
            BLOCK_SIZE_M,
            BLOCK_SIZE_N,
            KERNEL_SIZE,
            N_NONZERO_BLOCKS,
            NZB_BLOCK_SIZE,
            num_warps=8
    )
    # Undo the permutation to return [B, C, T], then crop any padded channels
    y = y_perm.permute(0, 1, 3, 2).contiguous().view(B, new_channels, n_time_steps)
    y = y[:, :n_channels, :]
    return y

def block_sparse_conv_1d_backward(
    dy, x, coo_block_coords, values,
    BLOCK_SIZE_M, BLOCK_SIZE_N,
    NZB_BLOCK_SIZE=16
):
    """
    Backward pass for the block-sparse 1D convolution.
    Now uses a permuted layout for both dx and dvalues computations.
    Returns:
      dx: same shape as x  ( [B, C, T] )
      dvalues: same shape as values ( [N_NONZERO_BLOCKS, M, M, K] )
    """
    B, n_channels, n_time_steps = x.shape
    N_NONZERO_BLOCKS = coo_block_coords.shape[0]
    KERNEL_SIZE = values.shape[-1]   # original is [N, M, M, K]

    # ----------
    # dx part (unchanged from your snippet)
    # ----------
    # Pad channels if needed for BLOCK_SIZE_M alignment:
    n_blocks = (n_channels + BLOCK_SIZE_M - 1) // BLOCK_SIZE_M
    new_channels = n_blocks * BLOCK_SIZE_M
    if new_channels != n_channels:
        dy_pad = F.pad(dy, (0, 0, 0, new_channels - n_channels))
        x_pad  = F.pad(x,  (0, 0, 0, new_channels - n_channels))
    else:
        dy_pad = dy
        x_pad  = x

    # Permute dy to [B, n_blocks, T, BLOCK_SIZE_M]
    dy_perm = (
        dy_pad.view(B, n_blocks, BLOCK_SIZE_M, n_time_steps)
               .permute(0, 1, 3, 2)
               .contiguous()
    )

    # We'll also need x in permuted form for the dW kernel,
    # so do that once here:
    x_perm = (
        x_pad.view(B, n_blocks, BLOCK_SIZE_M, n_time_steps)
              .permute(0, 1, 3, 2)
              .contiguous()
    )

    # Allocate dx in permuted layout:
    dx_perm = torch.zeros_like(dy_perm)

    # Permute the block-sparse values for the weight matrix as in forward:
    #   values:       [N, M, M, K]
    #   values_perm:  [N, K, M, M]
    values_perm = values.permute(0, 3, 1, 2).contiguous()

    # Launch the dx kernel
    n_time_tiles = (n_time_steps + BLOCK_SIZE_N - 1) // BLOCK_SIZE_N
    grid_dx = (
        (N_NONZERO_BLOCKS + NZB_BLOCK_SIZE - 1) // NZB_BLOCK_SIZE,
        n_time_tiles,
        B
    )
    with torch.cuda.device(x.device):
        block_sparse_conv_1d_bwd_dx_kernel[grid_dx](
            dy_perm, dx_perm, coo_block_coords.int(),
            values_perm,
            B, new_channels, n_time_steps,
            BLOCK_SIZE_M, BLOCK_SIZE_N, KERNEL_SIZE,
            N_NONZERO_BLOCKS, NZB_BLOCK_SIZE,
            num_warps=8
        )

    # Permute dx back to [B, C, T]
    dx = (
        dx_perm.permute(0, 1, 3, 2)
               .contiguous()
               .view(B, new_channels, n_time_steps)
    )
    dx = dx[:, :n_channels, :]  # crop any channel padding

    # ----------
    # dvalues part (UPDATED)
    # ----------
    # We'll accumulate into dvalues in the same permuted shape
    # used by forward: [N_NONZERO_BLOCKS, KERNEL_SIZE, BLOCK_SIZE_M, BLOCK_SIZE_M].
    dvalues_perm = torch.zeros_like(values_perm)

    # Launch our new kernel that reads x_perm & dy_perm in blocked layout
    # and accumulates into dvalues_perm:
    grid_dvalues = (
        (N_NONZERO_BLOCKS + NZB_BLOCK_SIZE - 1) // NZB_BLOCK_SIZE,
        n_time_tiles,
        B
    )
    with torch.cuda.device(x.device):
        block_sparse_conv_1d_bwd_dvalues_kernel[grid_dvalues](
            x_perm, dy_perm, coo_block_coords.int(),
            dvalues_perm,
            B, n_blocks, n_time_steps,
            BLOCK_SIZE_M, BLOCK_SIZE_N, KERNEL_SIZE,
            N_NONZERO_BLOCKS, NZB_BLOCK_SIZE,
            num_warps=4
        )

    # Finally, we want dvalues in the original layout [N, M, M, K].
    # So permute back from [N, K, M, M] => [N, M, M, K].
    dvalues = dvalues_perm.permute(0, 2, 3, 1).contiguous()

    return dx, dvalues



class BlockSparseConv1dFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, coo_block_coords, values,
                BLOCK_SIZE_M, BLOCK_SIZE_N):
        ctx.save_for_backward(x, coo_block_coords, values)
        ctx.BLOCK_SIZE_M = BLOCK_SIZE_M
        ctx.BLOCK_SIZE_N = BLOCK_SIZE_N

        y = block_sparse_conv_1d_forward(x, coo_block_coords, values,
                                         BLOCK_SIZE_M, BLOCK_SIZE_N)
        return y

    @staticmethod
    def backward(ctx, dy):
        x, coo_block_coords, values = ctx.saved_tensors
        BLOCK_SIZE_M = ctx.BLOCK_SIZE_M
        BLOCK_SIZE_N = ctx.BLOCK_SIZE_N
        dx, dvalues = block_sparse_conv_1d_backward(dy, x, coo_block_coords, values,
                                                    BLOCK_SIZE_M, BLOCK_SIZE_N)
        return dx, None, dvalues, None, None

def block_sparse_conv_1d_autograd(x, coo_block_coords, values,
                                  BLOCK_SIZE_M, BLOCK_SIZE_N):
    """
    Convenience API: call the autograd Function.
    """
    return BlockSparseConv1dFunction.apply(x, coo_block_coords, values, BLOCK_SIZE_M, BLOCK_SIZE_N)
