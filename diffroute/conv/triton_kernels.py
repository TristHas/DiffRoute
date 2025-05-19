import triton
import triton.language as tl

@triton.jit
def block_sparse_conv_1d_fwd_kernel(
    x_ptr,               # float32*; x_perm: [B, n_blocks, T, BLOCK_SIZE_M]
    coo_ptr,             # int32*;  coo_block_coords: [N_NONZERO_BLOCKS, 2]
    values_ptr,          # float32*; block_values: [N_NONZERO_BLOCKS, KERNEL_SIZE, BLOCK_SIZE_M, BLOCK_SIZE_M]
    y_ptr,               # float32*; y_perm: [B, n_blocks, T, BLOCK_SIZE_M]
    B: tl.int32,
    n_blocks: tl.int32,      
    n_time_steps: tl.int32,
    BLOCK_SIZE_M: tl.constexpr, 
    BLOCK_SIZE_N: tl.constexpr, 
    KERNEL_SIZE: tl.constexpr,
    N_NONZERO_BLOCKS: tl.int32,
    NZB_BLOCK_SIZE: tl.constexpr
):
    """
    Tiled block-sparse 1D convolution kernel with permuted weights layout:
      values: [N_NONZERO_BLOCKS, KERNEL_SIZE, BLOCK_SIZE_M, BLOCK_SIZE_M]
    """
    # ------------------------------------------------------------
    # Program IDs
    # ------------------------------------------------------------
    tile_nzb = tl.program_id(0)
    n_tile   = tl.program_id(1)
    b_idx    = tl.program_id(2)

    # ------------------------------------------------------------
    # Strides and base pointers
    # ------------------------------------------------------------
    batch_stride = n_blocks * n_time_steps * BLOCK_SIZE_M
    block_stride = n_time_steps * BLOCK_SIZE_M

    # Base offsets for the current batch
    b_offset = b_idx * batch_stride
    x_base = x_ptr + b_offset
    y_base = y_ptr + b_offset

    # ------------------------------------------------------------
    # Time tile: [n_tile * BLOCK_SIZE_N, ..., n_tile * BLOCK_SIZE_N + BLOCK_SIZE_N - 1]
    # ------------------------------------------------------------
    time_range = n_tile * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    time_mask  = time_range < n_time_steps
    time_offset = time_range * BLOCK_SIZE_M

    # Channel indices
    out_idx = tl.arange(0, BLOCK_SIZE_M)[:, None]  # shape [BLOCK_SIZE_M, 1]
    in_idx  = tl.arange(0, BLOCK_SIZE_M)[None, :]  # shape [1, BLOCK_SIZE_M]

    # Accumulator for merges on the same r_block
    group_acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), tl.float32)
    group_started = False
    prev_r_block = 0

    # ------------------------------------------------------------
    # Loop over the blocks assigned to this kernel instance
    # ------------------------------------------------------------
    for i in range(NZB_BLOCK_SIZE):
        nzb = tile_nzb * NZB_BLOCK_SIZE + i
        if nzb < N_NONZERO_BLOCKS:
            # load the current block's (r_block, c_block)
            curr_r_block = tl.load(coo_ptr + nzb * 2 + 0)
            curr_c_block = tl.load(coo_ptr + nzb * 2 + 1)
            # Base offset into the 4D weights (permuted to [nzb, k, out_ch, in_ch])
            #  => each block has shape [KERNEL_SIZE, BLOCK_SIZE_M, BLOCK_SIZE_M]
            block_weight_size = KERNEL_SIZE * BLOCK_SIZE_M * BLOCK_SIZE_M
            weight_base = nzb * block_weight_size
            # Input block offset for the current c_block
            x_block_offset = curr_c_block * block_stride
            # Temporary accumulator for current block
            tile_acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), tl.float32)

            # ------------------------------------------------------------
            # Sum over kernel_size
            # ------------------------------------------------------------
            for k in range(KERNEL_SIZE):
                w_k_offset = weight_base + k * (BLOCK_SIZE_M * BLOCK_SIZE_M)
                weight_index = w_k_offset + out_idx * BLOCK_SIZE_M + in_idx
                kernel_vals = tl.load(values_ptr + weight_index, mask=True, other=0.0)
                
                t_idx = time_range + k - (KERNEL_SIZE - 1)
                valid_time = (t_idx >= 0) & (t_idx < n_time_steps)
                x_ptrs = (
                    x_base
                    + x_block_offset
                    + (t_idx * BLOCK_SIZE_M)[None, :]
                    + tl.arange(0, BLOCK_SIZE_M)[:, None]
                )
                x_vals = tl.load(
                    x_ptrs, 
                    mask=(time_mask[None, :] & valid_time[None, :]),
                    other=0.0
                )
                tile_acc += tl.dot(kernel_vals, x_vals)

            # ------------------------------------------------------------
            # Accumulate into group_acc if same r_block, otherwise store
            # ------------------------------------------------------------
            if not group_started:
                group_acc = tile_acc
                prev_r_block = curr_r_block
                group_started = True
            else:
                if curr_r_block == prev_r_block:
                    group_acc += tile_acc
                else:
                    # commit the old group
                    y_ptrs = (
                        y_base 
                        + prev_r_block * block_stride
                        + time_offset[None, :]
                        + tl.arange(0, BLOCK_SIZE_M)[:, None]
                    )
                    tl.atomic_add(y_ptrs, group_acc, mask=time_mask[None, :])
                    # start new group
                    group_acc = tile_acc
                    prev_r_block = curr_r_block

    # ------------------------------------------------------------
    # Final commit
    # ------------------------------------------------------------
    if group_started:
        y_ptrs = (
            y_base
            + prev_r_block * block_stride
            + time_offset[None, :]
            + tl.arange(0, BLOCK_SIZE_M)[:, None]
        )
        tl.atomic_add(y_ptrs, group_acc, mask=time_mask[None, :])

@triton.jit
def block_sparse_conv_1d_bwd_dx_kernel(
    dy_ptr,            # float32*; dy_perm: [B, n_blocks, T, BLOCK_SIZE_M]
    dx_ptr,            # float32*; dx_perm: [B, n_blocks, T, BLOCK_SIZE_M]
    coo_ptr,           # int32*;  coo_block_coords: [N_NONZERO_BLOCKS, 2]
    values_ptr,        # float32*; block_values: [N_NONZERO_BLOCKS, KERNEL_SIZE, BLOCK_SIZE_M, BLOCK_SIZE_M]
    B: tl.int32,
    n_channels: tl.int32,   # padded number of channels (n_blocks * BLOCK_SIZE_M)
    n_time_steps: tl.int32,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    KERNEL_SIZE: tl.constexpr,
    N_NONZERO_BLOCKS: tl.constexpr,
    NZB_BLOCK_SIZE: tl.constexpr
):
    """
    Backward kernel to compute gradient with respect to input x.
    Assumes that dx and dy are in a permuted layout:
       [B, n_blocks, T, BLOCK_SIZE_M],
    where n_blocks = n_channels // BLOCK_SIZE_M.

    Key points of the "shift the store" method:
      - In forward, we did y[t_out] += W[k] * x[t_in].
      - So in backward wrt x, we must do dx[t_in] += W[k]^T * dy[t_out].
      - Here, t_in = t_out + (k - (KERNEL_SIZE-1)).
      - That means we keep dy loaded at time_range (the unshifted index) but
        we store into dx at the shifted index t_in.
    """

    # Grid indices: tile over nonzero blocks (tile_nzb) and time tiles (n_tile), plus batch b_idx.
    tile_nzb = tl.program_id(0)  # nonzero block tile index
    n_tile   = tl.program_id(1)  # time tile index
    b_idx    = tl.program_id(2)  # batch index

    # Time tile indices in [0..n_time_steps).
    time_range = n_tile * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    time_mask  = time_range < n_time_steps

    # Compute strides for permuted layout [B, n_blocks, T, BLOCK_SIZE_M].
    n_blocks = n_channels // BLOCK_SIZE_M
    batch_stride = n_blocks * n_time_steps * BLOCK_SIZE_M
    block_stride = n_time_steps * BLOCK_SIZE_M

    # Loop over the slice of nonzero blocks that this kernel instance handles.
    for i in range(NZB_BLOCK_SIZE):
        nzb = tile_nzb * NZB_BLOCK_SIZE + i
        if nzb < N_NONZERO_BLOCKS:
            # Load (r_block, c_block) => r_block is where dy is found, c_block is where dx is stored.
            r_block = tl.load(coo_ptr + nzb * 2 + 0)  # output-ch block index (dy)
            c_block = tl.load(coo_ptr + nzb * 2 + 1)  # input-ch block index (dx)

            # Base pointers for dy and dx in this batch.
            dy_base = dy_ptr + b_idx * batch_stride
            dx_base = dx_ptr + b_idx * batch_stride

            # Offsets for the relevant blocks in dy and dx.
            dy_block_ptr = dy_base + r_block * block_stride
            dx_block_ptr = dx_base + c_block * block_stride

            # Precompute a base offset in the weights array.
            # weights layout = [N_NONZERO_BLOCKS, KERNEL_SIZE, BLOCK_SIZE_M, BLOCK_SIZE_M].
            # Each block has size (KERNEL_SIZE * BLOCK_SIZE_M * BLOCK_SIZE_M).
            weight_block_base = nzb * (KERNEL_SIZE * BLOCK_SIZE_M * BLOCK_SIZE_M)

            # Indices for the 2D block in [BLOCK_SIZE_M, BLOCK_SIZE_M].
            out_idx = tl.arange(0, BLOCK_SIZE_M)[:, None]
            in_idx  = tl.arange(0, BLOCK_SIZE_M)[None, :]
            
            m_range = r_block * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
            n_range = c_block * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
            m_mask = m_range < n_channels
            n_mask = n_range < n_channels

            # ------------------------------------------
            # Loop over each kernel tap k
            # ------------------------------------------
            # For the forward pass:
            #   t_out = time_range
            #   t_in  = time_range + (k - (KERNEL_SIZE - 1))
            # So here we do dx[t_in] += W[k]^T * dy[t_out].
            # We'll do an atomic add *inside* this loop for each tap.
            for k in range(KERNEL_SIZE):
                # Compute shifted time index for x (dx).
                t_in = time_range + (k - (KERNEL_SIZE - 1))
                valid_time = (t_in >= 0) & (t_in < n_time_steps)

                # Load the appropriate slice of W. This is the k-th tap for the (r_block, c_block).
                w_offset_k = weight_block_base + k * (BLOCK_SIZE_M * BLOCK_SIZE_M)
                w_indices  = w_offset_k + out_idx * BLOCK_SIZE_M + in_idx
                # kernel_vals has shape [BLOCK_SIZE_M, BLOCK_SIZE_M],
                # matching forward's W[k].
                block_mask = m_mask[:, None] & n_mask[None, :]
                
                kernel_vals = tl.load(
                    values_ptr + w_indices,
                    mask=block_mask,
                    other=0.0
                )

                # Load dy at the *unshifted* time_range => t_out
                # forward used y[t_out], so backward uses dy[t_out].
                # We want a tile shaped [BLOCK_SIZE_M, BLOCK_SIZE_N], same as forward’s logic.
                dy_tile_ptr = (
                    dy_block_ptr
                    + time_range[None, :] * BLOCK_SIZE_M
                    + tl.arange(0, BLOCK_SIZE_M)[:, None]
                )
                dy_tile = tl.load(dy_tile_ptr, mask=time_mask[None, :], other=0.0)

                # partial_dx = W[k]^T * dy[t_out]
                partial_dx = tl.dot(tl.trans(kernel_vals), dy_tile)

                # Atomically add partial_dx into dx[t_in], for each valid time index.
                dx_tile_ptr = (
                    dx_block_ptr
                    + t_in[None, :] * BLOCK_SIZE_M
                    + tl.arange(0, BLOCK_SIZE_M)[:, None]
                )

                # Combine the existing time_mask (for t_out) with valid_time for t_in.
                # If time_range is out of bounds, dy_tile was loaded as 0 anyway.
                # But we definitely must not store if t_in is out of range.
                store_mask = time_mask & valid_time
                tl.atomic_add(dx_tile_ptr, partial_dx, mask=store_mask[None, :])
            # end for k
        # end if
    # end for

@triton.jit
def block_sparse_conv_1d_bwd_dvalues_kernel(
    x_ptr,               # float32*; x_perm:  [B, n_blocks, T, BLOCK_SIZE_M]
    dy_ptr,              # float32*; dy_perm: [B, n_blocks, T, BLOCK_SIZE_M]
    coo_ptr,             # int32*;   coo_block_coords: [N_NONZERO_BLOCKS, 2]
    dvalues_ptr,         # float32*; dvalues_perm: [N_NONZERO_BLOCKS, KERNEL_SIZE, BLOCK_SIZE_M, BLOCK_SIZE_M]
    B: tl.int32,
    n_blocks: tl.int32,
    n_time_steps: tl.int32,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    KERNEL_SIZE: tl.constexpr,
    N_NONZERO_BLOCKS: tl.int32,
    NZB_BLOCK_SIZE: tl.constexpr
):
    """
    Compute dW for block-sparse 1D convolution in the same permuted layout as forward.
    - x_perm and dy_perm both have shape: [B, n_blocks, T, BLOCK_SIZE_M].
    - We tile over time dimension (BLOCK_SIZE_N) and loop over NZB_BLOCK_SIZE nonzero blocks.

    For a given block (r_block, c_block):
      - r_block => which chunk of dy_perm we read (the 'out' channels).
      - c_block => which chunk of x_perm we read (the 'in' channels).
    Then for each time tile, we do an outer product of dy_tile and x_tile
    and accumulate into dvalues for that block.
    """

    # ------------------------------------------------------------
    # Program IDs
    # ------------------------------------------------------------
    tile_nzb = tl.program_id(0)  # which group of nonzero blocks we handle
    n_tile   = tl.program_id(1)  # which tile in time dimension
    b_idx    = tl.program_id(2)  # batch index

    # ------------------------------------------------------------
    # Time tile: [n_tile*BLOCK_SIZE_N, ..., n_tile*BLOCK_SIZE_N + BLOCK_SIZE_N - 1]
    # ------------------------------------------------------------
    time_range = n_tile * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    time_mask  = time_range < n_time_steps

    # ------------------------------------------------------------
    # Strides for [B, n_blocks, T, BLOCK_SIZE_M]
    # ------------------------------------------------------------
    batch_stride = n_blocks * n_time_steps * BLOCK_SIZE_M
    block_stride = n_time_steps * BLOCK_SIZE_M

    # Base offsets for current batch in x and dy
    x_base  = x_ptr  + b_idx * batch_stride
    dy_base = dy_ptr + b_idx * batch_stride

    # Channel indices for our local [BLOCK_SIZE_M, BLOCK_SIZE_M] block
    out_idx = tl.arange(0, BLOCK_SIZE_M)[:, None]  # shape (BLOCK_SIZE_M, 1)
    in_idx  = tl.arange(0, BLOCK_SIZE_M)[None, :]  # shape (1, BLOCK_SIZE_M)

    # ------------------------------------------------------------
    # Loop over a subset of the nonzero blocks assigned to this kernel instance
    # ------------------------------------------------------------
    for i in range(NZB_BLOCK_SIZE):
        nzb = tile_nzb * NZB_BLOCK_SIZE + i
        if nzb < N_NONZERO_BLOCKS:
            # load (r_block, c_block)
            r_block = tl.load(coo_ptr + nzb * 2 + 0)
            c_block = tl.load(coo_ptr + nzb * 2 + 1)

            # Offsets into [n_blocks, T, BLOCK_SIZE_M]
            dy_block_offset = r_block * block_stride
            x_block_offset  = c_block * block_stride

            # ------------------------------------------------------------
            # Loop over each kernel tap k
            # ------------------------------------------------------------
            for k in range(KERNEL_SIZE):
                # Shifted time index for x
                t_in = time_range + (k - (KERNEL_SIZE - 1))

                # Boolean mask for x's valid time range
                valid_time = (t_in >= 0) & (t_in < n_time_steps)

                # -----------------------------
                # Load dy tile: shape [BLOCK_SIZE_M, BLOCK_SIZE_N]
                #   from (r_block, time_range)
                # -----------------------------
                dy_ptrs = (
                    dy_base
                    + dy_block_offset
                    + time_range[None, :] * BLOCK_SIZE_M
                    + out_idx
                )
                dy_tile = tl.load(
                    dy_ptrs,
                    mask=time_mask[None, :],
                    other=0.0
                )

                # -----------------------------
                # Load x tile: shape [BLOCK_SIZE_M, BLOCK_SIZE_N]
                #   from (c_block, t_in)
                # -----------------------------
                x_ptrs = (
                    x_base
                    + x_block_offset
                    + t_in[None, :] * BLOCK_SIZE_M
                    + tl.arange(0, BLOCK_SIZE_M)[:, None]  # Use a column vector here.
                )
                x_tile = tl.load(
                    x_ptrs,
                    mask=valid_time[None, :],
                    other=0.0
                )

                # -----------------------------
                # Outer-product accumulation:
                #   partial_dW = dy_tile @ x_tile^T  => shape [BLOCK_SIZE_M, BLOCK_SIZE_M]
                # -----------------------------
                partial_dW = tl.dot(dy_tile, tl.trans(x_tile))

                # -----------------------------
                # Atomic-add into dvalues
                #   dvalues_perm: [N_NONZERO_BLOCKS, KERNEL_SIZE, BLOCK_SIZE_M, BLOCK_SIZE_M]
                # -----------------------------
                block_size     = BLOCK_SIZE_M * BLOCK_SIZE_M
                block_offset   = nzb * (KERNEL_SIZE * block_size)
                k_offset       = k * block_size
                dvalues_base   = block_offset + k_offset
                dvalues_index  = dvalues_base + out_idx * BLOCK_SIZE_M + in_idx

                tl.atomic_add(dvalues_ptr + dvalues_index, partial_dW)







def block_sparse_conv_1d_backward(dy, x, coo_block_coords, values, BLOCK_SIZE_M, BLOCK_SIZE_N):
    """
    Backward pass for the block-sparse 1D convolution.
    
    dy:               (B, C, T) gradient of the output
    x:                (B, C, T) original input
    coo_block_coords: (N_NONZERO_BLOCKS, 2) int32
    values:           (N_NONZERO_BLOCKS, BLOCK_SIZE_M, BLOCK_SIZE_M, KERNEL_SIZE) float32
    Returns:
       dx:      gradient with respect to x (shape: [B, C, T])
       dvalues: gradient with respect to the kernel values (same shape as values)
    """
    B, n_channels, n_time_steps = x.shape
    N_NONZERO_BLOCKS = coo_block_coords.shape[0]
    KERNEL_SIZE = values.shape[-1]

    dx = torch.zeros_like(x)
    dvalues = torch.zeros_like(values)

    grid = (
        (n_channels + BLOCK_SIZE_M - 1) // BLOCK_SIZE_M,
        (n_time_steps + BLOCK_SIZE_N - 1) // BLOCK_SIZE_N,
        B
    )
    with torch.cuda.device(x.device):
        block_sparse_conv_1d_bwd_dx_kernel[grid](
            dy, dx, coo_block_coords, values,
            B, n_channels, n_time_steps,
            BLOCK_SIZE_M, BLOCK_SIZE_N, KERNEL_SIZE, N_NONZERO_BLOCKS,
            num_warps=4
        )
    
        block_sparse_conv_1d_bwd_dvalues_kernel[grid](
            dy, x, dvalues, coo_block_coords,
            B, n_channels, n_time_steps,
            BLOCK_SIZE_M, BLOCK_SIZE_N, KERNEL_SIZE, N_NONZERO_BLOCKS,
            num_warps=4
        )

    return dx, dvalues

def block_sparse_conv_1d_backward_old(dy, x, coo_block_coords, values, BLOCK_SIZE_M, BLOCK_SIZE_N, NZB_BLOCK_SIZE=16):
    """
    Backward pass for the block-sparse 1D convolution.
    Computes gradients for input (dx) and kernel values (dvalues).
    """
    B, n_channels, n_time_steps = x.shape
    N_NONZERO_BLOCKS = coo_block_coords.shape[0]
    KERNEL_SIZE = values.shape[-1]

    dx = torch.zeros_like(x)
    dvalues = torch.zeros_like(values)

    # For dx, launch over NZB tiles and time tiles.
    n_time_tiles = (n_time_steps + BLOCK_SIZE_N - 1) // BLOCK_SIZE_N
    grid_dx = (
        (N_NONZERO_BLOCKS + NZB_BLOCK_SIZE - 1) // NZB_BLOCK_SIZE,  # nonzero block tiles
        n_time_tiles,                                              # time tiles
        B                                                          # batch dimension
    )
    with torch.cuda.device(x.device):
        block_sparse_conv_1d_bwd_dx_kernel[grid_dx](
            dy, dx, coo_block_coords, values,
            B, n_channels, n_time_steps,
            BLOCK_SIZE_M, BLOCK_SIZE_N, KERNEL_SIZE, N_NONZERO_BLOCKS, NZB_BLOCK_SIZE,
            num_warps=8
        )

        # dvalues kernel remains unchanged.
        grid = (
            (n_channels + BLOCK_SIZE_M - 1) // BLOCK_SIZE_M,
            (n_time_steps + BLOCK_SIZE_N - 1) // BLOCK_SIZE_N,
            B
        )
        block_sparse_conv_1d_bwd_dvalues_kernel[grid](
            dy, x, dvalues, coo_block_coords,
            B, n_channels, n_time_steps,
            BLOCK_SIZE_M, BLOCK_SIZE_N, KERNEL_SIZE, N_NONZERO_BLOCKS,
            num_warps=4
        )

    return dx, dvalues


def block_sparse_conv_1d_backward_old(dy, x, coo_block_coords, values, BLOCK_SIZE_M, BLOCK_SIZE_N, NZB_BLOCK_SIZE=16):
    """
    Backward pass for the block-sparse 1D convolution.
    Computes gradients for input (dx) and kernel values (dvalues).
    
    This version changes the layout for dx (and dy) so that channels are
    grouped into contiguous blocks, as in the forward pass.
    """
    B, n_channels, n_time_steps = x.shape
    N_NONZERO_BLOCKS = coo_block_coords.shape[0]
    KERNEL_SIZE = values.shape[-1]

    # --- dx computation with permuted layout ---
    # Compute number of channel blocks and pad channels if needed.
    n_blocks = (n_channels + BLOCK_SIZE_M - 1) // BLOCK_SIZE_M
    new_channels = n_blocks * BLOCK_SIZE_M
    if new_channels != n_channels:
        # Pad dy to match padded channel count.
        dy_pad = F.pad(dy, (0, 0, 0, new_channels - n_channels))
    else:
        dy_pad = dy
    # Permute dy to [B, n_blocks, T, BLOCK_SIZE_M]
    dy_perm = dy_pad.view(B, n_blocks, BLOCK_SIZE_M, n_time_steps).permute(0, 1, 3, 2).contiguous()
    # Allocate dx in the permuted layout.
    dx_perm = torch.zeros_like(dy_perm)
    values_perm = values.permute(0, 3, 1, 2).contiguous()

    n_time_tiles = (n_time_steps + BLOCK_SIZE_N - 1) // BLOCK_SIZE_N
    grid_dx = (
        (N_NONZERO_BLOCKS + NZB_BLOCK_SIZE - 1) // NZB_BLOCK_SIZE,  # tile over nonzero blocks
        n_time_tiles,                                              # time tiles
        B                                                          # batch
    )
    with torch.cuda.device(x.device):
        block_sparse_conv_1d_bwd_dx_kernel[grid_dx](
            dy_perm, dx_perm, coo_block_coords.int(), values_perm,
            B, new_channels, n_time_steps,
            BLOCK_SIZE_M, BLOCK_SIZE_N, KERNEL_SIZE, N_NONZERO_BLOCKS, NZB_BLOCK_SIZE,
            num_warps=8
        )

    # Permute dx back to [B, C, T] and remove any padded channels.
    dx = dx_perm.permute(0, 1, 3, 2).contiguous().view(B, new_channels, n_time_steps)[:, :n_channels, :]

    # --- dvalues kernel remains unchanged ---
    dvalues = torch.zeros_like(values)
    grid_dvalues = (
        (n_channels + BLOCK_SIZE_M - 1) // BLOCK_SIZE_M,
        (n_time_steps + BLOCK_SIZE_N - 1) // BLOCK_SIZE_N,
        B
    )
    with torch.cuda.device(x.device):
        block_sparse_conv_1d_bwd_dvalues_kernel[grid_dvalues](
            dy, x, dvalues, coo_block_coords,
            B, n_channels, n_time_steps,
            BLOCK_SIZE_M, BLOCK_SIZE_N, KERNEL_SIZE, N_NONZERO_BLOCKS,
            num_warps=4
        )

    return dx, dvalues