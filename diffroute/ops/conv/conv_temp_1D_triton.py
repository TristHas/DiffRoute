import triton
import triton.language as tl

@triton.jit
def block_sparse_conv_1d_fwd_kernel(
    x_ptr,               # [B, n_in_blocks,  T, BLOCK_SIZE_M]
    coo_ptr,             # [Nnz, 2] (r_block, c_block)
    values_ptr,          # [Nnz, K, BLOCK_SIZE_M, BLOCK_SIZE_M]  (perm layout)
    y_ptr,               # [B, n_out_blocks, T, BLOCK_SIZE_M]
    B: tl.int32,
    n_in_blocks: tl.int32,
    n_out_blocks: tl.int32,
    n_time_steps: tl.int32,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    KERNEL_SIZE: tl.constexpr,
    N_NONZERO_BLOCKS: tl.int32,
    NZB_BLOCK_SIZE: tl.constexpr,
):
    # ------------------------------------------------------------
    # Program-IDs
    # ------------------------------------------------------------
    tile_nzb = tl.program_id(0)   # which NZB group
    n_tile   = tl.program_id(1)   # which time tile
    b_idx    = tl.program_id(2)   # which batch

    # ------------------------------------------------------------
    # Strides
    # ------------------------------------------------------------
    in_batch_stride  = n_in_blocks  * n_time_steps * BLOCK_SIZE_M
    out_batch_stride = n_out_blocks * n_time_steps * BLOCK_SIZE_M
    in_block_stride  = n_time_steps * BLOCK_SIZE_M
    out_block_stride = n_time_steps * BLOCK_SIZE_M

    # Base pointers for this batch
    x_base = x_ptr + b_idx * in_batch_stride
    y_base = y_ptr + b_idx * out_batch_stride

    # Infer output dtype from y (works for fp16/fp32)
    # mask=False is OK; we just need dtype, the value doesn't matter.
    y0 = tl.load(y_base, mask=False, other=0)
    y_dtype = y0.dtype

    # ------------------------------------------------------------
    # Time tile indexes
    # ------------------------------------------------------------
    t_range = n_tile * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    t_mask  = t_range < n_time_steps
    t_offset_out = t_range * BLOCK_SIZE_M  # used for y writes

    out_idx = tl.arange(0, BLOCK_SIZE_M)[:, None]   # (M,1)
    in_idx  = tl.arange(0, BLOCK_SIZE_M)[None, :]   # (1,M)

    # Group accumulator: FP32 accumulation
    group_acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    group_active = False
    prev_r_block = 0

    # ------------------------------------------------------------
    # Loop over NZB_BLOCK_SIZE non-zero blocks
    # ------------------------------------------------------------
    for i in range(NZB_BLOCK_SIZE):
        nzb = tile_nzb * NZB_BLOCK_SIZE + i
        if nzb < N_NONZERO_BLOCKS:
            r_block = tl.load(coo_ptr + nzb * 2 + 0)  # output block
            c_block = tl.load(coo_ptr + nzb * 2 + 1)  # input  block

            # Offsets
            x_block_off = c_block * in_block_stride
            block_weight_base = nzb * (KERNEL_SIZE * BLOCK_SIZE_M * BLOCK_SIZE_M)

            # Tile accumulator: FP32 accumulation
            tile_acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

            # ----------------------------------------------------
            # Sum over kernel taps
            # ----------------------------------------------------
            for k in range(KERNEL_SIZE):
                w_k_off = block_weight_base + k * (BLOCK_SIZE_M * BLOCK_SIZE_M)
                w_idx   = w_k_off + out_idx * BLOCK_SIZE_M + in_idx

                # Load weights, promote to FP32 for accumulation
                w = tl.load(values_ptr + w_idx, mask=True, other=0.0).to(tl.float32)

                t_in = t_range + k - (KERNEL_SIZE - 1)
                in_time_valid = (t_in >= 0) & (t_in < n_time_steps)

                x_ptrs = (
                    x_base
                    + x_block_off
                    + (t_in * BLOCK_SIZE_M)[None, :]
                    + out_idx
                )

                # Load activations, promote to FP32 for accumulation
                x = tl.load(x_ptrs, mask=in_time_valid[None, :], other=0.0).to(tl.float32)

                tile_acc += tl.dot(w, x)  # FP32 accumulate

            # ----------------------------------------------------
            # Group reduction on identical r_block
            # ----------------------------------------------------
            if not group_active:
                group_acc = tile_acc
                prev_r_block = r_block
                group_active = True
            else:
                if r_block == prev_r_block:
                    group_acc += tile_acc
                else:
                    # commit previous group (cast to y dtype right before atomic)
                    y_ptrs = (
                        y_base
                        + prev_r_block * out_block_stride
                        + t_offset_out[None, :]
                        + out_idx
                    )
                    tl.atomic_add(y_ptrs, group_acc.to(y_dtype), mask=t_mask[None, :])

                    # start new group
                    group_acc = tile_acc
                    prev_r_block = r_block

    # Final commit
    if group_active:
        y_ptrs = (
            y_base
            + prev_r_block * out_block_stride
            + t_offset_out[None, :]
            + out_idx
        )
        tl.atomic_add(y_ptrs, group_acc.to(y_dtype), mask=t_mask[None, :])


# ------------------------------------------------------------
# Backward - dx kernel
# ------------------------------------------------------------
@triton.jit
def block_sparse_conv_1d_bwd_dx_kernel(
    dy_ptr,            # [B, n_out_blocks, T, M]
    dx_ptr,            # [B, n_in_blocks,  T, M]
    coo_ptr,           # [Nnz, 2]
    values_ptr,        # [Nnz, K, M, M]
    B: tl.int32,
    n_in_blocks: tl.int32,
    n_out_blocks: tl.int32,
    n_time_steps: tl.int32,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    KERNEL_SIZE: tl.constexpr,
    N_NONZERO_BLOCKS: tl.constexpr,
    NZB_BLOCK_SIZE: tl.constexpr,
):
    t_range = tl.program_id(1) * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    t_mask  = t_range < n_time_steps
    tile_nzb = tl.program_id(0)
    b_idx    = tl.program_id(2)

    # Strides
    out_batch_stride = n_out_blocks * n_time_steps * BLOCK_SIZE_M
    in_batch_stride  = n_in_blocks  * n_time_steps * BLOCK_SIZE_M
    out_block_stride = n_time_steps * BLOCK_SIZE_M
    in_block_stride  = n_time_steps * BLOCK_SIZE_M

    dy_base = dy_ptr + b_idx * out_batch_stride
    dx_base = dx_ptr + b_idx * in_batch_stride

    out_idx = tl.arange(0, BLOCK_SIZE_M)[:, None]
    in_idx  = tl.arange(0, BLOCK_SIZE_M)[None, :]

    padded_out_channels = n_out_blocks * BLOCK_SIZE_M
    padded_in_channels  = n_in_blocks  * BLOCK_SIZE_M

    for i in range(NZB_BLOCK_SIZE):
        nzb = tile_nzb * NZB_BLOCK_SIZE + i
        if nzb < N_NONZERO_BLOCKS:
            r_block = tl.load(coo_ptr + nzb * 2 + 0)  # dy block
            c_block = tl.load(coo_ptr + nzb * 2 + 1)  # dx block

            dy_block_ptr = dy_base + r_block * out_block_stride
            dx_block_ptr = dx_base + c_block * in_block_stride
            weight_block_base = nzb * (KERNEL_SIZE * BLOCK_SIZE_M * BLOCK_SIZE_M)

            # masks for kernel load
            m_range = r_block * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
            n_range = c_block * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
            m_mask  = m_range < padded_out_channels
            n_mask  = n_range < padded_in_channels
            block_mask = m_mask[:, None] & n_mask[None, :]

            for k in range(KERNEL_SIZE):
                t_in = t_range + (k - (KERNEL_SIZE - 1))
                valid_time = (t_in >= 0) & (t_in < n_time_steps)

                w_off = weight_block_base + k * (BLOCK_SIZE_M * BLOCK_SIZE_M)
                w_idx = w_off + out_idx * BLOCK_SIZE_M + in_idx
                kernel_vals = tl.load(values_ptr + w_idx, mask=block_mask, other=0.0)

                dy_ptrs = (
                    dy_block_ptr
                    + t_range[None, :] * BLOCK_SIZE_M
                    + out_idx
                )
                dy_tile = tl.load(dy_ptrs, mask=t_mask[None, :], other=0.0)

                partial_dx = tl.dot(tl.trans(kernel_vals), dy_tile)

                dx_ptrs = (
                    dx_block_ptr
                    + t_in[None, :] * BLOCK_SIZE_M
                    + tl.arange(0, BLOCK_SIZE_M)[:, None]
                )
                tl.atomic_add(dx_ptrs, partial_dx, mask=valid_time[None, :])
# ------------------------------------------------------------
# Backward - dW kernel
# ------------------------------------------------------------
@triton.jit
def block_sparse_conv_1d_bwd_dvalues_kernel(
    x_ptr,               # [B, n_in_blocks,  T, M]
    dy_ptr,              # [B, n_out_blocks, T, M]
    coo_ptr,             # [Nnz, 2]
    dvalues_ptr,         # [Nnz, K, M, M]
    B: tl.int32,
    n_in_blocks: tl.int32,
    n_out_blocks: tl.int32,
    n_time_steps: tl.int32,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    KERNEL_SIZE: tl.constexpr,
    N_NONZERO_BLOCKS: tl.int32,
    NZB_BLOCK_SIZE: tl.constexpr,
):
    tile_nzb = tl.program_id(0)
    n_tile   = tl.program_id(1)
    b_idx    = tl.program_id(2)

    t_range = n_tile * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    t_mask  = t_range < n_time_steps

    in_batch_stride  = n_in_blocks  * n_time_steps * BLOCK_SIZE_M
    out_batch_stride = n_out_blocks * n_time_steps * BLOCK_SIZE_M
    in_block_stride  = n_time_steps * BLOCK_SIZE_M
    out_block_stride = n_time_steps * BLOCK_SIZE_M

    x_base  = x_ptr  + b_idx * in_batch_stride
    dy_base = dy_ptr + b_idx * out_batch_stride

    out_idx = tl.arange(0, BLOCK_SIZE_M)[:, None]
    in_idx  = tl.arange(0, BLOCK_SIZE_M)[None, :]

    for i in range(NZB_BLOCK_SIZE):
        nzb = tile_nzb * NZB_BLOCK_SIZE + i
        if nzb < N_NONZERO_BLOCKS:
            r_block = tl.load(coo_ptr + nzb * 2 + 0)
            c_block = tl.load(coo_ptr + nzb * 2 + 1)

            dy_block_ptr = dy_base + r_block * out_block_stride
            x_block_ptr  = x_base  + c_block * in_block_stride

            for k in range(KERNEL_SIZE):
                t_in = t_range + (k - (KERNEL_SIZE - 1))
                valid_t = (t_in >= 0) & (t_in < n_time_steps)

                dy_ptrs = (
                    dy_block_ptr
                    + t_range[None, :] * BLOCK_SIZE_M
                    + out_idx
                )
                dy_tile = tl.load(dy_ptrs, mask=t_mask[None, :], other=0.0)

                x_ptrs = (
                    x_block_ptr
                    + t_in[None, :] * BLOCK_SIZE_M
                    + tl.arange(0, BLOCK_SIZE_M)[:, None]
                )
                x_tile = tl.load(x_ptrs, mask=valid_t[None, :], other=0.0)

                partial_dW = tl.dot(dy_tile, tl.trans(x_tile))

                block_size = BLOCK_SIZE_M * BLOCK_SIZE_M
                base = nzb * (KERNEL_SIZE * block_size) + k * block_size
                w_idx = base + out_idx * BLOCK_SIZE_M + in_idx
                tl.atomic_add(dvalues_ptr + w_idx, partial_dW)