import torch
import triton
import triton.language as tl

CHANNELS = 2 # This should be set within the AggregateIRFFunction context depending on the dtype of taus to be applicable to non-complex numbers

def stable_log(taus: torch.Tensor, epsilon: float = 1e-8) -> torch.Tensor:
    """
    Compute a stable complex logarithm:
      log(z) = log(max(|z|, epsilon)) + i * arg(z)
      This prevents the log from going to -inf for very small |z|.
    """
    amp = torch.abs(taus)
    stable_amp = amp.clamp_min(epsilon)
    return torch.log(stable_amp) + 1j * torch.angle(taus)

def log_aggregate_irf_triton(path_idxs: torch.Tensor,
                             path_nodes: torch.Tensor,
                             taus: torch.Tensor,
                             n_paths: int = None,
                             cascade=1) -> torch.Tensor:
    """
    Convenience wrapper for the custom autograd function.
    """
    log_taus = cascade * stable_log(taus) 
    agg_log_taus = AggregateIRFFunction.apply(path_idxs, path_nodes, log_taus, n_paths) 
    return torch.exp(agg_log_taus)
        
class AggregateIRFFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, path_idxs, path_nodes, log_taus, n_paths=None):
        """
        Forward pass for complex numbers.
          - path_idxs:  [N] int tensor with destination indices.
          - path_nodes: [N] int tensor with source indices.
          - taus:       [num_nodes, time_window] complex tensor.
          - n_paths:    Number of output rows (if None, computed as max(path_idxs)+1).
        """
        # Params
        if n_paths is None: n_paths = int(path_idxs.max().item()) + 1
        time_window = log_taus.shape[-1]
        time_channel_size = time_window * 2
        numel = path_idxs.numel()
        # Complex log form.
        log_taus_cat = torch.stack([log_taus.real, log_taus.imag], dim=-1)
        agg_log = torch.zeros((n_paths, time_window, 2), 
                              dtype=log_taus_cat.dtype, 
                              device=log_taus.device)
        # 
        log_taus_cat_flat = log_taus_cat.view(log_taus_cat.shape[0], -1)
        agg_log_flat = agg_log.view(agg_log.shape[0], -1)
        
        BLOCK_SIZE = 8
        TC_TILE = 16  
        
        grid = lambda meta: (triton.cdiv(numel, meta['BLOCK_SIZE']),)
        with torch.cuda.device(log_taus.device):
            scatter_kernel[grid](
                path_idxs, 
                path_nodes, 
                log_taus_cat_flat, 
                agg_log_flat,
                numel, time_channel_size,
                log_taus_cat_flat.stride(0), 
                log_taus_cat_flat.stride(1),
                agg_log_flat.stride(0), agg_log_flat.stride(1),
                BLOCK_SIZE=BLOCK_SIZE,
                TC_TILE=TC_TILE
            )

        # Convert the aggregated log (with separate channels) back to a complex tensor.
        agg_log_complex = torch.complex(agg_log[..., 0], agg_log[..., 1])

        # Save tensors for backward.
        ctx.save_for_backward(path_idxs, path_nodes, log_taus)
        ctx.n_paths = n_paths
        ctx.time_window = time_window
        return agg_log_complex

    @staticmethod
    def backward(ctx, grad_output):
        """
        Backward pass for complex numbers.
        grad_output: Gradient of the loss with respect to the output (complex tensor).
        """
        path_idxs, path_nodes, log_taus = ctx.saved_tensors
        n_paths = ctx.n_paths
        time_window = ctx.time_window

        # Compute gradient with respect to the aggregated log: dA = grad_output * result.
        dA = grad_output #* result  # complex multiplication
        # Split into two channels.
        dA_cat = torch.stack([dA.real, dA.imag], dim=-1)  # shape: [n_paths, time_window, 2]

        # Allocate gradient accumulator for log_taus.
        grad_log_taus_cat = torch.zeros((log_taus.shape[0], time_window, 2),
                                        dtype=dA_cat.dtype, device=log_taus.device)

        numel = path_idxs.numel()
        BLOCK_SIZE = 1024
        grid = lambda meta: (triton.cdiv(numel, meta['BLOCK_SIZE']),)
        
        with torch.cuda.device(log_taus.device):
            scatter_backward_kernel[grid](
                path_idxs, path_nodes, dA_cat, grad_log_taus_cat,
                numel, time_window, CHANNELS,
                dA_cat.stride(0), dA_cat.stride(1), dA_cat.stride(2),
                grad_log_taus_cat.stride(0), grad_log_taus_cat.stride(1), grad_log_taus_cat.stride(2),
                BLOCK_SIZE=BLOCK_SIZE
            )

        # Convert the gradient from channels back to a complex tensor.
        grad_log_taus_complex = torch.complex(grad_log_taus_cat[..., 0], grad_log_taus_cat[..., 1])
        # Chain rule for log: d/dτ log(τ) = 1/τ.
        #grad_taus = grad_log_taus_complex / taus

        return None, None, grad_log_taus_complex, None

@triton.jit
def scatter_kernel(
        path_idxs_ptr, path_nodes_ptr, log_taus_ptr, agg_log_ptr,
        numel, 
        time_channel_size: tl.constexpr,
        stride_log_taus_row: tl.constexpr, 
        stride_log_taus_tc: tl.constexpr,
        stride_agg_log_row: tl.constexpr, 
        stride_agg_log_tc: tl.constexpr,
        BLOCK_SIZE: tl.constexpr, 
        TC_TILE: tl.constexpr
    ):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < numel

    # Load destination and source indices for this block.
    dest_indices = tl.load(path_idxs_ptr + offsets, mask=mask)
    src_indices  = tl.load(path_nodes_ptr + offsets, mask=mask)

    # Iterate over the time-channel dimension in tiles.
    for tc_start in range(0, time_channel_size, TC_TILE):
        cur_tile = tl.arange(0, TC_TILE) + tc_start
        # Mask out invalid positions when time_channel_size is not a multiple of TC_TILE.
        tile_mask = cur_tile < time_channel_size
        final_mask = mask[:, None] & tile_mask[None, :]
        # Compute source pointers for the entire tile.
        src_ptrs = log_taus_ptr + src_indices[:, None] * stride_log_taus_row + cur_tile[None, :] * stride_log_taus_tc
        # Load the tile values with the combined mask.
        vals = tl.load(src_ptrs, mask=final_mask)
        dest_ptrs = agg_log_ptr + dest_indices[:, None] * stride_agg_log_row + cur_tile[None, :] * stride_agg_log_tc
        tl.atomic_add(dest_ptrs, vals, mask=final_mask)
        
@triton.jit
def scatter_backward_kernel(path_idxs_ptr, path_nodes_ptr, dA_ptr, grad_log_taus_ptr,
                            numel: tl.constexpr, time_window: tl.constexpr, CHANNELS: tl.constexpr,
                            stride_dA_row: tl.constexpr, stride_dA_col: tl.constexpr, stride_dA_ch: tl.constexpr,
                            stride_grad_log_taus_row: tl.constexpr, stride_grad_log_taus_col: tl.constexpr, stride_grad_log_taus_ch: tl.constexpr,
                            BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < numel

    # In the backward pass we "reverse" the scatter: for each index i,
    # grad_log_taus[path_nodes[i], j, c] += dA[path_idxs[i], j, c]
    dest_indices = tl.load(path_nodes_ptr + offsets, mask=mask)  # destination for grad_log_taus
    src_indices = tl.load(path_idxs_ptr + offsets, mask=mask)     # source for dA

    for j in range(time_window):
        for c in range(CHANNELS):
            src_ptr = (dA_ptr +
                       src_indices * stride_dA_row +
                       j * stride_dA_col +
                       c * stride_dA_ch)
            val = tl.load(src_ptr, mask=mask)
            dest_ptr = (grad_log_taus_ptr +
                        dest_indices * stride_grad_log_taus_row +
                        j * stride_grad_log_taus_col +
                        c * stride_grad_log_taus_ch)
            tl.atomic_add(dest_ptr, val)