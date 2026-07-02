import torch
import triton
import triton.language as tl
from .closure_sub import closure_sub
from .prefix_sum import prefix_sum


@triton.jit
def _coo_enum_exp_kernel(
    coords_ptr,
    freq_pair_ptr,
    prefix_ptr,
    edges_ptr,
    cumsum_ptr,
    n_nodes,
    n_freq: tl.constexpr,
    INCLUDE_SELF: tl.constexpr,
    BLOCK_FREQ: tl.constexpr,
):
    pid = tl.program_id(0)
    if pid >= n_nodes:
        return

    base = tl.load(cumsum_ptr + (pid - 1), mask=(pid > 0), other=0)
    dest = tl.where(INCLUDE_SELF, pid, tl.load(edges_ptr + pid))
    step = 0
    offs = tl.arange(0, BLOCK_FREQ)

    while dest != -1:
        row = base + step
        tl.store(coords_ptr + row * 2 + 0, dest)
        tl.store(coords_ptr + row * 2 + 1, pid)

        child = tl.load(edges_ptr + dest, mask=(dest >= 0) & (dest < n_nodes), other=-1)
        for f0 in range(0, n_freq, BLOCK_FREQ):
            f = f0 + offs
            m = f < n_freq
            real_col = 2 * f
            imag_col = real_col + 1

            real = tl.load(prefix_ptr + pid * (2 * n_freq) + real_col, mask=m, other=0.0)
            imag = tl.load(prefix_ptr + pid * (2 * n_freq) + imag_col, mask=m, other=0.0)
            if child >= 0:
                real -= tl.load(prefix_ptr + child * (2 * n_freq) + real_col, mask=m, other=0.0)
                imag -= tl.load(prefix_ptr + child * (2 * n_freq) + imag_col, mask=m, other=0.0)

            amp = tl.exp(real)
            out_real = amp * tl.cos(imag)
            out_imag = amp * tl.sin(imag)
            out_base = (row * n_freq + f) * 2
            tl.store(freq_pair_ptr + out_base + 0, out_real, mask=m)
            tl.store(freq_pair_ptr + out_base + 1, out_imag, mask=m)

        step += 1
        dest = tl.load(edges_ptr + dest)

def stable_log_flattened(taus: torch.Tensor, epsilon: float = 1e-8) -> torch.Tensor:
    """
    Compute a stable complex logarithm and format it as a float tensor.
        log(z) = log(max(|z|, epsilon)) + i * arg(z)
        This prevents the log from going to -inf for very small |z|.
    Inputs:

    Outputs:
    
    """
    amp = torch.abs(taus)
    log_amp = torch.log(amp.clamp_min(epsilon))
    angle = torch.angle(taus)
    return torch.stack([log_amp, angle], dim=-1).contiguous().view(amp.shape[0], -1)#.float()

def exp_complex(log_freqs):
    """
        
    """
    log_freqs   = log_freqs.view(-1, log_freqs.shape[1]//2, 2)               
    log_freqs   = torch.view_as_complex(log_freqs)
    freqs = torch.exp(log_freqs)
    return freqs


def closure_sub_exp(prefix: torch.Tensor,
                    edges: torch.Tensor,
                    path_cumsum: torch.Tensor,
                    include_self: bool = True,
                    block_f: int = 128):
    prefix = prefix.contiguous()
    edges = edges.contiguous()
    path_cumsum = path_cumsum.contiguous()

    n, f = prefix.shape
    n_freq = f // 2
    n_path = int(path_cumsum[-1].item())
    coords = torch.empty((n_path, 2), dtype=path_cumsum.dtype, device=prefix.device)
    freq_pairs = torch.empty((n_path, n_freq, 2), dtype=prefix.dtype, device=prefix.device)
    block_freq = max(1, min(block_f // 2, triton.next_power_of_2(n_freq)))

    with torch.cuda.device(prefix.device):
        _coo_enum_exp_kernel[(n,)](
            coords,
            freq_pairs,
            prefix,
            edges,
            path_cumsum,
            n,
            n_freq,
            INCLUDE_SELF=include_self,
            BLOCK_FREQ=block_freq,
        )
    return coords, torch.view_as_complex(freq_pairs)

def transitive_closure(irf, edges, path_cumsum,
                          include_self=True, block_f=128,
                          prefix_rounds=None):
    prefix     = prefix_sum(irf, edges, block_f, prefix_rounds)
    coords, v  = closure_sub(prefix, edges, path_cumsum,
                             include_self, block_f)
    return coords, v, prefix

def log_transitive_closure(irfs_freq, edges, path_cumsum,
                           *, include_self=True, block_f=128,
                           prefix_rounds=None):
    """
        
    """
    log_irfs_freq = stable_log_flattened(irfs_freq)
    coords, log_irfs_freq_agg, log_irfs_freq_prefix = transitive_closure(
        log_irfs_freq, edges, path_cumsum,
        include_self=include_self, block_f=block_f,
        prefix_rounds=prefix_rounds)
    irfs_freq_agg = exp_complex(log_irfs_freq_agg)
    return coords, irfs_freq_agg


def log_transitive_closure_no_grad(irfs_freq, edges, path_cumsum,
                                   *, include_self=True, block_f=128,
                                   prefix_rounds=None):
    log_irfs_freq = stable_log_flattened(irfs_freq)
    prefix = prefix_sum(log_irfs_freq, edges, block_f, prefix_rounds)
    return closure_sub_exp(prefix, edges, path_cumsum, include_self, block_f)
