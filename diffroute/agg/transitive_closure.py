import torch
from .closure import closure_sub
from .prefix import prefix_sum

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

def sparse_irf_coo_triton(irf, edges, path_cumsum,
                          include_self=True, block_f=128):
    prefix     = prefix_sum(irf, edges, block_f)
    coords, v  = closure_sub(prefix, edges, path_cumsum,
                             include_self, block_f)
    return coords, v, prefix

def sparse_irf_coo_complex(irfs_freq, edges, path_cumsum,
                           *, include_self=True, block_f=128):
    """
        
    """
    log_irfs_freq = stable_log_flattened(irfs_freq)
    coords, log_irfs_freq_agg, log_irfs_freq_prefix = sparse_irf_coo_triton(
        log_irfs_freq, edges, path_cumsum,
        include_self=include_self, block_f=block_f)
    irfs_freq_agg = exp_complex(log_irfs_freq_agg)
    return coords, irfs_freq_agg