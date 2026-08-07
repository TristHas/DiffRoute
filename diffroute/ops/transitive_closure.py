import torch
from .closure_sub import closure_sub
from .prefix_sum import prefix_sum

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

def downstream_prefixes(irf, edges, block_f=128):
    """Inclusive and exclusive downstream prefixes of ``irf``.

        P[i] = sum of irf over  i..root         (inclusive)
        Q[i] = sum of irf over  succ(i)..root   (exclusive; 0 at outlets)

    ``Q`` is a *gather* of ``P`` at the successor rather than ``P - irf``, so its
    values are bitwise equal to the inclusive prefix they stand for and no extra
    rounding is introduced on top of the path difference. Masking the outlets is
    load-bearing in backward too: it stops the ``clamp(min=0)`` used for the -1
    sentinel from routing every outlet's gradient into node 0.
    """
    P = prefix_sum(irf, edges, block_f)
    Q = P.index_select(0, edges.long().clamp(min=0)) \
        * (edges >= 0).unsqueeze(-1).to(P.dtype)
    return P, Q

def transitive_closure(irf, edges, path_cumsum,
                          route_src_reach=True, block_f=128):
    """Path sums over the downstream closure; see ``ops/closure_sub.py``.

    ``route_src_reach`` decides whether the source reach's own IRF is part of the
    path: it is when the runoff enters at the head of that reach, it is not when
    the runoff is already at its outlet. That choice is exactly the choice of
    which prefix the source end reads from.
    """
    P, Q       = downstream_prefixes(irf, edges, block_f)
    coords, v  = closure_sub(P if route_src_reach else Q, Q,
                             edges, path_cumsum,
                             route_src_reach, block_f)
    return coords, v, P

def log_transitive_closure(irfs_freq, edges, path_cumsum,
                           *, route_src_reach=True, block_f=128):
    """
        
    """
    log_irfs_freq = stable_log_flattened(irfs_freq)
    coords, log_irfs_freq_agg, log_irfs_freq_prefix = transitive_closure(
        log_irfs_freq, edges, path_cumsum,
        route_src_reach=route_src_reach, block_f=block_f)
    irfs_freq_agg = exp_complex(log_irfs_freq_agg)
    return coords, irfs_freq_agg