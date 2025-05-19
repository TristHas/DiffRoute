import torch
from .scatter_reduce_triton import log_aggregate_irf_triton

def stable_log(taus: torch.Tensor, epsilon: float = 1e-10) -> torch.Tensor:
    """
        Compute a stable complex logarithm:
              log(z) = log(max(|z|, epsilon)) + i * arg(z)
              This prevents the log from going to -inf for very small |z|.
    """
    amp = torch.abs(taus)
    stable_amp = amp.clamp_min(epsilon)
    return torch.log(stable_amp) + 1j * torch.angle(taus)

def log_aggregate_irf(path_idxs: torch.Tensor,
                         path_nodes: torch.Tensor,
                         taus: torch.Tensor,
                         n_paths: int = None) -> torch.Tensor:
    """
    """
    if n_paths is None: n_paths = path_idxs.max().item() + 1
    time_window = taus.shape[-1]

    log_taus = stable_log(taus)
    log_gathered = log_taus[path_nodes, ...]
    
    agg_log = torch.zeros((n_paths, time_window), dtype=taus.dtype, device=taus.device)
    agg_log.index_add_(0, path_idxs, log_gathered)

    result = torch.exp(agg_log)
    return result

def cpu_aggregate_irf(path_idxs: torch.Tensor,
                      path_nodes: torch.Tensor,
                      taus: torch.Tensor,
                      n_paths=None) -> torch.Tensor:
    """
    """
    time_window = taus.shape[-1]
    if n_paths is None: n_paths = path_idxs.max().item() + 1  # number of distinct paths
    
    device = "cpu"
    gathered = taus[path_nodes, :].to(device)
    result = torch.ones((n_paths, time_window), device=device, dtype=taus.dtype)
    idx_expanded = path_idxs.to(device).unsqueeze(-1).expand(-1, time_window)
    result.scatter_reduce_(
        dim=0,
        index=idx_expanded,
        src=gathered,
        reduce="prod"
    )

    return result.to(taus.device)

def euler_aggregate_irf(path_idxs: torch.Tensor,
                        path_nodes: torch.Tensor,
                        taus: torch.Tensor,
                        n_paths=None) -> torch.Tensor:
    """
    """
    time_window = taus.shape[-1]
    if n_paths is None: n_paths = path_idxs.max().item() + 1
    gathered = taus[path_nodes, :]  # stays on taus.device
    amplitudes = torch.abs(gathered)
    angles = torch.angle(gathered)

    idx_expanded = path_idxs.unsqueeze(-1).expand(-1, time_window)

    # Sum log amplitudes (turning product into sum)
    log_amplitudes = torch.log(amplitudes)
    agg_log_amp = torch.zeros((n_paths, time_window), 
                              dtype=log_amplitudes.dtype, 
                              device=taus.device)
    agg_log_amp.scatter_add_(dim=0, index=idx_expanded, src=log_amplitudes)
    amplitude_product = torch.exp(agg_log_amp)

    # Sum the angles
    agg_angle = torch.zeros((n_paths, time_window), 
                            dtype=angles.dtype, device=angles.device)
    agg_angle.scatter_add_(dim=0, index=idx_expanded, src=angles)

    # Reconstruct the aggregated complex value
    result = amplitude_product * torch.exp(1j * agg_angle)
    return result

IRF_AGGREGATE_FN = {
    "cpu":cpu_aggregate_irf,
    "log":log_aggregate_irf,
    "log_triton":log_aggregate_irf_triton,
    "euler":euler_aggregate_irf,
}
