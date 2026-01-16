import torch
import math

def irf_kernel_pure_lag(param, time_window=20, dt=1):
    """
        param[:,0] => delays
    """
    delays = param[:, 0].unsqueeze(1) / dt
    extended_time_steps = time_window * int(1/dt)
    t = torch.arange(extended_time_steps, device=delays.device, dtype=delays.dtype).unsqueeze(0)
    kernel = torch.clamp(1.0 - torch.abs(t - delays), min=0)
    return kernel

def irf_kernel_linear_storage(param, time_window=20, dt=1.):
    """
        param[:,0] => tau
    """
    taus = param[:, 0] / dt
    extended_time_steps = time_window * int(1/dt)
    t = torch.arange(extended_time_steps, device=taus.device, dtype=taus.dtype)
    x = 1.0 / (1.0 + taus.unsqueeze(1))
    kernel = x * (1.0 - x)**t
    return kernel

def irf_kernel_cascade_linear_storage(param, n=3, time_window=20, dt=1.):
    """
        param[:,0] => tau  (cascaded n times in closed form)
    """
    taus = param[:, 0] / dt
    extended_time_steps = time_window * int(1/dt)
    t = torch.arange(extended_time_steps, device=taus.device, dtype=taus.dtype)
    x = 1.0 / (1.0 + taus.unsqueeze(1))
    n_tensor = torch.tensor(n, dtype=taus.dtype, device=taus.device)
    binom_coef = torch.exp(torch.lgamma(t + n_tensor) - torch.lgamma(t + 1) - torch.lgamma(n_tensor))
    kernel = binom_coef * (x ** n) * ((1 - x) ** t)
    return kernel

def irf_kernel_muskingum(param, time_window=10, dt=1):
    """
        param[:,0] => x,  
        param[:,1] => k
    """
    x_vals = param[:, 0]
    k_vals = param[:, 1]
    extended_time_steps = time_window * int(1/dt)

    t_idx = torch.arange(extended_time_steps, device=x_vals.device, dtype=x_vals.dtype)
    k_constrained = k_vals
    denom = k_constrained * (1 - x_vals) + dt / 2.0
    C0 = (dt / 2.0 - k_constrained * x_vals) / denom
    C1 = (dt / 2.0 + k_constrained * x_vals) / denom
    C2 = (k_constrained * (1 - x_vals) - dt / 2.0) / denom
    mask_t0 = (t_idx == 0).float().unsqueeze(0)
    h_tail  = (C1 + C2*C0)[:,None] * (C2[:,None]**(t_idx.clamp(min=1) - 1))
    kernel = mask_t0 * C0[:,None] + (1. - mask_t0)*h_tail
    return kernel

def irf_kernel_linear_diffusion(param, time_window=20, dt=1, eps=-90):
    """
        param[:,0] => L
        param[:,1] => D
        param[:,2] => c
    """
    L, D, c = [x.unsqueeze(-1) for x in param.t()]
    t = torch.arange(1, time_window * int(1/dt)+1, 
                     device=param.device, 
                     dtype=param.dtype)[None] * dt 
    # Hayami formula
    h = L / (2.0 * torch.sqrt(torch.pi * D * t**3))
    exponent = -((L - c * (t-dt))**2) / (4.0 * D * t)
    # Needed for numerical accuracy, to avoid nans.
    mins = torch.max(exponent,axis=1, keepdims=True).values.detach()
    exponent_ = torch.where(mins > eps, exponent, exponent / mins * eps)
    kernel = h * torch.exp(exponent_)
    # normalize
    kernel = kernel / kernel.sum(-1, keepdim=True)
    return kernel

def irf_kernel_stable_hayami(param, time_window=20, dt=1.0):
    """
    """
    L, D, c = [x.unsqueeze(-1) for x in param.t()]
    t = torch.arange(1, int(time_window * (1/dt)) + 1, 
                     device=param.device, 
                     dtype=param.dtype)[None] * dt
    # Hayami formula in log space
    logh = torch.log(L)  \
         - math.log(2.0) \
         - 0.5 * (math.log(math.pi) \
                  + torch.log(D) \
                  + 3.0 * torch.log(t))
    exponent = -((L - c * (t-dt)) ** 2) / (4.0 * D * t)
    logk = logh + exponent
    # Normalization
    logk = logk - torch.logsumexp(logk, dim=-1, keepdim=True)
    kernel = torch.exp(logk)
    return kernel
    
IRF_FN = {
    "pure_lag": irf_kernel_pure_lag,
    "linear_storage": irf_kernel_linear_storage,
    "nash_cascade": irf_kernel_cascade_linear_storage,
    "muskingum": irf_kernel_muskingum,
    "hayami": irf_kernel_stable_hayami #irf_kernel_linear_diffusion
}

IRF_PARAMS = {
        "pure_lag":["delay"],
        "linear_storage":["tau"],
        "nash_cascade":["tau"],
        "muskingum":["x", "k"],
        "hayami":["L", "D", "c"]
}

def register_irf(name, func, params):
    """Register a custom impulse response function for routing.

    Args:
        name (str): Unique identifier for the IRF.
        func (Callable): Custom IRF generation function.
        params (Sequence[str]): Ordered parameter names expected by `func`.
    """
    IRF_FN[name]=func
    IRF_PARAMS[name]=params
