import torch 
from .irfs import IRF_FN
    
def register_irf(name, func):
    IRF_FN[name]=func

def downsample_irf(kernel: torch.Tensor, dt: float) -> torch.Tensor:
    """
    Sums blocks of size int(1/dt). If dt=1, no change.
    """
    if dt == 1.0:
        return kernel
    factor = int(1 / dt)
    kernel = kernel.unsqueeze(1)
    pooled = torch.nn.functional.avg_pool1d(kernel, kernel_size=factor, stride=factor) * factor
    return pooled.squeeze(1)

def repeatedly_convolve(kernel, dt):
    """
    
    """
    kernel_freq = torch.fft.rfft(kernel, n=kernel.shape[-1], dim=-1)
    kernel_freq = kernel_freq**int(1/dt)
    kernel = torch.fft.irfft(kernel_freq, n=kernel.shape[-1], dim=-1)
    return kernel

def convert_kernel_time_scale(kernel, dt):
    """
    Raises the frequency spectrum to an integer power (1/dt) -> repeated convolution,
    then block-sum downsamples.
    """
    kernel = repeatedly_convolve(kernel, dt)
    return downsample_irf(kernel, dt)

def build_irf(model_name: str,
              param: torch.Tensor,
              dt: float,
              time_window: int) -> torch.Tensor:
    """
        Dispatch function that calls the correct IRF generator based on model_name.
        
        Args:
          model_name (str): one of ["pure_lag", "linear_storage", "nash_cascade", "muskingum", "hayami"]
          param (Tensor):  shape depends on model_name
          dt (float):      time step
          time_window (int): number of output steps
        
        Returns:
          kernel (Tensor): shape => (n_cat, time_window)
    """
    if model_name not in IRF_FN:
        raise ValueError(f"Unsupported model_name='{model_name}'. Choose from {list(IRF_FN.keys())}.")

    kernel_fn = IRF_FN[model_name]
    return kernel_fn(param, time_window=time_window, dt=dt)



### Old
def baseline_generate_convolved_irfs(g, node_idxs=None, time_window=20):
    BASE_IRF = torch.zeros(time_window)
    BASE_IRF[0]=1

    if node_idxs is None: node_idxs=get_node_idxs(g)
    taus = generate_catchment_irfs(g, node_idxs=node_idxs, time_window=time_window)

    g = nx.relabel_nodes(g, node_idxs.to_dict())
    
    irfs = {}
    topo_order = list(nx.topological_sort(g))
    for dest in topo_order:
        irfs[(dest, dest)] = taus[dest] 
        w = irfs[(dest, dest)][None, None].flip(-1)  
        padding = w.shape[-1] - 1
        for pred in g.predecessors(dest):
            irfs[(dest, pred)] = taus[pred]
            
            for source in nx.ancestors(g, pred):
                x = irfs[(pred, source)][None, None]  # Shape: (1, 1, L)
                conv_result = torch.conv1d(x, w, padding=padding)
                irfs[(dest,source)] = conv_result[..., :time_window].squeeze()

    for dest in topo_order: del irfs[(dest,dest)]
        
    idxs = torch.tensor(list(irfs.keys()))
    vals = torch.stack(list(irfs.values()))
    return idxs, vals

def init_taus(g, node_idxs=None):
    """
    """
    if node_idxs is None: node_idxs=get_node_idxs(g)
    return torch.FloatTensor([g.nodes[n]["tau"] for n in node_idxs.index])

def generate_storage_irfs(g, node_idxs=None, time_window=20, dt=1):
    """
    """
    return irf_kernel_linear_storage(init_taus(g, node_idxs), time_window, dt)

def full_conv1d(a, b):
    """
        Perform a full 1D convolution of tensor 'a' with kernel 'b'.
    
        This function flips the kernel and applies 1D convolution with appropriate padding
        to simulate a full convolution (i.e. one that covers all overlapping positions).
    
        Parameters:
            a (Tensor): 1D input tensor.
            b (Tensor): 1D convolution kernel.
    
        Returns:
            Tensor: The result of the full 1D convolution.
    """
    b_flipped = torch.flip(b, dims=[0])
    a_ = a.unsqueeze(0).unsqueeze(0)  # shape (1, 1, L)
    b_ = b_flipped.unsqueeze(0).unsqueeze(0)  # shape (1, 1, kernel_size)
    conv_result = F.conv1d(a_, b_, padding=b.shape[0] - 1)
    return conv_result.squeeze()

def irf_kernel_cascade_linear_storage_old(taus, n=3, time_window=20, dt=1):
    """
        Generate a cascade of linear storage kernels by convolving a base kernel multiple times.
    
        The base kernel is first generated using the linear storage formulation.
        It is then convolved with itself (n-1) times to simulate cascading storage effects.
        If the resulting kernel is shorter than the desired time window, it is padded with zeros.
    
        Parameters:
            taus (Tensor): A 1D tensor of storage time constants for each category.
            n (int): The number of kernels to cascade (default: 3).
            time_window (int): The number of time steps for the final kernel (default: 20).
            dt (float): The time step duration (default: 1).
    
        Returns:
            Tensor: A 2D tensor of shape (n_categories, time_window) representing the cascaded kernel.
    """
    t = dt * torch.arange(time_window, device=taus.device, dtype=taus.dtype).unsqueeze(0)
    x = 1 / (1 + dt / taus[:, None])
    t_grid = torch.arange(time_window, device=taus.device)
    base = x * (1 - x) ** t_grid
    kernels = []
    for i in range(len(taus)):
        kernel = base[i]
        kernel_conv = kernel.clone()
        # Cascade the kernel n-1 times
        for _ in range(n - 1):
            kernel_conv = full_conv1d(kernel_conv, kernel)
        # Ensure the kernel has the desired length
        if kernel_conv.shape[0] < time_window:
            kernel_conv = F.pad(kernel_conv, (0, time_window - kernel_conv.shape[0]))
        else:
            kernel_conv = kernel_conv[:time_window]
        kernels.append(kernel_conv)
    return torch.stack(kernels, dim=0)