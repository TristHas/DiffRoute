from imports import *

from diffroute.utils import get_node_idxs
from diffroute.irfs import IRF_FN
from diffroute.agg.kernel_sampler import SubResolutionSampler

IRF_PARAMS = {
        "pure_lag":["delay"],
        "linear_storage":["tau"],
        "nash_cascade":["tau"],
        "muskingum":["x", "k"],
        "hayami":["D", "L", "c"]
}

def convolve_irfs(g, base_irfs, dt=1):
    """
    Assemble the dense kernel via recursive convolution.
    """
    time_window = base_irfs.shape[-1]
    node_list = sorted(g.nodes())
    irfs = {}
    for dest in nx.topological_sort(g):
        idx_dest = node_list.index(dest)
        irfs[dest] = {dest: base_irfs[idx_dest]}
        w = irfs[dest][dest][None, None].flip(-1)
        padding = time_window - 1 
        for pred in g.predecessors(dest):
            for source in nx.ancestors(g, pred) | {pred}:
                x_val = irfs[pred][source][None, None]
                conv_result = torch.conv1d(x_val, w, padding=padding)
                irfs[dest][source] = conv_result[..., :time_window].squeeze()
    return irfs

def irfs_to_dense_kernel(irfs, num_catchments, num_timesteps=10, dt=1.0):
    kernel = np.zeros((num_catchments, num_catchments, num_timesteps*int(1/dt)))
    for dest in irfs:
        for source in irfs[dest]:
            kernel[dest, source, :] = irfs[dest][source].cpu().numpy()
    return kernel

def conv(x, w):
    return F.conv1d(x, w, padding=w.shape[-1]-1 )[..., :x.shape[-1]].squeeze(0)
    
def read_params(g, model):
    p_name = IRF_PARAMS[model]
    params = torch.tensor([[g.nodes[n][p] for p in p_name] for n in get_node_idxs(g).index])
    return params
    
def gen_dense_k(g, time_window=10, model="muskingum", dt=1, **kwargs):
    params = read_params(g, model)
    base_irfs = IRF_FN[model](params, time_window=time_window, dt=dt, **kwargs)
    #base_irfs = repeatedly_convolve(base_irfs, dt) 
    irfs = convolve_irfs(g, base_irfs, dt)
    kernel = irfs_to_dense_kernel(irfs, len(g), time_window, dt)
    return np.ascontiguousarray(np.flip(kernel, -1))

def gen_dense_k_subresolution(g, time_window, model="pure_lag", sample_mode="avg", dt=1, **kwargs):
    kernel = gen_dense_k(g, time_window=time_window, model=model, dt=dt, **kwargs)
    kernel = torch.tensor(kernel).view(-1, kernel.shape[-1]).float()
    w_d = SubResolutionSampler(dt, sample_mode).phi_k(kernel)
    w_d = w_d.view(len(g), len(g), time_window).numpy()
    return w_d #/ dt

@torch.no_grad()
def run_dense_conv(x, g, model, time_window=10, dt=1, **kwargs):
    """
    """
    w = gen_dense_k(g, time_window=time_window, 
                    model=model, dt=dt, **kwargs)
    x = torch.tensor(x, dtype=torch.float32).unsqueeze(0)  # [1, num_catchments, num_timesteps]
    w = torch.tensor(w, dtype=torch.float32)               # Standard flip to account for F.conv1d doing cross-correlation.
    return conv(x, w).cpu().numpy()

@torch.no_grad()
def run_dense_conv_subresolution(x, g, model, time_window=10, dt=1, **kwargs):
    """
    """
    w = gen_dense_k_subresolution(g, time_window=time_window, 
                                  model=model, dt=dt, **kwargs)
    x = torch.tensor(x, dtype=torch.float32).unsqueeze(0)  # [1, num_catchments, num_timesteps]
    w = torch.tensor(w, dtype=torch.float32)               # Standard flip to account for F.conv1d doing cross-correlation.
    return conv(x, w).cpu().numpy()