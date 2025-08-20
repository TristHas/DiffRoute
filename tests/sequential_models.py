from imports import *
from diffroute.utils import get_node_idxs

def init_model(g, runoff_inputs, num_timesteps):
    """
    """
    if num_timesteps is None: num_timesteps=runoff_inputs.shape[-1]
    num_catchments = g.number_of_nodes()
    discharge = np.zeros((num_catchments, num_timesteps))
    node_to_idx = get_node_idxs(g)
    return num_timesteps, node_to_idx, discharge
    
def pure_lag_model(runoff_inputs, g, num_timesteps=None, dt=1.0):
    """
        Pure lag model supporting fractional dt=1/N.
        
        Parameters
        ----------
        runoff_inputs : np.ndarray
            Shape = (num_catchments, num_timesteps). Input runoff at each discrete time step.
        g : nx.DiGraph
            Network with "delay" parameter per node.
        num_timesteps : int
            Number of discrete steps (length of runoff_inputs in time).
        dt : float
            Time step. E.g., dt=0.5 means each index corresponds to 0.5 units of time.
        Returns
        -------
        discharge : np.ndarray
            Discharge for each catchment and time step (shape = (num_catchments, num_timesteps)).
    """
    num_timesteps, node_to_idx, discharge = init_model(g, runoff_inputs, num_timesteps)
    inflow = np.zeros_like(runoff_inputs)
    
    for t in range(num_timesteps):
        for node in nx.topological_sort(g):
            idx = node_to_idx[node]
            # Sum of upstream discharges at the current step
            upstream = sum(discharge[node_to_idx[pred], t] for pred in g.predecessors(node))
            inflow[idx, t] = runoff_inputs[idx, t] + upstream
            u = t - g.nodes[node]["delay"] / dt 
            
            if u>=0: # Too much in the past, no data
                i0 = int(np.floor(u))
                i1 = i0 + 1
                alpha = u - i0 
                
                if i1 < num_timesteps:
                    discharge[idx, t] = (1 - alpha)*inflow[idx, i0] + alpha*inflow[idx, i1]
                else:
                    discharge[idx, t] = inflow[idx, i0]
    
    return discharge
    
def linear_reservoir_model(runoff_inputs, g, num_timesteps, dt=1.0):
    num_timesteps, node_to_idx, discharge = init_model(g, runoff_inputs, num_timesteps)
    storage = np.zeros_like(runoff_inputs)

    for t in range(num_timesteps):
        for node in nx.topological_sort(g):
            idx = node_to_idx[node]
            inflow_from_upstream = sum(discharge[node_to_idx[pred], t] for pred in g.predecessors(node))
            total_inflow = runoff_inputs[idx, t] + inflow_from_upstream
            x = 1.0 / (1.0 + dt / g.nodes[node]["tau"])
            Q = x * (storage[idx, t] + total_inflow)
            discharge[idx, t] = Q
            if t < num_timesteps -1:
                storage[idx, t+1] = (1 - x) * (storage[idx, t] + total_inflow)
    return discharge

def nash_cascade_model(runoff_inputs, g, num_timesteps=None, n=3, dt=1.0):
    num_timesteps, node_to_idx, discharge = init_model(g, runoff_inputs, num_timesteps)
    state = np.zeros_like(runoff_inputs)
    
    for t in range(num_timesteps):
        for node in nx.topological_sort(g):
            idx = node_to_idx[node]
            upstream = sum(discharge[node_to_idx[pred], t] for pred in g.predecessors(node))
            inflow = runoff_inputs[idx, t] + upstream
            Q_in = inflow
            for i in range(n):
                tau = g.nodes[node]["tau"]
                Q = (state[idx, i] + Q_in*dt) / (1+ dt/tau)
                state[idx, i] = state[idx, i] + Q_in*dt - Q*dt
                Q_in = Q
            discharge[idx, t] = Q_in
    return discharge

def nash_cascade_model(runoff_inputs, g, num_timesteps=None, n=3, dt=1.0):
    num_timesteps, node_to_idx, discharge = init_model(g, runoff_inputs, num_timesteps)
    state = np.zeros((runoff_inputs.shape[0], n))
    
    for t in range(num_timesteps):
        for node in nx.topological_sort(g):
            idx = node_to_idx[node]
            upstream = sum(discharge[node_to_idx[pred], t] for pred in g.predecessors(node))
            inflow = runoff_inputs[idx, t] + upstream
            Q_in = inflow  
            for i in range(n):
                tau = g.nodes[node]["tau"]
                x = 1.0 / (1.0 + dt / tau)
                Q = x * (state[idx, i] + Q_in)
                state[idx, i] = (1 - x) * (state[idx, i] + Q_in)
                Q_in = Q
            discharge[idx, t] = Q_in
    return discharge

def muskingum_model(runoff_inputs, g, num_timesteps=None, dt=1.0, 
                    init_zero=False, rapid_shift=False):
    #dt = 1.0
    num_timesteps, node_to_idx, discharge = init_model(g, runoff_inputs, num_timesteps)
    I_prev = np.zeros(runoff_inputs.shape[0])
    
    for t in range(num_timesteps):
        for node in nx.topological_sort(g):
            idx = node_to_idx[node]
            K = g.nodes[node]["k"]
            X = g.nodes[node]["x"]
            inflow_from_upstream = sum(discharge[node_to_idx[pred], t] for pred in g.predecessors(node))
            inflow = runoff_inputs[idx, t] + inflow_from_upstream
            denom = 2*K*(1 - X) + dt
            C0 = (dt - 2*K*X) / denom
            C1 = (dt + 2*K*X) / denom
            C2 = (2*K*(1 - X) - dt) / denom
            if t == 0:
                if init_zero:
                    discharge[idx, t] = 0
                else:
                    discharge[idx, t] = C0 * inflow            
                I_prev[idx] = inflow
            else:
                if rapid_shift:
                    inflow_ = inflow = runoff_inputs[idx, t-1] + inflow_from_upstream
                else:
                    inflow_ = inflow
                discharge[idx, t] = C0 * inflow + C1 * I_prev[idx] + C2 * discharge[idx, t-1]
                I_prev[idx] = inflow
    return discharge

def linear_diffuse_model(runoff_inputs, g, num_timesteps=None, dt=1.0):
    """
    Sequential linear diffusion (Hayami) model.
    For each node, the inflow (local runoff + upstream discharge)
    is routed via convolution with its diffusion impulse response.
    This version updates discharge at every time step so that
    upstream contributions are properly propagated.
    """
    num_timesteps, node_to_idx, inflow, discharge = init_model(g, runoff_inputs, num_timesteps)

    # Precompute each node's diffusion IRF (in natural/casual order)
    kernels = {}
    for node in node_list:
        L = g.nodes[node]["L"]
        D = g.nodes[node]["D"]
        c = g.nodes[node]["c"]
        # Compute IRF kernel (h[0]=0, h[1:] from the Hayami formula)
        L_tensor = torch.tensor([L], dtype=torch.float32)
        D_tensor = torch.tensor([D], dtype=torch.float32)
        c_tensor = torch.tensor([c], dtype=torch.float32)
        kernel = irf_kernel_linear_diffusion(L_tensor, D_tensor, c_tensor, time_window=num_timesteps, dt=dt)
        # Keep the kernel in natural (causal) order.
        kernels[node] = kernel.numpy().flatten()

    # Sequential update: compute inflow and immediately compute discharge.
    # This way, when a node is processed, its discharge is available to downstream nodes.
    for t in range(num_timesteps):
        for node in nx.topological_sort(g):
            idx = node_to_idx[node]
            # Upstream discharge (already computed in the current time step)
            upstream = sum(discharge[node_to_idx[pred], t] for pred in g.predecessors(node))
            inflow[idx, t] = runoff_inputs[idx, t] + upstream
            h = kernels[node]
            # Compute discharge at time t via convolution with the past inflow:
            # discharge[t] = sum_{s=0}^{t} h[t-s] * inflow[s]
            discharge[idx, t] = np.sum(inflow[idx, :t+1][::-1] * h[:t+1])
    return discharge

MODELS = {
    "pure_lag":pure_lag_model,
    'linear_storage':linear_reservoir_model, 
    'nash_cascade':nash_cascade_model, 
    'muskingum':muskingum_model, 
    'hayami':linear_diffuse_model,
}