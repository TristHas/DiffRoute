# Single Stage Routing

Single-stage routing in DiffRoute is implemented by the `diffroute.router.LTIRouter` module. The router combines two internal phases: (1) it aggregates per-reach impulse response functions (IRFs) into a sparse routing kernel, and (2) it applies a causal block-sparse convolution against runoff time series. This page expands on the quickstart with practical code snippets and deeper explanations of model configuration.

## Quickstart recap

```python
import networkx as nx
import torch

from diffroute import LTIRouter, RivTree

device = "cuda:0"

# 1. Build a tiny river network with per-reach IRF parameters
G = nx.DiGraph()
G.add_node(0, tau=12.0)
G.add_node(1, tau=20.0)
G.add_node(2, tau=30.0)
G.add_edge(0, 1)
G.add_edge(1, 2)

g = RivTree(
    G,
    irf_fn="linear_storage",
    include_index_diag=False
).to(device)

# 2. Instantiate the router
router = LTIRouter(
    max_delay=72,
    dt=1.0,
    block_size=16,
    irf_fn="linear_storage",
    sampling_mode="avg",
    block_f=128
)

# 3. Route batched runoff
runoff = torch.rand(2, len(g), 168, device=device)
discharge = router(runoff, g)
print(discharge.shape)
```

## Router parameters

### Main Parameters:

- `max_delay`: Time span (in routing time steps) used to build the aggregated kernel. Pick a value large enough to cover the longest expected travel time through the network.
- `dt`: Temporal resolution of the routing relative to the runoff resolution. `dt <= 1.0` For daily runoff, `dt = 1.0` means the routing resolution is daily, `dt = 1.0 / 24` means the routing resolution is hourly.
- `irf_fn`: Name of the IRF to apply. Determines how reach-level parameters are interpreted (see the next section).

### Secondary Parameters
- `cascade`: Optional integer that repeats the IRF multiple times to emulate cascaded storage elements.
- `block_size`: Channel block size used when converting the sparse kernel to block-sparse form. Align this with the tiling strategy of your GPU; the default of 16 works well on most devices.
- `sampling_mode`: Strategy used by the `SubResolutionSampler` to subsample the kernel from routing resolution to native runoff resolution. Supported values are `avg`: returns the average daily discharge of hourly routed values and `sample`: returns the last hour discharge of each day.
- `block_f`: Controls the batch size for the frequency-domain aggregation (FFT and transitive closure). 
- `include_index_diag` (set on `RivTree`): When `False`, the routed discharge is added to the input runoff to form the final output. For most use-cases, leave it to the default value `True`

## Built-in IRFs and expected parameters

DiffRoute ships the following IRFs in `diffroute.irfs`. 
Each reach in your river graph must provide the parameter vector listed here. 
Parameters can be stored directly on NetworkX nodes (as in the quickstart) or supplied via a `pandas.DataFrame`.

| IRF name | Description | Expected parameters |
| --- | --- | --- |
| `pure_lag` | Unit hydrograph that delays runoff without attenuation | `delay` |
| `linear_storage` | Discrete linear reservoir | `tau` |
| `nash_cascade` | Closed-form cascade of `n` linear reservoirs (`n` set in `cascade`) | `tau` |
| `muskingum` | Classical Muskingum channel routing | `x`, `k` |
| `hayami` | Hayami diffusion wave approximation | `D` (diffusivity), `L` (reach length), `c` (wave celerity) |

Register custom IRFs with `diffroute.irfs.register_irf(name, func, params)`; provide a callable that accepts `(params, time_window, dt)` and returns kernels shaped `[n_reaches, window]`.

## Inspect the aggregated kernel

You can introspect the routing kernel produced by the aggregator before it is converted to block-sparse form.

```python
import matplotlib.pyplot as plt

kernel = router.aggregator(riv_tree)
# Show size of the sparse kernel.
print(kernel.size, kernel.coords.shape, kernel.values.shape)

dense_kernel = kernel.to_dense()
irf = dense_kernel[1,0] # Path-transverse IRF from node 0 to node 1
plt.plot(irf) # Plot the IRF
```

This is useful when debugging sparse connectivity, verifying kernel support, or exporting the kernel to custom tooling.

## Dynamic parameter inputs

Per-reach routing parameters are stored in `riv_tree.params`, shaped `[n_reaches, n_params]`, and ordered according to `riv_tree.nodes_idx`. 
When parameters remain static (typical operational routing) you can rely on the values embedded in the `RivTree`. 
For workflows where parameters evolve—such as calibration, data assimilation, or differentiable fitting—pass a parameter tensor directly to `LTIRouter.forward`.

```python
# Suppose we optimise tau for each reach during training.
# Build a learnable tensor aligned with riv_tree.nodes_idx.
learnable_tau = torch.nn.Parameter(riv_tree.params.clone())

# ... optimisation steps update learnable_tau ...

# Route with dynamic parameters (must follow the node ordering)
discharge = router(runoff, riv_tree, params=learnable_tau)
```

As long as the tensor matches the shape and ordering dictated by `riv_tree.nodes_idx`, the aggregator substitutes it in place of the static parameters. 
Refer to the **Data Structures** page for a deeper dive into `RivTree`, node ordering, and parameter handling.
