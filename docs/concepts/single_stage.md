# Basic Routing

Routing in DiffRoute is done by the forward function of the `LTIRouter` PyTorch module, which takes a `torch.Tensor` runoff and a `RivTree` description of the basin as inputs to compute an output routed `torch.Tensor` discharge. 
`LTIRouter` is configured at initialization with routing hyper-parameters. 
`RivTree` wraps a NetworkX `DiGraph` and the corresponding per-channel parameters in a GPU-ready format.

This page builds on the Quickstart snippet with richer context: first dissecting the main router hyper-parameters, then detailing the `RivTree` structure, before walking through the internal execution stages that turn runoff into discharge.

The router combines two internal phases: (1) it aggregates per-channel impulse response functions (IRFs) into a sparse routing kernel, and (2) it applies a causal block-sparse convolution against runoff time series. The sections below expand on each step with practical code snippets and configuration guidance.

## Core example

The canonical example shared in the **Overview** and **Quickstart** pages forms the foundation for the more advanced notes below.

```python
import numpy as np
import pandas as pd
import networkx as nx
import torch

from diffroute import RivTree, LTIRouter

b = 2
n = 20
T = 100
device = "cuda:0"

G = nx.gn_graph(n)
params = pd.DataFrame({"tau": np.random.rand(n)}, index=G.nodes)

riv_tree = RivTree(G, params=params, irf_fn="linear_storage").to(device)
router = LTIRouter(max_delay=48, dt=1)

runoff = torch.rand(b, n, T, device=device)
discharge = router(runoff, riv_tree)
```

This snippet builds a synthetic tree, attaches channel-level parameters, constructs a routing kernel once, and applies it to batched runoff in a differentiable manner. Replace the toy graph, parameter table, or device selection to match your own workflow.

## The LTIRouter module

### Main Parameters:

Two knobs govern most routing behaviour:

- `max_delay`: Temporal support (expressed in routing steps) for the aggregated kernel. Set it large enough to cover the slowest travel time across your basin; anything shorter risks truncating downstream contributions.
- `dt`: Ratio between the routing resolution and the native runoff resolution. With daily runoff, `dt = 1.0` keeps routing daily, while `dt = 1.0 / 24` lifts the computation to hourly resolution before optionally downsampling.

### Secondary Parameters

Additional hyper-parameters are available for specialised tuning, but their defaults usually perform well:

- `cascade`: Integer repeat count that cascades the IRF to emulate multi-reservoir responses.
- `block_size`: Channel block size used when converting the sparse kernel to block-sparse form. Align this with your GPU’s preferred tile size; the default value of 16 is a safe baseline.
- `sampling_mode`: Strategy used by the `SubResolutionSampler` to reduce high-resolution kernels back to native resolution. `avg` emits window averages, while `sample` returns the last sample of each window.
- `block_f`: Controls the frequency-domain aggregation batch size (FFT plus transitive closure). Larger values reduce kernel assembly passes at the expense of memory.
- `include_index_diag` (defined on `RivTree`): When `False`, routed discharge is added to the original runoff, effectively including identity links. Leave it at `True` unless you explicitly need cumulative routing.

## The RivTree structure

`RivTree` materialises the river network in tensor form so the router can consume it efficiently. Instantiating a `RivTree` looks like:

```python
riv_tree = RivTree(G, params=params, irf_fn="linear_storage")
```

The `params` DataFrame must expose one column per IRF parameter required by the selected `irf_fn`. The authoritative list lives in `diffroute.irfs.IRF_PARAMS` and is summarised in the table below.

Internally, `riv_tree` tracks three key tensors:

- `riv_tree.nodes_idx`: `pandas.Series` mapping each NetworkX node label to a contiguous integer. Reorder runoff tensors according to this mapping before routing.
- `riv_tree.edges`: `torch.Tensor` that stores the downstream successor indices for each channel in traversal order. The router uses it to build transitive closures.
- `riv_tree.params`: `torch.Tensor` shaped `[n_channels, n_params]` containing the parameter matrix ordered consistently with `nodes_idx`.

Additional buffers such as `riv_tree.path_cumsum` cache prefix sums used during block-sparse assembly, eliminating repeated graph traversals at execution time.

## Built-in IRFs and expected parameters

DiffRoute ships the following IRFs in `diffroute.irfs`. 
Each channel in your river graph must provide the parameter vector listed here. 
Parameters can be stored directly on NetworkX nodes (as in the quickstart) or supplied via a `pandas.DataFrame`.

| IRF name | Description | Expected parameters |
| --- | --- | --- |
| `pure_lag` | Unit hydrograph that delays runoff without attenuation | `delay` |
| `linear_storage` | Discrete linear reservoir | `tau` |
| `nash_cascade` | Closed-form cascade of `n` linear reservoirs (`n` set in `cascade`) | `tau` |
| `muskingum` | Classical Muskingum channel routing | `x`, `k` |
| `hayami` | Hayami diffusion wave approximation | `D` (diffusivity), `L` (channel length), `c` (wave celerity) |


### Registering custom routing schemes

`diffroute` suports easy integration of new custom LTI routing scheme through the `diffroute.irfs.register_irf(name, func, params)` utility function.
This utility function takes as imput:

- `name`: a string identifier to be passed to the `irf_fn` attribute of the `RivTree` initialization.
- `func`: a callable that accepts `(params, time_window, dt)` and returns kernels shaped `[n_channels, window]`
- `params`: a tuple of string describing the per-channel routing parameter of the routing scheme to register (i.e. (`x`, `k`) for the Muskingum scheme)

A [tutorial notebook](examples/custom_irf.md) is provided showing the XXX. 

## Internals of the routing procedure

The forward function of `LTIRouter` is intentionally compact:

```
def forward(self, runoff, riv_tree, params=None):
    kernel = self.aggregator(riv_tree, params)
    block_sparse = kernel.to_block_sparse(self.block_size)
    discharge = self.conv(runoff, block_sparse)
    return discharge if riv_tree.include_index_diag else runoff + discharge
```

Execution proceeds in three stages:

- (i) **IRF aggregation**: gathers per-channel kernels, applies routing parameters (including optional overrides), and accumulates path responses into a sparse tensor.
- (ii) **Block-sparse conversion**: slices the aggregated kernel into GPU-friendly tiles controlled by `block_size`, caching the layout for fast re-use.
- (iii) **Block-sparse convolution**: multiplies the cached tiles against runoff tensors to generate routed discharge, optionally adding the original runoff if self-links are excluded.

For research and development purposes, one can introspect the routing kernel produced by the aggregator before it is converted to block-sparse form.

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

`RivTree` stores per-channel routing parameters in the tensor `riv_tree.params`, ordered exactly like `riv_tree.nodes_idx`. You can populate these values either by attaching attributes to your NetworkX graph or by supplying a `pandas.DataFrame` at construction time.

Static routing scenarios can rely on the parameters embedded in the `RivTree`, but differentiable calibration often benefits from injecting learnable tensors into the forward pass.

```python
# Build a learnable tensor aligned with riv_tree.nodes_idx.
learnable_tau = torch.nn.Parameter(riv_tree.params.clone())

# Route with dynamic parameters (must follow the node ordering)
discharge = router(runoff, riv_tree, params=learnable_tau)
```

As long as the tensor matches the shape and ordering dictated by `riv_tree.nodes_idx`, the aggregator swaps it in for the stored parameters. Refer to the **Data Structures** page for a deeper dive into `RivTree`, node ordering, and parameter handling, and explore `diffhydro` if you need end-to-end calibration workflows.
