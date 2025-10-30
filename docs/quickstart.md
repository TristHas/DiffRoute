# DiffRoute Quickstart

This walkthrough unpacks the Overview code snippet step by step with deeper explanations. 
It shows an example of routing synthetic runoff through a compact river network using `diffroute`,
but the same pattern can applied to real catchments and calibrated parameters.

## 1. Install and import

```bash
pip install git+https://github.com/TristHas/DiffRoute.git
```

Installs the latest version of `diffroute` from GitHub; PyTorch, NetworkX, and the remaining dependencies are automatically installed.

```python
import numpy as np
import pandas as pd
import networkx as nx
import torch

from diffroute import RivTree, LTIRouter
```

## 2. Describe the river network

Define the toy routing experiment by choosing batch size, number of reaches, simulation length, and target device.
The device parameter is a torch parameter setting on which GPU the computations are carried.

```python
b = 2       
n = 20       
T = 100      
device = "cuda:0"  
```

DiffRoute uses NetworkX to describe river graph connectivity.
Any connectivity format can be easily read into a Networkx graph.
`nx.gn_graph` quickly supplies a directed tree with a single outlet, which is perfect for demonstration purposes.

```python
G = nx.gn_graph(n, seed=0)
```

Routing also requires per-channel parameters.
The set of parameters required depends on the routing scheme used.
Here, we use the `linear_storage` scheme, which parameterizes channels with a unique parameter `tau`. 
We randomly sample `tau` values for the sake of illustration.

```python
params = pd.DataFrame({"tau": np.random.rand(n)}, index=G.nodes)
```

The `RivTree` class is `diffroute`'s native representations of river graphs.
This structure holds the graph connectivity, routing scheme and parameters as torch tensors.
As `RivTree` holds the tensor representation of the graph, it needs to be given a `device` for storage through the helper `.to(device)`.
It can be instantiated from a NetworkX graph for connectivity and pandas DataFrame for parameters.

```python
riv_tree = RivTree(G, params=params, irf_fn="linear_storage").to(device)
```

## 3. Instantiate the router

`LTIRouter` is the primary PyTorch module provided by DiffRoute and can be dropped into any nn.Module graph.
You configure it once with routing hyper-parameters: `max_delay` controls the temporal support of the aggregated kernel, 
and `dt` sets the routing resolution relative to the runoff time step.

```python
router = LTIRouter(
    max_delay=48,  # Time steps to cover all upstream travel times
    dt=1           # Routing resolution relative to runoff resolution
)
```

## 4. Route runoff

The router consumes input runoff tensor and `RivTree` structure to output a discharge tensor.
Both inputs must reside on the same device, or an error will be raised.
Runoff tensors follow the `[batch, catchments, time]` layout.
The output discharge is a tensor on the same device as the input and with the same shape as the input runoff.
The output discharge is fully differentiable with respect to both input runoff and RivTree parameters.

```python
runoff = torch.rand(b, n, T, device=device)
discharge = router(runoff, riv_tree)
print(discharge.shape)
```

## Next
- Go through the **Concepts** section for deeper explanations of the core abstractions and configuration details.
- Browse the **Examples** section for end-to-end workflows that cover IO, execution at scale, and custom IRF integration.
- Visit the `diffhydro` documentation for advanced learning, calibration, and larger pipeline integrations.
