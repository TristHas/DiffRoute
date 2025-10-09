# DiffRoute Quickstart

This walkthrough shows how to route synthetic runoff through a tiny river network using `diffroute`. You can adapt the same steps to real networks and calibrated parameters.

## 1. Install and import

```bash
pip install git+https://github.com/TristHas/DiffRoute.git
```

```python
import networkx as nx
import torch
from diffroute import LTIRouter, RivTree
```

## 2. Describe the river network

Create a directed acyclic graph with hydrological attributes. 
Each node stores the impulse-response parameters required by the chosen IRF function (parameter `tau` for routing scheme `linear_storage`).

```python
G = nx.DiGraph()

# Define river channels as nodes
G.add_node(0, tau=12.0) # Upstream headwater
G.add_node(1, tau=20.0) # Confluence reach
G.add_node(2, tau=30.0) # Outlet reach

# Define connectivity as edges
G.add_edge(0, 1)
G.add_edge(1, 2)
```

Wrap the NetworkX graph in a `RivTree`, which precomputes sparse routing structures and stores reach parameters.

```python
g = RivTree(
    G,
    irf_fn="linear_storage",   # Choose from diffroute.irfs.IRF_FN
    include_index_diag=False     # Add identity skip if your runoff already includes local discharge
)
```

## 3. Instantiate the router

```python
router = LTIRouter(
    max_delay=48,   # Maximum time step delay for upstream runoff to reach downstream
    dt=1,           # Temporal resolution of routing relative to runoff resolution
)
```

## 4. Route batched runoff

Runoff tensors follow the `[batch, catchments, time]` convention. Below we feed two runoff scenarios over a 7-day horizon (168 hours).

```python
batch = 2
time_steps = 168
device = "cuda:0"

runoff = torch.rand(batch, len(g), time_steps, device=device)
g = g.to(device)

discharge = router(runoff, g)
print(discharge.shape)  # torch.Size([2, 3, 168])
```

Gradients flow through the router by design, so you can differentiate with respect to runoff or IRF parameters, or embed the operator inside a larger neural model.

## Next
- Go through the **Concepts** section to see more detailed basic usage and explanations on the implementation.
- Browse through the **Example** section for practical size routing IO and execution and custom routing scheme integration.
- Visit the `diffhydro` documentation for advanced learning and calibration use-cases.
