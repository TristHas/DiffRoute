# Defining a Custom IRF

```python
import torch

from diffroute import LTIRouter, RivTree, register_irf
```

```python
# 1. Implement the custom impulse response
def geomorphic_unit_hydrograph(params, time_window, dt):
    """
    params[:, 0] -> reference celerity (m/s)
    params[:, 1] -> diffusion (m^2/s)
    """
    c = params[:, 0].unsqueeze(1)
    d = params[:, 1].unsqueeze(1)
    support = time_window * int(1 / dt)
    t = torch.arange(support, device=params.device, dtype=params.dtype).unsqueeze(0) * dt
    pulse = (c / torch.sqrt(4 * torch.pi * d * t.clip(min=dt))) * torch.exp(-(c ** 2) * t / (4 * d))
    pulse = pulse / pulse.sum(dim=-1, keepdim=True)
    return pulse
```

```python
# 2. Register it under a new name with parameter labels
register_irf(
    "geomorphic_uh",
    geomorphic_unit_hydrograph,
    params=["celerity", "diffusivity"]
)
```

```python
# 3. Build a river network that uses the new IRF
import networkx as nx
g = nx.DiGraph()
g.add_node(0, celerity=2.4, diffusivity=120.0)
g.add_node(1, celerity=2.1, diffusivity=150.0)
g.add_edge(0, 1, delay=2)

riv_tree = RivTree(
    g,
    irf_fn="geomorphic_uh",
    include_index_diag=False
)
```

```python
# 4. Route runoff with the customised kernel
router = LTIRouter(
    max_delay=48,
    dt=1.0,
    block_size=16,
    irf_fn="geomorphic_uh"
)

runoff = torch.rand(1, len(riv_tree), 168)
discharge = router(runoff, riv_tree)
print(discharge.shape)
```

Key points:

- Custom IRFs must accept the signature `(params, time_window, dt)` and return a tensor of shape `[n_reaches, window]`.
- Registering the IRF updates the global `IRF_FN` and `IRF_PARAMS` registries, so `RivTree` knows how to assemble reach-level parameter tensors.