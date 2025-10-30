# Defining a Custom IRF

DiffRoute lets you extend the library with custom impulse response functions (IRFs). 
In this example, we will implement the **Weibull-based Instantaneous Unit Hydrograph (IUH)** as an example to demonstrate how to register such custom IRFs.
The Weibull-based Instantaneous Unit Hydrograph (IUH) is a flexible probability-density IRF 
that emerged from the probabilistic IUH literature as an alternative to linear reservoirs, Nash cascades, and Muskingum routing. 
Studies such as Bhunya et al. (2008) and Nadarajah (2007) highlight the Weibull IUH’s ability to reproduce skewed hydrograph shapes using only the shape (`k`) and scale (`λ`) parameters,
with an optional onset delay. 

We will implement the IRF in PyTorch, register it within `diffroute`, and demonstrate how to route runoff using the new kernel.

## 1. Import dependencies

Bring in PyTorch along with the DiffRoute helpers, plus the scientific Python stack used to synthesise a random river network and parameter table.

```python
import torch
import numpy as np
import pandas as pd
import networkx as nx

from diffroute import LTIRouter, RivTree, register_irf
```

## 2. Implement the Weibull IUH

The IRF signature follows DiffRoute’s convention: `(params, time_window, dt) -> tensor[n_reaches, window]`. Each row of the returned tensor must be normalised to unit mass. We expose three parameters—`shape`, `scale`, and `onset`—to mirror the standard Weibull PDF with an optional translation.

```python
def weibull_irf(params, time_window, dt):
    """
    params[:, 0] -> shape (k)     : controls peak sharpness
    params[:, 1] -> scale (lam)   : stretches the hydrograph in time
    params[:, 2] -> onset (t0)    : optional delay before response
    """
    shape = params[:, 0].unsqueeze(1)
    scale = params[:, 1].unsqueeze(1)
    onset = params[:, 2].unsqueeze(1) if params.size(1) > 2 else torch.zeros_like(shape)

    steps = max(1, int(round(time_window / dt)))
    t = torch.arange(1, steps + 1, device=params.device, dtype=params.dtype).unsqueeze(0) * dt

    te = (t - onset).clamp(min=0)
    eps = torch.finfo(params.dtype).eps
    z = te / scale.clamp(min=eps)
    mask = (te > 0).to(params.dtype)

    pulse = mask * (shape / scale.clamp(min=eps)) * torch.pow(z.clamp(min=eps), shape - 1) * torch.exp(-torch.pow(z, shape))
    pulse = pulse / (pulse.sum(dim=-1, keepdim=True) + eps)
    return pulse
```

## 3. Register the IRF

`register_irf` adds the implementation to DiffRoute’s global registry and associates human-readable parameter names with the kernel. These labels must match the attributes you store on NetworkX nodes (or columns in your `pandas.DataFrame`), so keep the naming consistent.

```python
register_irf(
    "weibull_iuh",
    weibull_irf,
    params=["shape", "scale", "onset"]
)
```

After registration, `RivTree` knows that any network using `irf_fn="weibull_iuh"` requires the parameters `shape`, `scale`, and `onset`, and it will pack them in that exact order when building tensors for the router.

## 4. Build a river network with Weibull parameters

Create a random river tree using the same `nx.gn_graph` helper employed in the Overview example. Store the Weibull parameters in a `pandas.DataFrame`, with one independently generated column per parameter. Passing the DataFrame to `RivTree` keeps the parameter ordering consistent with the registration labels.

```python
n_reaches = 20
rng = np.random.default_rng(seed=0)
G = nx.gn_graph(n_reaches, seed=0)

param_df = pd.DataFrame(
    {
        "shape": rng.uniform(1.2, 2.5, size=n_reaches),
        "scale": rng.uniform(10.0, 24.0, size=n_reaches),
        "onset": rng.uniform(0.0, 4.0, size=n_reaches),
    },
    index=G.nodes
)

riv_tree = RivTree(
    G,
    params=param_df,
    irf_fn="weibull_iuh",
    include_index_diag=False
)
```

## 5. Route runoff with the custom kernel

Instantiate `LTIRouter` with the same `irf_fn` name and feed it a sample runoff tensor. DiffRoute automatically looks up the registered kernel and hands the per-reach parameter tensor—ordered as `["shape", "scale", "onset"]`—to the Weibull IUH implementation.

```python
router = LTIRouter(
    max_delay=96,
    dt=1.0,
    irf_fn="weibull_iuh"
)

runoff = torch.rand(1, len(riv_tree), 168)
discharge = router(runoff, riv_tree)
print(discharge.shape)
```

## Key points

- Custom IRFs must implement the `(params, time_window, dt)` interface and return `[n_reaches, window]` kernels normalised to unit mass.
- `register_irf` couples the kernel implementation with parameter labels; these labels must match the attributes used when constructing `RivTree`.
- Using the same `irf_fn` identifier during registration, network construction, and routing ensures DiffRoute wires the correct parameters into the custom kernel.
