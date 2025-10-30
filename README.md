# DiffRoute

**Differentiable and scalable LTI routing operator integrated into PyTorch**

---

## Overview

This repository contains the code for the `diffroute` model presented in the paper:

> *Differentiable river routing for end-to-end learning of hydrological processes at diverse scales, ESS Open Archive . May 27, 2025.*

`diffroute` provides a differentiable and GPU-accelerated formulation of classical Linear Time Invariant (LTI) river routing models. 
Every component is written in PyTorch, which keeps routing differentiable and easy to integrate in deep learning workflows.

For advanced use-cases, we recommend using diffroute through the companion module `diffhydro`, 
which layers additional data structures and utilities to integrate DiffRoute into more complex differentiable hydrological pipelines, with minimal dependency.
However, DiffRoute can also be used as a standalone component as showcased in this documentation.

Please note that `diffroute` is in an early development stage and thus subject to changes.
Nevertheless, the explanations included in this documentation are limited to API and structural components that are unlikely to change in the short to middle term.

## Key Features
- **GPU Acceleration**: Formulating LTI River Routing operations as 1D Convolution layers allow for efficient GPU execution.
- **Differentiable**: DiffRoute is integrated to Pytorch, which allows for efficient gradient computations through Automatic Differentiation.
- **Scalable**: Computations scale to very large river graphs (up to millions of river channels) using staged computations.
- **Generality**: DiffRoute allows to formulate *any* LTI River Routing scheme. Currenlty implemented routing schemes include the *Muskingum*, *Linear Diffusive Wave*, *Linear Storage*, *Nash Cascade* and *Pure Lag* schemes. Custom schemes can easily be added as shown in this example. 
- **Batching**: Accepts batched runoff tensors with shape `[batch, catchments, time]` and routes them efficiently. This simplifies batched training of parameters and inference of ensemble predictions.
- **Composability**: Integration to Pytorch Automatic Differentiation framework also aims to combine `diffroute` with other learnable components. The companion module `diffhydro` aims to simplify the assembly of more complex hydrological pipelines.
- **Pure Python**: DiffRoute is written in pure python. As such it is easily hackable for users interested in experimenting with variations.

## Installation

You can install `diffroute` from pip repository:

```bash
pip install diffroute
```

Or install the latest version from GitHub:
```bash
pip install git+https://github.com/TristHas/DiffRoute.git
```

Or clone and install locally:

```bash
git clone https://github.com/TristHas/DiffRoute.git
cd DiffRoute; pip install .
```

### Dependencies

The package depends on:

- `torch>=2.0`
- `networkx`
- `pandas`
- `tqdm`

These are automatically installed when using `pip`.

---

## Quickstart

```python
import numpy as np
import pandas as pd
import networkx as nx
import torch

from diffroute import RivTree, LTIRouter

b = 2   # Batch (or ensemble size)
n = 20  # Number of channels
T = 100 # Number of time steps
device = "cuda:0" # GPU device to use

G = nx.gn_graph(n) # Toy example tree with a unique outlet.
# Define per-node routing parameters. For a Linear Storage scheme, only one parameter "tau"
params = pd.DataFrame({"tau": np.random.rand(n)}, index=G.nodes) 
# River Tree data structure
riv_tree = RivTree(G, params=params, irf_fn="linear_storage").to(device) 
# Generate random input runoff
runoff = torch.rand(b, n, T, device=device)
# Instantiate the routing model with desired parameters
router = LTIRouter(max_delay=48, dt=1)
# Compute output discharges from input graph and runoffs
discharge = router(runoff, riv_tree)
```

## Documentation

TODO

## Publications and Citation

The motivation for, mathematical derivations behind, and illustrative use-cases of DiffRoute are presented in the paper "Differentiable river routing for end-to-end learning of hydrological processes".

> "Differentiable river routing for end-to-end learning of hydrological processes at diverse scales"  
> ESS Open Archive . May 27, 2025.
> DOI: 10.22541/essoar.174835108.87664030/v1

If you use `diffroute` in your academic work, please cite the above reference.
The original code used for the experiment in the paper at the time of submission can be found in a standalone repository

However, We recommend using `diffroute` through the companion package `diffhydro`, which layers additional data structures and utilities to ease the use of DiffRoute, with minimal dependency.
The original experiments of the paper have been included as example notebooks within the `diffhydro` package.

---

## License

This project is licensed under the terms of the MIT license.  
See the [LICENSE](LICENSE) file for details.
