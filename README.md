# DiffRoute

**Differentiable and scalable LTI routing operator integrated into PyTorch**

---

## Overview

This repository contains the code for the `diffroute` model presented in the paper:

> *Differentiable river routing for end-to-end learning of hydrological processes at diverse scales*

`diffroute` provides a GPU-accelerated, differentiable formulation of Linear Time Invariant (LTI) river routing models, enabling seamless integration into deep learning pipelines.

---

## Features

- Fast, differentiable implementation of LTI routing schemes (e.g., Muskingum, Hayami, Nash cascade)
- Fully GPU-accelerated using PyTorch
- Modular and scalable to large river networks
- Designed for integration into ML workflows (e.g., forecasting, assimilation, inverse modeling)
---

## Installation

You can install the latest version of `diffroute` directly from GitHub:

```bash
pip install git+https://github.com/TristHas/DiffRoute.git
```

Or clone and install locally:

```bash
git clone https://github.com/TristHas/DiffRoute.git
cd DiffRoute
pip install .
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
from diffroute import RoutingModel  

model = RoutingModel(g, ...)    # instantiate model with the river network graph g and desired parameters
Q_out = model(runoff_input)     # apply routing to input runoffs
```

For more examples and to reproduce the experiments of the original paper, see the companion repository:  
👉 https://github.com/TristHas/diffrouteExperiments

---

## Tutorials

The repository includes two simple tutorial notebooks to get you started.

### RAPID IO

In this tutorial, you can see how to use the DiffRoute model to accelerate existing simulations.
We show you how to load routing parameters from existing RAPID projects and execute them with DiffRoute.

### Custom IRF

In this tutorial, we show you how to add your own routing scheme IRF and scale it to large river networks/.

## Citation

If you use this work, please cite:

> "Differentiable river routing for end-to-end learning of hydrological processes at diverse scales"  
> [Citation coming soon — preprint in preparation]

---

## License

This project is licensed under the terms of the MIT license.  
See the [LICENSE](LICENSE) file for details.
