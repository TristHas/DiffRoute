# DiffRoute Overview

`diffroute` provides a differentiable, GPU-accelerated formulation of classical Linear Time Invariant (LTI) river routing models. 
Every component is written in PyTorch, which keeps routing differentiable and easy to integrate in deep learning workflows.

The motivation for, mathematical derivations behind, and illustrative use-cases of DiffRoute are presented in the paper *"Differentiable river routing for end-to-end learning of hydrological processes"*.

We recommend using diffroute through the companion package `diffhydro`, which layers additional data structures and utilities to ease the use of DiffRoute, with minimal dependency.
Nevertheless, DiffRoute can be used as a standalone component as showcased in this documentation.

Please note that `diffroute` is in an early development stage and thus subject to changes.
Nevertheless, the explanations included in this documentation are limited to API and structural components that are unlikely to change in the short to middle term.

## Key Functionalities
- **GPU Acceleration**: Formulating LTI River Routing operations as 1D Convolution layers allow for efficient GPU execution.
- **Differentiable**: DiffRoute is integrated to Pytorch, which allows for efficient gradient computations through Automatic Differentiation.
- **Scalable**: Computations scale to very large river graphs (up to millions of river channels) using staged computations.
- **Generality**: DiffRoute allows to formulate *any* LTI River Routing scheme. Currenlty implemented routing schemes include the *Muskingum*, *Linear Diffusive Wave*, *Nash Cascade* and *Pure Lag* schemes. Utility functions allow to easily define and integrate custom LTI routing schemes. 
- **Batching**: Accepts batched runoff tensors with shape `[batch, catchments, time]` and routes them efficiently.

## Library Layout
- `diffroute.router.LTIRouter`: single-stage router suitable for limited size river graphs (up to tens of thousands of river channels).
- `diffroute.staged_router.LTIStagedRouter`: orchestration layer for staged routing across large river networks  (up to millions of river channels).
- `diffroute.agg.IRFAggregator`: Operator for the first stage of the routing procedure: Kernel aggregation.
- `diffroute.conv.BlockSparseCausalConv`: Operator for the first stage of the routing procedure: Causal block-sparse convolution.
- `diffroute.irfs`: Reference implementations for classical routing scheme IRFs plus a utility registry for custom scheme definition.
- `diffroute.structs`: Typed containers for sparse kernels and river-network metadata.
- `diffroute.ops`: Low-level implementation of various operators in triton.

Please see the **Code Structure** section for more details.

## Typical Workflow

1. Build a `RivTree` (or `RivTreeCluster` for large river networks) graph that encodes river channel connectivity and parameters.
2. Instantiate an `LTIRouter` (or `LTIStagedRouter` for clustered graphs) with the desired routing parameters.
3. Call the router with a runoff tensor `runoff: torch.Tensor[batch, catchments, time]`.
4. Use the routed discharge for downstream forecasting, calibration, or training.

Continue with the quickstart to see a minimal runnable example and recommended configuration patterns.
