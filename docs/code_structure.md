# Code Structure

DiffRoute is organised by routing stage, with clear separation between high-level orchestration, sparse tensor operators, and supporting utilities. The overview below maps each top-level package or module to its role in the pipeline.

- **`diffroute/agg`**  
  Implements the first stage of routing: building impulse-response kernels. The main class, `IRFAggregator`, runs an FFT-based transitive-closure to accumulate upstream IRFs, rescales them with `SubResolutionSampler`, and outputs a `SparseKernel`. Its `forward` method accepts a `RivTree` (plus optional parameter overrides) and returns the aggregated kernel ready for convolution.

- **`diffroute/conv`**  
  Hosts the second stage: block-sparse convolutions. `BlockSparseCausalConv.forward` expects the runoff tensor and a `BlockSparseKernel`, then calls either the Triton implementation (`diffroute.ops.block_sparse_conv_1d`) or a PyTorch fallback. The layer preserves causality and trims padding so outputs match the input horizon.

- **`diffroute/ops`**  
  Contains low-level kernels and graph algorithms. `transitive_closure` powers stage one by summing IRFs in the Fourier domain, while convolution routines (Triton and PyTorch variants) drive stage two. Use these if you need custom operator extensions or want to experiment with alternative kernels.

- **`diffroute/structs`**  
  Provides data containers for river graphs (`RivTree`, `RivTreeCluster`) and routing kernels (`SparseKernel`, `BlockSparseKernel`). Utility helpers cover index initialisation, COO/blocked conversions, and transfer table construction for staged routing.

- **`diffroute/graph_utils`**  
  Offers helpers for graph segmentation, currently centred around `define_schedule`. The function splits large river networks into stageable clusters and returns both the subgraphs and the inter-cluster transfer map. Expect more clustering strategies here in future releases.

- **`diffroute/io.py`**  
  Handles RAPID interoperability. `read_rapid_graph` and `read_multiple_rapid_graphs` parse RAPID CSV exports, attach Muskingum parameters, and return ready-to-route `RivTree` or `RivTreeCluster` objects.

- **`diffroute/irfs.py`**  
  Defines the built-in impulse-response functions (Muskingum, Nash cascade, linear storage, Hayami, pure lag) and exposes the `register_irf` hook for custom kernels.

- **`diffroute/router.py`**  
  Implements `LTIRouter`, the single-stage operator that combines `IRFAggregator` and `BlockSparseCausalConv` inside a PyTorch `nn.Module`.

- **`diffroute/staged_router.py`**  
  Implements `LTIStagedRouter`, which orchestrates sequential routing over `RivTreeCluster` instances, moves discharge between clusters, and reuses the single-stage router internally.
