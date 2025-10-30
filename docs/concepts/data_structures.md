# Data Structures

DiffRoute pairs differentiable operators with light-weight data structures that describe river graphs and sparse routing kernels. 
Understanding these containers makes it easier to efficiently use and tweak the library.

## River graph representation

### `RivTree`

- Wraps a directed acyclic `networkx.DiGraph` and precomputes the tensors needed for routing (edge lists, path prefixes, parameter matrices).
- Defines an ordering of the nodes. This ordering can either be set at initialization (`nodes_idx`) or defaults to a depth-first traversal if not provided.
- Stores per-reach IRF parameters in `RivTree.params`, a floating tensor shaped `[n_channels, n_params]`.

### `RivTreeCluster`

- Bundles multiple `RivTree` instances produced by graph segmentation.
- Maintains `node_ranges`, a lookup that maps each cluster to the slice of global reach indices it owns.
- Tracks inter-cluster transfers through `src_transfer`, `dst_transfer`, and `tot_transfer` buffers so staged routing can move discharge between clusters.
- Offers `__iter__` and `__getitem__` helpers to iterate over constituent `RivTree`s.

## Routing kernel representations

### `SparseKernel`

- Stores routing kernels in coordinate (COO) form with `coords` `[nnz, 2]`, `vals` `[nnz, window]`, and `size` `(rows, cols, window)`.
- Returned by `IRFAggregator` and easy to convert to dense tensors via `to_dense()`.
- Provides `.to_block_sparse(block_size)` to build a block-structured view.

### `BlockSparseKernel`

- Packs kernels into block-sparse tiles that align with GPU-friendly convolution implementations.
- Keeps `block_indices` `[n_blocks, 2]`, `block_values` `[n_blocks, block_size, block_size, window]`, and the overall `size`.
- Supports conversions: `from_sparse_kernel`, `from_coo`, `to_dense`, and `to_coo`.
- The staged convolution layer (`BlockSparseCausalConv`) consumes this representation directly.

Together, these structures bridge the gap between graph-based hydrological metadata and efficient tensor operations, enabling scalable routing on modern accelerators.
