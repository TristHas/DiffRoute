# Multi-stage Routing

Large basins quickly push single-stage routing to its memory limits because the aggregated routing kernel grows up to the square of the number of reaches. 
DiffRoute addresses this through staged routing: the river network is segmented into clusters, each routed independently, while inter-cluster discharges are exchanged through transfer buffers.

## Why staging matters

The worst-case memory complexity of a dense routing kernel is \(O(N^2 \times W / dt)\), where:
- \(N\) is the number of reaches (graph nodes),
- \(W\) is the impulse response window expressed in the runoff time step unit (e.g. 10 days for daily runoffs),
- \(dt\) is the routing temportal resolution relative to runoff (e.g. 1/24 for hourly routing of daily runoffs).

Doubling the graph size can up-to-quadruple the kernel footprint before considering temporal expansion. 
For continental-scale networks at fine spatial resolution, this becomes intractable. 
Staging keeps each subgraph small so the per-cluster kernels fit comfortably in GPU memory.

## Segmenting the graph

DiffRoute ships a basic segmentation utility, `diffroute.graph_utils.define_schedule`, that:
1. Scans the directed acyclic river graph and flags breakpoints when upstream routing paths exceed the `plength_thr` threshold.
2. Cuts breakpoint edges to produce weakly connected components.
3. Groups components into clusters that respect the `node_thr` size limit.
4. Tracks transfers between clusters so routed discharge can flow downstream.

The segmentation produces:
- `clusters_g`: a list of NetworkX subgraphs (one per cluster).
- `node_transfer`: a dictionary describing which nodes exchange discharge across cluster boundaries.

A `RivTreeCluster` object can be instantiated from `clusters_g` and `node_transfer`, and directly consumed by `LTIStagedRouter` that will schedule the routing through the different sub-graphs.

## Working with `LTIStagedRouter`

The staged router wraps a standard `LTIRouter` and reuses the same IRF catalogue and parameters. A typical workflow is shown below.

```python
import networkx as nx
import torch

from diffroute import LTIStagedRouter, RivTreeCluster
from diffroute.graph_utils import define_schedule

device = "cuda:0"
# 1. Build or load the full river network (must be acyclic)
G = nx.DiGraph()
# ... populate nodes with IRF parameters and edges with delays ...

# 2. Segment the graph into manageable clusters
clusters_g, node_transfer = define_schedule(
    G,
    plength_thr=25_000,   # breakpoint when cumulative path length exceeds this
    node_thr=800          # maximum reaches per cluster
)

# 3. Wrap the segmented network in a RivTreeCluster
gs = RivTreeCluster(
    clusters_g,
    node_transfer=node_transfer,
    irf_fn="linear_storage",
    include_index_diag=False
).to(device)

# 4. Route runoff in stages
router = LTIStagedRouter(
    max_delay=72,
    dt=1.0,
    block_size=16,
    block_f=128
)

runoff = torch.rand(2, len(gs.nodes_idx), 168, device=device)  # [batch, reaches, time]
discharge = router.route_all_clusters(runoff, gs)
print(discharge.shape)
```

### Choosing segmentation thresholds

- Start with `node_thr` in the 500–1,000 range for GPUs with 24 GB of memory. Lower values trade more staging passes for smaller kernels.
- Set `plength_thr` high enough to let most tributaries stay intact, but low enough to break long serial chains that inflate the kernel width.
- For complex basins, consider manual seeding (e.g., by pre-clustering with hydrological regions) before running `define_schedule`.

### Visualising the segmentation

To inspect the clustering outcome, iterate over `clusters_g` and plot each subgraph, or summarise them as shown:

```python
for cid, cluster in enumerate(clusters_g):
    n_nodes = cluster.number_of_nodes()
    n_edges = cluster.number_of_edges()
    print(f"Cluster {cid:02d}: {n_nodes} reaches, {n_edges} edges")

print(f"Total transfer links: {gs.tot_transfer}")
```

This helps confirm that clusters remain balanced and that the transfer buffers stay manageable.
