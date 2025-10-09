# IO Utilities

DiffRoute can ingest RAPID routing configurations and convert them into ready-to-route `RivTree` or `RivTreeCluster` instances. The helpers live in `diffroute.io` and handle both single-VPU and multi-VPU projects.

## Reading a single RAPID project

```python
from pathlib import Path

from diffroute.io import read_rapid_graph

vpu_root = Path("/path/to/rapid/VPU1234")

# Without segmentation: returns a single RivTree
g = read_rapid_graph(vpu_root)

# With segmentation: returns a RivTreeCluster
clustered = read_rapid_graph(
    vpu_root,
    plength_thr=50_000,
    node_thr=1_000
)

print(type(g), len(g))
print(type(clustered), len(clustered))
```

- RAPID parameters (`k`, `x`) are converted to Muskingum IRF coefficients and stored in the resulting data structure.
- When `plength_thr` and `node_thr` are provided, the function runs the graph segmentation pipeline (`define_schedule`) before building the `RivTreeCluster`.

## Reading multiple VPUs at once

```python
from pathlib import Path

from diffroute.io import read_multiple_rapid_graphs

vpu_roots = [
    Path("/path/to/rapid/VPU0101"),
    Path("/path/to/rapid/VPU0102"),
]

gs = read_multiple_rapid_graphs(
    vpu_roots,
    plength_thr=40_000,
    node_thr=800
)

print(f"Global reaches: {len(gs.nodes_idx)}")
```

`read_multiple_rapid_graphs` merges the VPU graphs into a single NetworkX DAG, combines their Muskingum parameters, and applies the same optional segmentation logic as the single-project loader.
