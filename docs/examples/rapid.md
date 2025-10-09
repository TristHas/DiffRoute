# Routing a RAPID Project

```python
from pathlib import Path
import torch

from diffroute import LTIStagedRouter
from diffroute.io import read_rapid_graph
```

```python
# 1. Load the RAPID configuration
vpu_root = Path("/data/rapid/VPU1201")
gs = read_rapid_graph(
    vpu_root,
    plength_thr=30_000,  # optional: segment the network
    node_thr=600
)
```

```python
# 2. Build the router (handles both RivTree and RivTreeCluster)
router = LTIStagedRouter(
    max_delay=96,
    dt=1.0,
    block_size=16
)
```

```python
# 3. Route runoff forcings (here random test data)
runoff = torch.rand(3, len(gs.nodes_idx), 240) if hasattr(gs, "nodes_idx") else torch.rand(3, len(gs), 240)
discharge = router.route_all_clusters(runoff, gs) if hasattr(gs, "node_ranges") else router(runoff, gs)
print(discharge.shape)
```

- `read_rapid_graph` parses RAPID CSV files (`rapid_connect.csv`, `k.csv`, `x.csv`, `riv_bas_id.csv`) and maps Muskingum parameters to DiffRoute IRF inputs.
- The same router instance seamlessly handles single-cluster (`RivTree`) and staged (`RivTreeCluster`) networks.
- For real applications, replace the random runoff tensor with your model or observation-driven runoff forcings.
