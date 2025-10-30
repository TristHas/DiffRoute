# Routing a RAPID Project

This tutorial walks through configuring DiffRoute against RAPID inputs, covering graph ingestion, staged routing, and preparation of runoff forcings stored as NetCDF. 
Each step mirrors how you would wire DiffRoute into an operational RAPID deployment.
In this tutorial, we suppose a standard RAPID project with CSV definition files (`rapid_connect.csv`, `k.csv`, `x.csv`, `riv_bas_id.csv`) 
and NetCDF input runoff file (`input.nc`) stored in a same directory.

## 1. Import dependencies

Start by importing the utilities needed to read RAPID configurations, manage paths, load NetCDF runoff with `xarray`, and move data into PyTorch tensors.

```python
from pathlib import Path
import torch
import xarray as xr

from diffroute import LTIStagedRouter
from diffroute.io import read_rapid_graph
```

## 2. Load the RAPID configuration

`read_rapid_graph` parses the standard RAPID CSV files (`rapid_connect.csv`, `k.csv`, `x.csv`, `riv_bas_id.csv`) 
and materialises them as either a `RivTree` or a `RivTreeCluster`, depending on the partitioning thresholds you provide.

```python
vpu_root = Path("/data/rapid/VPU1201")
gs = read_rapid_graph(
    vpu_root,
    plength_thr=30_000,  # optional: segment the network
    node_thr=600
)
```

## 3. Configure the staged router

`LTIStagedRouter` accepts the same high-level hyper-parameters as `LTIRouter` while handling clustered networks transparently. 
You can tune `max_delay` and `dt` to match your hydrological assumptions.

```python
router = LTIStagedRouter(
    max_delay=96,
    dt=1.0,
)
```

## 4. Load RAPID runoff forcings from NetCDF

Load the NetCDF input runoff into an xarray and order the catchments according to the `gs.node_idxs` ordering.
Then convert the xarray data into the `[batch, catchments, time]` tensor layout expected by the router

```python
runoff_da = xr.open_dataarray(vpu_root / "input.nc")
runoff_da = runoff_ds.transpose("reach_id", "time").sel(reach_id=gs.node_idxs.index)
runoff = torch.from_numpy(runoff_da.values).unsqueeze(0).to(device)
```

## 5. Route the forcings

Once the forcings follow the expected device and layout, invoke the router. 

```python
discharge = router(runoff, gs)
print(discharge.shape)
```

- The pipeline above is fully differentiable, so you can backpropagate through runoff inputs or Muskingum parameters when calibrating against observations.
- Swap the synthetic NetCDF for your production runoff forcings to reproduce an end-to-end RAPID routing workflow with DiffRoute.
