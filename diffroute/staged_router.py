from typing import Any, Dict, List, Sequence, Tuple
import copy, random, sys
from tqdm.auto import tqdm

import pandas as pd
import torch
import torch.nn as nn

from .router import LTIRouter
from .ops import write_slice, index_add_inplace

class LTIStagedRouter(nn.Module):
    """Staged router that orchestrates block-sparse routing over clusters.

    Wraps `LTIRouter` while managing inter-cluster transfers entirely in
    PyTorch tensors for batched execution.
    """
    def __init__(
        self,
        max_delay: int = 32,
        dt: float = 1.0,
        sampling_mode: str = "avg",
        block_size: int = 16,
        block_f: int = 128,
        cascade: int = 1,
    ) -> None:
        """Configure the staged router and construct its base LTI model.

        Args:
            max_delay (int): Maximum impulse response length in time-steps.
            dt (float): Temporal resolution of the runoff inputs.
            sampling_mode (str): Strategy for sampling cascade parameters.
            block_size (int | None): Optional override for kernel block size.
            block_f (int): Hidden dimensionality for kernel factorization.
            cascade (int): Number of cascaded IRFs combined by the aggregator.
        """
        super().__init__()
        self.model = LTIRouter(max_delay=max_delay, 
                               block_size=block_size,
                               dt=dt, cascade=cascade, 
                               sampling_mode=sampling_mode,
                               block_f=block_f)

    def forward(self, x: torch.Tensor, gs, params=None):
        return self.route_all_clusters(x, gs, params) 
    
    def _init_transfer_bucket(self, runoff: torch.Tensor, gs) -> torch.Tensor:
        return torch.zeros(runoff.shape[0], gs.tot_transfer, runoff.shape[-1],
                           dtype=runoff.dtype, device=runoff.device)

    def _apply_incoming(self, runoff: torch.Tensor, gs, cluster_idx: int,
                        transfer_bucket: torch.Tensor | None) -> torch.Tensor:
        """Merge upstream transfers into the current cluster runoff."""
        if transfer_bucket is None or cluster_idx not in gs.dst_transfer:
            return runoff
        dst_idx, g_idx = gs.dst_transfer[cluster_idx] 
        runoff = runoff.index_add(1, dst_idx, transfer_bucket[:,g_idx])
        return runoff

    def _store_outgoing(self, discharge: torch.Tensor, gs, cluster_idx: int,
                        transfer_bucket: torch.Tensor | None) -> torch.Tensor | None:
        """Accumulate downstream transfers produced by the current cluster."""
        if transfer_bucket is None or cluster_idx not in gs.src_transfer:
            return transfer_bucket
        src_idx, g_idx = gs.src_transfer[cluster_idx] 
        #transfer_bucket = transfer_bucket.index_add(1, g_idx, discharge[:, src_idx])
        transfer_bucket = index_add_inplace(transfer_bucket, g_idx, discharge[:, src_idx], dim=1)
        return transfer_bucket

    def route_one_cluster(self, runoff: torch.Tensor, gs, cluster_idx: int,
                           params=None,
                           transfer_bucket: torch.Tensor | None = None
                           ) -> Tuple[torch.Tensor, torch.Tensor | None]:
        """Route a single cluster and update the transfer bucket.

        Args:
            runoff (torch.Tensor): Cluster runoff shaped `[B, n_c, T]`.
            gs: Clustered river structure providing routing metadata.
            cluster_idx (int): Identifier of the cluster to route.
            params (Any | None): Optional parameter bundle for the cluster.
            transfer_bucket (torch.Tensor | None): Global transfer storage.

        Returns:
            Tuple[torch.Tensor, torch.Tensor | None]: Routed discharge for
            the cluster and the updated transfer bucket.
        """
        runoff = self._apply_incoming(runoff, gs, cluster_idx, transfer_bucket).clone()
        y_c = self.model(runoff, gs[cluster_idx], params)  # [n_c, T]
        transfer_bucket = self._store_outgoing(y_c, gs, cluster_idx, transfer_bucket)
        return y_c, transfer_bucket

    def route_all_clusters(self, x: torch.Tensor, gs, params: List[Any] | None = None,
                           display_progress: bool = False) -> torch.Tensor:
        """Route all clusters sequentially and assemble the full discharge.

        Args:
            x (torch.Tensor): Full runoff tensor shaped `[B, N_nodes, T]`.
            gs: Clustered river structure with index metadata.
            params (List[Any] | None): Optional per-cluster parameter list.
            display_progress (bool): Whether to wrap the loop with a tqdm bar.

        Returns:
            torch.Tensor: Routed discharge tensor shaped `[B, N_nodes, T]`.
        """
        pbar = tqdm if display_progress else lambda y: y
        if params is None: params = [None] * len(gs)

        transfer_bucket = self._init_transfer_bucket(x, gs)
        out = torch.empty(x.shape[0], len(gs.nodes_idx), x.shape[-1],
                          device=x.device, dtype=x.dtype)

        start = 0
        for cid, param in enumerate(pbar(params)):
            s, e = gs.node_ranges[cid, 0].item(), gs.node_ranges[cid, 1].item()
            y_c, transfer_bucket = self.route_one_cluster(x[:,s:e], gs, cid, param, transfer_bucket)
            end = start + y_c.shape[1]
            out = write_slice(out, y_c, start, end)
            start = end

        return out 

    def route_all_clusters_yield(self, xs: List[torch.Tensor], gs, 
                                 params: List[Any] | None = None,
                                 display_progress: bool = False):
        """Yield per-cluster discharges lazily for streamed routing.

        Args:
            xs (List[torch.Tensor]): Sequence of cluster runoff tensors.
            gs: Clustered river structure with transfer metadata.
            params (List[Any] | None): Optional per-cluster parameter list.
            display_progress (bool): Whether to wrap iteration in tqdm.

        Yields:
            torch.Tensor: Discharge tensor for each cluster in order.
        """
        pbar = tqdm if display_progress else lambda y: y
        if params is None: params = [None] * len(gs)

        transfer_bucket = None
        for idx, (x_c, param) in enumerate(pbar(zip(xs, params))):
            if idx == 0: transfer_bucket = self._init_transfer_bucket(x_c, gs)
            out_c, transfer_bucket = self.route_one_cluster(x_c, gs, idx, param, transfer_bucket)
            yield out_c  

    def init_upstream_discharges(self, xs: List[torch.Tensor], gs, cluster_idx: int,
                                 params: List[Any] | None = None,
                                 display_progress: bool = False) -> torch.Tensor:
        """Warm up the staged router until the target cluster.

        Args:
            xs (List[torch.Tensor]): Sequence of cluster runoff tensors.
            gs: Clustered river structure with transfer metadata.
            cluster_idx (int): Cluster index to stop before routing.
            params (List[Any] | None): Optional per-cluster parameter list.
            display_progress (bool): Whether to wrap iteration in tqdm.

        Returns:
            torch.Tensor: Transfer bucket capturing upstream discharges.
        """
        pbar = tqdm if display_progress else lambda y: y
        if params is None: params = [None] * len(gs)
        transfer_bucket = None
        
        for idx, (x_c, param) in enumerate(pbar(zip(xs, params))):
            if idx == 0: transfer_bucket = self._init_transfer_bucket(x_c, gs)
            if idx == cluster_idx: return transfer_bucket
            _, transfer_bucket = self.route_one_cluster(x_c, gs, idx, param, transfer_bucket)
        return transfer_bucket
