from typing import Any, Dict, List, Sequence, Tuple
import copy, random, sys
from tqdm.auto import tqdm

import pandas as pd
import torch
import torch.nn as nn

from .router import LTIRouter
from .ops import write_slice

class LTIStagedRouter(nn.Module):
    """
        Tensor-only staged router over clusters.
        All public methods return/consume torch.Tensor.
    """
    def __init__(
        self,
        max_delay: int = 32,
        block_size: int | None = None,
        block_f: int = 128,
        dt: float = 1.0,
        cascade: int = 1,
        sampling_mode: str = "avg",
    ) -> None:
        super().__init__()
        self.model = LTIRouter(max_delay=max_delay, 
                               block_size=block_size,
                               dt=dt, cascade=cascade, 
                               sampling_mode=sampling_mode,
                               block_f=block_f)

    def _init_transfer_bucket(self, runoff: torch.Tensor, gs) -> torch.Tensor:
        return torch.zeros(runoff.shape[0], gs.tot_transfer, runoff.shape[-1],
                           dtype=runoff.dtype, device=runoff.device)

    def _apply_incoming(self, runoff: torch.Tensor, gs, cluster_idx: int,
                        transfer_bucket: torch.Tensor | None) -> torch.Tensor:
        """
            runoff: [1, n_c, T]
        """
        if transfer_bucket is None or cluster_idx not in gs.dst_transfer:
            return runoff
        dst_idx, g_idx = gs.dst_transfer[cluster_idx] 
        runoff = runoff.index_add(1, dst_idx, transfer_bucket[:,g_idx])
        return runoff

    def _store_outgoing(self, discharge: torch.Tensor, gs, cluster_idx: int,
                        transfer_bucket: torch.Tensor | None) -> torch.Tensor | None:
        """
        discharge: [B, n_c, T]
        """
        if transfer_bucket is None or cluster_idx not in gs.src_transfer:
            return transfer_bucket
        src_idx, g_idx = gs.src_transfer[cluster_idx] 
        transfer_bucket = transfer_bucket.index_add(1, g_idx, discharge[:, src_idx])
        return transfer_bucket

    def route_one_cluster(self, runoff: torch.Tensor, gs, cluster_idx: int,
                           params=None,
                           transfer_bucket: torch.Tensor | None = None
                           ) -> Tuple[torch.Tensor, torch.Tensor | None]:
        """
            x_c: [n_c, T] for cluster `cluster_idx`
            Returns: (y_c: [n_c, T], updated transfer_bucket)
        """
        runoff = self._apply_incoming(runoff, gs, cluster_idx, transfer_bucket).clone()
        y_c = self.model(runoff, gs[cluster_idx], params)  # [n_c, T]
        transfer_bucket = self._store_outgoing(y_c, gs, cluster_idx, transfer_bucket)
        return y_c, transfer_bucket

    def route_all_clusters(self, x: torch.Tensor, gs, params: List[Any] | None = None,
                           display_progress: bool = False) -> torch.Tensor:
        """
        Args:
            x: [N_total_nodes, T]
            gs: graph/cluster structure with node_ranges, nodes_idx, etc.
            params: list per cluster or None
        Returns:
            out: [N_total_nodes, T]
        """
        if not display_progress: tqdm = lambda y: y
        if params is None: params = [None] * len(gs)

        transfer_bucket = self._init_transfer_bucket(x, gs)
        out = torch.empty(x.shape[0], len(gs.nodes_idx), x.shape[-1],
                          device=x.device, dtype=x.dtype)

        start = 0
        for cid, param in enumerate(tqdm(params)):
            s, e = gs.node_ranges[cid, 0].item(), gs.node_ranges[cid, 1].item()
            y_c, transfer_bucket = self.route_one_cluster(x[:,s:e], gs, cid, param, transfer_bucket)
            end = start + y_c.shape[1]
            out = write_slice(out, y_c, start, end)
            start = end

        return out 

    def route_all_clusters_yield(self, xs: List[torch.Tensor], gs, 
                                 params: List[Any] | None = None,
                                 display_progress: bool = False):
        """
        Batch helper: xs is a list/iterable of [n_c, T] tensors per cluster.
        Yields: tensors [n_c, T].
        """
        if not display_progress: tqdm = lambda y: y
        if params is None: params = [None] * len(gs)

        transfer_bucket = None
        for idx, (x_c, param) in enumerate(tqdm(zip(xs, params))):
            if idx == 0: transfer_bucket = self._init_transfer_bucket(x_c, gs)
            out_c, transfer_bucket = self.route_one_cluster(x_c, gs, idx, param, transfer_bucket)
            yield out_c  

    def init_upstream_discharges(self, xs: List[torch.Tensor], gs, cluster_idx: int,
                                 params: List[Any] | None = None,
                                 display_progress: bool = False) -> torch.Tensor:
        """
        Run sequentially up to (but not including) `cluster_idx` and return the transfer bucket.
        """
        if not display_progress: tqdm = lambda y: y
        if params is None: params = [None] * len(gs)
        transfer_bucket = None
        
        for idx, (x_c, param) in enumerate(tqdm(zip(xs, params))):
            if idx == 0: transfer_bucket = self._init_transfer_bucket(x_c, gs)
            if idx == cluster_idx: return transfer_bucket
            _, transfer_bucket = self.route_one_cluster(x_c, gs, idx, param, transfer_bucket)
        return transfer_bucket
