import torch
from torch import nn as nn

from .temporal_sampler import SubResolutionSampler
from ..ops import log_transitive_closure
from ..irfs import IRF_FN
from ..structs import SparseKernel

def aggregate_irf(params, irf_fn,
                  edges, path_cumsum,
                  dt, time_window,
                  cascade=1,
                  include_index_diag=True,
                  block_f=128):
    """
    """
    irfs = irf_fn(params, time_window=time_window, dt=dt).squeeze()
    time_window_expanded = irfs.shape[-1]
    assert time_window_expanded == time_window * int( 1 / dt )

    irfs_freq = torch.fft.rfft(irfs, n=time_window_expanded, dim=-1)
    coords, irfs_freq_agg = log_transitive_closure(
        irfs_freq, edges, path_cumsum,
        include_self=include_index_diag,
        block_f=block_f
    )
    irfs_agg = torch.fft.irfft(irfs_freq_agg, n=time_window_expanded, dim=-1)
    return coords, irfs_agg

class IRFAggregator(nn.Module):
    def __init__(self,
                 max_delay=6,
                 block_size=16,
                 dt=1,
                 sampling_mode="avg",
                 cascade=1,
                 block_f=128,
                 **kwargs):
        """
            g (nx.Digraph): river network stage graph
            nodes_idx (Collection): Node ordering (optional).
            irf_fn:
            irf_agg:
            index_precomp
        """
        super().__init__()
        self.dt = dt
        self.cascade = cascade
        self.max_delay = max_delay
        self.block_size = block_size
        self.block_f = block_f
        self.sampler = SubResolutionSampler(dt=dt, out_mode=sampling_mode)

    def forward(self, g, params=None):
        if params is None: params = g.params
        irf_fn = IRF_FN[g.irf_fn]

        coords, irfs_agg = aggregate_irf( params,
                                          irf_fn=irf_fn,
                                          edges=g.edges,
                                          path_cumsum=g.path_cumsum,
                                          dt=self.dt,
                                          time_window=self.max_delay,
                                          cascade=self.cascade,
                                          include_index_diag=g.include_index_diag,
                                          block_f=self.block_f)

        irfs_agg = torch.relu(irfs_agg)
        irfs_agg = self.sampler.phi_k(irfs_agg).flip(-1)
        irfs_agg /= irfs_agg.sum(-1, keepdims=True)

        kernel_size = (len(g), len(g), irfs_agg.shape[-1])
        return SparseKernel(coords, irfs_agg, kernel_size)
