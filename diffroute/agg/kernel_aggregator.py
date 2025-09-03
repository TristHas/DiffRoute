from imports import *

from .index_precompute import init_pre_indices
from .transitive_closure import sparse_irf_coo_complex
from .kernel_sampler import SubResolutionSampler

from ..block_sparse_tensor import BlockSparseTensor
from ..irfs import IRF_FN

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
    coords, irfs_freq_agg = sparse_irf_coo_complex(
        irfs_freq, edges, path_cumsum,
        include_self=include_index_diag, 
        block_f=block_f
    )
    irfs_agg = torch.fft.irfft(irfs_freq_agg, n=time_window_expanded, dim=-1)
    return coords, irfs_agg

class RoutingIRFAggregator(nn.Module):
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
        
    def forward(self, g):
        irf_fn = IRF_FN[g.irf_fn]
        n_nodes = len(g.nodes_idx)
        out_size = (n_nodes, n_nodes, self.max_delay)

        coords, irfs_agg = aggregate_irf( g.params, 
                                          irf_fn=irf_fn,
                                          edges=g.edges, 
                                          path_cumsum=g.path_cumsum, 
                                          dt=self.dt, 
                                          time_window=self.max_delay, 
                                          cascade=self.cascade, 
                                          include_index_diag=g.include_index_diag, 
                                          block_f=self.block_f)
        
        irfs_agg = torch.relu(irfs_agg)
        irfs_agg = self.sampler.phi_k(irfs_agg.flip(-1))
        irfs_agg /= irfs_agg.sum(-1, keepdims=True) 
        
        return BlockSparseTensor.from_coo(coords, irfs_agg, 
                                          block_size=self.block_size, 
                                          size=out_size,
                                          flip_values=False)