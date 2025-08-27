from imports import *

from .index_precompute import init_pre_indices
from .scatter_reduce import IRF_AGGREGATE_FN
from .transitive_closure import sparse_irf_coo_complex
from .kernel_sampler import SubResolutionSampler

from ..utils import get_node_idxs
from ..block_sparse_tensor import BlockSparseTensor
from ..irfs import IRF_FN

def aggregate_irf(params, irf_fn,
                  edges, path_cumsum, nodes_idx, 
                  dt, time_window, 
                  cascade=1, 
                  include_index_diag=True, 
                  block_f=128):
    """
    """
    #device = params.device
    #edges = edges.to(device)
    #path_cumsum = path_cumsum.to(device)

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
    def __init__(self, g, 
                 nodes_idx=None, 
                 max_delay=6, 
                 block_size=16,
                 irf_fn="linear_storage", 
                 include_index_diag=False,
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
        self.irf_fn = IRF_FN[irf_fn]

        self.g = g
        self.dt = dt
        self.cascade = cascade
        self.max_delay = max_delay
        self.block_size = block_size
        self.include_index_diag = include_index_diag
        self.block_f = block_f
        
        self.nodes_idx = get_node_idxs(g) if nodes_idx is None else nodes_idx
        self.out_size = (len(self.nodes_idx), len(self.nodes_idx), self.max_delay)
        self.sampler = SubResolutionSampler(dt=dt, out_mode=sampling_mode)
        
        # Variables needed for buffer init
        edges, path_cumsum, _ = init_pre_indices(g, nodes_idx, 
                                                 include_self=include_index_diag)
        self.register_buffer("edges", edges)
        self.register_buffer("path_cumsum", path_cumsum)
        
    def forward(self, params):
        coords, irfs_agg = aggregate_irf( params, 
                                          irf_fn=self.irf_fn,
                                          edges=self.edges, 
                                          path_cumsum=self.path_cumsum, 
                                          nodes_idx=self.nodes_idx, 
                                          dt=self.dt, 
                                          time_window=self.max_delay, 
                                          cascade=self.cascade, 
                                          include_index_diag=self.include_index_diag, 
                                          block_f=self.block_f)
        irfs_agg = torch.relu(irfs_agg)
        irfs_agg = self.sampler.phi_k(irfs_agg.flip(-1))
        irfs_agg /= irfs_agg.sum(-1, keepdims=True) 
        return BlockSparseTensor.from_coo(coords, irfs_agg, 
                                          block_size=self.block_size, 
                                          size=self.out_size,
                                          flip_values=False)