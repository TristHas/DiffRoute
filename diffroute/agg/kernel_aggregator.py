import networkx as nx
import torch
from torch import nn

from .index_precompute import INDEX_PRECOMPUTE
from .scatter_reduce import IRF_AGGREGATE_FN
from .kernel_sampler import SubResolutionSampler

from ..utils import get_node_idxs
from ..block_sparse_tensor import BlockSparseTensor
from ..irfs import IRF_FN

class RoutingIRFAggregator(nn.Module):
    def __init__(self, g, nodes_idx=None, 
                 max_delay=6, block_size=4,
                 irf_fn="linear_storage", 
                 irf_agg="log_triton", 
                 index_precomp="cpu",
                 include_index_diag=False,
                 dt=1, sampling_mode="avg",
                 device="cpu",
                 cascade=1):
        """
            g (nx.Digraph): river network stage graph
            nodes_idx (Collection): Node ordering (optional).
            irf_fn:
            irf_agg: 
            index_precomp
        """
        super().__init__()
        self.irf_agg_fn = IRF_AGGREGATE_FN[irf_agg]
        self.irf_fn = IRF_FN[irf_fn]
        self.dt = dt
        self.cascade = cascade
        self.nodes_idx = get_node_idxs(g) if nodes_idx is None else nodes_idx
        self.n_nodes = len(self.nodes_idx)
        self.max_delay = max_delay
        self.block_size = block_size
        self.out_size = (len(self.nodes_idx), len(self.nodes_idx), self.max_delay )
        self.sampler = SubResolutionSampler(dt=dt, out_mode=sampling_mode)
        
        # Variables needed for buffer init
        self.g = g
        self.index_precomp = index_precomp
        self.include_index_diag = include_index_diag
        self.buffers_initialized = False

    def to(self, device, **kwargs):
        super().to(device, **kwargs)
        if not self.buffers_initialized:
            self.init_buffers(device)
        return self
        
    def init_buffers(self, device):
        if device=="cpu":
            path_idx, path_node, coords = INDEX_PRECOMPUTE[self.index_precomp](self.g, self.nodes_idx, 
                                                                              include_self=self.include_index_diag)
        else:
            path_idx, path_node, coords = INDEX_PRECOMPUTE["gpu"](self.g, node_idxs=self.nodes_idx, 
                                                                  device=device,
                                                                  include_self=self.include_index_diag)
        self.register_buffer("coords", coords)
        self.register_buffer("path_nodes", path_node)
        self.register_buffer("path_idxs", path_idx)
        self.buffers_initialized = True

    def aggregate_irf(self, params):
        """
        Args:
            params (_type_): _description_
        """
        if not self.buffers_initialized: self.init_buffers(params.device)
        time_window = self.max_delay
        irfs = self.irf_fn(params, time_window=time_window, dt=self.dt).squeeze()
        #irfs = torch.relu(irfs)
        #irfs = irfs / irfs.sum(-1, keepdims=True)
        # Be careful here: unsqueezed yield bug, needs to be clarified
        time_window_expanded = irfs.shape[-1]
        assert time_window_expanded == self.max_delay * int( 1 / self.dt )
        irfs_freq = torch.fft.rfft(irfs, n=time_window_expanded, dim=-1)
        irfs_freq_agg = self.irf_agg_fn(self.path_idxs, self.path_nodes, 
                                        irfs_freq,
                                        cascade=self.cascade)
        irfs_agg = torch.fft.irfft(irfs_freq_agg, n=time_window_expanded, dim=-1)
        irfs_agg = torch.relu(irfs_agg)
        irfs_agg = self.sampler.phi_k(irfs_agg.flip(-1))
        irfs_agg /= irfs_agg.sum(-1, keepdims=True) #.detach()
        return BlockSparseTensor.from_coo(self.coords, irfs_agg, 
                                          block_size=self.block_size, 
                                          size=self.out_size,
                                          flip_values=False)
        
    def forward(self, x):
        """
        """
        return self.aggregate_irf(x)

    @classmethod
    def create_aggregator(cls, g, nodes_idx=None, max_delay=None, 
                          block_size=4,
                          irf_fn="linear_storage", 
                          irf_agg="log_triton", 
                          index_precomp="optimized",
                          dt=1):
        """
        """
        return cls(g, nodes_idx, max_delay,
                    block_size=block_size,
                    irf_fn=irf_fn, 
                    irf_agg=irf_agg, 
                    index_precomp=index_precomp,
                    dt=dt)
