import torch.nn as nn
from .agg import RoutingIRFAggregator
from .conv import BlockSparseCausalConv

class LTIRouter(nn.Module):
    def __init__(self, g, nodes_idx=None, 
                 max_delay=100, 
                 block_size=16,
                 irf_fn="linear_storage", 
                 irf_agg="log_triton", 
                 index_precomp="cpu",
                 runoff_to_output=False,
                 dt=1, cascade=1,
                 sampling_mode="avg",
                 device="cpu"):
        """
            Args:
                
            
        """
        super().__init__()
        self.residual = runoff_to_output
        self.aggregator = RoutingIRFAggregator(g, nodes_idx=nodes_idx,
                                               max_delay=max_delay, 
                                               block_size=block_size,
                                               irf_fn=irf_fn, 
                                               irf_agg=irf_agg, 
                                               index_precomp=index_precomp,
                                               include_index_diag=not self.residual,
                                               dt=dt, cascade=cascade, 
                                               sampling_mode=sampling_mode,
                                               device=device)   
        self.conv = BlockSparseCausalConv()
        
    def forward(self, x, params):
        """
        """
        kernel = self.aggregator(params)
        output = self.conv(x, kernel)
        if self.residual: output = x + output
        return output
