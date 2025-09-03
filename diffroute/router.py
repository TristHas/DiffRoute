import torch.nn as nn
from .agg import RoutingIRFAggregator
from .conv import BlockSparseCausalConv

class LTIRouter(nn.Module):
    def __init__(self,
                 max_delay=100, 
                 block_size=16,
                 irf_fn="linear_storage", 
                 dt=1, cascade=1,
                 sampling_mode="avg",
                 block_f=128,
                 **kwargs):
        """
            Args:
        """
        super().__init__()
        self.residual = not g.include_index_diag
        self.aggregator = RoutingIRFAggregator(max_delay=max_delay, 
                                               block_size=block_size,
                                               dt=dt, cascade=cascade, 
                                               sampling_mode=sampling_mode,
                                               block_f=block_f)   
        self.conv = BlockSparseCausalConv()
        
    def forward(self, x, g):
        """
        """
        kernel = self.aggregator(g)
        output = self.conv(x, kernel)
        if self.residual: output = x + output
        return output

class LTIRouter(nn.Module):
    def __init__(self, aggregator, **kwargs):
        """
        """
        super().__init__()
        self.aggregator = aggregator
        self.conv = BlockSparseCausalConv()
        
    def forward(self, x, g):
        """
        """
        kernel = self.aggregator(g)
        output = self.conv(x, kernel)
        if not g.include_index_diag: output = x + output
        return output
