import torch
import torch.nn as nn

from .agg import IRFAggregator
from .conv import BlockSparseCausalConv

class LTIRouter(nn.Module):
    """
    """
    def __init__(self,
                 max_delay=100, 
                 block_size=16,
                 irf_fn="linear_storage", 
                 dt=1, cascade=1,
                 sampling_mode="avg",
                 block_f=128,
                 **kwargs):
        super().__init__()
        self.block_size = block_size
        self.aggregator = IRFAggregator(max_delay=max_delay, 
                                        #block_size=block_size,
                                        dt=dt, cascade=cascade, 
                                        sampling_mode=sampling_mode,
                                        block_f=block_f)
        self.conv = BlockSparseCausalConv()

    def forward(self, runoff: torch.Tensor, g, params=None) -> torch.Tensor:
        """
        Args:
            runoff: torch.Tensor [B, C, T]
            g: RivTree
            params: optional per-cluster params for aggregator. If None, then use the g.params
        Returns:
            discharge: torch.Tensor [B, C, T]
        """
        if runoff.ndim != 3: raise ValueError(f"runoff must be [B, C, T], got {runoff.shape}")
        # Stage 1: Aggregate kernel
        kernel = self.aggregator(g, params)
        kernel = kernel.to_block_sparse(self.block_size)
        # Stage 2: Convolution
        y = self.conv(runoff, kernel)
        # Handle residual if needed
        if not g.include_index_diag: y = runoff + y 
        return y

