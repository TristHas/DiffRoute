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
        self.aggregator = IRFAggregator(max_delay=max_delay, 
                                        block_size=block_size,
                                        dt=dt, cascade=cascade, 
                                        sampling_mode=sampling_mode,
                                        block_f=block_f)
        self.conv = BlockSparseCausalConv()

    def forward(self, runoff: torch.Tensor, g, params=None) -> torch.Tensor:
        """
        Args:
            runoff: [n_series, T]
            g: graph/cluster object used by IRFAggregator
            params: optional per-cluster params for aggregator
        Returns:
            discharge: [n_series, T]
        """
        if runoff.ndim != 3: raise ValueError(f"runoff must be [B, N, T], got {runoff.shape}")
        kernel = self.aggregator(g, params)
        y = self.conv(runoff, kernel)
        if not g.include_index_diag: y = runoff + y 
        return y

