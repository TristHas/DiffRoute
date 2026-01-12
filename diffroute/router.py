import torch
import torch.nn as nn

from .agg import IRFAggregator
from .conv import BlockSparseCausalConv

class LTIRouter(nn.Module):
    """Linear time-invariant river routing module.

    Combines an impulse response function aggregator with a block-sparse
    convolution to transform runoff into downstream discharge.
    """
    def __init__(self,
                 max_delay=100, 
                 dt=1,
                 sampling_mode="avg",
                 block_size=16,
                 block_f=128,
                 cascade=1,
                 **kwargs):
        """Initialize the router with aggregation and convolution settings.

        Args:
            max_delay (int): Maximum impulse response length in time-steps.
            dt (float): Temporal resolution of the runoff inputs.
            sampling_mode (str): Strategy for sampling cascade parameters.
            block_size (int): Spatial block size used for block-sparse kernels.
            block_f (int): Hidden dimensionality for kernel factorization.
            cascade (int): Number of cascaded IRFs combined by the aggregator.
            **kwargs: Unused keyword arguments kept for legacy compatibility.
        """
        super().__init__()
        self.block_size = block_size
        self.aggregator = IRFAggregator(max_delay=max_delay, 
                                        dt=dt, cascade=cascade, 
                                        sampling_mode=sampling_mode,
                                        block_f=block_f)
        self.conv = BlockSparseCausalConv()

    def forward(self, runoff: torch.Tensor, g, params=None) -> torch.Tensor:
        """Compute routed discharge for a set of runoff inputs.

        Args:
            runoff (torch.Tensor): Tensor shaped `[B, C, T]` with batch,
                channel (node), and time dimensions.
            g (RivTree): River network containing kernel parameters.
            params (torch.Tensor | None): Optional per-cluster parameters;
                defaults to attributes stored on `g`.

        Returns:
            torch.Tensor: Routed discharge with shape `[B, C, T]`.

        Raises:
            ValueError: If `runoff` does not have three dimensions.
        """
        if runoff.ndim != 3: raise ValueError(f"runoff must be [B, C, T], got {runoff.shape}")
        # Stage 1: Aggregate kernel
        kernel = self.aggregator(g, params).to(runoff.device)
        kernel = kernel.to_block_sparse(self.block_size)
        # Stage 2: Convolution
        y = self.conv(runoff, kernel)
        # Handle residual if needed
        if not g.include_index_diag: y = runoff + y 
        return y
