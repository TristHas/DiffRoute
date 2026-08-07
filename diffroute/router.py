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

        Any number of leading (batch-like) dimensions are supported: every
        dimension before the trailing channel (node) and time dimensions is
        merged into a single batch dimension for routing, then restored on the
        output. So `[B, C, T]`, `[B, E, C, T]` (with an ensemble axis), etc. are
        all accepted.

        Args:
            runoff (torch.Tensor): Tensor shaped `[..., C, T]` with any number
                of leading batch dimensions, then channel (node) and time.
            g (RivTree): River network containing kernel parameters.
            params (torch.Tensor | None): Optional per-cluster parameters;
                defaults to attributes stored on `g`.

        Returns:
            torch.Tensor: Routed discharge, `[..., C, T]` by default and
            `[..., len(g.output_reach), T]` once `g.set_output_reach` has
            narrowed it, with rows in the order that call asked for.

        Raises:
            ValueError: If `runoff` has fewer than three dimensions.
        """
        if runoff.ndim < 3:
            raise ValueError(f"runoff must be [..., C, T] with at least one "
                             f"leading batch dimension, got {runoff.shape}")
        *lead, C, T = runoff.shape
        x = runoff.contiguous().view(-1, C, T)            # merge leading dims -> [B, C, T]
        if g.n_paths:
            # Stage 1: Aggregate kernel -- (n_out x n_in), so restricting the
            # output shrinks the closure and the block count together
            kernel = self.aggregator(g, params).to(x.device)
            kernel = kernel.to_block_sparse(self.block_size)
            # Stage 2: Convolution
            y = self.conv(x, kernel)
        else:
            # No path carries anything: a single reach with route_src_reach=False
            # emits its input unchanged, and there is no kernel to build.
            y = x.new_zeros(x.shape[0], g.n_out, T)
        # With route_src_reach=False the runoff is already at the reach outlet, so the
        # kernel holds only strictly-downstream paths and the diagonal is the
        # identity -- added here rather than carried in the sparse kernel. The
        # weight is per node: it is 0 where the reach IS traversed (head entry,
        # diagonal already in the kernel) and 0 at transition nodes, whose value
        # was already reported by the cluster they came from.
        if g.has_residual:
            # the identity diagonal, taken at the rows actually being returned
            y = y + (x * g.residual_weight if g.full_output
                     else x.index_select(1, g.out_positions) * g.residual_weight_out)
        return y.reshape(*lead, g.n_out, T)               # restore leading dims
