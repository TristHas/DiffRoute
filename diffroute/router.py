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
                 conv_imp="auto",
                 block_n=None,
                 block_n_dx=None,
                 block_n_dw=None,
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
        self.conv = BlockSparseCausalConv(conv_imp=conv_imp,
                                          block_n=block_n,
                                          block_n_dx=block_n_dx,
                                          block_n_dw=block_n_dw)
        self._kernel_cache = {"key": None, "value": None}

    def clear_kernel_cache(self) -> None:
        """Drop any cached non-gradient block-sparse routing kernel."""
        self._kernel_cache["key"] = None
        self._kernel_cache["value"] = None

    def _kernel_cache_key(self, g, params: torch.Tensor, device: torch.device):
        return (
            id(g),
            getattr(g, "irf_fn", None),
            bool(getattr(g, "include_index_diag", True)),
            getattr(g, "prefix_jump_rounds", None),
            int(g.edges.data_ptr()),
            int(getattr(g.edges, "_version", 0)),
            tuple(g.edges.shape),
            int(g.path_cumsum.data_ptr()),
            int(getattr(g.path_cumsum, "_version", 0)),
            tuple(g.path_cumsum.shape),
            int(params.data_ptr()),
            int(getattr(params, "_version", 0)),
            tuple(params.shape),
            tuple(params.stride()),
            int(params.storage_offset()),
            str(params.dtype),
            str(params.device),
            str(device),
            self.block_size,
            self.aggregator.max_delay,
            self.aggregator.dt,
            self.aggregator.cascade,
            self.aggregator.block_f,
            self.aggregator.sampler.factor,
            self.aggregator.sampler.out_mode,
        )

    def _block_sparse_kernel(self, g, params: torch.Tensor, device: torch.device):
        if params.requires_grad:
            kernel = self.aggregator(g, params).to(device)
            return kernel.to_block_sparse(self.block_size)

        key = self._kernel_cache_key(g, params, device)
        if self._kernel_cache["key"] == key:
            return self._kernel_cache["value"]

        with torch.no_grad():
            kernel = self.aggregator(g, params).to(device)
            block_kernel = kernel.to_block_sparse(self.block_size)
        self._kernel_cache["key"] = key
        self._kernel_cache["value"] = block_kernel
        return block_kernel

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
            torch.Tensor: Routed discharge with the same shape as `runoff`.

        Raises:
            ValueError: If `runoff` has fewer than three dimensions.
        """
        if runoff.ndim < 3:
            raise ValueError(f"runoff must be [..., C, T] with at least one "
                             f"leading batch dimension, got {runoff.shape}")
        if params is None:
            params = g.params
        *lead, C, T = runoff.shape
        x = runoff.contiguous().view(-1, C, T)            # merge leading dims -> [B, C, T]
        # Stage 1: Aggregate and blockize the routing kernel.
        kernel = self._block_sparse_kernel(g, params, x.device)
        # Stage 2: Convolution
        y = self.conv(x, kernel)
        # Handle residual if needed
        if not g.include_index_diag: y = x + y
        return y.reshape(*lead, C, T)                     # restore leading dims
