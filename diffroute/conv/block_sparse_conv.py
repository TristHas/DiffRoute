import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..ops import block_sparse_conv_1d
from ..structs import BlockSparseKernel
from ..backend import select_conv_config

def conv1d_block_sparse(x, block_values, cols, rows):
    """
    Args:
        x: dense input of shape (batch_size, n_channels, t)
        block_values: block sparse kernel values of shape (n_blocks, block_size, block_size, ks)
        cols: column indices for block selection
        rows: row indices for block aggregation
    Returns:
        y: convolution output
    """
    #print("Using legacy conv")
    batch_size, n_channels, t = x.shape
    n_blocks, block_size, _, ks = block_values.shape
    n_block_row = math.ceil(n_channels / block_size)
    padding_channels = n_block_row * block_size - n_channels

    if padding_channels > 0:
        x = F.pad(x, (0, 0, 0, padding_channels))  # Pad the channel dimension

    # Reshape and select input blocks
    x = x.view(batch_size, n_block_row, block_size, t)  # Shape: (batch_size, n_block_row, block_size, t)
    X_unfold = x[:, cols]  # Select relevant blocks; Shape: (batch_size, n_blocks, block_size, t)

    # Prepare for group convolution
    X_unfold = X_unfold.reshape(1, batch_size * n_blocks * block_size, t)
    W = block_values.view(n_blocks * block_size, block_size, ks)
    W = W.repeat(batch_size, 1, 1)  # Repeat weights for batch size
    W = W.view(batch_size * n_blocks * block_size, block_size, ks)

    # Perform group convolution
    o = F.conv1d(X_unfold, W, groups=batch_size * n_blocks, padding=ks-1)[...,:t]
    o = o.view(batch_size, n_blocks, block_size, t)

    # Aggregate outputs
    y = torch.zeros(batch_size, n_block_row, block_size, t, device=o.device, dtype=o.dtype)
    idx = rows.unsqueeze(0).unsqueeze(-1).unsqueeze(-1).expand(batch_size, -1, block_size, t)
    y.scatter_add_(1, idx, o)

    # Reshape output and remove padding
    y = y.view(batch_size, n_block_row * block_size, t)
    y = y[:, :n_channels, :]
    return y

class BlockSparseCausalConv(nn.Module):
    def __init__(self, bs_kernel=None, 
                 conv_imp="auto",
                 block_m=None, 
                 block_n=None,
                 block_n_dx=None,
                 block_n_dw=None):
        super().__init__()
        self.bs_kernel = bs_kernel
        self.conv_imp = conv_imp
        self.block_n = block_n
        self.block_n_dx = block_n_dx
        self.block_n_dw = block_n_dw
        self.block_m = block_m
    
    def forward(self, x, w=None):
        """
        Args:
            x: Input tensor of shape (batch_size, n_channels, t)
        Returns:
            y: Output tensor of shape (batch_size, n_channels, t)
        
        """
        if w is None: w = self.bs_kernel
        assert isinstance(w, BlockSparseKernel), "Kernel must be provided either at init or at forward"
        config = select_conv_config(device=x.device,
                                    impl=self.conv_imp,
                                    block_m=self.block_m,
                                    block_n=self.block_n,
                                    block_n_dx=self.block_n_dx,
                                    block_n_dw=self.block_n_dw)
        BLOCK_SIZE_M = w.block_size if config.block_m is None else config.block_m
        if config.impl=="triton":
            return block_sparse_conv_1d(x, 
                                        w.block_indices, 
                                        w.block_values,
                                        w.size,
                                        BLOCK_SIZE_M, 
                                        config.block_n,
                                        BLOCK_SIZE_N_DX=config.block_n_dx,
                                        BLOCK_SIZE_N_DVALUES=config.block_n_dw,
                                        DX_BLOCK_ORDER=w.block_col_order,
                                        DX_BLOCK_COL_OFFSETS=w.block_col_offsets)
        else:        
            return conv1d_block_sparse(
                x, 
                w.block_values, 
                w.block_indices[:,1],  # cols
                w.block_indices[:,0]   # rows
            )

    def to(self, device, **kwargs):
        super().to(device, **kwargs)
        if isinstance(self.bs_kernel, BlockSparseKernel):
            self.bs_kernel = self.bs_kernel.to(device)
            return self
