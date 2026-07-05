import torch
import torch.nn as nn

class SparseKernel(nn.Module):
    """COO-formatted sparse kernel used for routing convolutions."""
    def __init__(self, coords, vals, size):
        """Initialize kernel coordinates, values, and target size.

        Args:
            coords (torch.Tensor): Integer indices of non-zero entries `[N, 2]`.
            vals (torch.Tensor): Kernel values aligned with `coords` `[N, ks]`.
            size (Tuple[int, int, int]): Height, width, and kernel length.
        """
        super().__init__()
        self.register_buffer("coords", coords)  # [n_blocks, 2]
        self.register_buffer("vals", vals)    # [n_blocks, block_size, block_size, ks]
        self.size = size

    def to_block_sparse(self, block_size):
        """Convert the kernel to a block-sparse representation."""
        return BlockSparseKernel.from_sparse_kernel(self, block_size=block_size)

    def to_dense(self):
        """Materialize the sparse kernel as a dense tensor."""
        H, W, ks = self.size
        dense = torch.zeros(self.size, dtype=self.vals.dtype, device=self.vals.device)
        flat_idx = self.coords[:, 0] * W + self.coords[:, 1]
        dense.view(-1, ks).index_add_(0, flat_idx, self.vals)
        return dense
        
class BlockSparseKernel(nn.Module): 
    """Block-sparse tensor storing convolution kernels."""
    def __init__(
        self,
        block_indices,
        block_values,
        block_size,
        size,
        block_col_order=None,
        block_col_offsets=None,
    ):
        """Store block-sparse indices and values for convolution.

        Args:
            block_indices (torch.Tensor): Block positions shaped `[B, 2]`.
            block_values (torch.Tensor): Block data shaped
                `[B, block_size, block_size, ks]`.
            block_size (int): Spatial size of an individual block.
            size (Tuple[int, int, int]): Overall tensor dimensions `(H, W, ks)`.
        """
        super().__init__()
        self.register_buffer("block_indices", block_indices)  # [n_blocks, 2]
        self.register_buffer("block_values", block_values)    # [n_blocks, block_size, block_size, ks]
        if block_col_order is None:
            block_col_order = self._make_col_order(block_indices, block_size, size)
        self.register_buffer("block_col_order", block_col_order)
        if block_col_offsets is None:
            block_col_offsets = self._make_col_offsets(block_indices, block_size, size)
        self.register_buffer("block_col_offsets", block_col_offsets)
        self.block_size = block_size        # Block size
        self.size = size                    # Overall size of the tensor [H, W, ks]

    @staticmethod
    def _make_col_order(block_indices, block_size, size):
        """Return block ids sorted by input block, then output block."""
        if block_indices.numel() == 0:
            return torch.empty((0,), dtype=torch.int32, device=block_indices.device)
        n_row_blocks = (int(size[0]) + block_size - 1) // block_size
        sort_key = block_indices[:, 1].long() * n_row_blocks + block_indices[:, 0].long()
        return torch.argsort(sort_key).to(torch.int32)

    @staticmethod
    def _make_col_offsets(block_indices, block_size, size):
        """Return CSR offsets into `block_col_order` for each input block."""
        n_col_blocks = (int(size[1]) + block_size - 1) // block_size
        offsets = torch.empty((n_col_blocks + 1,), dtype=torch.int32, device=block_indices.device)
        offsets[0] = 0
        if n_col_blocks == 0:
            return offsets
        cols = block_indices[:, 1].long()
        counts = torch.bincount(cols, minlength=n_col_blocks).to(torch.int32)
        offsets[1:] = torch.cumsum(counts, dim=0)
        return offsets

    def to(self, device):
        """Move block indices and values to a target device."""
        self.block_indices = self.block_indices.to(device)
        self.block_values = self.block_values.to(device)
        self.block_col_order = self.block_col_order.to(device)
        self.block_col_offsets = self.block_col_offsets.to(device)
        return self

    def to_dense(self):
            """Convert the block-sparse tensor to a dense representation.

            Returns:
                torch.Tensor: Dense tensor of shape `(H, W, ks)`.
            """
            H, W, ks = self.size
            B = self.block_size
            # Initialize a dense tensor with zeros
            dense_tensor = torch.zeros((H, W, ks), dtype=self.block_values.dtype, device=self.block_values.device)
            # Iterate over blocks and populate the dense tensor
            for idx, (row_block, col_block) in enumerate(self.block_indices):
                row_start = row_block * B
                col_start = col_block * B
                row_end = min(row_start + B, H)
                col_end = min(col_start + B, W)
                # Extract the block values and assign them to the dense tensor
                dense_tensor[row_start:row_end, col_start:col_end, :] += self.block_values[idx, :row_end - row_start, :col_end - col_start, :]
    
            return dense_tensor    

    def to_coo(self, drop_zero_rows=True):
        """Convert the block-sparse tensor to COO indices and values.

        Args:
            drop_zero_rows (bool): Remove locations whose channels are all zero.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Coordinate tensor `[N, 2]`
            and value tensor `[N, ks]`.
        """
        device = self.block_values.device
        dtype  = self.block_values.dtype
        H, W, ks = self.size
        B = self.block_size
    
        n_blocks = self.block_indices.shape[0]
        if n_blocks == 0:
            return (torch.empty((0, 2), dtype=torch.long, device=device),
                    torch.empty((0, ks), dtype=dtype, device=device))
    
        # Block start positions
        rb = self.block_indices[:, 0].long()            # [n_blocks]
        cb = self.block_indices[:, 1].long()            # [n_blocks]
        row_base = rb * B                                # [n_blocks]
        col_base = cb * B                                # [n_blocks]
    
        # Offsets within a block [0..B-1]
        r_off = torch.arange(B, device=device, dtype=torch.long)  # [B]
        c_off = torch.arange(B, device=device, dtype=torch.long)  # [B]
    
        # Global row/col for each block cell (broadcast to [n_blocks, B, B])
        rows = row_base[:, None, None] + r_off[None, :, None]     # [n_blocks, B, 1] + [1, B, 1]
        cols = col_base[:, None, None] + c_off[None, None, :]     # [n_blocks, 1, 1] + [1, 1, B]
        rows = rows.expand(n_blocks, B, B)
        cols = cols.expand(n_blocks, B, B)
    
        # Flatten to per-element COO
        coords = torch.stack([rows.reshape(-1), cols.reshape(-1)], dim=1)  # [(n_blocks*B*B), 2]
        values = self.block_values.reshape(-1, ks)                         # [(n_blocks*B*B), ks]
    
        # Mask out-of-bounds cells (for edge/truncated blocks)
        in_bounds = (coords[:, 0] < H) & (coords[:, 1] < W)
        if not torch.all(in_bounds):
            coords = coords[in_bounds]
            values = values[in_bounds]
    
        # Optionally drop all-zero rows across ks channels
        if drop_zero_rows:
            nz = values.ne(0).any(dim=-1)
            if not torch.all(nz):
                coords = coords[nz]
                values = values[nz]
    
        return coords.long(), values
        
    @classmethod
    def from_coo(cls, coords, values, block_size, size=None, flip_values=False):
        """Construct a block-sparse kernel from COO inputs.

        Args:
            coords (torch.Tensor): Coordinate tensor shaped `[N, 2]`.
            values (torch.Tensor): Value tensor shaped `[N, ks]`.
            block_size (int): Spatial size of each block.
            size (Tuple[int, int, int] | None): Optional full tensor shape.
            flip_values (bool): Reverse kernel direction along the time axis.

        Returns:
            BlockSparseKernel: Block-sparse representation built from inputs.
        """
        values = values.flip(-1) if flip_values else values
        if coords.is_cuda and values.is_cuda and size is not None and coords.numel() > 0:
            return cls._from_coo_dense_keys(coords, values, block_size, size)

        B = block_size
        ks = values.shape[-1]
        
        block_coords = coords // B  
        block_local_coords = coords % B
        
        unique_blocks, block_indices = torch.unique(block_coords, dim=0, return_inverse=True)
        n_blocks = unique_blocks.size(0)
        
        # Compute linear indices for flattening
        linear_indices = block_indices * (B * B) + block_local_coords[:,0] * B + block_local_coords[:,1]
        block_values = torch.zeros((n_blocks * B * B, ks), dtype=values.dtype, device=values.device)
        block_values = block_values.index_put((linear_indices,), values)
        block_values = block_values.reshape(n_blocks, B, B, ks)
        
        # Compute the overall size of the tensor
        if size is None:
            max_coords = coords.max(dim=0)[0] + 1  # Add 1 because indices start from 0
            size = (max_coords[0].item(), max_coords[1].item(), ks)

        return cls(unique_blocks, block_values, block_size, size)

    @classmethod
    def _from_coo_dense_keys(cls, coords, values, block_size, size):
        """CUDA COO-to-block conversion using a dense block-key map."""
        B = block_size
        H, W, ks = size
        n_row_blocks = (int(H) + B - 1) // B
        n_col_blocks = (int(W) + B - 1) // B

        block_coords = coords // B
        block_local_coords = coords % B
        block_keys = (
            block_coords[:, 0].long() * n_col_blocks + block_coords[:, 1].long()
        ).contiguous()

        present = torch.zeros(
            (n_row_blocks * n_col_blocks,),
            dtype=torch.bool,
            device=coords.device,
        )
        present.scatter_(0, block_keys, True)
        unique_keys = torch.nonzero(present, as_tuple=False).flatten()
        n_blocks = unique_keys.numel()

        key_to_block = torch.empty_like(present, dtype=torch.long)
        key_to_block[unique_keys] = torch.arange(
            n_blocks,
            dtype=torch.long,
            device=coords.device,
        )
        block_indices = torch.stack(
            (unique_keys // n_col_blocks, unique_keys % n_col_blocks),
            dim=1,
        ).to(coords.dtype)

        linear_indices = (
            key_to_block[block_keys] * (B * B)
            + block_local_coords[:, 0].long() * B
            + block_local_coords[:, 1].long()
        )
        block_values = torch.zeros(
            (n_blocks * B * B, ks),
            dtype=values.dtype,
            device=values.device,
        )
        block_values.index_copy_(0, linear_indices.contiguous(), values)
        block_values = block_values.reshape(n_blocks, B, B, ks)

        present_blocks = present.view(n_row_blocks, n_col_blocks)
        col_counts = present_blocks.sum(dim=0, dtype=torch.int32)
        block_col_offsets = torch.empty(
            (n_col_blocks + 1,),
            dtype=torch.int32,
            device=coords.device,
        )
        block_col_offsets[0] = 0
        block_col_offsets[1:] = torch.cumsum(col_counts, dim=0)

        col_major_positions = torch.nonzero(
            present_blocks.t().contiguous().view(-1),
            as_tuple=False,
        ).flatten()
        col_major_rows = col_major_positions % n_row_blocks
        col_major_cols = col_major_positions // n_row_blocks
        col_major_keys = col_major_rows * n_col_blocks + col_major_cols
        block_col_order = key_to_block[col_major_keys].to(torch.int32)

        return cls(
            block_indices,
            block_values,
            block_size,
            size,
            block_col_order,
            block_col_offsets,
        )

    @classmethod
    def from_sparse_kernel(cls, kernel, block_size):
        """Create a block-sparse kernel from a `SparseKernel` instance."""
        return cls.from_coo(kernel.coords, kernel.vals, 
                            block_size=block_size, 
                            size=kernel.size)
    
    @classmethod
    def from_irfs(cls, irfs, block_size):
        """Translate precomputed IRFs into a block-sparse kernel."""
        coords, values, nodes = [],[],set()
        for dest in irfs:
            for source in irfs[dest]:
                coords.append([dest, source])
                values.append(irfs[dest][source])
                nodes.add(source)
                
        coords = torch.tensor(coords, dtype=torch.int64)
        values = torch.tensor(values, dtype=torch.float32)  # Shape [N, ks]
        size = [len(nodes), len(nodes), values.shape[-1]]
        return cls.from_coo(coords, values, block_size, size=size)
