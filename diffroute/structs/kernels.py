import torch
import torch.nn as nn

class SparseKernel(nn.Module):
    def __init__(self, coords, vals):
        self.coords = coords
        self.vals = vals

class BlockSparseKernel(nn.Module): 
    def __init__(self, block_indices, block_values, block_size, size):
        """
            This class provides the block sparse tensor datastructure to interface stage (i) and (ii)
        """
        super().__init__()
        self.register_buffer("block_indices", block_indices)  # [n_blocks, 2]
        self.register_buffer("block_values", block_values)    # [n_blocks, block_size, block_size, ks]
        self.block_size = block_size        # Block size
        self.size = size                    # Overall size of the tensor [H, W, ks]

    def to(self, device):
        """
        """
        self.block_indices = self.block_indices.to(device)
        self.block_values = self.block_values.to(device)
        return self

    def to_dense(self):
            """
            Convert the block-sparse tensor to a dense representation.
            
            Returns:
                A dense tensor of shape (H, W, ks), where H, W, and ks are defined by the size of the tensor.
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
        """
            Convert the block-sparse tensor to a COO representation.
        
            Args: drop_zero_rows: If True, remove entries whose entire ks-channel is zero.
        
            Returns:
                coords: LongTensor [N, 2] of (row, col)
                values: Tensor     [N, ks] of values
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
        """
        Create a BlockSparseTensor from COO format.

        Args:
            coords: Coordinate tensor of shape [N, 2]
            values: Value tensor of shape [N, ks]
            block_size: Size of the blocks
            size: Overall size of the tensor [H, W, ks]
        """
        values = values.flip(-1) if flip_values else values
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
    def from_sparse_kernel(cls, kernel, block_size, size=None):
        return cls.from_coo(kernel.coords, kernel.vals, 
                            block_size=block_size, 
                            size=size)
    
    @classmethod
    def from_irfs(cls, irfs, block_size):
        """
        """
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