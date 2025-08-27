import torch
import torch.nn as nn

class BlockSparseTensor(nn.Module): # TODO: Should not be a Module, remove that later
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
        #block_values = block_values.index_add_(0, linear_indices, values)
        #block_values = block_values.index_add(0, linear_indices, values)
        block_values = block_values.index_put((linear_indices,), values)
        block_values = block_values.reshape(n_blocks, B, B, ks)
        
        # Compute the overall size of the tensor
        if size is None:
            max_coords = coords.max(dim=0)[0] + 1  # Add 1 because indices start from 0
            size = (max_coords[0].item(), max_coords[1].item(), ks)

        return cls(unique_blocks, block_values, block_size, size)

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


import torch

def _roundtrip_equal(bst: BlockSparseTensor):
    """Check that to_coo -> from_coo -> to_dense reproduces the original dense tensor."""
    dense_original = bst.to_dense()
    coords, values = bst.to_coo(drop_zero_rows=True)
    bst2 = BlockSparseTensor.from_coo(coords, values, block_size=bst.block_size, size=bst.size)
    dense_reconstructed = bst2.to_dense()
    torch.testing.assert_close(dense_reconstructed, dense_original)

def test_random_roundtrip(device="cpu", seed=0):
    torch.manual_seed(seed)

    # Randomized parameters including edge-case shapes
    H, W, ks = 53, 47, 5          # not multiples of B to test edge clipping
    B = 8
    dtype = torch.float32
    n_block_rows = (H + B - 1) // B
    n_block_cols = (W + B - 1) // B
    max_blocks = n_block_rows * n_block_cols

    # Choose a random subset of block positions without replacement
    n_blocks = min(max_blocks, 30)  # cap to avoid exploding memory in test
    all_block_coords = torch.stack(torch.meshgrid(
        torch.arange(n_block_rows), torch.arange(n_block_cols), indexing='ij'
    ), dim=-1).reshape(-1, 2)  # [max_blocks, 2]

    perm = torch.randperm(all_block_coords.size(0))[:n_blocks]
    block_indices = all_block_coords[perm].to(torch.long).to(device)   # [n_blocks, 2]

    # Random block values (some zeros included)
    block_values = torch.randn((n_blocks, B, B, ks), dtype=dtype, device=device)

    # Make some rows all-zero on purpose
    mask_zero = torch.rand_like(block_values[..., 0]) < 0.15  # 15% chance a (n,b,b) position is zero across ks
    block_values[mask_zero] = 0

    bst = BlockSparseTensor(block_indices, block_values, B, size=(H, W, ks)).to(device)
    _roundtrip_equal(bst)

def test_small_deterministic(device="cpu"):
    H, W, ks = 5, 6, 3
    B = 4

    # Two blocks: (0,0) and (1,1) -- the second is an edge block (truncated)
    block_indices = torch.tensor([[0, 0], [1, 1]], dtype=torch.long, device=device)

    block_values = torch.zeros((2, B, B, ks), dtype=torch.float32, device=device)

    # Fill some identifiable values
    # Block (0,0)
    block_values[0, 0, 0, :] = torch.tensor([1., 2., 3.], device=device)
    block_values[0, 1, 2, :] = torch.tensor([4., 5., 6.], device=device)

    # Block (1,1) starts at (4,4); with H=5,W=6,B=4 it clips to (rows 4..4, cols 4..5)
    block_values[1, 0, 0, :] = torch.tensor([7., 0., 0.], device=device)   # (4,4)
    block_values[1, 0, 1, :] = torch.tensor([0., 8., 0.], device=device)   # (4,5)
    block_values[1, 1:, :, :] = 0  # rest won't fit anyway, but keep explicit

    bst = BlockSparseTensor(block_indices, block_values, B, size=(H, W, ks)).to(device)
    _roundtrip_equal(bst)
