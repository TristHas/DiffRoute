import torch
from .kernels import BlockSparseKernel

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