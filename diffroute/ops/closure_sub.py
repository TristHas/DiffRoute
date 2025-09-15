import math, torch, triton, triton.language as tl
from torch.autograd import Function

@triton.jit
def _coo_enum_sum_kernel(coords_ptr, vals_ptr, 
                         prefix_ptr, edges_ptr, cumsum_ptr, 
                         n_nodes, 
                         n_feat: tl.constexpr,
                         INCLUDE_SELF: tl.constexpr,
                         BLOCK_F: tl.constexpr):
    pid  = tl.program_id(0)  # start node
    if pid >= n_nodes: return

    # global write offset for this start node
    base = tl.load(cumsum_ptr + (pid - 1), mask=(pid > 0), other=0)
    dest = tl.where(INCLUDE_SELF, pid, tl.load(edges_ptr + pid))
    step = 0
    offs = tl.arange(0, BLOCK_F)

    while dest != -1:
        row = base + step

        tl.store(coords_ptr + row*2 + 0, dest)
        tl.store(coords_ptr + row*2 + 1, pid)

        # compute path-sum via prefix difference
        child = tl.load(edges_ptr + dest, mask=(dest >= 0) & (dest < n_nodes), other=-1)
        for b in range(0, n_feat, BLOCK_F):
            m   = offs + b < n_feat
            p_i = tl.load(prefix_ptr + pid  * n_feat + b + offs, mask=m, other=0.)
            val = p_i
            if child >= 0:
                p_c = tl.load(prefix_ptr + child * n_feat + b + offs, mask=m, other=0.)
                val = val - p_c
            tl.store(vals_ptr + row*n_feat + b + offs, val, mask=m)

        step += 1
        dest  = tl.load(edges_ptr + dest)


@triton.jit
def _vals_to_prefix_grad_kernel(coords_ptr, gvals_ptr,
                                edges_ptr, gpre_ptr,
                                n_feat: tl.constexpr,
                                BLOCK_F: tl.constexpr):
    row  = tl.program_id(0)
    offs = tl.arange(0, BLOCK_F)

    start = tl.load(coords_ptr + row*2 + 1).to(tl.int32)
    dest  = tl.load(coords_ptr + row*2 + 0).to(tl.int32)
    child = tl.load(edges_ptr + dest)

    for base in range(0, n_feat, BLOCK_F):
        m = offs + base < n_feat
        g = tl.load(gvals_ptr + row * n_feat + base + offs, mask=m, other=0.)
        tl.atomic_add(gpre_ptr + start * n_feat + base + offs, g, mask=m)
        if child >= 0:
            tl.atomic_add(gpre_ptr + child * n_feat + base + offs, -g, mask=m)


# ------------------------------------------------------------------
# Low-level helpers
# ------------------------------------------------------------------
def _closure_enum_fwd(prefix: torch.Tensor,
                      edges: torch.Tensor,
                      path_cumsum: torch.Tensor,
                      include_self: bool = True,
                      block_f: int = 128):
    """
    Triton forward: enumerate (dest,start) pairs & prefix differences.
    """
    prefix      = prefix.contiguous()
    edges       = edges.contiguous()
    path_cumsum = path_cumsum.to(torch.int32).contiguous()  # safe cast

    n, f   = prefix.shape
    N_path = int(path_cumsum[-1].item())
    coords = torch.empty((N_path, 2), dtype=path_cumsum.dtype, device=prefix.device)
    vals   = torch.empty((N_path, f), dtype=prefix.dtype, device=prefix.device)

    with torch.cuda.device(prefix.device):
        _coo_enum_sum_kernel[(n,)](coords, vals,
                                   prefix, edges, path_cumsum,
                                   n, f,
                                   INCLUDE_SELF=include_self,
                                   BLOCK_F=block_f)
    return coords, vals


def _closure_enum_bwd(g_vals: torch.Tensor,
                      coords: torch.Tensor,
                      edges: torch.Tensor,
                      n_feat: int,
                      block_f: int = 128):
    """
    Triton backward: accumulate ∂L/∂vals -> ∂L/∂prefix.
    """
    g_vals = g_vals.contiguous()
    coords = coords.contiguous()
    edges  = edges.contiguous()
    n      = edges.numel()

    g_pre = torch.zeros((n, n_feat), dtype=g_vals.dtype, device=g_vals.device)

    with torch.cuda.device(g_vals.device):
        _vals_to_prefix_grad_kernel[(coords.shape[0],)](
            coords, g_vals, edges, g_pre,
            n_feat=n_feat, BLOCK_F=block_f)
    return g_pre


# ------------------------------------------------------------------
# Autograd wrapper
# ------------------------------------------------------------------
class ClosureSub(Function):
    @staticmethod
    def forward(ctx, prefix, edges, path_cumsum,
                include_self: bool = True, block_f: int = 128):
        coords, vals = _closure_enum_fwd(prefix, edges, path_cumsum,
                                         include_self, block_f)
        ctx.save_for_backward(edges, coords)
        ctx.block_f  = block_f
        ctx.n_feat   = prefix.shape[1]
        return coords, vals

    @staticmethod
    def backward(ctx, g_coords, g_vals):
        if g_vals is None:                       # no grad flows
            return None, None, None, None, None
        edges, coords = ctx.saved_tensors
        g_prefix = _closure_enum_bwd(g_vals, coords, edges,
                                     ctx.n_feat, ctx.block_f)
        return g_prefix, None, None, None, None


def closure_sub(prefix, edges, path_cumsum,
                include_self: bool = True, block_f: int = 128):
    return ClosureSub.apply(prefix, edges, path_cumsum, include_self, block_f)
