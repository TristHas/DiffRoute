"""Enumerate the (dest, src) path sums of the transitive closure.

Convention
----------
For a river DAG with ``edges[i] = succ(i)`` (-1 at outlets), the routing kernel
entry ``K(d, s)`` is the convolution of the reach IRFs along the path ``s -> d``.
Which endpoint reaches are traversed is set per node by ``own_reach``, the low-level
form of ``RivTree(route_src_reach=...)``. A clustered graph mixes the two within one
subgraph, which is why this is a mask and not a flag:

``own_reach[s] = True``   (``route_src_reach=True``, the default)
    runoff enters at the HEAD of its own reach, so it traverses that reach::

        K(d, s) = conv of irf over  s..d   (both ends included)
        K(s, s) = irf(s)                   (the diagonal is emitted)

``own_reach[s] = False``  (``route_src_reach=False``)
    runoff is already at the OUTLET of its own reach (this is what a
    catchment-outlet runoff model produces), so its own reach is NOT traversed::

        K(d, s) = conv of irf over  succ(s)..d   (SOURCE EXCLUDED)
        K(s, s) = delta                          (no diagonal is emitted;
                                                  LTIRouter adds ``y = x + Kx``)

In log space a path sum is a difference of downstream prefixes. With

    P[i] = sum of f over  i..root          (inclusive, from ``prefix_sum``)
    Q[i] = sum of f over  succ(i)..root    (exclusive, = P[succ(i)]; 0 at outlets)

both cases are the *same* expression, differing only in which prefix the source
end reads from::

    val(d, s) = head[s] - Q[d],      head[s] = P[s] if own_reach[s] else Q[s]

That is why this module takes two prefix tensors and needs no edge lookup at the
destination: ``Q[d]`` already is "everything strictly downstream of d".

History: before the v1 line the source end always read ``P[s]``, so outlet entry
dropped the diagonal but still convolved every path with the source reach's own IRF
-- identical off-diagonal kernels to head entry. See ``tests/test_semantics.py``.
"""
import math, torch, triton, triton.language as tl
from torch.autograd import Function

@triton.jit
def _coo_enum_sum_kernel(coords_ptr, vals_ptr,
                         head_ptr, tail_ptr,
                         edges_ptr, cumsum_ptr, own_ptr, out_row_ptr,
                         n_nodes,
                         n_feat: tl.constexpr,
                         BLOCK_F: tl.constexpr):
    pid  = tl.program_id(0)  # start node
    if pid >= n_nodes: return
    # global write offset for this start node
    base = tl.load(cumsum_ptr + (pid - 1), mask=(pid > 0), other=0)
    # first destination: the node itself when its own reach is traversed,
    # otherwise the next reach downstream. Per-node, because a clustered graph
    # mixes conventions (see structs/utils.resolve_self_flags).
    own  = tl.load(own_ptr + pid)
    dest = tl.where(own != 0, pid, tl.load(edges_ptr + pid))
    step = 0
    offs = tl.arange(0, BLOCK_F)

    while dest != -1:
        # `out_row` is the destination's row in the OUTPUT, or -1 when that node
        # was not requested. It is both the predicate and the remap, so the
        # emitted kernel is (n_out x n_in) rather than (n x n) with holes.
        out_row = tl.load(out_row_ptr + dest)
        if out_row >= 0:
            row = base + step

            tl.store(coords_ptr + row*2 + 0, out_row)
            tl.store(coords_ptr + row*2 + 1, pid)

        # path-sum via prefix difference: head[pid] already excludes the source
        # reach under outlet entry, tail[dest] always excludes dest.
            for b in range(0, n_feat, BLOCK_F):
                m   = offs + b < n_feat
                p_s = tl.load(head_ptr + pid  * n_feat + b + offs, mask=m, other=0.)
                p_d = tl.load(tail_ptr + dest * n_feat + b + offs, mask=m, other=0.)
                tl.store(vals_ptr + row*n_feat + b + offs, p_s - p_d, mask=m)

            step += 1
        dest  = tl.load(edges_ptr + dest)


@triton.jit
def _vals_to_prefix_grad_kernel(coords_ptr, gvals_ptr,
                                ghead_ptr, gtail_ptr,
                                n_feat: tl.constexpr,
                                BLOCK_F: tl.constexpr):
    row  = tl.program_id(0)
    offs = tl.arange(0, BLOCK_F)

    dest  = tl.load(coords_ptr + row*2 + 0).to(tl.int32)
    start = tl.load(coords_ptr + row*2 + 1).to(tl.int32)

    for base in range(0, n_feat, BLOCK_F):
        m = offs + base < n_feat
        g = tl.load(gvals_ptr + row * n_feat + base + offs, mask=m, other=0.)
        tl.atomic_add(ghead_ptr + start * n_feat + base + offs,  g, mask=m)
        tl.atomic_add(gtail_ptr + dest  * n_feat + base + offs, -g, mask=m)


# ------------------------------------------------------------------
# Low-level helpers
# ------------------------------------------------------------------
def _closure_enum_fwd(head: torch.Tensor,
                      tail: torch.Tensor,
                      edges: torch.Tensor,
                      path_cumsum: torch.Tensor,
                      own_reach: torch.Tensor,
                      out_row: torch.Tensor,
                      block_f: int = 128):
    """
    Triton forward: enumerate (dest,start) pairs & prefix differences.
    """
    head        = head.contiguous()
    tail        = tail.contiguous()
    edges       = edges.contiguous()
    path_cumsum = path_cumsum.contiguous()  # safe cast .to(torch.int32)
    own_reach   = own_reach.to(torch.int8).contiguous()
    out_row     = out_row.to(torch.int32).contiguous()

    n, f   = head.shape
    N_path = int(path_cumsum[-1].item())
    coords = torch.empty((N_path, 2), dtype=path_cumsum.dtype, device=head.device)
    vals   = torch.empty((N_path, f), dtype=head.dtype, device=head.device)

    with torch.cuda.device(head.device):
        _coo_enum_sum_kernel[(n,)](coords, vals,
                                   head, tail, edges, path_cumsum,
                                   own_reach, out_row,
                                   n, f,
                                   BLOCK_F=block_f)
    return coords, vals


def _closure_enum_bwd(g_vals: torch.Tensor,
                      coords: torch.Tensor,
                      n_nodes: int,
                      n_feat: int,
                      block_f: int = 128):
    """
    Triton backward: accumulate dL/dvals -> (dL/dhead, dL/dtail).

    ``head`` and ``tail`` may be the same tensor (they are when every node uses
    outlet entry); autograd then sums the two returned buffers, which is exactly
    the gradient that tensor should receive.
    """
    g_vals = g_vals.contiguous()
    coords = coords.contiguous()

    g_head = torch.zeros((n_nodes, n_feat), dtype=g_vals.dtype, device=g_vals.device)
    g_tail = torch.zeros_like(g_head)

    with torch.cuda.device(g_vals.device):
        _vals_to_prefix_grad_kernel[(coords.shape[0],)](
            coords, g_vals, g_head, g_tail,
            n_feat=n_feat, BLOCK_F=block_f)
    return g_head, g_tail


# ------------------------------------------------------------------
# Autograd wrapper
# ------------------------------------------------------------------
class ClosureSub(Function):
    @staticmethod
    def forward(ctx, head, tail, edges, path_cumsum, own_reach, out_row,
                block_f: int = 128):
        coords, vals = _closure_enum_fwd(head, tail, edges, path_cumsum,
                                         own_reach, out_row, block_f)
        ctx.save_for_backward(coords)
        ctx.block_f  = block_f
        ctx.n_nodes  = head.shape[0]
        ctx.n_feat   = head.shape[1]
        return coords, vals

    @staticmethod
    def backward(ctx, g_coords, g_vals):
        if g_vals is None:                       # no grad flows
            return None, None, None, None, None, None, None
        coords, = ctx.saved_tensors
        g_head, g_tail = _closure_enum_bwd(g_vals, coords, ctx.n_nodes,
                                           ctx.n_feat, ctx.block_f)
        return g_head, g_tail, None, None, None, None, None


def closure_sub(head, tail, edges, path_cumsum, own_reach, out_row,
                block_f: int = 128):
    """Path sums ``val(d, s) = head[s] - tail[d]`` over the downstream closure.

    Args:
        head: prefix the SOURCE end reads from -- the inclusive prefix ``P`` when
            ``own_reach[s]`` is true, the exclusive prefix ``Q`` otherwise.
        tail: prefix the DESTINATION end reads from -- always the exclusive
            prefix ``Q`` (see the module docstring).
        edges: ``edges[i] = succ(i)``, -1 at outlets.
        path_cumsum: per-node write offsets from ``downstream_path_stats``.
        own_reach: per-node bool; whether the diagonal ``(s, s)`` is emitted.
        out_row: per-node int32; the destination's row in the output, or -1 to
            drop every path ending there.
    """
    return ClosureSub.apply(head, tail, edges, path_cumsum, own_reach, out_row,
                            block_f)
