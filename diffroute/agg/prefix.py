import math, torch, triton, triton.language as tl
from torch.autograd import Function

def prefix_sum_bwd_ref(g_prefix, edges):
    """
    Correct gradient for inclusive downstream prefix.
    g_prefix: [n,f] upstream grads wrt prefix outputs.
    edges: int32[n] downstream pointer (-1 for outlet).
    Returns g_irf: [n,f].
    """
    n, f = g_prefix.shape
    g_irf = torch.zeros_like(g_prefix)
    e = edges.tolist()
    for start in range(n):
        g = g_prefix[start]
        j = start
        while j != -1:
            g_irf[j] = g_irf[j] + g
            j = e[j]
    return g_irf
    
@triton.jit
def _prefix_jump_kernel(prev_ptr, next_ptr,
                        edges_ptr, jump_ptr,
                        n_nodes, n_feat: tl.constexpr,
                        BLOCK_F: tl.constexpr):
    pid = tl.program_id(0)
    if pid >= n_nodes: return
    dst   = tl.load(jump_ptr + pid)
    valid = dst >= 0
    offs  = tl.arange(0, BLOCK_F)
    for base in range(0, n_feat, BLOCK_F):
        m   = offs + base < n_feat
        acc = tl.load(prev_ptr + pid * n_feat + base + offs, mask=m, other=0.)
        if valid:
            add = tl.load(prev_ptr + dst * n_feat + base + offs, mask=m, other=0.)
            acc += add
        tl.store(next_ptr + pid * n_feat + base + offs, acc, mask=m)
    # write 2-hop jump table
    tl.store(edges_ptr + pid, tl.where(valid, tl.load(jump_ptr + dst), -1))


@triton.jit
def _gradprefix_jump_kernel(prev_ptr, next_ptr,
                            edges_ptr, jump_ptr,
                            n_nodes, n_feat: tl.constexpr,
                            BLOCK_F: tl.constexpr):
    pid = tl.program_id(0)
    if pid >= n_nodes: return
    dst   = tl.load(edges_ptr + pid)
    valid = dst >= 0
    offs  = tl.arange(0, BLOCK_F)
    for base in range(0, n_feat, BLOCK_F):
        m = offs + base < n_feat
        g = tl.load(prev_ptr + pid * n_feat + base + offs, mask=m, other=0.)
        # self
        tl.atomic_add(next_ptr + pid * n_feat + base + offs, g, mask=m)
        # downstream
        if valid:
            tl.atomic_add(next_ptr + dst * n_feat + base + offs, g, mask=m)
    # build 2-hop table for next round
    tl.store(jump_ptr + pid, tl.where(valid, tl.load(edges_ptr + dst), -1))


# ------------------------------------------------------------------
# Low-level helper: forward
# ------------------------------------------------------------------
def _prefix_jump_fwd(irf: torch.Tensor,
                     edges: torch.Tensor,
                     block_f: int = 128) -> torch.Tensor:
    """
    Fast inclusive prefix (node -> outlet) via pointer-jumping.
    """
    irf   = irf.contiguous()
    edges = edges.contiguous()
    n, f  = irf.shape

    buf0  = irf.clone()
    buf1  = torch.empty_like(buf0)
    #print(irf.device, buf0.device)
    # running jump table that gets contracted each round
    e_run  = edges.clone()
    e_snap = torch.empty_like(e_run)   # scratch snapshot

    rounds = math.ceil(math.log2(max(1, n)))
    grid   = (n,)
    with torch.cuda.device(irf.device):
        for _ in range(rounds):
            #print(buf0.device)
            # snapshot current jump table (so kernel reads stable values)
            e_snap.copy_(e_run)
            _prefix_jump_kernel[grid](buf0, buf1, e_run, e_snap,
                                      n, f, BLOCK_F=block_f)
            buf0, buf1 = buf1, buf0
            if (e_run < 0).all():
                break

    return buf0


@triton.jit
def _prefix_bwd_push_kernel(
    cur_ptr,          # [n, f] gradients to push this round
    next_ptr,         # [n, f] (zero-filled before launch) receive buffer
    edges_ptr,        # [n] int32 downstream pointer (-1 if outlet)
    n_nodes,          # int
    n_feat: tl.constexpr,
    BLOCK_F: tl.constexpr,
):
    pid = tl.program_id(0)
    if pid >= n_nodes:
        return

    # downstream node
    dst = tl.load(edges_ptr + pid)
    valid = dst >= 0

    offs = tl.arange(0, BLOCK_F)

    # loop over feature dimension in tiles
    for base in range(0, n_feat, BLOCK_F):
        m = offs + base < n_feat

        g = tl.load(
            cur_ptr + pid * n_feat + base + offs,
            mask=m,
            other=0.0,
        )

        # push downstream (atomic because many parents can target same child)
        if valid:
            tl.atomic_add(
                next_ptr + dst * n_feat + base + offs,
                g,
                mask=m,
            )


def _prefix_jump_bwd(g_prefix: torch.Tensor,
                     edges: torch.Tensor,
                     *,
                     max_depth: int | None = None,
                     block_f: int = 128,
                     stream=None) -> torch.Tensor:
    """
    Parallel Triton implementation of prefix_sum_bwd_ref.

    Parameters
    ----------
    g_prefix : [n, f] tensor (float32/float64)  -- upstream grads wrt prefix outputs.
    edges    : [n] int32 tensor                 -- downstream pointer (-1 for outlet).
    max_depth: optional int upper bound on max path length; if None, estimated.
    block_f  : Triton tile size along feature dim. Tune for performance.
    stream   : optional CUDA stream.

    Returns
    -------
    g_irf : [n, f] tensor (same dtype/device) -- grads wrt original irfs.
    """
    assert g_prefix.ndim == 2, "g_prefix must be [n, f]"
    assert edges.ndim == 1, "edges must be [n]"
    n, f = g_prefix.shape
    assert edges.shape[0] == n, "edges length mismatch"
    assert edges.dtype in (torch.int32, torch.int64)
    if edges.dtype != torch.int32:
        edges = edges.to(torch.int32)

    device = g_prefix.device
    dtype = g_prefix.dtype
    # working buffers
    g_irf = g_prefix.clone()          # start with self-contribution
    cur   = g_prefix.clone()          # mass to push this round
    next_ = torch.zeros_like(g_prefix)

    # crude (safe) max_depth estimate if not provided: chase once on CPU
    if max_depth is None:
        # NOTE: O(n * depth) worst case but done once and cheap for typical n.
        e = edges.cpu().tolist()
        md = 0
        for i in range(n):
            d = 0
            j = i
            while j != -1:
                j = e[j]
                d += 1
            md = max(md, d)
        max_depth = md

    # grid
    grid = lambda META: (triton.cdiv(n, 1),)

    # push level by level
    with torch.cuda.device(device):
        for _ in range(max_depth):
            # zero receive buffer
            next_.zero_()
    
            _prefix_bwd_push_kernel[grid](
                cur,
                next_,
                edges,
                n,
                n_feat=f,
                BLOCK_F=min(block_f, triton.next_power_of_2(f)),
                #stream=stream,
            )
    
            # accumulate what arrived this round
            g_irf += next_
    
            # stop early if nothing moved
            if torch.count_nonzero(next_) == 0:
                break
    
            # next round pushes newly arrived mass further downstream
            cur, next_ = next_, cur  # swap buffers; we'll zero() the new next_ at top

    return g_irf



# ------------------------------------------------------------------
# Autograd wrapper
# ------------------------------------------------------------------
class PrefixSum(Function):
    @staticmethod
    def forward(ctx, irf, edges, block_f: int = 128):
        prefix = _prefix_jump_fwd(irf, edges, block_f)
        ctx.save_for_backward(edges)
        ctx.block_f = block_f
        return prefix

    @staticmethod
    def backward(ctx, g_prefix):
        edges, = ctx.saved_tensors
        block_f = ctx.block_f
        g_irf = _prefix_jump_bwd(g_prefix, edges) #, block_f)
        #g_irf = prefix_sum_bwd_ref(g_prefix, edges)
        return g_irf, None, None


def prefix_sum(irf, edges, block_f: int = 128):
    return PrefixSum.apply(irf, edges, block_f)
