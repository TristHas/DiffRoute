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
    # widen before pid/dst * n_feat is computed below, or the product silently
    # wraps at large n (Triton does not trap on int32 overflow) -- same fix as
    # closure_sub.py's PATH-indexed offsets, here for NODE-indexed ones: this
    # kernel's addressing was never widened, unlike that file's, so it wraps
    # once n_nodes * n_feat exceeds 2**31 (e.g. n_feat=770 hourly taps, n
    # beyond ~2.79M -- above today's largest single routed cluster, but not
    # structurally bounded to stay there).
    pid = tl.program_id(0).to(tl.int64)
    if pid >= n_nodes: return
    dst   = tl.load(jump_ptr + pid).to(tl.int64)
    valid = dst >= 0
    offs  = tl.arange(0, BLOCK_F)

    for base in range(0, n_feat, BLOCK_F):
        m   = offs + base < n_feat
        acc = tl.load(prev_ptr + pid * n_feat + base + offs, mask=m, other=0.)
        add = tl.load(prev_ptr + dst * n_feat + base + offs, mask=m & valid, other=0.)
        acc += add
        tl.store(next_ptr + pid * n_feat + base + offs, acc, mask=m)
    # write 2-hop jump table
    nxt = tl.load(jump_ptr + dst, mask=valid, other=-1)
    tl.store(edges_ptr + pid, nxt)


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
                     block_f: int = 128,
                     return_jumps: bool = False):
    """
    Fast inclusive prefix (node -> outlet) via pointer-jumping.

    return_jumps: also return the per-round jump table each round actually
        used (e_snap before that round's doubling). Backward replays these in
        reverse instead of re-deriving them one graph edge at a time -- see
        _prefix_jump_bwd_from_history.
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
    jump_history = [] if return_jumps else None

    rounds = math.ceil(math.log2(max(1, n)))
    grid   = (n,)
    with torch.cuda.device(irf.device):
        for _ in range(rounds):
            #print(buf0.device)
            # snapshot current jump table (so kernel reads stable values)
            e_snap.copy_(e_run)
            if return_jumps:
                jump_history.append(e_snap.clone())
            _prefix_jump_kernel[grid](buf0, buf1, e_run, e_snap,
                                      n, f, BLOCK_F=block_f)
            buf0, buf1 = buf1, buf0
            if (e_run < 0).all():
                break

    return (buf0, jump_history) if return_jumps else buf0


@triton.jit
def _prefix_bwd_push_kernel(
    cur_ptr,          # [n, f] gradients to push this round
    next_ptr,         # [n, f] (zero-filled before launch) receive buffer
    edges_ptr,        # [n] int32 downstream pointer (-1 if outlet)
    n_nodes,          # int
    n_feat: tl.constexpr,
    BLOCK_F: tl.constexpr,
):
    # see _prefix_jump_kernel's comment: widen before the *n_feat products
    # below, or they silently wrap at large n.
    pid = tl.program_id(0).to(tl.int64)
    if pid >= n_nodes:
        return

    # downstream node
    dst = tl.load(edges_ptr + pid).to(tl.int64)
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


def _prefix_jump_bwd_from_history(g_prefix: torch.Tensor,
                                  jump_history: list[torch.Tensor],
                                  block_f: int = 128) -> torch.Tensor:
    """
    Backward via the forward's own pointer-jump rounds, replayed in reverse.

    Each forward round computed ``V[i] += V[jump[i]]`` for that round's jump
    table; backing that one round out is a scatter-add of the incoming
    gradient to both ``i`` and ``jump[i]`` (`_prefix_bwd_push_kernel`, reused
    unchanged from the single-hop driver this replaces -- it already treats
    its third argument as "the jump table for this round", so a per-round
    table works exactly like the single-hop `edges` did). Replaying every
    round this way, latest round first, is exactly the len(jump_history)
    rounds the forward pass needed (its own early exit already bounds this
    to ceil(log2(max path length)), never one round per graph edge) --
    O(depth) sequential rounds becomes O(log depth).

    g_prefix : [n, f] tensor -- upstream grads wrt prefix outputs.
    jump_history : per-round jump tables from _prefix_jump_fwd(return_jumps=True).
    """
    n, f = g_prefix.shape
    cur = g_prefix.contiguous()
    block_f = min(block_f, triton.next_power_of_2(f))
    grid = (n,)

    with torch.cuda.device(g_prefix.device):
        for jump in reversed(jump_history):
            next_ = cur.clone()  # self-contribution: V_r[i] feeds V_{r+1}[i] directly
            _prefix_bwd_push_kernel[grid](
                cur, next_, jump, n, n_feat=f, BLOCK_F=block_f,
            )
            cur = next_

    return cur



# ------------------------------------------------------------------
# Autograd wrapper
# ------------------------------------------------------------------
class PrefixSum(Function):
    @staticmethod
    def forward(ctx, irf, edges, block_f: int = 128):
        prefix, jump_history = _prefix_jump_fwd(irf, edges, block_f, return_jumps=True)
        # plain ctx attribute, not save_for_backward: these are internal
        # round-by-round jump tables, not an input or output of this
        # Function, so they don't need autograd's input/output version
        # tracking -- save_for_backward is for tensors that are.
        ctx.jump_history = jump_history
        ctx.block_f = block_f
        return prefix

    @staticmethod
    def backward(ctx, g_prefix):
        g_irf = _prefix_jump_bwd_from_history(g_prefix, ctx.jump_history, ctx.block_f)
        return g_irf, None, None


def prefix_sum(irf, edges, block_f: int = 128):
    return PrefixSum.apply(irf, edges, block_f)
