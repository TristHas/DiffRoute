"""
Backward (gradient) correctness tests for DiffRoute.

Each test checks that the custom Triton / autograd backward passes compute
gradients that match a pure-PyTorch reference built from standard ops.
Loss is always MSE with a rand_like target so that sign or index errors
that would cancel in a plain sum() cannot hide.

Only Hayami IRF is used throughout.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
import torch.nn.functional as F
import networkx as nx

from diffroute.irfs import IRF_FN
from diffroute.ops.prefix_sum import prefix_sum
from diffroute.ops.closure_sub import closure_sub
from diffroute.agg.kernel_aggregator import aggregate_irf
from diffroute.ops.conv import block_sparse_conv_1d
from diffroute.ops.in_place import write_slice, index_add_inplace
from diffroute.structs.riv_graphs import init_node_idxs
from diffroute.structs.utils import init_pre_indices

DEVICE = "cuda:0"
HAYAMI = IRF_FN["hayami"]

# ─────────────────────────────────────────────────────────────────────────────
# Shared test-graph constructors
# ─────────────────────────────────────────────────────────────────────────────

def _chain_graph():
    """0 → 1 → 2  (simplest linear chain)."""
    g = nx.DiGraph()
    g.add_node(0, L=10.0, D=1.0, c=1.0)
    g.add_node(1, L=8.0,  D=0.8, c=1.2)
    g.add_node(2, L=5.0,  D=0.5, c=1.5)
    g.add_edge(0, 1)
    g.add_edge(1, 2)
    return g


def _hayami_params(g, device):
    nidx = init_node_idxs(g)
    p = torch.tensor(
        [[g.nodes[n]["L"], g.nodes[n]["D"], g.nodes[n]["c"]] for n in nidx.index],
        dtype=torch.float32, device=device,
    )
    return p, nidx


# ─────────────────────────────────────────────────────────────────────────────
# Pure-PyTorch reference implementations (autograd-traceable, no custom kernels)
# ─────────────────────────────────────────────────────────────────────────────

def ref_prefix_sum(irf, edges):
    """Inclusive downstream prefix sum via explicit for-loops."""
    e = edges.tolist()
    rows = []
    for i in range(len(e)):
        p = irf[i]
        j = e[i]
        while j != -1:
            p = p + irf[j]
            j = e[j]
        rows.append(p)
    return torch.stack(rows)


def ref_exclusive(prefix, edges, one=False):
    """Exclusive downstream prefix: Q[i] = prefix[succ(i)], neutral at outlets.

    ``one=True`` returns the multiplicative identity for the frequency-domain
    reference (which divides instead of subtracting).
    """
    e = edges.tolist()
    neutral = (lambda r: torch.ones_like(r)) if one else (lambda r: torch.zeros_like(r))
    return [prefix[e[i]] if e[i] != -1 else neutral(prefix[i]) for i in range(len(e))]


def ref_closure_sub(head, tail, edges, route_src_reach):
    """Enumerate path-differences via for-loops (matches kernel coord order).

    ``val(dest, start) = head[start] - tail[dest]`` where ``tail`` is the
    EXCLUSIVE downstream prefix, so the destination needs no child lookup and the
    source end excludes its own reach exactly when ``route_src_reach`` is false.
    See ``diffroute/ops/closure_sub.py`` for the convention.
    """
    e = edges.tolist()
    coords_list, vals_list = [], []
    for start in range(len(e)):
        dest = start if route_src_reach else e[start]
        if dest == -1:
            continue
        while dest != -1:
            coords_list.append([dest, start])
            vals_list.append(head[start] - tail[dest])
            dest = e[dest]
    coords = torch.tensor(coords_list, dtype=torch.long, device=head.device)
    return coords, torch.stack(vals_list)


def ref_aggregate_irf(params, irf_fn, edges, route_src, dt, time_window):
    """
    Reference aggregation using standard PyTorch ops only.

    Computes path IRFs by direct frequency-domain multiplication
    (no log-space trick, no custom Triton kernels).
    Division by cum[child] is numerically safe for Hayami IRF because its
    frequency components stay well above zero for these parameter ranges.
    """
    irfs = irf_fn(params, time_window=time_window, dt=dt)   # [n, K]
    n, K = irfs.shape
    e = edges.tolist()
    freq = torch.fft.rfft(irfs, n=K, dim=-1)                # [n, F] complex

    cum = []
    for i in range(n):
        p = freq[i]
        j = e[i]
        while j != -1:
            p = p * freq[j]
            j = e[j]
        cum.append(p)

    # exclusive product (everything strictly downstream); the source end uses it
    # instead of ``cum`` when the reach's own IRF is not on the path
    cum_ex = ref_exclusive(cum, edges, one=True)
    head = cum if route_src else cum_ex

    coords_list, vals_list = [], []
    for start in range(n):
        dest = start if route_src else e[start]
        if dest == -1:
            continue
        while dest != -1:
            path_freq = head[start] / cum_ex[dest]
            coords_list.append([dest, start])
            vals_list.append(torch.fft.irfft(path_freq, n=K))
            dest = e[dest]

    coords = torch.tensor(coords_list, dtype=torch.long, device=params.device)
    return coords, torch.stack(vals_list)


# ─────────────────────────────────────────────────────────────────────────────
# Group 1 – prefix_sum backward
# ─────────────────────────────────────────────────────────────────────────────

def test_prefix_sum_backward_chain():
    """Triton backward matches reference on a simple linear chain."""
    torch.manual_seed(0)
    n, f = 8, 64
    edges = torch.arange(1, n + 1, dtype=torch.int32, device=DEVICE)
    edges[-1] = -1

    irf = torch.randn(n, f, device=DEVICE, requires_grad=True)
    prefix_t = prefix_sum(irf, edges)
    target = torch.rand_like(prefix_t)
    F.mse_loss(prefix_t, target).backward()
    grad_triton = irf.grad.clone()

    irf2 = irf.detach().clone().requires_grad_(True)
    prefix_r = ref_prefix_sum(irf2, edges)
    F.mse_loss(prefix_r, target).backward()
    grad_ref = irf2.grad.clone()

    err = (grad_triton - grad_ref).abs().max().item()
    assert torch.allclose(grad_triton, grad_ref, rtol=1e-3, atol=1e-3), \
        f"prefix_sum backward (chain): max_err={err:.2e}"


def test_prefix_sum_backward_fanin():
    """Triton backward matches reference on a fan-in tree (two tributaries)."""
    torch.manual_seed(1)
    # 0 → 2, 1 → 2, 2 → 3  (manual index order)
    edges = torch.tensor([2, 2, 3, -1], dtype=torch.int32, device=DEVICE)
    n, f = 4, 64

    irf = torch.randn(n, f, device=DEVICE, requires_grad=True)
    prefix_t = prefix_sum(irf, edges)
    target = torch.rand_like(prefix_t)
    F.mse_loss(prefix_t, target).backward()
    grad_triton = irf.grad.clone()

    irf2 = irf.detach().clone().requires_grad_(True)
    prefix_r = ref_prefix_sum(irf2, edges)
    F.mse_loss(prefix_r, target).backward()
    grad_ref = irf2.grad.clone()

    err = (grad_triton - grad_ref).abs().max().item()
    assert torch.allclose(grad_triton, grad_ref, rtol=1e-3, atol=1e-3), \
        f"prefix_sum backward (fan-in): max_err={err:.2e}"


# ─────────────────────────────────────────────────────────────────────────────
# Group 2 – closure_sub backward
# ─────────────────────────────────────────────────────────────────────────────

def _closure_sub_backward(route_src):
    torch.manual_seed(2)
    g = _chain_graph()
    nidx = init_node_idxs(g)
    edges, path_cumsum, own = init_pre_indices(g, nidx, route_src_reach=route_src)
    edges = edges.to(DEVICE)
    path_cumsum = path_cumsum.to(DEVICE)
    own = own.to(DEVICE)
    # full output: every node keeps its own row
    out_row = torch.arange(len(nidx), dtype=torch.int32, device=DEVICE)
    n, f = len(nidx), 64

    head_raw = torch.randn(n, f, device=DEVICE)
    tail_raw = torch.randn(n, f, device=DEVICE)

    head_t = head_raw.clone().requires_grad_(True)
    tail_t = tail_raw.clone().requires_grad_(True)
    _, vals_t = closure_sub(head_t, tail_t, edges, path_cumsum, own, out_row)
    target = torch.rand_like(vals_t)
    F.mse_loss(vals_t, target).backward()

    head_r = head_raw.clone().requires_grad_(True)
    tail_r = tail_raw.clone().requires_grad_(True)
    _, vals_r = ref_closure_sub(head_r, tail_r, edges, route_src)
    F.mse_loss(vals_r, target).backward()

    for name, gt, gr in (("head", head_t.grad, head_r.grad),
                         ("tail", tail_t.grad, tail_r.grad)):
        err = (gt - gr).abs().max().item()
        assert torch.allclose(gt, gr, rtol=1e-3, atol=1e-3), \
            f"closure_sub backward {name} (route_src={route_src}): max_err={err:.2e}"


def test_closure_sub_backward_shared_prefix():
    """``transitive_closure`` passes the exclusive prefix as BOTH head and tail
    when route_src_reach=False, so autograd has to accumulate the two contributions
    into one leaf. Guards that the Function returns per-argument gradients."""
    torch.manual_seed(4)
    g = _chain_graph()
    nidx = init_node_idxs(g)
    edges, path_cumsum, own = init_pre_indices(g, nidx, route_src_reach=False)
    edges = edges.to(DEVICE)
    path_cumsum = path_cumsum.to(DEVICE)
    own = own.to(DEVICE)
    # full output: every node keeps its own row
    out_row = torch.arange(len(nidx), dtype=torch.int32, device=DEVICE)
    n, f = len(nidx), 64

    raw = torch.randn(n, f, device=DEVICE)

    p_t = raw.clone().requires_grad_(True)
    _, vals_t = closure_sub(p_t, p_t, edges, path_cumsum, own, out_row)
    target = torch.rand_like(vals_t)
    F.mse_loss(vals_t, target).backward()

    p_r = raw.clone().requires_grad_(True)
    _, vals_r = ref_closure_sub(p_r, p_r, edges, False)
    F.mse_loss(vals_r, target).backward()

    err = (p_t.grad - p_r.grad).abs().max().item()
    assert torch.allclose(p_t.grad, p_r.grad, rtol=1e-3, atol=1e-3), \
        f"closure_sub backward (shared prefix): max_err={err:.2e}"


def test_closure_sub_backward_diag_true():
    """Closure-sub backward correct with route_src=True."""
    _closure_sub_backward(True)


def test_closure_sub_backward_diag_false():
    """Closure-sub backward correct with route_src=False."""
    _closure_sub_backward(False)


# ─────────────────────────────────────────────────────────────────────────────
# Group 3 – full aggregation backward  (params → irfs_agg)
# ─────────────────────────────────────────────────────────────────────────────

def _aggregate_irf_backward(route_src):
    torch.manual_seed(3)
    g = _chain_graph()
    nidx = init_node_idxs(g)
    edges, path_cumsum, own = init_pre_indices(g, nidx, route_src_reach=route_src)
    edges = edges.to(DEVICE)
    path_cumsum = path_cumsum.to(DEVICE)
    own = own.to(DEVICE)
    # full output: every node keeps its own row
    out_row = torch.arange(len(nidx), dtype=torch.int32, device=DEVICE)
    dt, tw = 1.0, 30
    params_raw, _ = _hayami_params(g, DEVICE)

    params_t = params_raw.clone().requires_grad_(True)
    _, irfs_t = aggregate_irf(params_t, HAYAMI, edges, path_cumsum,
                              dt=dt, time_window=tw,
                              own_reach=own, out_row=out_row)
    target = torch.rand_like(irfs_t)
    F.mse_loss(irfs_t, target).backward()
    grad_opt = params_t.grad.clone()

    params_r = params_raw.clone().requires_grad_(True)
    _, irfs_r = ref_aggregate_irf(params_r, HAYAMI, edges, route_src,
                                   dt=dt, time_window=tw)
    F.mse_loss(irfs_r, target).backward()
    grad_ref = params_r.grad.clone()

    # Wider tolerance: optimized uses log-space (numerically distinct path)
    err = (grad_opt - grad_ref).abs().max().item()
    assert torch.allclose(grad_opt, grad_ref, rtol=5e-2, atol=5e-2), \
        f"aggregate_irf backward (route_src={route_src}): max_err={err:.2e}"


def test_aggregate_irf_backward_diag_true():
    """Aggregation backward correct with route_src=True."""
    _aggregate_irf_backward(True)


def test_aggregate_irf_backward_diag_false():
    """Aggregation backward correct with route_src=False."""
    _aggregate_irf_backward(False)


# ─────────────────────────────────────────────────────────────────────────────
# Group 4 – block-sparse convolution backward
# ─────────────────────────────────────────────────────────────────────────────
#
# Reference: materialize the same kernel as a dense [C_out, C_in, K] tensor
# and use F.conv1d.  The Triton kernel computes y[t] = Σ_k W[k]@x[t-(K-1)+k],
# which equals F.conv1d(x, W, padding=K-1)[:, :, :T]  (cross-correlation, no flip).

def _dense_from_bsparse(coo, values_detached, C, K, BS):
    dense = torch.zeros(C, C, K, dtype=values_detached.dtype, device=values_detached.device)
    for idx, (r, c) in enumerate(coo.tolist()):
        dense[r * BS:(r + 1) * BS, c * BS:(c + 1) * BS, :] = values_detached[idx]
    return dense


def _conv_backward(coo, values, x, BS):
    B, C, T = x.shape
    K = values.shape[-1]
    kernel_shape = (C, C, K)

    y_sp = block_sparse_conv_1d(x, coo, values, kernel_shape, BS, 64)
    target = torch.rand_like(y_sp)
    F.mse_loss(y_sp, target).backward()
    dx_sp = x.grad.clone()
    dv_sp = values.grad.clone()

    x_d = x.detach().clone().requires_grad_(True)
    dense = _dense_from_bsparse(coo, values.detach(), C, K, BS)
    dense_p = dense.requires_grad_(True)
    y_d = F.conv1d(x_d, dense_p, padding=K - 1)[:, :, :T]
    F.mse_loss(y_d, target).backward()
    dx_d = x_d.grad.clone()
    dv_d = torch.stack([
        dense_p.grad[r * BS:(r + 1) * BS, c * BS:(c + 1) * BS, :]
        for r, c in coo.tolist()
    ])

    err_dx = (dx_sp - dx_d).abs().max().item()
    err_dv = (dv_sp - dv_d).abs().max().item()
    assert torch.allclose(dx_sp, dx_d, rtol=1e-3, atol=1e-3), \
        f"conv backward dx: max_err={err_dx:.2e}"
    assert torch.allclose(dv_sp, dv_d, rtol=1e-3, atol=1e-3), \
        f"conv backward dvalues: max_err={err_dv:.2e}"


def test_conv_backward_diagonal():
    """Convolution backward correct for diagonal-only block pattern."""
    torch.manual_seed(10)
    B, C, T, K, BS = 2, 32, 64, 20, 16   # BS >= 16 required by tl.dot
    n_blk = C // BS
    coo = torch.tensor([[i, i] for i in range(n_blk)], dtype=torch.long, device=DEVICE)
    values = torch.randn(n_blk, BS, BS, K, device=DEVICE, requires_grad=True)
    x = torch.randn(B, C, T, device=DEVICE, requires_grad=True)
    _conv_backward(coo, values, x, BS)


def test_conv_backward_offdiagonal():
    """Convolution backward correct for off-diagonal block pattern."""
    torch.manual_seed(11)
    B, C, T, K, BS = 2, 32, 64, 20, 16
    n_blk = C // BS
    off = [(r, c) for r in range(n_blk) for c in range(n_blk) if r != c]
    coo = torch.tensor(off, dtype=torch.long, device=DEVICE)
    values = torch.randn(len(off), BS, BS, K, device=DEVICE, requires_grad=True)
    x = torch.randn(B, C, T, device=DEVICE, requires_grad=True)
    _conv_backward(coo, values, x, BS)


def test_conv_backward_mixed():
    """Convolution backward correct for mixed diagonal + off-diagonal blocks."""
    torch.manual_seed(12)
    B, C, T, K, BS = 3, 32, 128, 32, 16
    n_blk = C // BS
    pairs = [(r, c) for r in range(n_blk) for c in range(n_blk)]
    coo = torch.tensor(pairs, dtype=torch.long, device=DEVICE)
    values = torch.randn(len(pairs), BS, BS, K, device=DEVICE, requires_grad=True)
    x = torch.randn(B, C, T, device=DEVICE, requires_grad=True)
    _conv_backward(coo, values, x, BS)


# ─────────────────────────────────────────────────────────────────────────────
# Group 5 – custom in-place ops backward
# ─────────────────────────────────────────────────────────────────────────────

def test_write_slice_backward():
    """grad_src equals the gradient slice the loss has w.r.t. that region."""
    torch.manual_seed(20)
    B, C, T = 2, 10, 16
    rs, re = 3, 7
    src = torch.randn(B, re - rs, T, device=DEVICE, requires_grad=True)

    target = write_slice(torch.zeros(B, C, T, device=DEVICE), src, rs, re)
    tgt = torch.rand_like(target)
    F.mse_loss(target, tgt).backward()

    with torch.no_grad():
        expected = 2.0 / target.numel() * (target[:, rs:re].detach() - tgt[:, rs:re])

    err = (src.grad - expected).abs().max().item()
    assert torch.allclose(src.grad, expected, rtol=1e-5, atol=1e-6), \
        f"write_slice backward: max_err={err:.2e}"


def test_index_add_inplace_backward():
    """grad_src[k] equals grad_target at position index[k]."""
    torch.manual_seed(21)
    B, C, T = 2, 8, 16
    index = torch.tensor([1, 3, 6], dtype=torch.long, device=DEVICE)
    src = torch.randn(B, len(index), T, device=DEVICE, requires_grad=True)

    target = index_add_inplace(torch.zeros(B, C, T, device=DEVICE), index, src, dim=1)
    tgt = torch.rand_like(target)
    F.mse_loss(target, tgt).backward()

    with torch.no_grad():
        grad_full = 2.0 / target.numel() * (target.detach() - tgt)
        expected = grad_full.index_select(1, index)

    err = (src.grad - expected).abs().max().item()
    assert torch.allclose(src.grad, expected, rtol=1e-5, atol=1e-6), \
        f"index_add_inplace backward: max_err={err:.2e}"


# ─────────────────────────────────────────────────────────────────────────────
# Script runner (human-readable summary, can be invoked directly)
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import traceback

    tests = {
        "prefix_sum_chain":         test_prefix_sum_backward_chain,
        "prefix_sum_fanin":         test_prefix_sum_backward_fanin,
        "closure_sub_diag_true":    test_closure_sub_backward_diag_true,
        "closure_sub_diag_false":   test_closure_sub_backward_diag_false,
        "closure_sub_shared":       test_closure_sub_backward_shared_prefix,
        "aggregate_irf_diag_true":  test_aggregate_irf_backward_diag_true,
        "aggregate_irf_diag_false": test_aggregate_irf_backward_diag_false,
        "conv_diagonal":            test_conv_backward_diagonal,
        "conv_offdiagonal":         test_conv_backward_offdiagonal,
        "conv_mixed":               test_conv_backward_mixed,
        "write_slice":              test_write_slice_backward,
        "index_add_inplace":        test_index_add_inplace_backward,
    }

    results = {}
    for name, fn in tests.items():
        try:
            fn()
            results[name] = True
        except AssertionError as e:
            print(f"  FAIL  {name}: {e}")
            results[name] = False
        except Exception as e:
            print(f"  ERROR {name}: {e}")
            traceback.print_exc()
            results[name] = False

    print("\n" + "=" * 60)
    n_pass = sum(results.values())
    for name, ok in results.items():
        print(f"  {'PASS' if ok else 'FAIL'}  {name}")
    print(f"\n{n_pass}/{len(results)} tests passed")
