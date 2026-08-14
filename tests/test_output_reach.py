"""Selecting output reaches must restrict the result, never change it.

Training only ever compares gauged reaches, so routing every reach to every
downstream reach is mostly waste. `set_output_reach` drops the paths that end
somewhere nobody asked about, which shrinks the closure *and* the convolution
kernel: its row dimension becomes the number of selected reaches rather than the
number of nodes.

What it must not touch is the graph, the node ordering, or the runoff tensor the
router expects -- selecting outputs is not a way to build a smaller network.
"""
from collections import defaultdict

import networkx as nx
import numpy as np
import pandas as pd
import pytest
import torch

from diffroute import RivTree, RivTreeCluster
from diffroute.router import LTIRouter
from diffroute.staged_router import LTIStagedRouter

DEVICE = "cuda:0"
TW = 32


def chain(n=6):
    return nx.DiGraph([(i, i + 1) for i in range(n - 1)])


def confluence():
    return nx.DiGraph([(0, 2), (1, 2), (2, 3), (5, 3), (3, 4)])


def tree(n=15, seed=1):
    rng = np.random.default_rng(seed)
    g = nx.DiGraph()
    g.add_nodes_from(range(n))
    for i in range(n - 1):
        g.add_edge(i, int(rng.integers(i + 1, n)))
    return g


def relabelled(g, offset=1000):
    """Same topology, non-contiguous labels: positions must come from a lookup,
    never from the label value."""
    return nx.relabel_nodes(g, {n: offset + 7 * n for n in g.nodes})


def _rt(g, route_src_reach=True, seed=0, **kw):
    rng = np.random.default_rng(seed)
    nodes = sorted(g.nodes)
    pdf = pd.DataFrame({"L": rng.uniform(1.0, 6.0, len(nodes)),
                        "D": rng.uniform(0.4, 1.2, len(nodes)),
                        "c": rng.uniform(0.8, 2.0, len(nodes))}, index=nodes)
    return RivTree(g, irf_fn="hayami", param_df=pdf,
                   route_src_reach=route_src_reach,
                   param_names=["L", "D", "c"], **kw).to(DEVICE)


# ------------------------------------------------------- the contract
@pytest.mark.parametrize("route_src_reach", [True, False])
@pytest.mark.parametrize("graph", [chain(6), confluence(), tree(), relabelled(tree())])
def test_selection_matches_full(graph, route_src_reach):
    """Every selected row equals the corresponding row of the full routing."""
    rt = _rt(graph, route_src_reach)
    router = LTIRouter(max_delay=TW, dt=1).to(DEVICE)
    T = 3 * TW
    torch.manual_seed(0)
    x = torch.rand(1, len(rt), T, device=DEVICE)

    full = router(x, rt, rt.params)
    labels = rt.nodes.tolist()

    rng = np.random.default_rng(0)
    for _ in range(4):
        k = int(rng.integers(1, len(labels) + 1))
        sel = rng.choice(labels, size=k, replace=False).tolist()
        rt.set_output_reach(sel)
        sub = router(x, rt, rt.params)
        assert sub.shape == (1, k, T)
        for i, node in enumerate(sel):
            pos = labels.index(node)
            err = (sub[0, i] - full[0, pos]).abs().max().item()
            scale = max(full[0, pos].abs().max().item(), 1e-6)
            assert err <= 1e-5 * scale, (
                f"route_src_reach={route_src_reach}: node {node} differs from "
                f"the full routing by {err:.3e}")
    rt.set_output_reach(None)
    assert torch.equal(router(x, rt, rt.params), full), "None did not restore full routing"


def test_kernel_is_not_square():
    """The convolution kernel must shrink with the selection, not just the
    returned tensor -- that is the whole point."""
    g = tree(30, seed=4)
    rt = _rt(g)
    router = LTIRouter(max_delay=TW, dt=1).to(DEVICE)
    outlets = [n for n in g.nodes if g.out_degree(n) == 0]

    full_paths = rt.n_paths
    kernel_full = router.aggregator(rt, rt.params)
    assert kernel_full.size[0] == kernel_full.size[1] == len(rt)

    rt.set_output_reach(outlets)
    kernel_sub = router.aggregator(rt, rt.params)
    assert kernel_sub.size == (len(outlets), len(rt), kernel_full.size[2]), (
        f"kernel stayed {kernel_sub.size}, wanted ({len(outlets)}, {len(rt)}, ...)")
    assert 0 < rt.n_paths < full_paths, (
        f"closure did not shrink: {rt.n_paths} paths against {full_paths}")
    # and the block-sparse form carries fewer blocks
    b_full = kernel_full.to_block_sparse(16).block_indices.shape[0]
    b_sub = kernel_sub.to_block_sparse(16).block_indices.shape[0]
    assert b_sub < b_full, f"blocks did not shrink: {b_sub} against {b_full}"


def test_graph_and_input_layout_are_untouched():
    """Selecting outputs must not alter the graph, the node order, or the
    runoff layout the router expects."""
    g = tree(20, seed=2)
    rt = _rt(g)
    before_nodes = rt.nodes.tolist()
    before_edges = rt.edges.clone()
    before_len = len(rt)

    rt.set_output_reach(before_nodes[:3])
    assert rt.nodes.tolist() == before_nodes, "node ordering changed"
    assert torch.equal(rt.edges, before_edges), "edges changed"
    assert len(rt) == before_len, "node count changed"
    assert rt.g.number_of_nodes() == len(before_nodes), "graph was pruned"

    # the router still takes one row per node
    router = LTIRouter(max_delay=TW, dt=1).to(DEVICE)
    y = router(torch.rand(1, before_len, 64, device=DEVICE), rt, rt.params)
    assert y.shape == (1, 3, 64)


def test_row_order_follows_the_request():
    g = chain(6)
    rt = _rt(g)
    router = LTIRouter(max_delay=TW, dt=1).to(DEVICE)
    x = torch.rand(1, len(rt), 64, device=DEVICE)
    rt.set_output_reach([5, 0, 3]); a = router(x, rt, rt.params)
    rt.set_output_reach([3, 0, 5]); b = router(x, rt, rt.params)
    assert torch.equal(a[0, 0], b[0, 2])
    assert torch.equal(a[0, 1], b[0, 1])
    assert torch.equal(a[0, 2], b[0, 0])


def test_set_at_construction_matches_setter():
    g = confluence()
    sel = [4, 2]
    a = _rt(g, output_reach=sel)
    b = _rt(g); b.set_output_reach(sel)
    router = LTIRouter(max_delay=TW, dt=1).to(DEVICE)
    x = torch.rand(1, len(a), 64, device=DEVICE)
    assert torch.equal(router(x, a, a.params), router(x, b, b.params))


def test_single_reach_selection():
    """A one-row output, including the degenerate no-path case."""
    g = chain(4)
    router = LTIRouter(max_delay=TW, dt=1).to(DEVICE)
    x = torch.rand(1, 4, 64, device=DEVICE)
    for src in (True, False):
        rt = _rt(g, route_src_reach=src, output_reach=[0])   # headwater only
        y = router(x, rt, rt.params)
        assert y.shape == (1, 1, 64)
        if not src:                       # no path ends at 0, only the identity
            assert rt.n_paths == 0
            assert torch.allclose(y[0, 0], x[0, 0])


def test_bad_selection_is_rejected():
    rt = _rt(chain(6))
    with pytest.raises(KeyError):
        rt.set_output_reach([99])
    with pytest.raises(ValueError, match="duplicate"):
        rt.set_output_reach([1, 1])


def _split_chain(g, cuts):
    """Split a chain-like graph at each (u, v) in ``cuts`` (given in
    downstream order), one cluster per segment. Each downstream cluster gets
    a COPY of its cut's ``u`` wired to ``v``, matching ``define_schedule``'s
    convention (see test_clusters.py's ``_split``, generalized to more than
    one cut so a middle cluster can be a pure pass-through).
    """
    h = g.copy()
    for u, v in cuts:
        h.remove_edge(u, v)
    order = list(nx.topological_sort(g))
    comps = sorted(nx.weakly_connected_components(h),
                   key=lambda c: min(order.index(n) for n in c))
    clusters_g = [g.subgraph(c).copy() for c in comps]
    for i, (u, v) in enumerate(cuts):
        clusters_g[i + 1].add_edge(u, v)
    idxs = [pd.Series(np.arange(len(cg)), index=list(nx.dfs_preorder_nodes(cg)))
           for cg in clusters_g]
    node_transfer = defaultdict(list)
    transition_nodes = defaultdict(set)
    for i, (u, v) in enumerate(cuts):
        node_transfer[i].append((i + 1, int(idxs[i].loc[u]), int(idxs[i + 1].loc[u])))
        transition_nodes[i + 1].add(u)
    return clusters_g, dict(node_transfer), dict(transition_nodes), idxs


def _cluster_params(g, seed=0):
    rng = np.random.default_rng(seed)
    nodes = sorted(g.nodes)
    return pd.DataFrame({"L": rng.uniform(1.0, 6.0, len(nodes)),
                         "D": rng.uniform(0.4, 1.2, len(nodes)),
                         "c": rng.uniform(0.8, 2.0, len(nodes))}, index=nodes)


def test_clustered_selection_matches_full():
    """Narrowing a clustered graph must match the same graph's full output,
    including for a selection that never touches the transfer-source node
    directly -- the internal mechanism (keep it anyway, hide it again) has
    to fire without the caller asking for it by name.
    """
    g = chain(9)
    pdf = _cluster_params(g)
    clusters, transfer, transition, nidx = _split_chain(g, [(2, 3)])
    gs = RivTreeCluster(clusters, transfer, irf_fn="hayami", param_df=pdf,
                        param_names=["L", "D", "c"], nodes_idx=nidx,
                        transition_nodes=transition).to(DEVICE)
    staged = LTIStagedRouter(max_delay=TW, dt=1).to(DEVICE)
    x = torch.rand(1, len(gs.nodes), 3 * TW, device=DEVICE)
    full = staged(x, gs, gs.params)
    labels = gs.nodes.tolist()

    for sel in ([8], [4, 6], [1, 3, 5, 7]):     # none of these is node 2, the transfer source
        gs.set_output_reach(sel)
        sub = staged(x, gs, gs.params)
        assert sub.shape == (1, len(sel), x.shape[-1])
        for i, node in enumerate(sel):
            pos = labels.index(node)
            err = (sub[0, i] - full[0, pos]).abs().max().item()
            scale = max(full[0, pos].abs().max().item(), 1e-6)
            assert err <= 1e-3 * scale, f"node {node} differs from full by {err:.3e}"


def test_clustered_selection_matches_unsplit():
    """The same selection, routed clustered vs unsplit -- the two must agree,
    same as the unnarrowed contract in test_clusters.py, now also under
    narrowing.

    chain(6): matches test_clusters.py's own scale. _params's own docstring
    notes its Hayami range is chosen to keep kernels "well inside TW" for
    a SINGLE reach -- composed across enough cascaded reaches (a long
    enough chain) the EFFECTIVE combined width can still exceed the window,
    and the clustered path then truncates-and-renormalizes it twice (once
    per cluster segment) against the flat path's once, so error compounds
    faster than plain fp noise as the chain grows past this scale. That is
    a pre-existing property of long IRF cascades against a fixed max_delay
    (present in test_clusters.py's own machinery too, unrelated to
    set_output_reach), not something narrowing should be judged against.
    """
    g = chain(6)
    pdf = _cluster_params(g)
    rt = RivTree(g, irf_fn="hayami", param_df=pdf, param_names=["L", "D", "c"]).to(DEVICE)
    router = LTIRouter(max_delay=TW, dt=1).to(DEVICE)
    x = torch.rand(1, len(rt), 3 * TW, device=DEVICE)
    rt.set_output_reach([4, 5])
    y_flat = router(x, rt, rt.params)

    clusters, transfer, transition, nidx = _split_chain(g, [(2, 3)])
    gs = RivTreeCluster(clusters, transfer, irf_fn="hayami", param_df=pdf,
                        param_names=["L", "D", "c"], nodes_idx=nidx,
                        transition_nodes=transition).to(DEVICE)
    staged = LTIStagedRouter(max_delay=TW, dt=1).to(DEVICE)
    gs.set_output_reach([4, 5])
    y_cl = staged(x, gs, gs.params)

    err = (y_cl - y_flat).abs().max().item()
    scale = y_flat.abs().max().item()
    assert err <= 3e-3 * max(scale, 1e-6)


def test_inactive_cluster_is_skipped():
    """A cluster with nothing needed in it, directly or through a transfer to
    something needed, is not routed at all -- and dropping it must not
    change the requested output.
    """
    g = chain(9)
    pdf = _cluster_params(g)
    clusters, transfer, transition, nidx = _split_chain(g, [(2, 3), (5, 6)])
    gs = RivTreeCluster(clusters, transfer, irf_fn="hayami", param_df=pdf,
                        param_names=["L", "D", "c"], nodes_idx=nidx,
                        transition_nodes=transition).to(DEVICE)
    staged = LTIStagedRouter(max_delay=TW, dt=1).to(DEVICE)
    x = torch.rand(1, len(gs.nodes), 3 * TW, device=DEVICE)

    gs.set_output_reach([1])                   # upstream-most cluster only
    assert gs.active == [True, False, False], (
        "clusters 1 and 2 feed nothing the request needs and must be skipped")
    y_cl = staged(x, gs, gs.params)

    rt = RivTree(g, irf_fn="hayami", param_df=pdf, param_names=["L", "D", "c"],
                output_reach=[1]).to(DEVICE)
    router = LTIRouter(max_delay=TW, dt=1).to(DEVICE)
    y_flat = router(x, rt, rt.params)

    err = (y_cl - y_flat).abs().max().item()
    scale = y_flat.abs().max().item()
    assert err <= 1e-3 * max(scale, 1e-6)


def test_clustered_restore_and_duplicate_rejection():
    g = chain(9)
    pdf = _cluster_params(g)
    clusters, transfer, transition, nidx = _split_chain(g, [(2, 3)])
    gs = RivTreeCluster(clusters, transfer, irf_fn="hayami", param_df=pdf,
                        param_names=["L", "D", "c"], nodes_idx=nidx,
                        transition_nodes=transition).to(DEVICE)
    staged = LTIStagedRouter(max_delay=TW, dt=1).to(DEVICE)
    x = torch.rand(1, len(gs.nodes), 64, device=DEVICE)
    full = staged(x, gs, gs.params)

    gs.set_output_reach([4, 8])
    gs.set_output_reach(None)
    restored = staged(x, gs, gs.params)
    err = (restored - full).abs().max().item()
    scale = full.abs().max().item()
    assert err <= 1e-3 * max(scale, 1e-6), "None did not restore full routing"

    with pytest.raises(ValueError, match="duplicate"):
        gs.set_output_reach([4, 4])


def test_clustered_selection_gradients_match_full():
    """d loss / d gs.params through narrowed clustered routing must equal the
    same loss on the fully-routed cluster with rows selected afterwards --
    the clustered analogue of test_selection_gradients_match_full, and for
    the same reason: forward tests cannot see a backward-only bug (the
    2026-08 output_reach regression this mirrors was exactly that).
    """
    g = chain(6)
    pdf = _cluster_params(g)
    clusters, transfer, transition, nidx = _split_chain(g, [(2, 3)])
    gs = RivTreeCluster(clusters, transfer, irf_fn="hayami", param_df=pdf,
                        param_names=["L", "D", "c"], nodes_idx=nidx,
                        transition_nodes=transition).to(DEVICE)
    staged = LTIStagedRouter(max_delay=TW, dt=1).to(DEVICE)
    T = 3 * TW
    torch.manual_seed(1)
    x = torch.rand(1, len(gs.nodes), T, device=DEVICE)
    labels = gs.nodes.tolist()
    sel = [4, 5]                                # downstream of the transfer, not the source itself
    pos = [labels.index(n) for n in sel]
    target = torch.rand(1, len(sel), T, device=DEVICE)

    gs.set_output_reach(None)
    P_full = gs.params.clone().requires_grad_(True)
    y_full = staged(x, gs, P_full)[:, pos]
    ((y_full - target) ** 2).mean().backward()

    gs.set_output_reach(sel)
    P_sub = gs.params.clone().requires_grad_(True)
    y_sub = staged(x, gs, P_sub)
    ((y_sub - target) ** 2).mean().backward()

    assert torch.allclose(y_sub, y_full, rtol=1e-3, atol=1e-6), \
        "narrowed clustered forward diverged from full"
    scale = float(P_full.grad.abs().max())
    err = float((P_full.grad - P_sub.grad).abs().max()) / max(scale, 1e-9)
    assert err < 1e-2, f"narrowed clustered gradients diverge from full: rel {err:.3e}"


def test_conv_to_returns_self():
    """`.to()` must return the module whether or not a kernel was set at init.

    The `return self` used to sit inside the `isinstance` branch, so the common
    `BlockSparseCausalConv().to(dev)` silently evaluated to None. LTIRouter never
    hit it because nn.Module.to recurses into children instead.
    """
    from diffroute.conv import BlockSparseCausalConv
    assert BlockSparseCausalConv().to(DEVICE) is not None
    conv = BlockSparseCausalConv()
    assert conv.to(DEVICE) is conv


@pytest.mark.parametrize("route_src_reach", [True, False])
@pytest.mark.parametrize("graph", [chain(6), confluence(), tree(),
                                   tree(24, seed=9)])
def test_selection_gradients_match_full(graph, route_src_reach):
    """Narrowing must not change gradients: d loss / d params through the
    narrowed routing must equal the same loss on the full routing with the
    rows selected afterwards.

    Regression (2026-08): the forward stores the destination's remapped OUTPUT
    row in ``coords[:, 0]``; the backward used that value as a *node* index,
    scattering every path's tail gradient onto whichever nodes occupied rows
    0..n_out-1. Forward tests cannot see it -- only this equivalence can.
    """
    rt = _rt(graph, route_src_reach)
    router = LTIRouter(max_delay=TW, dt=1).to(DEVICE)
    T = 3 * TW
    torch.manual_seed(1)
    x = torch.rand(1, len(rt), T, device=DEVICE)
    labels = rt.nodes.tolist()
    rng = np.random.default_rng(1)
    sel = rng.choice(labels, size=max(2, len(labels) // 3),
                     replace=False).tolist()
    pos = [labels.index(n) for n in sel]
    target = torch.rand(1, len(sel), T, device=DEVICE)

    rt.set_output_reach(None)
    P_full = rt.params.clone().requires_grad_(True)
    y_full = router(x, rt, P_full)[:, pos]
    ((y_full - target) ** 2).mean().backward()

    rt.set_output_reach(sel)
    P_sub = rt.params.clone().requires_grad_(True)
    y_sub = router(x, rt, P_sub)
    ((y_sub - target) ** 2).mean().backward()

    assert torch.allclose(y_sub, y_full, rtol=1e-4, atol=1e-6), \
        "narrowed forward diverged from full"
    scale = float(P_full.grad.abs().max())
    err = float((P_full.grad - P_sub.grad).abs().max()) / max(scale, 1e-9)
    assert err < 1e-3, f"narrowed gradients diverge from full: rel {err:.3e}"
    rt.set_output_reach(None)
