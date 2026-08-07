"""Selecting output reaches must restrict the result, never change it.

Training only ever compares gauged reaches, so routing every reach to every
downstream reach is mostly waste. `set_output_reach` drops the paths that end
somewhere nobody asked about, which shrinks the closure *and* the convolution
kernel: its row dimension becomes the number of selected reaches rather than the
number of nodes.

What it must not touch is the graph, the node ordering, or the runoff tensor the
router expects -- selecting outputs is not a way to build a smaller network.
"""
import networkx as nx
import numpy as np
import pandas as pd
import pytest
import torch

from diffroute import RivTree, RivTreeCluster
from diffroute.router import LTIRouter

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


def test_clusters_refuse_selection():
    """Refused rather than silently wrong: a transfer source must stay in its
    own cluster's output."""
    g = chain(6)
    pdf = pd.DataFrame({"L": [3.0] * 6, "D": [0.8] * 6, "c": [1.4] * 6}, index=sorted(g.nodes))
    up, dn = g.subgraph({0, 1, 2}).copy(), g.subgraph({3, 4, 5}).copy()
    gs = RivTreeCluster([up, dn], {}, irf_fn="hayami", param_df=pdf,
                        param_names=["L", "D", "c"])
    with pytest.raises(NotImplementedError, match="transfer"):
        gs.set_output_reach([5])
    gs.set_output_reach(None)              # the no-op case still works


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
