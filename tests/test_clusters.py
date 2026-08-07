"""A clustered graph must route exactly like the same graph unsplit.

That is the whole contract of ``RivTreeCluster``: splitting is a scheduling
device, not a modelling choice, so it may not change a single output value.

Before the v1 line this only held with route_src_reach=True. The transfer hands an upstream
cluster's *routed discharge* to the downstream cluster as if it were runoff, and
that only reconstructs the unsplit network if the receiving row still traverses
one reach -- true when the source reach is routed, false when it is not, where a node's input is
by definition already past its own reach.

v1 resolves it by handing the value to a *copy* of the upstream breakpoint node
placed in the downstream cluster, and giving that copy route_src_reach=False
regardless of the global setting. It is therefore per node, and these tests pin that
the equivalence now holds either way.
"""
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


# ---------------------------------------------------------------- fixtures
def chain(n=6):
    return nx.DiGraph([(i, i + 1) for i in range(n - 1)])


def confluence():
    """0,1 -> 2 -> 3 -> 4, plus a tributary 5 -> 3"""
    return nx.DiGraph([(0, 2), (1, 2), (2, 3), (5, 3), (3, 4)])


def deep_tree(n=14, seed=0):
    rng = np.random.default_rng(seed)
    g = nx.DiGraph()
    g.add_nodes_from(range(n))
    for i in range(n - 1):
        g.add_edge(i, int(rng.integers(i + 1, n)))
    return g


def _params(g, seed=0):
    """Hayami params keyed by node, in a range that keeps kernels well inside TW."""
    rng = np.random.default_rng(seed)
    nodes = sorted(g.nodes)
    return pd.DataFrame(
        {"L": rng.uniform(1.0, 6.0, len(nodes)),
         "D": rng.uniform(0.4, 1.2, len(nodes)),
         "c": rng.uniform(0.8, 2.0, len(nodes))},
        index=nodes)


def _split(g, cut):
    """Split ``g`` at edge ``cut = (u, v)`` the way define_schedule does.

    Upstream cluster keeps u; downstream cluster gets a COPY of u wired to v.
    Returns (clusters, node_transfer, transition_nodes).
    """
    u, v = cut
    h = g.copy()
    h.remove_edge(u, v)
    comps = list(nx.weakly_connected_components(h))
    up = next(c for c in comps if u in c)
    down = next(c for c in comps if v in c)
    if up is down:
        pytest.skip(f"edge {cut} is not a cut edge")

    g_up = g.subgraph(up).copy()
    g_down = g.subgraph(down).copy()
    g_down.add_edge(u, v)                      # the copy of u

    idx_up = pd.Series(np.arange(len(g_up)), index=list(nx.dfs_preorder_nodes(g_up)))
    idx_dn = pd.Series(np.arange(len(g_down)), index=list(nx.dfs_preorder_nodes(g_down)))
    transfer = {0: [(1, int(idx_up.loc[u]), int(idx_dn.loc[u]))]}
    return [g_up, g_down], transfer, {1: {u}}, [idx_up, idx_dn]


# ---------------------------------------------------------------- the contract
@pytest.mark.parametrize("route_src_reach", [True, False])
@pytest.mark.parametrize("graph,cut", [
    (chain(6), (2, 3)),
    (chain(6), (0, 1)),
    (confluence(), (2, 3)),
    (confluence(), (3, 4)),
    (deep_tree(), None),
])
def test_clustered_equals_unsplit(graph, cut, route_src_reach):
    """Routing the split graph reproduces the unsplit result, node for node."""
    if cut is None:                            # pick any edge whose removal splits
        cut = next((u, v) for u, v in graph.edges
                   if nx.number_weakly_connected_components(
                       nx.restricted_view(graph, [], [(u, v)])) > 1)
    pdf = _params(graph)
    T = 4 * TW
    torch.manual_seed(0)

    rt = RivTree(graph, irf_fn="hayami", param_df=pdf,
                 route_src_reach=route_src_reach, param_names=["L", "D", "c"]).to(DEVICE)
    router = LTIRouter(max_delay=TW, dt=1).to(DEVICE)
    x_ref = torch.rand(1, len(rt.nodes), T, device=DEVICE)
    y_ref = router(x_ref, rt, rt.params)[0]
    ref = dict(zip(rt.nodes.tolist(), y_ref))

    clusters, transfer, transition, nidx = _split(graph, cut)
    gs = RivTreeCluster(clusters, transfer, irf_fn="hayami", param_df=pdf,
                        route_src_reach=route_src_reach, param_names=["L", "D", "c"],
                        nodes_idx=nidx, transition_nodes=transition).to(DEVICE)
    staged = LTIStagedRouter(max_delay=TW, dt=1).to(DEVICE)

    # same runoff, re-ordered into the cluster layout (one row per real node)
    order = gs.nodes.tolist()
    assert sorted(order) == sorted(rt.nodes.tolist()), "cluster layout lost a node"
    x_cl = torch.stack([x_ref[0, int(rt.nodes_idx.loc[n])] for n in order]).unsqueeze(0)
    y_cl = staged(x_cl, gs, gs.params)[0]

    for i, n in enumerate(order):
        err = (y_cl[i] - ref[n]).abs().max().item()
        scale = ref[n].abs().max().item()
        assert err <= 3e-3 * max(scale, 1e-6), (
            f"route_src_reach={route_src_reach}: node {n} differs between clustered and unsplit "
            f"routing by {err:.3e} (signal {scale:.3e})")


@pytest.mark.parametrize("route_src_reach", [True, False])
def test_transition_nodes_are_hidden(route_src_reach):
    """The copy is internal: it must not appear in the node layout or the output."""
    g = chain(6)
    clusters, transfer, transition, nidx = _split(g, (2, 3))
    gs = RivTreeCluster(clusters, transfer, irf_fn="hayami", param_df=_params(g),
                        route_src_reach=route_src_reach, param_names=["L", "D", "c"],
                        nodes_idx=nidx, transition_nodes=transition).to(DEVICE)

    assert gs.has_transitions
    assert sorted(gs.nodes.tolist()) == sorted(g.nodes)        # no duplicate
    assert len(gs.internal_nodes) == len(gs.nodes) + 1         # the copy exists
    assert list(gs.internal_nodes).count(2) == 2

    staged = LTIStagedRouter(max_delay=TW, dt=1).to(DEVICE)
    y = staged(torch.rand(1, len(gs.nodes), 64, device=DEVICE), gs, gs.params)
    assert y.shape[1] == len(gs.nodes)


def test_transition_node_carries_outlet_entry():
    """Even with route_src_reach=True, the copy must not route its own reach again --
    the upstream cluster already did."""
    g = chain(6)
    clusters, transfer, transition, nidx = _split(g, (2, 3))
    gs = RivTreeCluster(clusters, transfer, irf_fn="hayami", param_df=_params(g),
                        route_src_reach=True, param_names=["L", "D", "c"],
                        nodes_idx=nidx, transition_nodes=transition)
    down = gs[1]
    pos = int(down.nodes_idx.loc[2])
    assert not bool(down.own_reach[pos]), "transition node still routes its own reach"
    assert bool(down.own_reach.sum()) > 0, "the rest of the cluster lost route_src_reach"
    assert float(down.residual_weight[pos]) == 0.0, "transition node re-emits its input"
    assert not down.uniform, "cluster should be mixed"
