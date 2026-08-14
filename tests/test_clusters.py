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


# --------------------------------------------------------- Task D, Bug 2
# ``RivTreeCluster.params`` used to be a property re-concatenating the
# sub-trees' own buffers on every access. Routing reads ``gs.params`` ONCE
# (route_all_clusters slices it per cluster via node_ranges), so that was
# fine for params built directly from a param_df -- but an in-place edit
# AFTER construction (e.g. scaling k from days to hours, the standard way
# to switch a graph to hourly resolution) landed on nothing routing ever
# read: the property recomputed a fresh torch.cat() on the NEXT access,
# discarding the edit. VPU 209 hourly clustered routed with k still in
# days as a result (0.977 median NSE vs 0.99977 flat, error accumulating
# downstream -- the signature this test's tolerance is chosen to catch).
def test_in_place_param_edit_is_routed():
    """The exact failure mode: mutate gs.params in place, then route --
    must match a graph built directly from the already-scaled params."""
    g = chain(8)
    clusters, transfer, transition, nidx = _split(g, (3, 4))
    pdf = _params(g)
    gs = RivTreeCluster(clusters, transfer, irf_fn="hayami", param_df=pdf,
                        route_src_reach=True, param_names=["L", "D", "c"],
                        nodes_idx=nidx, transition_nodes=transition).to(DEVICE)

    before = gs.params.clone()
    gs.params[:, 2] *= 3.0                      # celerity, in place -- as k *= 24 does
    assert torch.equal(gs.params[:, 2], before[:, 2] * 3.0), (
        "in-place edit did not persist on gs.params -- params is still a "
        "snapshot/property rather than the buffer routing reads")

    staged = LTIStagedRouter(max_delay=TW, dt=1).to(DEVICE)
    x = torch.rand(1, len(gs.nodes), 4 * TW, device=DEVICE)
    y_scaled = staged(x, gs, gs.params)[0]

    pdf_scaled = pdf.copy()
    pdf_scaled["c"] *= 3.0
    gs_ref = RivTreeCluster(clusters, transfer, irf_fn="hayami", param_df=pdf_scaled,
                            route_src_reach=True, param_names=["L", "D", "c"],
                            nodes_idx=nidx, transition_nodes=transition).to(DEVICE)
    y_ref = staged(x, gs_ref, gs_ref.params)[0]

    # gs and gs_ref are separate instances -> separate closure/conv kernel
    # launches, whose atomic-add accumulation order is not guaranteed
    # identical (see the run-to-run noise floor documented across
    # camels_tdx/scripts/d_certify_bitwise.py); a relative tolerance is the
    # right bar, not torch.equal -- the point of this test is that the
    # PARAMETERS were used at all, not bit-for-bit reproducibility.
    err = (y_scaled - y_ref).abs().max().item()
    scale = y_ref.abs().max().item()
    assert err <= 1e-3 * max(scale, 1e-6), (
        f"routing did not use the in-place-edited params (Bug 2: stale "
        f"RivTreeCluster.params): diff {err:.3e} vs signal {scale:.3e}")


def test_clustered_hourly_muskingum_matches_unsplit():
    """The crashes README's own Bug 2 discriminator: a two-cluster toy
    chain at hourly settings (Muskingum, k in HOURS, dt=1, matching the
    real VPU 209 repro convention -- see camels_tdx
    README_DIFFROUTE_CRASHES.md SS Bug 2), diffing clustered against the
    unsplit graph directly rather than against an external reference.
    """
    n, cut = 300, (150, 151)
    g = nx.DiGraph([(i, i + 1) for i in range(n - 1)])
    rng = np.random.default_rng(0)
    nodes = sorted(g.nodes)
    pdf = pd.DataFrame({"x": rng.uniform(0.1, 0.4, n),
                        "k": rng.uniform(0.5, 3.0, n)}, index=nodes)
    TW_H = 32 * 24

    rt = RivTree(g, irf_fn="muskingum", param_df=pdf, route_src_reach=True).to(DEVICE)
    router = LTIRouter(max_delay=TW_H, dt=1).to(DEVICE)
    T = 4 * TW_H
    x_ref = torch.rand(1, len(rt.nodes), T, device=DEVICE)
    y_ref = router(x_ref, rt, rt.params)[0]
    ref = dict(zip(rt.nodes.tolist(), y_ref))

    clusters, transfer, transition, nidx = _split(g, cut)
    gs = RivTreeCluster(clusters, transfer, irf_fn="muskingum", param_df=pdf,
                        route_src_reach=True, param_names=["x", "k"],
                        nodes_idx=nidx, transition_nodes=transition).to(DEVICE)
    staged = LTIStagedRouter(max_delay=TW_H, dt=1).to(DEVICE)
    order = gs.nodes.tolist()
    x_cl = torch.stack([x_ref[0, int(rt.nodes_idx.loc[nd])] for nd in order]).unsqueeze(0)
    y_cl = staged(x_cl, gs, gs.params)[0]

    for i, nd in enumerate(order):
        err = (y_cl[i] - ref[nd]).abs().max().item()
        scale = ref[nd].abs().max().item()
        assert err <= 1e-2 * max(scale, 1e-6), (
            f"node {nd} differs between hourly clustered and unsplit routing "
            f"by {err:.3e} (signal {scale:.3e})")
