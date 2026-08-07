"""Routing SEMANTICS of ``route_src_reach`` — behaviour only, no internals.

Convention under test.  For ``edges[i] = succ(i)`` (-1 at outlets), the routing
kernel entry ``K(d, s)`` is the convolution of the reach IRFs along ``s -> d``:

  route_src_reach=True    runoff enters at the HEAD of its own reach
                            K(d, s) = conv of irf over  s..d   (both ends in)
                            K(s, s) = irf(s)

  route_src_reach=False   runoff is already at the OUTLET of its own reach
                            K(d, s) = conv of irf over  succ(s)..d  (source OUT)
                            K(s, s) = delta   (LTIRouter adds ``y = x + Kx``)

These tests are deliberately independent of how the closure is computed: the
existing ``tests/test_backward.py`` only checks the Triton kernels against a
reference that encodes the same formula, so it cannot see a semantics error.

Intended home: DiffRoute/tests/test_semantics.py
"""
import networkx as nx
import numpy as np
import pandas as pd
import pytest
import torch

from diffroute import RivTree
from diffroute.router import LTIRouter
from diffroute.structs.riv_graphs import init_node_idxs
from diffroute.structs.utils import downstream_path_stats

DEVICE = "cuda:0"
TW = 32                      # max_delay / time window, in steps (dt = 1)


# ---------------------------------------------------------------- fixtures
def chain(n=3):
    return nx.DiGraph([(i, i + 1) for i in range(n - 1)])


def confluence():
    """two headwaters 0,1 -> junction 2 -> outlet 3"""
    return nx.DiGraph([(0, 2), (1, 2), (2, 3)])


def random_tree(n=12, seed=0):
    """Random river tree: every node drains to exactly one higher-indexed node."""
    rng = np.random.default_rng(seed)
    g = nx.DiGraph()
    g.add_nodes_from(range(n))
    for i in range(n - 1):
        g.add_edge(i, int(rng.integers(i + 1, n)))
    return g


def _tree(g, delays, route_src, irf="pure_lag"):
    pdf = pd.DataFrame({"delay": [float(delays[n]) for n in sorted(g.nodes)]},
                       index=sorted(g.nodes))
    rt = RivTree(g, irf_fn=irf, param_df=pdf, route_src_reach=route_src,
                 param_names=["delay"]).to(DEVICE)
    pos = dict(zip(rt.nodes_idx.index.tolist(), rt.nodes_idx.values.tolist()))
    return rt, pos, LTIRouter(max_delay=TW, dt=1).to(DEVICE)


def _impulse_response(router, rt, pos, src, n_nodes, T=TW):
    x = torch.zeros(1, n_nodes, T, device=DEVICE)
    x[0, pos[src], 0] = 1.0
    return router(x, rt, rt.params)[0].detach()


def _expected_nodes(g, s, d, route_src):
    """Reaches whose IRF must appear in K(d, s)."""
    if s == d:
        return [s] if route_src else []
    path = nx.shortest_path(g, s, d)
    return path if route_src else path[1:]


# ------------------------------------------------- 1. path composition
@pytest.mark.parametrize("route_src", [True, False])
@pytest.mark.parametrize("graph,delays", [
    (chain(3),      {0: 1, 1: 2, 2: 4}),
    (chain(4),      {0: 1, 1: 2, 2: 4, 3: 8}),
    (confluence(),  {0: 1, 1: 2, 2: 4, 3: 8}),
    (random_tree(), {i: 1 + (i % 3) for i in range(12)}),
])
def test_path_composition(graph, delays, route_src):
    """A pure_lag path IRF is a delta at the SUM of the delays on the path.

    So the arrival lag reads off exactly which reach IRFs the kernel used.
    """
    rt, pos, router = _tree(graph, delays, route_src)
    n = graph.number_of_nodes()
    for s in sorted(graph.nodes):
        y = _impulse_response(router, rt, pos, s, n)
        for d in sorted(graph.nodes):
            if s != d and not nx.has_path(graph, s, d):
                continue
            v = y[pos[d]].cpu().numpy()
            if v.max() <= 1e-4:
                pytest.fail(f"no signal {s}->{d} (route_src={route_src})")
            want = sum(delays[k] for k in _expected_nodes(graph, s, d, route_src))
            assert int(v.argmax()) == want, (
                f"route_src={route_src}: {s}->{d} arrived at lag "
                f"{int(v.argmax())}, expected {want} "
                f"(reaches {_expected_nodes(graph, s, d, route_src)})")


def test_offdiag_is_not_ondiag():
    """Guard the guard: the two modes must actually differ off the diagonal.

    Without this, a fix that leaves the source IRF in place would still pass
    ``test_path_composition[True]`` and silently make [False] identical to it.
    """
    delays = {0: 1, 1: 2, 2: 4}
    got = {}
    for diag in (True, False):
        rt, pos, router = _tree(chain(3), delays, diag)
        y = _impulse_response(router, rt, pos, 0, 3)
        got[diag] = int(y[pos[2]].cpu().numpy().argmax())
    assert got[True] == 7 and got[False] == 6, got


# ------------------------------------------------- 2. headwater gradient
@pytest.mark.parametrize("route_src,expect_zero", [(True, False), (False, True)])
def test_headwater_gradient(route_src, expect_zero):
    """With the diagonal off, a headwater's OWN reach is never traversed, so the
    loss cannot depend on its routing parameter."""
    g = chain(3)
    delays = {0: 1.5, 1: 2.5, 2: 3.5}          # non-integer: smooth in delay
    rt, pos, router = _tree(g, delays, route_src)
    P = rt.params.detach().clone().requires_grad_(True)
    torch.manual_seed(0)
    x = torch.rand(1, 3, TW, device=DEVICE)
    router(x, rt, P)[0, pos[2]].sum().backward()
    gr = P.grad[:, 0].abs()
    head = float(gr[pos[0]])
    if expect_zero:
        assert head == 0.0, f"headwater gradient {head:.3e}, expected exactly 0"
    else:
        assert head > 0.0, "headwater gradient vanished with the diagonal on"
    assert float(gr[pos[1]]) > 0.0, "interior reach must always carry gradient"


# ------------------------------------------------- 3. cross-mode equivalence
@pytest.mark.parametrize("graph,delays", [
    (chain(4),      {0: 1, 1: 2, 2: 3, 3: 4}),
    (confluence(),  {0: 2, 1: 1, 2: 3, 3: 2}),
    (random_tree(), {i: 1 + (i % 3) for i in range(12)}),
])
def test_cross_mode_equivalence(graph, delays):
    """route_True(x) == route_False(irf_i * x_i).

    Exact identity:  irf_j x_j + sum_{i<j} (prod_{succ(i)..j}) irf_i x_i
                   = sum_{i<=j} (prod_{i..j}) x_i.
    Ties the two modes together; the only way to satisfy it is for the False
    kernel to omit exactly one copy of the source IRF.
    """
    n = graph.number_of_nodes()
    T = 4 * TW
    torch.manual_seed(0)
    x = torch.rand(1, n, T, device=DEVICE)

    rt_t, pos, router = _tree(graph, delays, True)
    rt_f, pos_f, _ = _tree(graph, delays, False)
    assert pos == pos_f                              # same node ordering

    # pre-route each node's input through its OWN reach (integer pure_lag = shift)
    xs = torch.zeros_like(x)
    for node in sorted(graph.nodes):
        k = int(delays[node])
        xs[0, pos[node], k:] = x[0, pos[node], : T - k] if k else x[0, pos[node]]

    a = router(x,  rt_t, rt_t.params)
    b = router(xs, rt_f, rt_f.params)
    rel = (a - b).abs().max().item() / a.abs().max().item()
    # The log/exp round trip in ``log_transitive_closure`` is float32, which puts
    # a ~8e-4 relative floor under ANY comparison here (measured against the exact
    # integer-shift answer, in BOTH modes). A source-IRF error shows up at ~5e-1,
    # so 3e-3 leaves ~3x headroom over the floor and ~170x margin below a failure.
    assert rel < 3e-3, f"cross-mode identity broken: max relative err {rel:.3e}"


# ------------------------------------------------- 4. enumeration guard
@pytest.mark.parametrize("route_src", [True, False])
def test_enumeration(route_src):
    """The diagonal is present iff route_src_reach, and the number of emitted
    (dest, src) pairs matches ``downstream_path_stats``."""
    g = random_tree()
    delays = {i: 1 + (i % 3) for i in range(g.number_of_nodes())}
    rt, pos, _ = _tree(g, delays, route_src)
    counts = downstream_path_stats(g, route_src)
    assert int(rt.path_cumsum[-1]) == sum(counts.values())
    # rebuild the (dest, src) set the aggregator will emit
    e = rt.edges.tolist()
    pairs = set()
    for s in range(len(e)):
        d = s if route_src else e[s]
        while d != -1:
            pairs.add((d, s))
            d = e[d]
    n_self = sum(1 for (d, s) in pairs if d == s)
    assert n_self == (len(e) if route_src else 0)
    assert len(pairs) == int(rt.path_cumsum[-1])
