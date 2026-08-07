import numpy as np
import pandas as pd
import networkx as nx
import torch


def resolve_self_flags(g: nx.DiGraph, node_idxs: pd.Series, route_src_reach):
    """Normalise ``route_src_reach`` to one bool per node, in ``node_idxs`` order.

    Accepts a single bool (uniform), a mapping ``{node: bool}``, or an array-like
    already in ``node_idxs`` order. Per-node values exist because a clustered
    graph mixes conventions: a transition node carries an upstream cluster's
    routed discharge, which is already at that reach's outlet, while the rest of
    the cluster follows the global ``route_src_reach``.
    """
    labels = list(node_idxs.index)
    if isinstance(route_src_reach, (bool, np.bool_)):
        return np.full(len(labels), bool(route_src_reach), dtype=bool)
    if hasattr(route_src_reach, "get"):
        return np.fromiter((bool(route_src_reach.get(n, False)) for n in labels),
                           dtype=bool, count=len(labels))
    arr = np.asarray(route_src_reach, dtype=bool).reshape(-1)
    if arr.shape[0] != len(labels):
        raise ValueError(f"route_src_reach has {arr.shape[0]} entries for "
                         f"{len(labels)} nodes")
    return arr


def init_pre_indices(g: nx.DiGraph,
                     node_idxs: pd.Series,
                     route_src_reach=False):
    """
    Parameters
    ----------
    g : nx.DiGraph
        Original graph whose nodes can be any hashable objects.
    node_idxs : pd.Series, optional
        Mapping node_label → integer index.
    route_src_reach : bool | Mapping | array-like, default False
        Whether a node's own reach is on the paths leaving it, i.e. whether the
        diagonal (n, n) is emitted. See `resolve_self_flags`.

    Returns
    -------
    edges        : torch.IntTensor  (shape = [n])
                   edges[i] = j  →  successor of node *i* in integer space,
                   or -1 if *i* is a sink.
    path_cumsum  : torch.IntTensor  (shape = [n])
                   write offsets into the emitted (dest, src) rows.
    own_reach    : torch.BoolTensor (shape = [n])
                   the resolved per-node flags, in `node_idxs` order.
    """
    n = len(node_idxs)
    edges_np = np.full(n, -1, dtype=np.int32)
    if g.number_of_edges():                    # a cluster can be a single reach
        src_labels, dst_labels = zip(*g.edges)
        src_idx = node_idxs.loc[list(src_labels)].to_numpy(dtype=np.int32)
        dst_idx = node_idxs.loc[list(dst_labels)].to_numpy(dtype=np.int32)
        edges_np[src_idx] = dst_idx
    edges = torch.from_numpy(edges_np).int()

    own_np = resolve_self_flags(g, node_idxs, route_src_reach)
    flags = dict(zip(node_idxs.index, own_np))
    count_paths = downstream_path_stats(g, flags)
    count_paths = np.fromiter((count_paths[n_] for n_ in node_idxs.index), dtype=np.int32)

    path_cumsum = torch.from_numpy(np.cumsum(count_paths)).int()
    own_reach = torch.from_numpy(own_np.copy())

    return edges, path_cumsum, own_reach


def downstream_path_stats(g, route_src_reach):
    """Number of (dest) rows emitted per source node.

    ``count[u] = route_src_reach[u] + (number of nodes strictly downstream of u)``.
    Splitting it that way is what lets ``route_src_reach`` vary per node: the
    descendant count does not depend on the flags at all.

    Parameters
    ----------
    route_src_reach : bool | Mapping[node, bool]
    """
    if isinstance(route_src_reach, (bool, np.bool_)):
        route_src_reach = {node: bool(route_src_reach) for node in g.nodes()}

    n_desc = {node: 0 for node in g.nodes()}
    for u in reversed(list(nx.topological_sort(g))):
        n_desc[u] = sum(1 + n_desc[v] for v in g.successors(u))

    return {u: int(bool(route_src_reach[u])) + n_desc[u] for u in g.nodes()}
