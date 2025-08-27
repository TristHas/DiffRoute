import numpy as np
import pandas as pd
import networkx as nx
import torch

from ..utils import get_node_idxs

def init_pre_indices(g: nx.DiGraph,
                         node_idxs: pd.Series | None = None,
                         include_self: bool = False):
    """
    Parameters
    ----------
    g : nx.DiGraph
        Original graph whose nodes can be any hashable objects.
    node_idxs : pd.Series, optional
        Mapping node_label → integer index.  If None, DFS preorder is used
        (identical to previous behaviour).
    include_self : bool, default False
        Forwarded to `downstream_path_stats`.

    Returns
    -------
    edges        : torch.IntTensor  (shape = [n])
                   edges[i] = j  →  successor of node *i* in integer space,
                   or -1 if *i* is a sink.
    path_cumsum  : torch.IntTensor  (shape = [n])
    length_cumsum: torch.IntTensor  (shape = [n])
    """
    if node_idxs is None: node_idxs = get_node_idxs(g)
    n = len(node_idxs)
    edges_np = np.full(n, -1, dtype=np.int32)
    src_labels, dst_labels = zip(*g.edges)
    src_idx = node_idxs.loc[list(src_labels)].to_numpy(dtype=np.int32)
    dst_idx = node_idxs.loc[list(dst_labels)].to_numpy(dtype=np.int32)
    edges_np[src_idx] = dst_idx
    edges = torch.from_numpy(edges_np).int()

    count_paths, sum_lengths = downstream_path_stats(g, include_self)
    count_paths = np.fromiter((count_paths[n] for n in node_idxs.index), dtype=np.int32)
    sum_lengths = np.fromiter((sum_lengths[n] for n in node_idxs.index), dtype=np.int32)
    
    path_cumsum   = torch.from_numpy(np.cumsum(count_paths)).int()
    length_cumsum = torch.from_numpy(np.cumsum(sum_lengths)).int()

    return edges, path_cumsum, length_cumsum

def downstream_path_stats(g, include_self=True):
    """
    """
    init = 1 if include_self else 0
    update = 0 if include_self else 1
    count_paths = {node: init for node in g.nodes()}
    sum_lengths = {node: init for node in g.nodes()}

    topo_order = list(nx.topological_sort(g))
    for u in reversed(topo_order):
        for v in g.successors(u):
            count_paths[u] += update + count_paths[v]
            sum_lengths[u] += update + sum_lengths[v] + count_paths[v]
            
    return count_paths, sum_lengths