import torch
import pandas as pd
import numpy as np
import networkx as nx
from tqdm import tqdm_notebook as tqdm
    
def find_roots(g):
    out = pd.Series(dict(g.out_degree))
    return out[out==0].index

def get_node_idxs(g):
    dfs_order = np.fromiter(nx.dfs_preorder_nodes(g), dtype=int)
    return pd.Series(np.arange(len(dfs_order)), index=dfs_order)

def get_node_idxs_old(g):
    """
        Order nodes in depth-first fashion
    """
    dfs_order = [int(x) for x in nx.dfs_preorder_nodes(g)]
    return pd.Series(np.arange(len(dfs_order)), index=dfs_order)

def get_node_idxs_correct_slow(g):
    roots = find_roots(g)
    g_ = g.reverse()
    x = [np.fromiter(nx.dfs_preorder_nodes(g_, root), dtype=int) for root in roots]
    x = np.concatenate(x)
    return pd.Series(np.arange(len(x)), index=x)

def upstream_path_stats_w_breakpoints(G, threshold=10**9):
    """
    """
    topo_order = list(nx.topological_sort(G))
    count_paths     = {node: 1 for node in G.nodes()}
    max_length      = {node: 1 for node in G.nodes()}
    sum_lengths     = {node: 1 for node in G.nodes()}
    sum_all_lengths = {node: 0 for node in G.nodes()}
    for node in G.nodes(): G.nodes[node]['breakpoint'] = False
        
    for u in tqdm(topo_order, desc="Computing breakpoints"):
        for v in G.successors(u):
            candidate = sum_all_lengths[v] + (sum_all_lengths[u] + sum_lengths[u])
            if candidate > threshold:
                G.nodes[u]['breakpoint'] = True
                max_length[v]      = 1
                count_paths[v]     = 1
                sum_lengths[v]     = 1
                sum_all_lengths[v] = 0
                sum_all_lengths[u] += sum_lengths[u]
            else:
                max_length[v] = max(max_length[v], max_length[u] + 1)
                count_paths[v] += count_paths[u] 
                sum_lengths[v] += sum_lengths[u] + count_paths[u]
                sum_all_lengths[v] += sum_all_lengths[u] + sum_lengths[u]
    return max_length, count_paths, sum_lengths, sum_all_lengths

def annotate_downstream_path_stats(g, include_self=True):
    """
    """
    init = 1 if include_self else 0
    update = 0 if include_self else 1
    for n in g.nodes: g.nodes[n]["count_paths"]=init
    for n in g.nodes: g.nodes[n]["sum_lengths"]=init

    topo_order = list(nx.topological_sort(g))
    for u in reversed(topo_order):
        for v in g.successors(u):
            g.nodes[u]["count_paths"] += update + g.nodes[v]["count_paths"] 
            g.nodes[u]["sum_lengths"] += update + g.nodes[v]["sum_lengths"] \
                                                + g.nodes[v]["count_paths"]
            
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
            
    # Convert to pandas Series for later cumulative-sum computations.
    count_paths = pd.Series(count_paths).sort_index()
    sum_lengths = pd.Series(sum_lengths).sort_index()
    return count_paths, sum_lengths

def downstream_path_stats_new_bug(g, node_idxs, include_self=True):
    """
    """
    init = 1 if include_self else 0
    update = 0 if include_self else 1
    count_paths = np.zeros(len(node_idxs), dtype=int)
    sum_lengths = np.zeros(len(node_idxs), dtype=int)
    node_idxs = node_idxs.to_dict()
    topo_order = np.fromiter(nx.topological_sort(g), dtype=int)
    for u in topo_order[::-1]:
        u_ = node_idxs[u]
        for v in g.successors(u):
            v_ = node_idxs[v]
            count_paths[u_] += update + count_paths[v_]
            sum_lengths[u_] += update + sum_lengths[v_] + count_paths[v_]
    return count_paths, sum_lengths

def compute_kernel(g, delays, nodes_idx=None):
    """
    """
    aggregator = RoutingDelayAggregator(g, nodes_idx=nodes_idx)
    return aggregator.aggregate_discrete_delays(delays).to_dense()

def run_routing(g, delays, runoffs, device="cuda:0"):
    """
        Attrs:
            g
            delays
            runoffs
        Output:
            y
    """
    # Take care of ordering
    node_idxs = get_node_idxs()
    inputs = torch.from_numpy(runoffs[node_idxs.index].values.T).to(device)
    delays = torch.from_numpy(delays[node_idxs.index].values).to(device)
    
    with torch.no_grad():
        router = LinearRouter(g, delays)
        output = router(inputs[None]).squeeze().cpu().numpy()
    
    output = pd.DataFrame(output[node_idxs[runoffs.columns]].T, 
                          columns=inputs.columns, 
                          index=inputs.index)
    
    return output