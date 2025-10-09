from collections import defaultdict, deque
from tqdm.auto import tqdm

import numpy as np
import pandas as pd
import networkx as nx

###
### The following are helpers for the segment_graph_into_blocks component
###
def cut_edge(G, u):
    """
    """
    down = list(G.successors(u))
    if down: 
        down = down[0]
        G.remove_edge(u, down)
        return down
    
def count_upstream_nodes(G, u, count_nodes):
    """
    """
    candidate = count_nodes[u]
    for v in G.predecessors(u): candidate += count_nodes[v]
    return candidate

def process_exact_cut(G, u, clusters, cluster_links, count_nodes):
    """
    """
    clusters[u] = list(nx.ancestors(G, u) | {u})
    down = cut_edge(G, u) 
    if down is not None: 
        cluster_links[u] = [(u, down)]
        n = 1
    else:
        n = 0
    
def process_over_cut(G, u, clusters, cluster_links, count_nodes, threshold):
    """
    """
    upstream = list(nx.ancestors(G, u) | {u})
    topo_order = list(nx.topological_sort(G.subgraph(upstream)))
    
    for n in topo_order: count_nodes[n]=1
    current_nodes = []
    
    for v in topo_order:
        count_nodes[v]=count_upstream_nodes(G, v, count_nodes)
        current_nodes.append(v)
        
        if (len(current_nodes) % threshold) == 0:
            
            clusters[v] = current_nodes
            
            out_links = []
            for n in current_nodes:
                for down in list(G.successors(n)):
                    if not (down in current_nodes):
                        G.remove_edge(n, down)
                        out_links.append((n, down))
            if out_links:
                cluster_links[v] = out_links            
            
            current_nodes = []
            
    for v in topo_order:
        if count_nodes[v]!=0:
            assert count_nodes[v]==len(nx.ancestors(G, v))+1

def process_under_cut(G, u, clusters):
    """
    """
    clusters[u] = list(nx.ancestors(G, u) | {u})

def segment_graph_into_blocks(G, threshold=16):
    """
    """
    topo_order = list(tqdm(nx.topological_sort(G)))
    count_nodes = {node: 1 for node in G.nodes()}
    clusters = {}      # Dictionnary of {cluster_id: subrgraph}
    cluster_links = {} # Disctionnary {cluster_id: [list of (input, output) node links between clusters]}
        
    for u in tqdm(topo_order, desc="Computing clusters"):
        upstream_nodes = count_upstream_nodes(G, u, count_nodes)
        count_nodes[u] = upstream_nodes
        if upstream_nodes == threshold: 
            process_exact_cut(G, u, clusters, cluster_links, count_nodes)
        elif upstream_nodes > threshold:
            process_over_cut(G, u, clusters, cluster_links, count_nodes, threshold)
        if (G.out_degree[u]==0) and (upstream_nodes != threshold):
            process_under_cut(G, u, clusters)
            
    return clusters, cluster_links, count_nodes

###
### The following are helpers for block-sized cluster grouping logic
###

def build_cluster_edges(clusters, cluster_links):
    """
    """
    cluster_map = {node:cluster_idx \
                   for cluster_idx,nodes in tqdm(clusters.items()) \
                   for node in nodes}
    
    edges = [] #defaultdict(list)
    for (src_cluster, node_edges) in tqdm(cluster_links.items()):
        for src_node, dst_node in node_edges:
            dst_cluster = cluster_map[dst_node]
            edges.append([src_cluster, dst_cluster, src_node, dst_node])
    return pd.DataFrame(edges, columns=["src_cluster", "dst_cluster", "src_node", "dst_node"])

def test_clusters(clusters, cluster_size, g):
    s = pd.Series({k:len(v) for k,v in clusters.items()})
    s = s.value_counts().sort_index()
    assert s.index.max()<=cluster_size
    assert (s.values*s.index.values).sum() == len(g)

def pack_small_clusters(clusters, threshold=16):
    """
    clusters: {cid -> [nodes]}
    threshold: target size to pack to (e.g., 16)
    Returns:
        new_clusters: {new_id -> [nodes]}
        cluster_idx_map: {original_cid -> new_id_of_grouped_cluster}
    """
    # --- prep
    size_of = {cid: len(nodes) for cid, nodes in clusters.items()}
    by_size = defaultdict(list)
    for cid, sz in size_of.items():
        if sz <= threshold:
            by_size[sz].append(cid)

    new_clusters = {}
    cluster_idx_map = {}

    # keep exact == threshold and any > threshold as-is
    for cid, sz in size_of.items():
        if sz >= threshold:
            new_clusters[cid] = clusters[cid]
            cluster_idx_map[cid] = cid

    next_id = (max(clusters) if clusters else 0) + 1

    # working pools for sizes strictly less than threshold
    avail = {s: deque(by_size.get(s, [])) for s in range(1, threshold)}
    leftovers = {s: [] for s in range(1, threshold)}

    # --- 1) pair complements (k, T-k)
    # only do k < T-k to avoid double counting; handle mid (T even) separately
    for k in range(1, (threshold // 2) + 1):
        j = threshold - k
        if k == j:  # only when threshold is even: pair (T/2, T/2)
            while len(avail[k]) >= 2:
                i1, i2 = avail[k].pop(), avail[k].pop()
                new_clusters[next_id] = clusters[i1] + clusters[i2]
                cluster_idx_map[i1] = next_id
                cluster_idx_map[i2] = next_id
                next_id += 1
        elif k < j:  # normal complements
            a, b = avail[k], avail[j]
            while a and b:
                i, q = a.pop(), b.pop()
                new_clusters[next_id] = clusters[i] + clusters[q]
                cluster_idx_map[i] = next_id
                cluster_idx_map[q] = next_id
                next_id += 1

    # --- 2) greedy fill-from-large with tiny exact remainder fill (capacity <= threshold-1)
    def try_fill(rem, counts):
        # counts: {size -> available_count}; return {size -> take} summing to rem or None
        counts = counts.copy()
        def dfs(rem, smax):
            if rem == 0:
                return {}
            upper = min(smax, rem)
            for s in range(upper, 0, -1):
                c = counts.get(s, 0)
                if c <= 0:
                    continue
                max_take = min(c, rem // s)
                for t in range(max_take, 0, -1):
                    counts[s] -= t
                    sub = dfs(rem - s * t, s)
                    if sub is not None:
                        sub[s] = sub.get(s, 0) + t
                        return sub
                    counts[s] += t
            return None
        return dfs(rem, threshold - 1)

    for s in range(threshold - 1, 0, -1):  # starters: big -> small
        while avail[s]:
            starter = avail[s].pop()
            rem = threshold - s
            used = try_fill(rem, {t: len(avail[t]) for t in range(1, threshold)})
            if used is None:
                leftovers[s].append(starter)
                continue
            ids = [starter]
            for t, c in used.items():
                for _ in range(c):
                    ids.append(avail[t].pop())
            # materialize new cluster
            new_clusters[next_id] = [n for cid in ids for n in clusters[cid]]
            for cid in ids:
                cluster_idx_map[cid] = next_id
            next_id += 1

    # --- 3) put all leftovers into a single cluster
    leftover_ids = [cid for ids in leftovers.values() for cid in ids]
    leftover_ids += [cid for s in range(1, threshold) for cid in avail[s]]
    if leftover_ids:
        new_id = next_id
        next_id += 1
        buf = []
        for cid in leftover_ids:
            buf.extend(clusters[cid])
            cluster_idx_map[cid] = new_id
        new_clusters[new_id] = buf

    return new_clusters, cluster_idx_map


def assemble_cluster_topologically(new_clusters, edges, thr=10_000, cluster_size=16):

    cg = nx.DiGraph()
    cg.add_nodes_from(list(new_clusters.keys()))
    cg.add_edges_from(edges[["src_cluster", "dst_cluster"]].values)
    
    topo_layers = list(nx.topological_generations(cg))
    
    clusters = []
    cluster_weight = []
    
    # Process each generation and subdivide it if the combined weight exceeds thr.
    for layer in topo_layers:
        # Convert layer (a set) to list and optionally sort by weight (heavier first).
        layer_nodes = list(layer)
        #layer_nodes.sort(key=lambda x: subgraph_weights.get(x, 0), reverse=True)
        
        current_cluster = []
        current_weight = 0
        
        for sub in layer_nodes:
            w = cluster_size
            # If adding this subgraph would exceed the threshold, flush the current cluster.
            if current_cluster and (current_weight + w > thr):
                clusters.append(current_cluster)
                cluster_weight.append(current_weight)
                current_cluster = [sub]
                current_weight = w
            else:
                current_cluster.append(sub)
                current_weight += w
        
        # Append any remaining subgraphs from this generation.
        if current_cluster:
            clusters.append(current_cluster)
            cluster_weight.append(current_weight)
            
    cluster_idx_map = {c:i for i,group in enumerate(clusters) for c in group}
    new_clusters = [np.concatenate([new_clusters[x] for x in v]) for v in clusters]

    edges["src_cluster"]=edges["src_cluster"].apply(lambda x: cluster_idx_map[x])
    edges["dst_cluster"]=edges["dst_cluster"].apply(lambda x: cluster_idx_map[x])
    edges = edges[edges["src_cluster"] != edges["dst_cluster"]]
    return new_clusters, edges

def define_block_size_aware_schedule(G, cluster_size, max_nodes=10_000):
    """
    """
    g = G.copy()
    clusters, cluster_links, count_nodes = segment_graph_into_blocks(g, cluster_size)
    test_clusters(clusters, cluster_size, g)
    edges = build_cluster_edges(clusters, cluster_links)
    new_clusters, cluster_idx_map = pack_small_clusters(clusters)
    
    edges["src_cluster"]=edges["src_cluster"].apply(lambda x: cluster_idx_map[x])
    edges["dst_cluster"]=edges["dst_cluster"].apply(lambda x: cluster_idx_map[x])
    edges = edges[edges["src_cluster"] != edges["dst_cluster"]]
    
    nc, ne = assemble_cluster_topologically(new_clusters, edges, 
                                            thr=max_nodes, 
                                            cluster_size=cluster_size)
    node_idxs = [pd.Series(np.arange(len(x)), index=x) for x in nc]
    
    ne["dst_node"]=ne.apply(lambda x:node_idxs[x["dst_cluster"]][x["dst_node"]], axis=1)
    ne["src_node"]=ne.apply(lambda x:node_idxs[x["src_cluster"]][x["src_node"]], axis=1)
    
    node_transfer = ne.set_index("src_cluster").apply(lambda x: x.tolist(), axis=1)#.to_dict()
    node_transfer = node_transfer.groupby(node_transfer.index).apply(list).to_dict()
    
    cluster_g = [g.subgraph(c) for c in nc]
    return cluster_g, node_transfer, node_idxs