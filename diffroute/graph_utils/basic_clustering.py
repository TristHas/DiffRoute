from collections import defaultdict
from tqdm.notebook import tqdm

import pandas as pd
import networkx as nx

from ..structs import get_node_idxs

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
    
def segment_graph_by_breakpoints(G, add_edge=False):
    """
    """
    G_mod = G.copy()
    removed_edges = []  # To store edges removed from breakpoints

    print("Removing edges...")
    for node in tqdm(list(G_mod.nodes())):
        if G_mod.nodes[node].get("breakpoint", False):
            succs = list(G_mod.successors(node))
            if succs:
                succ = succs[0]
                G_mod.remove_edge(node, succ)
                removed_edges.append((node, succ))

    print("Segment graph into connected components....")
    clusters_components = list(nx.weakly_connected_components(G_mod))

    print("Build subgraphs for each cluster and node-cluster map...")
    cluster_subgraphs = {}
    node_to_cluster = {}
    for i, comp in enumerate(tqdm(clusters_components)):
        cluster_subgraphs[i] = G_mod.subgraph(comp).copy()
        for n in comp: node_to_cluster[n] = i

    print("Establish dependencies between clusters...")
    dependencies = []
    for u, v in tqdm(removed_edges):
        cluster_u = node_to_cluster.get(u)
        cluster_v = node_to_cluster.get(v)
        if cluster_u is not None and cluster_v is not None and cluster_u != cluster_v:
            dependencies.append((cluster_u, cluster_v))

        if add_edge:
            # Add edge to cluster_v:
            assert (not u in cluster_subgraphs[cluster_v].nodes) and (not v in cluster_subgraphs[cluster_u].nodes)
            assert (u in cluster_subgraphs[cluster_u].nodes) and (v in cluster_subgraphs[cluster_v].nodes)
            cluster_subgraphs[cluster_v].add_edge(u, v)
            # Copy attributes
            for key,val in cluster_subgraphs[cluster_u].nodes[u].items():
                if key!="breakpoint":
                    cluster_subgraphs[cluster_v].nodes[u][key]=val
        #assert (u in cluster_subgraphs[cluster_v].nodes) and (not v in cluster_subgraphs[cluster_u].nodes)
    return cluster_subgraphs, dependencies, removed_edges



def group_subraphs_to_cluster_sequence(cluster_subgraphs, dependencies, edges, subgraph_weights, thr):
    """
    """
    print("Initialize dependencies...")
    # Compute weights for each subgraph
    subgraph_weights = subgraph_weights.sort_values(ascending=False)

    # Build dependency graph from given dependencies.
    dep_graph = nx.DiGraph()
    dep_graph.add_edges_from(dependencies)

    # Compute topological generations (layers). Each layer contains subgraphs that are independent 
    # (i.e. none in the same layer depends on another).
    topo_layers = list(nx.topological_generations(dep_graph))
    
    clusters = []
    cluster_weight = []
    
    # Process each generation and subdivide it if the combined weight exceeds thr.
    for layer in topo_layers:
        # Convert layer (a set) to list and optionally sort by weight (heavier first).
        layer_nodes = list(layer)
        layer_nodes.sort(key=lambda x: subgraph_weights.get(x, 0), reverse=True)
        
        current_cluster = []
        current_weight = 0
        
        for sub in layer_nodes:
            w = subgraph_weights.get(sub, 0)
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

    # Mark subgraphs already assigned to clusters.
    done = set().union(*[set(cluster) for cluster in clusters]) if clusters else set()

    print("Associate clusters for remaining subgraphs...")
    # For any subgraph not present in the dependency graph, add it to an existing cluster if it fits,
    # or create a new cluster otherwise.
    for subg, weight in tqdm(subgraph_weights.items()):
        if subg not in done:
            added = False
            # Try to add to the cluster with the smallest weight first.
            sorted_indices = sorted(range(len(cluster_weight)), key=lambda i: cluster_weight[i])
            for i in sorted_indices:
                if cluster_weight[i] + weight < thr:
                    clusters[i].append(subg)
                    cluster_weight[i] += weight
                    done.add(subg)
                    added = True
                    break
            if not added:
                clusters.append([subg])
                cluster_weight.append(weight)
                done.add(subg)

    print("Merging graphs...")
    # Merge subgraphs within each cluster into a single graph.
    clusters_g = [nx.compose_all([cluster_subgraphs[x] for x in cluster]) for cluster in tqdm(clusters)]
    print("Computing merged graphs node idxs...")
    node_idxs = [get_node_idxs(g) for g in tqdm(clusters_g)]
    
    # Build a mapping from subgraph id to its cluster index for matching breakpoint nodes.
    cluster_map = {}
    for idx, cluster in enumerate(clusters):
        for sub in cluster:
            cluster_map[sub] = idx

    print("Match breakpoint nodes across clusters...")
    node_transfer = defaultdict(list)

    for (start, end), (u, v) in zip(dependencies, edges):
        # Use cluster_map to get the cluster indices.
        if start not in cluster_map or end not in cluster_map:
            continue
        start_cluster_idx = cluster_map[start]
        end_cluster_idx = cluster_map[end]
        start_edge_idx = node_idxs[start_cluster_idx].loc[u]
        end_edge_index = node_idxs[end_cluster_idx].loc[v]  # Adjust u or v per your scheme
        node_transfer[start_cluster_idx].append((end_cluster_idx, start_edge_idx, end_edge_index))
    
    return clusters_g, node_transfer

def define_schedule(G, plength_thr=10**5, node_thr=10**4, runoff_to_output=False):
    print("#### Upstream stats computations ... ####")
    _, _, _, sum_all_lengths = upstream_path_stats_w_breakpoints(G, plength_thr)
    print("#### Segmentation into subgraphs ... ####")
    cluster_subgraphs, dependencies, edges = segment_graph_by_breakpoints(G)
    subgraph_weights = pd.Series({k: len(v)for k, v in cluster_subgraphs.items()})
    print("#### Grouping subgraphs to cluster and infering dependencies ... ####")
    clusters_g, node_transfer = group_subraphs_to_cluster_sequence(cluster_subgraphs, dependencies, 
                                                                   edges, subgraph_weights, thr=node_thr)
    #print("#### Cluster Annotations ... ####")
    #for g in clusters_g: annotate_downstream_path_stats(g, include_self=not runoff_to_output)
    return clusters_g, node_transfer