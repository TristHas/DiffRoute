import numpy as np
import pandas as pd
import networkx as nx
import torch
import triton
import triton.language as tl

from ..utils import get_node_idxs, downstream_path_stats, annotate_downstream_path_stats

def precompute_indices_baseline(g, node_idxs=None, include_self=False):
    """
    """
    if node_idxs is None: node_idxs=get_node_idxs(g)
    g = nx.relabel_nodes(g, node_idxs.to_dict())

    start_idx = None if include_self else 1
    condition_index = 0 if include_self else 1
    start, end, path = list(zip(*[[start, dest, pat[start_idx:]] \
                             for start, paths in (nx.all_pairs_dijkstra_path(g)) \
                             for dest, pat in paths.items() if len(pat)>condition_index]))
    x = [[i]*len(p) for i,p in enumerate(path)]
    
    path_idx = torch.from_numpy(np.concatenate(x))
    path_node = torch.from_numpy(np.concatenate(path))
    node_coords = torch.stack([torch.tensor(end), torch.tensor(start)], dim=1)
    return path_idx, path_node, node_coords

def precompute_indices_cpu(g, node_idxs=None, include_self=False):
    """
        CPU optimized version of the slow baseline
    """
    if node_idxs is None: node_idxs = get_node_idxs(g)
    g = nx.relabel_nodes(g, node_idxs.to_dict())
    topo_order = list(nx.topological_sort(g.reverse(copy=True)))

    # Build full paths from the root for each node
    paths = {}
    for node in topo_order:
        child = list(g.successors(node))
        assert len(child) <= 1, "Each node should have at most one successor in a tree."
        if not child: paths[node] = [node]
        else: paths[node] = paths[child[0]] + [node]

    # Build all the pathes from original nodes to all nodes on the way to the root
    # This organization of the computation is done to avoid costly long lists and instead slice existing allocated arrays.
    subpaths_tuples = []
    for node, full_path in paths.items():
        full_path_arr = np.array(full_path, dtype=np.int32)
        L = len(full_path_arr)

        for i in range(L):
            destination = full_path_arr[i]
            if include_self: subpath = full_path_arr[i:]
            else: subpath = full_path_arr[i:-1]  # Skip the last node (the source) 
            if len(subpath) > 0: subpaths_tuples.append((destination, node, subpath))

    # Sort all subpaths by (source, destination) – that is, by (second, first)
    subpaths_tuples.sort(key=lambda x: (x[1], x[0]))
    sorted_sources      = [t[0] for t in subpaths_tuples]  # originally 'destination'
    sorted_destinations = [t[1] for t in subpaths_tuples]  # originally 'source'
    sorted_subpaths     = [t[2] for t in subpaths_tuples]
    
    path_node = torch.from_numpy(np.concatenate(sorted_subpaths))
    node_coords = torch.stack([
        torch.tensor(sorted_sources,      dtype=torch.int64),
        torch.tensor(sorted_destinations, dtype=torch.int64)
    ], dim=1)
    
    # Build path_idx by repeating each subpath index subpath_length times
    subpath_lengths = np.array([p.shape[0] for p in sorted_subpaths], dtype=np.int64)
    indices = np.arange(len(sorted_subpaths), dtype=np.int64)
    path_idx_np = np.repeat(indices, subpath_lengths)
    path_idx = torch.from_numpy(path_idx_np)
    
    return path_idx, path_node, node_coords


@triton.jit
def precompute_kernel(
        path_idx_ptr,       # pointer to int64 output, shape [total_subpath_elements]
        path_node_ptr,      # pointer to int32 output, shape [total_subpath_elements]
        node_coords_ptr,    # pointer to int64 output, shape [total_subpaths, 2]
        edges_ptr,          # pointer to int32 array of parent pointers, shape [n_nodes]
        cumsum_paths_ptr,   # pointer to int64 cumulative subpath counts per node
        cumsum_lengths_ptr, # pointer to int64 cumulative subpath lengths per node
        n_nodes,
        include_self: tl.constexpr
    ):
    pid = tl.program_id(0)
    if pid >= n_nodes:
        return
    start_node = pid

    # Offsets for writing out subpaths for this start node.
    subpaths_base  = tl.where(start_node > 0, tl.load(cumsum_paths_ptr   + (start_node - 1)), 0)
    path_node_base = tl.where(start_node > 0, tl.load(cumsum_lengths_ptr + (start_node - 1)), 0)

    path_counter = 0  # number of subpaths written for start_node
    global_offset = 0  # total number of nodes written for start_node

    # For include_self, we want to include the trivial (self) path.
    # Otherwise, start with the child of start_node.
    if include_self:
        dest = start_node
    else:
        dest = tl.load(edges_ptr + start_node)

    # Outer loop: for each destination along the unique chain.
    while dest != -1:
        subpath_index = subpaths_base + path_counter
        tl.store(node_coords_ptr + subpath_index * 2 + 0, dest)
        tl.store(node_coords_ptr + subpath_index * 2 + 1, start_node)

        # Inner loop: build the stored subpath.
        # For include_self=True, write the entire path (starting at start_node).
        # For include_self=False, skip the trivial start and begin with its child.
        if include_self:
            path = start_node
        else:
            path = tl.load(edges_ptr + start_node)
        local_length = 0
        done = False
        while not done:
            curr_offset = path_node_base + global_offset + local_length
            tl.store(path_node_ptr + curr_offset, path)
            tl.store(path_idx_ptr + curr_offset, subpath_index)
            local_length += 1
            if path == dest:
                done = True
            else:
                path = tl.load(edges_ptr + path)
        global_offset += local_length
        path_counter += 1
        dest = tl.load(edges_ptr + dest)

def read_edges(g, node_idxs=None):
    node_idxs = get_node_idxs(g) if node_idxs is None else node_idxs
    edges = np.array(g.edges)
    edges = torch.from_numpy(pd.Series(node_idxs.loc[edges[:,1]].values, 
                                       index=node_idxs.loc[edges[:,0]].values).reindex(node_idxs.values).fillna(-1).astype(int).values).int()    
    return edges

def read_downstream_path_stats(g, node_idxs=None):
    
    index = np.fromiter(g.nodes, dtype=int)
    index_int = get_node_idxs(g)[index]
    
    count_path = np.fromiter((g.nodes[n]["count_paths"] for n in index), dtype=int)
    count_path = pd.Series(count_path, index=index_int).sort_index().cumsum()
    count_path = torch.from_numpy(count_path.values).int()
    
    sum_lengths = np.fromiter((g.nodes[n]["sum_lengths"] for n in index), dtype=int)
    sum_lengths = pd.Series(sum_lengths, index=index_int).sort_index().cumsum()  
    sum_lengths = torch.from_numpy(sum_lengths.values).int()
    
    return count_path, sum_lengths

def read_pre_indices(g, node_idxs=None):
    node_idxs = get_node_idxs(g) if node_idxs is None else node_idxs
    edges = read_edges(g, node_idxs)
    path_cumsum, length_cumsum = read_downstream_path_stats(g, node_idxs)
    return edges, path_cumsum, length_cumsum

def init_pre_indices(g, node_idxs=None, include_self=False):
    node_idxs = get_node_idxs(g) if node_idxs is None else node_idxs
    g = nx.relabel_nodes(g, node_idxs)
    edges = np.array(g.edges)
    edges = torch.from_numpy(
        pd.Series(edges[:,1], edges[:,0])
        .reindex(node_idxs.values).fillna(-1).astype(int).values
    ).int()
    count_paths, sum_lengths = downstream_path_stats(g, include_self)
    path_cumsum = torch.from_numpy(count_paths.cumsum().values).int()
    length_cumsum = torch.from_numpy(sum_lengths.cumsum().values).int()
    return edges, path_cumsum, length_cumsum

def generate_indices(edges, path_cumsum, length_cumsum, include_self, device):
    n_nodes = edges.shape[0]
    total_subpaths = int(path_cumsum[-1].item()) if n_nodes > 0 else 0
    total_elements = int(length_cumsum[-1].item()) if n_nodes > 0 else 0
    # Allocate output buffers on GPU.
    path_idx_out = torch.empty(total_elements, dtype=torch.int64, device=device)
    path_node_out = torch.empty(total_elements, dtype=torch.int32, device=device)
    node_coords_out = torch.empty((total_subpaths, 2), dtype=torch.int64, device=device)
    # Transfer inputs to GPU.
    edges_gpu = edges.to(device).int()
    path_cumsum_gpu = path_cumsum.to(device).int()
    length_cumsum_gpu = length_cumsum.to(device).int()
    # Launch the Triton kernel.
    grid = (n_nodes,)
    with torch.cuda.device(device):
        precompute_kernel[grid](path_idx_out, path_node_out, node_coords_out,
                                edges_gpu, path_cumsum_gpu, length_cumsum_gpu,
                                n_nodes, include_self)
    return path_idx_out, path_node_out, node_coords_out

def precompute_indices_gpu(g, node_idxs=None, include_self=False, device="cuda:0", read_indices=False):
    if read_indices:
        edges, path_cumsum, length_cumsum = read_pre_indices(g, node_idxs)
    else:
        edges, path_cumsum, length_cumsum = init_pre_indices(g, node_idxs, include_self)
    path_idx_out, path_node_out, node_coords_out = generate_indices(edges, path_cumsum, length_cumsum, 
                                                                    include_self, device)
    return path_idx_out, path_node_out, node_coords_out
    
INDEX_PRECOMPUTE = {
    "djikstra": precompute_indices_baseline,
    "cpu": precompute_indices_cpu,
    "gpu": precompute_indices_gpu
}

def test_pre_indices_eq(include_self=True):
    annotate_downstream_path_stats(g, include_self=include_self)
    e,p,l = init_pre_indices(g, include_self=include_self)
    e1,p1,l1 = read_pre_indices(g)
    
    assert torch.equal(e, e1)
    assert torch.equal(p, p1)
    assert torch.equal(l, l1)
    print("Test Passed")