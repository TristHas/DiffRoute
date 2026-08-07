import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import networkx as nx

from typing import Dict, List, Tuple
from tqdm.auto import tqdm

from ..irfs import IRF_PARAMS
from .utils import init_pre_indices, downstream_path_stats

def check_route_src_reach(route_src_reach):
    """Validate a uniform ``route_src_reach`` or a per-node mapping of them."""
    if hasattr(route_src_reach, "items"):
        return {k: bool(v) for k, v in route_src_reach.items()}
    return bool(route_src_reach)


def _src_reach_flags(route_src_reach, nodes, transition_nodes):
    """Per-node "is this node's own reach on the paths leaving it" flags.

    Transition nodes are forced to False whatever the global setting: the value
    handed to them is an upstream cluster's routed discharge, which is by
    construction already at that reach's outlet.
    """
    transition = set(transition_nodes or ())
    if isinstance(route_src_reach, bool):
        base = {n: route_src_reach for n in nodes}
    else:
        base = {n: bool(route_src_reach.get(n, True)) for n in nodes}
    return {n: base[n] and (n not in transition) for n in nodes}


class RivTree(nn.Module):
    """River network wrapper that stores IRF parameters per node."""
    def __init__(self, g, irf_fn,
                 route_src_reach=True,
                 param_df=None,
                 param_names=None,
                 nodes_idx=None,
                 transition_nodes=None,
                 output_reach=None):
        """Initialize river network metadata and parameter buffers.

        Args:
            g (networkx.DiGraph): Directed river network graph.
            irf_fn (str | None): Name of the IRF parameterization to use.
            route_src_reach (bool): Whether a node's runoff is routed through
                that node's OWN reach.
                True  (default) -- the runoff enters at the reach head, so
                    ``K(d, s)`` convolves the IRFs over ``s..d`` and the diagonal
                    ``K(s, s) = irf(s)`` is part of the kernel.
                False -- the runoff is already at the reach outlet, which is what
                    a catchment-outlet runoff model produces, so the source reach
                    is skipped: ``K(d, s)`` convolves ``succ(s)..d`` and the
                    diagonal is the identity, added by ``LTIRouter`` as
                    ``y = x + Kx``.
                Note the consequence for headwaters: with False their own reach
                parameters are never used and carry no gradient.
                A ``{node: bool}`` mapping sets it per node.
            param_df (pd.DataFrame | None): Optional parameter table.
            param_name (Iterable | None): Optional parameter names.
            nodes_idx (pd.Series | None): Precomputed node ordering.
            transition_nodes (Iterable | None): Nodes that carry an upstream
                cluster's routed discharge rather than local runoff. They are
                forced to route_src_reach=False, and they emit no output of
                their own -- the cluster they came from already reported it.
            output_reach (Iterable | None): Nodes the router should return; see
                ``set_output_reach``. ``None`` returns every node.
        """
        super().__init__()
        self.g = g
        self.nodes_idx = nodes_idx if nodes_idx is not None else init_node_idxs(g)
        self.route_src_reach = check_route_src_reach(route_src_reach)
        self.transition_nodes = set(transition_nodes or ())
        self.irf_fn = irf_fn

        labels = list(self.nodes_idx.index)
        flags = _src_reach_flags(self.route_src_reach, labels, self.transition_nodes)
        edges, path_cumsum, own_reach = init_pre_indices(g, self.nodes_idx,
                                                         route_src_reach=flags)

        # Diagonal weight for the residual in LTIRouter: a node contributes its
        # own input to its own output exactly when its reach is not traversed
        # (route_src_reach False) AND it is a real node, not a transition copy.
        emit = (~own_reach).float()
        if self.transition_nodes:
            trans = torch.tensor([n in self.transition_nodes for n in labels])
            emit = emit.masked_fill(trans, 0.0)
            self.register_buffer("transition", trans)
        else:
            self.register_buffer("transition", torch.zeros(len(labels), dtype=torch.bool))
        self.n_paths = int(path_cumsum[-1]) if len(path_cumsum) else 0
        self.uniform_src_reach = (True if bool(own_reach.all()) else
                                  False if not bool(own_reach.any()) else None)
        self.has_residual = bool((emit != 0).any())

        self.register_buffer("edges", edges)
        self.register_buffer("own_reach", own_reach)
        self.register_buffer("residual_weight", emit.view(-1, 1))
        self._full_path_cumsum = path_cumsum
        self.set_output_reach(output_reach)
        if irf_fn is not None:
            self.init_params(param_df, param_names)

    # ------------------------------------------------------------------
    def set_output_reach(self, output_reach=None):
        """Restrict which reaches the router returns.

        Paths that end at a reach nobody asked for are dropped from the closure
        AND from the convolution kernel, whose row dimension becomes
        ``len(output_reach)`` instead of the number of nodes. With a handful of
        gauges on a large network that is most of the work.

        What it does NOT change: the graph, the node ordering, and the runoff
        tensor the router expects, which stays one row per node in graph order.
        ``prefix_sum`` and the IRF transform therefore still run over every node;
        narrowing those is a separate problem.

        Args:
            output_reach: node labels to return, in the order wanted, or ``None``
                to return every node in graph order. Callable at any time.
        """
        n = len(self.nodes_idx)
        device = self.edges.device
        if output_reach is None:
            self.output_reach = None
            self.n_out = n
            self.full_output = True
            out_row = torch.arange(n, dtype=torch.int32)
            out_positions = torch.arange(n, dtype=torch.long)
            path_cumsum = self._full_path_cumsum
        else:
            labels = list(output_reach)
            if len(set(labels)) != len(labels):
                raise ValueError("output_reach contains duplicate nodes")
            # structural, so the label lookup happens once here on the host --
            # no per-call lookup, hence no device-side index needed
            positions = self.nodes_idx.loc[labels].to_numpy(dtype=np.int64)
            self.output_reach = labels
            self.n_out = len(labels)
            self.full_output = False
            out_row = torch.full((n,), -1, dtype=torch.int32)
            out_row[positions] = torch.arange(len(labels), dtype=torch.int32)
            out_positions = torch.from_numpy(positions)

            keep = dict(zip(self.nodes_idx.index, (out_row >= 0).numpy()))
            flags = dict(zip(self.nodes_idx.index, self.own_reach.cpu().numpy()))
            counts = downstream_path_stats(self.g, flags, keep)
            counts = np.fromiter((counts[k] for k in self.nodes_idx.index), dtype=np.int32)
            path_cumsum = torch.from_numpy(np.cumsum(counts)).int()

        self.register_buffer("out_row", out_row.to(device))
        self.register_buffer("out_positions", out_positions.to(device))
        self.register_buffer("path_cumsum", path_cumsum.to(device))
        self.register_buffer("residual_weight_out",
                             self.residual_weight[out_positions.to(device)])
        self.n_paths = int(path_cumsum[-1]) if len(path_cumsum) else 0
        return self

    def init_params(self, param_df, param_names):
        """Load impulse response parameters from a graph or dataframe.

        Args:
            param_df (pd.DataFrame | None): Optional parameter values keyed
                by node indices; if `None`, parameters are read from `self.g`.
        """
        if param_names is None:
            param_names = IRF_PARAMS[self.irf_fn]
        params = init_params_from_g(self.g, self.irf_fn, param_names, self.nodes_idx) \
                if param_df is None\
                else init_params_from_df(param_df, self.irf_fn, param_names, self.nodes_idx)            
        self.register_buffer("params", params)

    def __len__(self):
        return len(self.nodes_idx)

    @property
    def uniform(self) -> bool:
        """Whether every node shares the same ``route_src_reach``.

        The per-node form is the ``own_reach`` buffer, which is what the kernels
        consume: it is exactly "is the diagonal ``K(s, s)`` part of the sparse
        kernel".
        """
        return self.uniform_src_reach is not None

    @property
    def nodes(self):
        return self.nodes_idx.index.values

class RivTreeCluster(nn.Module):
    """Collection of river subgraphs with optional inter-cluster transfers."""
    def __init__(self, clusters_g, node_transfer,
                 irf_fn=None,
                 route_src_reach=True,
                 param_df=None,
                 param_names=None,
                 nodes_idx=None,
                 transition_nodes=None):
        """Assemble clustered river networks and transfer bookkeeping.

        Args:
            clusters_g (Sequence[networkx.DiGraph]): Clustered river graphs.
            node_transfer (Dict[int, List[Tuple[int, int, int]]] | None):
                Mapping describing inter-cluster transfers.
            irf_fn (str): Name of the IRF parameterization to use.
            route_src_reach (bool): See ``RivTree``. Applies to every real node;
                transition nodes are always False whatever this says, which is
                what lets a split network reproduce the unsplit one either way.
            param_df (pd.DataFrame | None): Optional parameter table.
            nodes_idx (Sequence[pd.Series] | None): Custom node orderings.
            transition_nodes (Dict[int, Iterable] | None): Per cluster, the copies
                of upstream breakpoint nodes that receive a transfer instead of
                local runoff (from ``define_schedule``). They are excluded from
                the user-facing node layout -- the cluster they were copied from
                already reports them.
        """
        super().__init__()
        if nodes_idx is None: nodes_idx = [None]*len(clusters_g)
        transition_nodes = transition_nodes or {}
        self.irf_fn = irf_fn
        self.route_src_reach = check_route_src_reach(route_src_reach)
        self.gs = nn.ModuleList([RivTree(g, irf_fn=irf_fn,
                                         route_src_reach=self.route_src_reach,
                                         param_df=param_df,
                                         param_names=param_names,
                                         nodes_idx=nodes_idx[i],
                                         transition_nodes=transition_nodes.get(i)) \
                                 for i,g in enumerate(tqdm(clusters_g))])
        self.node_transfer = node_transfer
        all_nodes = np.concatenate([g.nodes_idx.index.values for g in self.gs])
        self.nodes_idx = pd.Series(np.arange(len(all_nodes)),
                                   index=all_nodes)

        # Duplicate-free view. `nodes_idx` spans the internal layout, transition
        # copies included; `keep_pos` selects one row per real node, which is what
        # the router takes in and hands back.
        keep = torch.cat([g.transition for g in self.gs]).logical_not()
        self.register_buffer("keep_pos", torch.nonzero(keep, as_tuple=False).flatten())
        self.out_nodes_idx = pd.Series(np.arange(int(keep.sum())),
                                       index=all_nodes[keep.numpy()])
        self.has_transitions = bool((~keep).any())

        # Init coordinate indexors
        lengths = np.array([len(g) for g in self.gs], dtype=np.int64)
        starts  = np.zeros_like(lengths)
        starts[1:] = np.cumsum(lengths[:-1])
        ends    = starts + lengths
        self.node_ranges = np.stack([starts, ends], axis=1)  # shape [M, 2]
        
        # Init node transfers
        if node_transfer is not None:
            src_map, dst_map, tot = build_transfer_tables(
                node_transfer, dtype=torch.long
            )
            self.tot_transfer = tot
            self.src_transfer = BufferDict(src_map)
            self.dst_transfer = BufferDict(dst_map)
        else:
            self.tot_transfer = 0
            self.src_transfer = BufferDict({})
            self.dst_transfer = BufferDict({})

    def set_output_reach(self, output_reach=None):
        """Not supported on a clustered graph yet.

        A cluster's boundary node has to stay in its own cluster's output, since
        that value is what `node_transfer` hands downstream. Selecting outputs
        here therefore means keeping the requested nodes *plus* every transfer
        source, then hiding the latter again on the way out -- worth doing, but
        it is not the same bookkeeping as the single-graph case.
        """
        if output_reach is not None:
            raise NotImplementedError(
                "set_output_reach is not supported on RivTreeCluster: transfer "
                "sources must stay in their cluster's output. Route the graph "
                "unsplit, or open an issue if you need this.")
        return self

    def __len__(self):
        return len(self.gs)

    def __iter__(self):
        return iter(self.gs) 

    def __getitem__(self, idx):
        return self.gs[idx]

    @property
    def nodes(self):
        """Real nodes, one entry each, in router input/output order."""
        return self.out_nodes_idx.index.values

    @property
    def internal_nodes(self):
        """Every row of the internal layout, transition copies included."""
        return self.nodes_idx.index.values

    @property
    def params(self):
        return torch.cat([g.params for g in self.gs])

def init_node_idxs(g):
    """Derive a depth-first traversal ordering for nodes in the graph.

    Args:
        g (networkx.DiGraph): River network graph.

    Returns:
        pd.Series: Mapping from node ids to contiguous indices.
    """
    dfs_order = np.fromiter(nx.dfs_preorder_nodes(g), dtype=int)
    return pd.Series(np.arange(len(dfs_order)), index=dfs_order)

def get_node_idxs(g):
    """Return cached node indices or compute them from the graph.

    Args:
        g: Graph-like object optionally carrying a `nodes_idx` attribute.

    Returns:
        pd.Series: Mapping from node ids to contiguous indices.
    """
    if hasattr(g, "nodes_idx"): return g.nodes_idx
    else: return init_node_idxs(g)

def init_params_from_g(g, model_name, param_names, nodes_idx=None):
    """Extract IRF parameters stored on graph nodes.

    Args:
        g (networkx.DiGraph): River network with node attributes.
        model_name (str | None): Registered IRF identifier.
        nodes_idx (pd.Series | None): Optional node ordering.

    Returns:
        torch.Tensor | None: Parameters ordered by `nodes_idx`.
    """    
    nodes_idx = get_node_idxs(g) if nodes_idx is None else nodes_idx
    candidates = set(g.nodes[nodes_idx.index[0]])
    if not set(param_names).issubset(candidates):
                print(f"WARNING - init_params_from_df - param_names ({param_names}) not included in g.nodes attributes ({candidates})")

    params = torch.tensor([[g.nodes[n].get(p, 0) for p in param_names] \
                           for n in get_node_idxs(g).index])
    return params.float()

def init_params_from_df(param_df, model_name=None, param_names=None, nodes_idx=None):
    """Load IRF parameters from a DataFrame.

    Args:
        param_df (pd.DataFrame): Table of parameter values indexed by nodes.
        model_name (str | None): Registered IRF identifier.
        nodes_idx (pd.Series): Node ordering to align with parameters.

    Returns:
        torch.Tensor: Parameter tensor ordered by `nodes_idx`.
    """
    if not set(param_names).issubset(param_df.columns):
        print(f"WARNING - init_params_from_df - param_names ({param_names}) not included in param_df.columns ({param_df.columns})")
    params = param_df.loc[nodes_idx.index].reindex(columns=param_names, fill_value=0)
    return torch.from_numpy(params.values).float()

def read_params(g, model_name, nodes_idx):
    """Retrieve parameters from the graph or compute defaults.

    Args:
        g: Graph-like object that may store parameters.
        model_name (str): Registered IRF identifier.
        nodes_idx (pd.Series): Node ordering to align parameters.

    Returns:
        torch.Tensor | None: Parameter tensor if available.
    """
    if hasattr(g, "params"): return g.garams
    else: return init_params_from_g(g, model_name, nodes_idx)

def build_transfer_tables(
    node_transfer: Dict[int, List[Tuple[int, int, int]]],
    *,
    dtype: torch.dtype = torch.long,
    device: torch.device | str | None = None,
) -> Tuple[Dict[int, torch.Tensor], Dict[int, torch.Tensor], int]:
    """
    Build transfer maps in stacked form.
    Returns:
      src_tensor_map:  Dict[src_cluster, Tensor([2, N_src_edges])]
      dest_tensor_map: Dict[dst_cluster, Tensor([2, N_dst_edges])]
      tot_transfer:    total number of global transfers (N_total)
    """
    src_map: Dict[int, Tuple[List[int], List[int]]] = {}
    dest_buckets: Dict[int, Tuple[List[int], List[int]]] = {}

    next_global = 0
    for src_cluster, edges in node_transfer.items():
        s_bucket = src_map.setdefault(src_cluster, ([], []))
        s_local, s_gids = s_bucket

        for dest_cluster, src_idx, dest_idx in edges:
            s_local.append(src_idx)
            s_gids.append(next_global)

            d_bucket = dest_buckets.setdefault(dest_cluster, ([], []))
            d_local, d_gids = d_bucket
            d_local.append(dest_idx)
            d_gids.append(next_global)

            next_global += 1

    to_tensor = lambda seq: torch.as_tensor(seq, dtype=dtype, device=device)

    src_tensor_map: Dict[int, torch.Tensor] = {
        c: torch.stack((to_tensor(local), to_tensor(gids)), dim=0)
        for c, (local, gids) in src_map.items()
    }
    dest_tensor_map: Dict[int, torch.Tensor] = {
        c: torch.stack((to_tensor(local), to_tensor(gids)), dim=0)
        for c, (local, gids) in dest_buckets.items()
    }

    return src_tensor_map, dest_tensor_map, next_global

class BufferDict(nn.Module):
    def __init__(self, buffers=None, *, persistent: bool = True):
        super().__init__()
        self._name_map = {}
        if buffers:
            names = [self._as_name(k) for k in buffers]
            if len(set(names)) != len(names):
                dup = next(n for n in names if names.count(n) > 1)
                raise ValueError(f"Duplicate stringified key '{dup}'")
            for k, v in buffers.items():
                s = self._as_name(k)
                self._name_map[k] = s
                self.register_buffer(s, v, persistent=persistent)

    def __getitem__(self, key):
        if key in self._name_map:
            return getattr(self, self._name_map[key])
        s = key if isinstance(key, str) else self._as_name(key)
        if s not in self._buffers:  
            raise KeyError(key)
        return getattr(self, s)

    def __contains__(self, key) -> bool:
        if key in self._name_map:
            return True
        s = key if isinstance(key, str) else self._as_name(key)
        return s in self._buffers  

    @staticmethod
    def _as_name(key):
        s = str(key)
        if not s or "." in s:
            raise KeyError("Buffer name must be non-empty and contain no '.'")
        return s
