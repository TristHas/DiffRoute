import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import networkx as nx

from typing import Dict, List, Tuple
from tqdm.auto import tqdm

from ..irfs import IRF_PARAMS
from .utils import init_pre_indices

class RivTree(nn.Module):
    def __init__(self, g, 
                 irf_fn=None, 
                 include_index_diag=True,
                 param_df=None):
        """ """
        super().__init__()
        self.g = g
        self.nodes_idx = init_node_idxs(g)
        self.include_index_diag = include_index_diag
        self.irf_fn = irf_fn
        
        edges, path_cumsum, _ = init_pre_indices(g, self.nodes_idx, 
                                                 include_self=include_index_diag)

        self.register_buffer("edges", edges)
        self.register_buffer("path_cumsum", path_cumsum)
        self.init_params(param_df)

    def init_params(self, param_df):
        """ """
        params = init_params_from_g(self.g, self.irf_fn, self.nodes_idx) if param_df is None\
            else init_params_from_df(param_df, self.irf_fn, self.nodes_idx)            
        self.register_buffer("params", params)

    def __len__(self):
        return len(self.nodes_idx)

    @property
    def nodes(self):
        return self.nodes_idx.index.values

class RivTreeCluster(nn.Module):
    def __init__(self, clusters_g, node_transfer, 
                 irf_fn=None, 
                 include_index_diag=True,
                 param_df=None):
        super().__init__()
        self.gs = nn.ModuleList([RivTree(g, irf_fn=irf_fn,
                                         include_index_diag=include_index_diag,
                                         param_df=param_df) \
                                 for g in tqdm(clusters_g)])
        self.node_transfer = node_transfer
        all_nodes = np.concatenate([g.nodes_idx.index.values for g in self.gs])
        self.nodes_idx = pd.Series(np.arange(len(all_nodes)),
                                   index=all_nodes)

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

    def __len__(self):
        return len(self.gs)

    def __iter__(self):
        return iter(self.gs) 

    def __getitem__(self, idx):
        return self.gs[idx]

    @property
    def nodes(self):
        return self.nodes_idx.index.values

def init_node_idxs(g):
    """ """
    dfs_order = np.fromiter(nx.dfs_preorder_nodes(g), dtype=int)
    return pd.Series(np.arange(len(dfs_order)), index=dfs_order)

def get_node_idxs(g):
    """ """
    if hasattr(g, "nodes_idx"): return g.nodes_idx
    else: return init_node_idxs(g)

def init_params_from_g(g, model_name, nodes_idx=None):
    """ """
    if model_name is None: return
    nodes_idx = get_node_idxs(g) if nodes_idx is None else nodes_idx
    p_name = IRF_PARAMS[model_name]
    params = torch.tensor([[g.nodes[n][p] for p in p_name] for n in get_node_idxs(g).index])
    return params.float()

def init_params_from_df(param_df, model_name=None, nodes_idx=None):
    """ 
        
    """
    if model_name is not None:
        p_name = IRF_PARAMS[model_name]
        params = torch.from_numpy(param_df.loc[nodes_idx.index, p_name].values).float()
    else:
        params = torch.from_numpy(param_df.loc[nodes_idx.index].values).float()
    return params

def read_params(g, model_name, nodes_idx):
    """ """
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