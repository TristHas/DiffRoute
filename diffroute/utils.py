from tqdm import tqdm_notebook as tqdm
import pandas as pd
import numpy as np
import networkx as nx

import torch
import torch.nn as nn

from diffroute.agg.index_precompute import init_pre_indices
from diffroute.irfs import IRF_PARAMS

def find_roots(g):
    out = pd.Series(dict(g.out_degree))
    return out[out==0].index

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
    nodes_idx = get_node_idxs(g) if nodes_idx is None else nodes_idx
    p_name = IRF_PARAMS[model_name]
    params = torch.tensor([[g.nodes[n][p] for p in p_name] for n in get_node_idxs(g).index])
    return params.float()

def init_params_from_df(param_df, model_name, nodes_idx=None):
    """ """
    p_name = IRF_PARAMS[model_name]
    params = torch.from_numpy(param_df.loc[nodes_idx.index, p_name].values)
    return params.float()

def read_params(g, model_name, nodes_idx):
    """ """
    if hasattr(g, "params"): return g.garams
    else: return init_params(g, model_name, nodes_idx)

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
        if self.irf_fn is None: return
        params = init_params_from_g(g, self.irf_fn, self.nodes_idx) if param_df is None else \
                 init_params_from_df(param_df, self.irf_fn, self.nodes_idx)
        self.register_buffer("params", params)

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

    def __len__(self):
        return len(self.gs)

    def __iter__(self):
        return iter(self.gs) 

    def __getitem__(self, idx):
        return self.gs[idx]