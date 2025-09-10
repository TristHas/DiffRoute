from tqdm import tqdm_notebook as tqdm
import pandas as pd
import numpy as np
import networkx as nx

import torch
import torch.nn as nn
from torch.utils.dlpack import to_dlpack
import cudf
import cupy as cp

from diffroute.agg.index_precompute import init_pre_indices
from diffroute.irfs import IRF_PARAMS

class DataFrameTh(nn.Module):
    """
        Torch-friendly 2D table with pandas-like column/index labels.
        Internally stores `values` transposed (shape = [n_cols, n_rows])
        to match your original code.
    """
    def __init__(
        self,
        values: torch.Tensor,          
        columns,                       
        index,                         
        *,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
        make_contiguous: bool = True
    ):
        super().__init__()
        if not isinstance(values, torch.Tensor): raise TypeError("values must be a torch.Tensor")
        if dtype is not None and values.dtype != dtype: values = values.to(dtype)
        if device is not None: values = values.to(device)
        if values.ndim != 2: raise ValueError("values must be 2D (matrix)")

        n_cols, n_rows = values.shape
        columns = list(columns)
        index   = list(index)

        if len(columns) != n_cols: raise ValueError(f"len(columns)={len(columns)} != n_cols={n_cols}")
        if len(index) != n_rows: raise ValueError(f"len(index)={len(index)} != n_rows={n_rows}")

        self.map_inp = pd.Series(np.arange(n_cols, dtype=np.int64), index=pd.Index(columns, name="columns"))
        self.index   = pd.Series(np.arange(n_rows, dtype=np.int64), index=pd.Index(index, name="index"))
        self.register_buffer("values", values)

    @staticmethod
    def from_pandas(
            df: pd.DataFrame,
            *,
            device: torch.device | str | None = None,
            dtype: torch.dtype | None = None,
        ):
        """
        Build from a pandas DataFrame. Preserves df's index/columns.
        """
        arr = df.to_numpy(copy=False)
        if not isinstance(arr, np.ndarray): arr = np.asarray(arr)
        t = torch.from_numpy(arr).t()
        return DataFrameTh(
            t, df.columns, df.index,
            device=device, dtype=dtype,  
        )

    @property
    def device(self): return self.values.device

    def __getitem__(self, cols):
        return self.values[self.map_inp[cols].values]

    # --------- conversions ----------
    def to_pandas(self):
        """
            Return a pandas DataFrame on CPU with original labels.
        """
        v = self.values.T.detach().cpu().numpy()
        return pd.DataFrame(v, index=self.index.index.copy(), columns=self.map_inp.index.copy())

    def to_cudf(self, device_index: int | None = None):
        """
            Return a cuDF DataFrame allocated on the specified CUDA device.
            If `self.values` is already on CUDA and `device_index` is None, uses its device.
            If `self.values` is on CPU, you MUST provide `device_index`.
        """
        if device_index is None:
            if self.values.is_cuda:
                device_index = int(self.values.get_device())
            else:
                raise ValueError(
                    "self.values is on CPU. Please pass `device_index` to choose a CUDA device."
                )

        # Ensure torch tensor is on the correct CUDA device
        tgt_dev = torch.device(f"cuda:{device_index}")
        V = self.values
        if not V.is_cuda or V.get_device() != device_index:
            V = V.to(tgt_dev, non_blocking=True)

        # Convert to row-major [n_rows, n_cols] for DataFrame construction
        V_row_major = V.T.contiguous()

        # One-shot cupy array via DLPack (no device copy, zero-copy view)
        with cp.cuda.Device(device_index):
            cp_mat = cp.fromDlpack(to_dlpack(V_row_major))  # shape [n_rows, n_cols]
            gdf = cudf.DataFrame(cp_mat)
            gdf.columns = list(self.map_inp.index)
            #gdf.index = cudf.RangeIndex(self.n_rows)  # cheap default
            return gdf

    def clone(self) -> "DataFrameTh":
        """
        Deep-ish clone: values cloned; labels reused (immutable enough).
        """
        new = DataFrameTh(
            self.values.T.clone(),              # back to row-major for ctor
            self.map_inp.index.copy(),
            self.index.index.copy(),
            device=self.device,
            dtype=self.values.dtype,
            values_is_row_major=True,
            make_contiguous=True
        )
        return new

    def __repr__(self) -> str:
        return (f"DataFrameTh(shape={self.values.shape}\n"
                f"dtype={self.values.dtype}, device={self.values.device})")

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
        params = init_params_from_g(self.g, self.irf_fn, self.nodes_idx) if param_df is None else \
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
        self.all_nodes = np.concatenate([g.nodes_idx.index.values for g in self.gs])

    def __len__(self):
        return len(self.gs)

    def __iter__(self):
        return iter(self.gs) 

    def __getitem__(self, idx):
        return self.gs[idx]

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
    else: return init_params_from_g(g, model_name, nodes_idx)