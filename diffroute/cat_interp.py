from tqdm import tqdm

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from .utils import get_node_idxs

def index_precompute(cin, cout,            # (N,) int64
                     map_river, map_pixel, # (R,) int64  (unordered OK)
                     map_weight):          # (R,) float32/64

    ###
    ### Part 1: Index precomputation
    ###
    N = cin.shape[0]
    device = cin.device
    map_weight = map_weight.to(device=device)

    # ---- 1. sort mapping by river so we can binary-search ----
    sort_idx   = torch.argsort(map_river)
    map_river  = map_river[sort_idx].to(dtype=torch.int64, device=device, copy=False).contiguous()
    map_pixel  = map_pixel[sort_idx].to(dtype=torch.int64, device=device, copy=False)
    map_weight = map_weight[sort_idx]

    # ---- 2. find contiguous blocks per river ----
    left  = torch.searchsorted(map_river, cin, right=False)
    right = torch.searchsorted(map_river, cin, right=True)
    cnt   = right - left                            # how many pixels per row in `vals`
    keep  = cnt > 0                                 # drop rivers with no pixel mapping

    if not keep.all():
        raise NotImplementedError

    prefix = torch.zeros_like(cnt)
    prefix[1:] = torch.cumsum(cnt[:-1], 0)
    tot = int((prefix + cnt).max().item())          # total number of exploded rows

    # ---- 3. explode rows -> one entry per (pixel, river-row) combination ----
    row_id   = torch.repeat_interleave(torch.arange(N, device=device, dtype=torch.int64), cnt)   # (tot,)
    global_i = torch.arange(tot, device=device, dtype=torch.int64)
    rel_i    = global_i - prefix[row_id]                                                # offset within each river
    map_i    = left[row_id] + rel_i                                                     # index in map_pixel/weight

    pixel  = map_pixel[map_i]      # (tot,) int64
    weight = map_weight[map_i]     # (tot,) float32/64
    c_out  = cout[row_id]          # (tot,) int64

    # (tot, F)

    # ---- 4. pack (pixel, c_out) into a 64-bit key and aggregate ----
    key         = (pixel << 32) | (c_out & 0xffffffff)
    key_s, idx  = torch.sort(key)                       # stable sort so unique_consecutive works
    uniq_key, inverse = torch.unique_consecutive(key_s, return_inverse=True)
    M = uniq_key.numel()

    p_in  = (uniq_key >> 32).to(torch.int64)
    c_out_unique = (uniq_key & 0xffffffff).to(torch.int64)
    return weight, idx, row_id, M, inverse, p_in, c_out_unique

def aggregate(vals, weight, idx, row_id, M, inverse, p_in, c_out_unique):
    ###
    ### Part 2: Actual aggregation
    ###
    N, F = vals.shape
    contrib = vals[row_id] * weight.unsqueeze(1)                                        
    contrib_s   = contrib[idx]
    out = torch.zeros(M, F, dtype=vals.dtype, device=device)
    out.scatter_add_(0, inverse.unsqueeze(1).expand(-1, F), contrib_s)
    return out

def river_to_pixel_gpu_pt(vals,                # (N, F) float32/64, **must** be on CUDA
                          cin, cout,           # (N,) int64
                          map_river, map_pixel,# (R,) int64  (unordered OK)
                          map_weight):
    (weight, idx, row_id, M, 
     inverse, p_in, c_out_unique) = index_precompute(cin, cout,            
                                                     map_river, map_pixel, 
                                                     map_weight)
    out = aggregate(vals, weight, idx, row_id, M, inverse, p_in, c_out_unique)
    return out, c_out_unique, p_in
    
class DataFrameTh(nn.Module):
    def __init__(self, df):
        super().__init__()
        self.map_inp = pd.Series(np.arange(len(df.columns)), index=df.columns)
        self.register_buffer("values", torch.from_numpy(df.values).t().contiguous())
        
class CI(nn.Module):
    def __init__(self, g, weight_df):
        """
        """
        super().__init__()
        map_out = get_node_idxs(g) 
        weight_subset = weight_df.loc[map_out.index].copy()
        pix_idxs = weight_subset["pixel_idx"].unique()
        map_inp = pd.Series(np.arange(len(pix_idxs)), index=pix_idxs)
        self.n_cats = len(map_out)
        self.register_buffer("dest_idxs", torch.tensor(map_out[weight_subset.index].values, dtype=torch.long))
        self.register_buffer("src_idxs",  torch.tensor(map_inp.loc[weight_subset["pixel_idx"]].values, dtype=torch.long))
        self.register_buffer("weights",   torch.tensor(weight_subset["area_sqm_total"].values, dtype=torch.float))
        self.register_buffer("pix_idxs",  torch.tensor(pix_idxs, dtype=torch.long))

    def interpolate_runoff(self, runoff):
        """
        """
        local_runoff = runoff[self.pix_idxs]
        x = local_runoff[self.src_idxs]
        weighted_x = x * self.weights[:, None]  # broadcasts over the time dimension
        out = torch.zeros(self.n_cats, x.shape[1], dtype=x.dtype, device=runoff.device)
        out.index_add_(0, self.dest_idxs, weighted_x)
        return out[None]

    def interpolate_kernel(self, irfs_agg, coords):
        """
            Here, coords_pixel are aligned to pix_idxs ordering.
        """
        irfs_agg_pixel, co, pi = river_to_pixel_gpu_pt(irfs_agg,                      # [N, F] float32/64  (GPU)
                                                      coords[:,1], coords[:,0],       # [N] int32/int64
                                                      self.dest_idxs, self.src_idxs,  # [R] int32/int64 (NOT necessarily sorted)
                                                      self.weights)
        coords_pixel = torch.stack([co, pi]).t()
        kernel_size =  (self.n_cats, self.pix_idxs.shape[0])
        return irfs_agg_pixel, coords_pixel, kernel_size

    def forward(self, x):
        raise NotImplementedError
        
class CatchmentInterpolator(nn.Module):
    def __init__(self, gs, runoff, weight_df):
        super().__init__()
        self.runoff = DataFrameTh(runoff)
        self.weight_df = weight_df.copy()
        self.weight_df["pixel_idx"].values[:] = self.runoff.map_inp.loc[weight_df["pixel_idx"]].values
        self.weight_df = self.weight_df.sort_values("river_id").set_index("river_id")
        self.cis = nn.ModuleList([CI(g, self.weight_df) for g in tqdm(gs)])
            
    def read_pixels(self, idx):
        return self.runoff.values[self.cis[idx].pix_idxs][None]

    def interpolate_runoff(self, idx):
        return self.cis[idx].interpolate_runoff(self.runoff.values)
    
    def interpolate_kernel(self, idx, irfs_agg, coords):
        return self.cis[idx].interpolate_kernel(irfs_agg, coords)
    
class CatchmentInterpolatorOld:
    def __init__(self, gs, runoff, weight_df, device="cpu"):
        """
        Parameters:
          gs: list of subgraphs
          runoff: DataFrame of runoff (rows: time, columns: pixels)
          weight_df: DataFrame with columns: 'pixel_idx', 'river_id', 'area_sqm_total'
          device: device to store tensors (e.g. "cpu" or "cuda")
        """
        self.device = device
        self.map_inp = pd.Series(np.arange(len(runoff.columns)), index=runoff.columns)
        self.runoff = torch.from_numpy(runoff.values).to(device).t().contiguous()
        self.gs = {i: g for i, g in enumerate(gs)}
        weight_df = weight_df.copy()
        weight_df["pixel_idx"].values[:] = self.map_inp.loc[weight_df["pixel_idx"]].values
        self.weight_df = weight_df.sort_values("river_id").set_index("river_id")
        self.init_all_indices()

    def init_all_indices(self):
        """
            Precompute interpolation indices for all subgraphs.
        """
        print("CatchmentInterpolator initializes index table")
        self.indices = {k: self.init_indices(k) for k in tqdm(self.gs)}

    def init_indices(self, idx):
        """
        For a given subgraph, precompute the interpolation indices.
        
        Returns:
          n_cats: int, the number of catchments (output columns) in the subgraph.
          src_idxs: 1D torch.Tensor containing indices into runoff (source columns).
          dest_idxs: 1D torch.Tensor containing destination indices for scatter-add.
          weights: 1D torch.Tensor of weights (area_sqm_total) corresponding to each pixel.
        
        Assumes get_node_idxs(g) returns a pandas Series whose index are the river_ids and whose values
        provide the output order for each catchment.
        """
        g = self.gs[idx]
        map_out = get_node_idxs(g) 
        weight_subset = self.weight_df.loc[map_out.index].copy()
        pix_idxs = weight_subset["pixel_idx"].unique()
        map_inp = pd.Series(np.arange(len(pix_idxs)), index=pix_idxs)
    
        # Destination indices: using the order provided by map_out.
        dest_idxs = torch.tensor(map_out[weight_subset.index].values, 
                                 dtype=torch.long, device=self.device)
        src_idxs = torch.tensor(map_inp.loc[weight_subset["pixel_idx"]].values,
                                dtype=torch.long, device=self.device)
        pix_idxs = torch.tensor(pix_idxs, dtype=torch.long, device=self.device)
        # Weights: the area_sqm_total values as a float tensor.
        weights = torch.tensor(weight_subset["area_sqm_total"].values,
                               dtype=torch.float, device=self.device)
        n_cats = len(map_out)
        
        return n_cats, pix_idxs, src_idxs, dest_idxs, weights

    def read_catchment(self, idx):
        """
        Apply the interpolation for a given subgraph.
        
        Returns:
          out: torch.Tensor of shape (T, n_cats) containing the aggregated (weighted) runoff.
        """
        # Retrieve precomputed indices.
        n_cats, pix_idxs, src_idxs, dest_idxs, weights = self.indices[idx] #init_indices(self, idx)
        # Gather the runoff data for the required pixels (columns).
        # Note: runoff is of shape (T, num_pixels), so we select along dim 1.
        local_runoff = self.runoff[pix_idxs]
        x = local_runoff[src_idxs]  # shape: (T, K)
        # Multiply each selected column by its weight.
        weighted = x * weights[:, None]  # broadcasts over the time dimension
        # Create an output tensor to accumulate the weighted sums.
        out = torch.zeros(n_cats, x.shape[1], dtype=x.dtype, device=self.device)
        # Scatter-add: for each column in weighted, add it to the corresponding catchment column in out.
        out.index_add_(0, dest_idxs, weighted)
        return out[None]

    def __getitem__(self, idx):
        return self.read_catchment(idx)
    
    def __iter__(self):
        for i in range(len(self)):
            return self[i]

    def __len__(self):
        return len(self.gs)

def test():
    cat = CatchmentInterpolator(clusters_g, runoff, interp_df).to(device)
    catold = CatchmentInterpolatorOld(clusters_g, runoff, interp_df, device="cuda:5")
    
    x = cat.read_catchment(0)
    y = catold.read_catchment(0)
    (y.squeeze().to(device)-x).nonzero()#.shape
    
    assert torch.allclose(x, y.to(device), rtol=.001, atol=.001)