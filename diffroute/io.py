from tqdm.auto import tqdm
import pandas as pd
import networkx as nx

from .graph_utils import define_schedule
from .structs import RivTree, RivTreeCluster

def _read_rapid_graph(vpu_root,
                     rapid_connect="rapid_connect.csv",
                     riv_bas_id="riv_bas_id.csv",
                     k="k.csv",
                     x="x.csv"):
    """Load a single RAPID routing graph and Muskingum parameters.

    Args:
        vpu_root: Directory containing RAPID outputs (e.g., a `pathlib.Path`).
        rapid_connect: Name of the RAPID connectivity file (two-column CSV).
        riv_bas_id: Name of the RAPID reach ID file.
        k: Name of the RAPID Muskingum `k` parameter file.
        x: Name of the RAPID Muskingum `x` parameter file.

    Returns:
        Tuple[nx.DiGraph, pandas.DataFrame]: Directed graph of river reaches and
        a DataFrame indexed by reach ID with columns `k` (converted to days) and
        `x`.
    """
    df = pd.read_csv(vpu_root / rapid_connect, header=None)
    g = nx.DiGraph()
    g.add_edges_from(df[[0,1]].values)
    g.remove_node(-1)

    idx = pd.read_csv(vpu_root / riv_bas_id, header=None).values.squeeze()
    k   = pd.read_csv(vpu_root / k, header=None).values.squeeze() / (3600*24)
    x   = pd.read_csv(vpu_root / x, header=None).values.squeeze()
    params = pd.DataFrame({"k":k, "x":x}, index=idx)
    return g, params

def _read_multiple_rapid_graphs(vpu_roots, **kwargs):
    """Aggregate several RAPID graphs and parameters into a single graph.

    Args:
        vpu_roots: Iterable of directories containing RAPID outputs.
        **kwargs: Additional keyword arguments forwarded to `_read_rapid_graph`.

    Returns:
        Tuple[nx.DiGraph, pandas.DataFrame]: Composed directed graph covering all
        VPUs and a concatenated parameter DataFrame aligned to reach IDs.
    """
    gs, params = zip(*[_read_rapid_graph(vpu_root, **kwargs) \
                   for vpu_root in tqdm(vpu_roots)])
    g = nx.compose_all(gs)
    params = pd.concat(params)
    return g, params
    
def read_rapid_graph(vpu_root, plength_thr=None, node_thr=None, **rapid_kwargs):
    """Build a `RivTree` or `RivTreeCluster` from RAPID outputs.

    Args:
        vpu_root: Directory containing RAPID outputs for a single VPU.
        plength_thr: Optional path length threshold for clustering.
        node_thr: Optional node threshold for clustering.
        **rapid_kwargs: Optional overrides forwarded to `_read_rapid_graph`,
            such as `rapid_connect`, `riv_bas_id`, `k`, or `x`.

    Returns:
        RivTree | RivTreeCluster: Tree-like structure with Muskingum parameters
        attached to each reach.
    """
    g, params = _read_rapid_graph(vpu_root, **rapid_kwargs)
    
    if (plength_thr is not None) and (node_thr is not None):
        clusters_g, node_transfer = define_schedule(g, plength_thr=plength_thr, 
                                                    node_thr=node_thr)
        g = RivTreeCluster(clusters_g, 
                           node_transfer,
                           irf_fn="muskingum", 
                           include_index_diag=True,
                           param_df=params)
    else:
        g = RivTree(g, irf_fn="muskingum", 
                    include_index_diag=True,
                    param_df=params)
    return g

def read_multiple_rapid_graphs(vpu_roots, plength_thr=None, node_thr=None, **rapid_kwargs):
    """Build a `RivTree`/`RivTreeCluster` that spans multiple RAPID VPUs.

    Args:
        vpu_roots: Iterable of directories containing RAPID outputs.
        plength_thr: Optional path length threshold for clustering.
        node_thr: Optional node threshold for clustering.
        **rapid_kwargs: Optional overrides forwarded to `_read_multiple_rapid_graphs`,
            such as `rapid_connect`, `riv_bas_id`, `k`, or `x`.

    Returns:
        RivTree | RivTreeCluster: Composite river tree or clustered variant for
        the provided VPUs.
    """
    gs, params = _read_multiple_rapid_graphs(vpu_roots, **rapid_kwargs)
    
    if (plength_thr is not None) and (node_thr is not None):
        clusters_g, node_transfer = define_schedule(g, plength_thr=plength_thr, 
                                                    node_thr=node_thr)
        g = RivTreeCluster(clusters_g, 
                           node_transfer,
                           irf_fn="muskingum", 
                           include_index_diag=True,
                           param_df=params)
    else:
        g = RivTree(g, irf_fn="muskingum", 
                    include_index_diag=True,
                    param_df=params)
    return g
