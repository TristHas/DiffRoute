import pandas as pd
import networkx as nx

from .schedule import define_schedule
from .utils import RivTree, RivTreeCluster

def read_rapid_graph(vpu_root, plength_thr=None, node_thr=None):
    df = pd.read_csv(vpu_root / "rapid_connect.csv", header=None)
    g = nx.DiGraph()
    g.add_edges_from(df[[0,1]].values)
    g.remove_node(-1)

    idx = pd.read_csv(vpu_root / "riv_bas_id.csv", header=None).values.squeeze()
    k   = pd.read_csv(vpu_root / "k.csv", header=None).values.squeeze() / (3600*24)
    x   = pd.read_csv(vpu_root / "x.csv", header=None).values.squeeze()
    params = pd.DataFrame({"k":k, "x":x}, index=idx)

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