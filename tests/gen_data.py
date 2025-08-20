from imports import *
from diffroute import get_node_idxs
from diffroute.agg.kernel_aggregator import RoutingIRFAggregator, get_node_idxs
from diffroute.utils import annotate_downstream_path_stats

def generate_mini_data(num_timesteps=20):
    """
    Generate a simple directed network with three catchments.
    Each node carries parameters for:
      - pure lag routing (delay),
      - linear reservoir routing (tau),
      - Muskingum routing (x and k),
      - linear diffusion routing (L, D, c).
    """
    g = nx.DiGraph()
    # Pure lag: delay (pure delay).
    # Nash Cascade & linear reservoir: tau (storage time constant).
    # Muskingum: x (weight) and k (delay parameter).
    # Linear diffusion: L (channel length), D (diffusivity), c (wave celerity).
    g.add_node(0, tau=2.0, x=0.2, k=2.0, delay=2.0, L=10.0, D=1.0, c=1.0)
    g.add_node(1, tau=1.5, x=0.3, k=1.5, delay=1.5, L=8.0,  D=0.8, c=1.2)
    g.add_node(2, tau=1.0, x=0.4, k=1.0, delay=1.0, L=5.0,  D=0.5, c=1.5)

    # Define flow connections.
    g.add_edge(0, 1)
    g.add_edge(1, 2)
    
    # Generate runoff inputs.
    np.random.seed(0)
    runoff_inputs = np.random.rand(3, num_timesteps)
    return g, runoff_inputs

def sparse_k_from_g(g, max_delay, block_size, device):
    """
    """
    nodes_idx = get_node_idxs(g)
    delays = torch.FloatTensor([g.nodes[n]["tau"] for n in nodes_idx.index]).to(device).unsqueeze(-1)
    agg = RoutingIRFAggregator(g, nodes_idx, max_delay=max_delay, block_size=block_size).to(device)
    k = agg(delays)
    return k.to(device)
    
def generate_inputs(n_trees, max_heights, max_delay, block_size, n_time_steps, batch_size, device="cuda:0"):
    """
    """
    g = generate_river_forest(n_trees, max_heights)
    annotate_downstream_path_stats(g)
    k = sparse_k_from_g(g, max_delay, block_size, device)
    x = torch.relu(torch.randn(batch_size, len(g), n_time_steps, dtype=torch.float32, device=device)).requires_grad_()
    return g, x, k

###
### More realistic tree forest
###
def generate_river_forest(n_trees=3, 
                          max_depth=10, 
                          branch_prob=0.7, 
                          branch_factor_options=[1, 2],
                          branch_factor_probs=[0.6, 0.4],
                          tau_mean=3.0, 
                          tau_sigma=0.5):
    """
    Generates a forest (disjoint union) of river networks.
    Each tree is generated independently using the more realistic branching model.
    The nodes in the forest are relabeled to ensure uniqueness.
    """
    forest = nx.DiGraph()
    next_offset = 0
    for i in range(n_trees):
        main_length = max_depth #np.random.choice(max_depth)
        tree = generate_fixed_length_river(main_length=main_length,
                                           branch_prob=branch_prob,
                                           branch_factor_options=branch_factor_options,
                                           branch_factor_probs=branch_factor_probs,
                                           tau_mean=tau_mean,
                                           tau_sigma=tau_sigma)
        # Relabel nodes to ensure uniqueness across trees
        mapping = {node: node + next_offset for node in tree.nodes()}
        tree = nx.relabel_nodes(tree, mapping)
        forest = nx.disjoint_union(forest, tree)
        next_offset = max(forest.nodes()) + 1
    forest = nx.relabel_nodes(forest, {node: node + 1 for node in forest.nodes()})
    return forest

def generate_fixed_length_river(main_length=100, 
                                branch_prob=0.7, 
                                branch_factor_options=[1, 2], 
                                branch_factor_probs=[0.6, 0.4],
                                branch_max_depth=10, 
                                tau_mean=3.0, 
                                tau_sigma=0.5):
    """
    Generates a river network with a fixed-length main trunk and tributaries attached to it.
    
    Parameters:
      - main_length: Number of nodes in the main channel (backbone).
      - branch_prob: Probability of adding tributaries at a given node.
      - branch_factor_options & branch_factor_probs: Options and probabilities for the number of tributaries.
      - branch_max_depth: Maximum recursion depth for tributary generation.
      - tau_mean, tau_sigma: Parameters for the lognormal distribution used for delay (tau).
    
    Returns:
      - G: A networkx.DiGraph representing the river network.
    """
    G = nx.DiGraph()
    
    # 1. Create the fixed-length main trunk (a simple path).
    trunk_nodes = list(range(main_length))
    G.add_nodes_from(trunk_nodes)
    for i in range(main_length - 1):
        # Create an edge from upstream (i) to downstream (i+1)
        G.add_edge(trunk_nodes[i], trunk_nodes[i+1])
    # Assign delays for the trunk nodes.
    for node in trunk_nodes:
        G.nodes[node]["tau"] = np.random.lognormal(mean=tau_mean, sigma=tau_sigma)
    
    # 2. Attach tributaries to nodes on the main trunk.
    next_node = main_length  # Counter for new node IDs.
    
    def add_tributaries(node, depth):
        nonlocal next_node
        if depth >= branch_max_depth:
            return
        # Decide whether to add tributaries.
        if np.random.rand() > branch_prob:
            return
        # Determine number of tributaries.
        num_tributaries = np.random.choice(branch_factor_options, p=branch_factor_probs)
        for _ in range(num_tributaries):
            new_node = next_node
            next_node += 1
            G.add_node(new_node, tau=np.random.lognormal(mean=tau_mean, sigma=tau_sigma))
            # Attach the new tributary node to the current node.
            G.add_edge(new_node, node)
            # Recursively generate further upstream tributaries.
            add_tributaries(new_node, depth + 1)
    
    # Apply tributary generation for each node along the trunk.
    for node in trunk_nodes:
        add_tributaries(node, depth=0)
    
    return G

###
### Balanced tree forests
###
def generate_balanced_tree(height):
    """
        Generates a balanced binary tree of a given height using networkx.
        Note: A balanced binary tree with height h has (2^(h+1)-1) nodes.
    """
    T = nx.balanced_tree(r=2, h=height)
    return nx.bfs_tree(T, 0).reverse()

def generate_random_forest(n_trees=3, max_heights=1000):
    """
        Generates a forest (disjoint union) of binary trees.
        Each tree gets a random height chosen from `heights`.
        Each node in the forest is assigned a tau (delay) uniformly sampled from [1, 100].
        Returns a directed graph.
    """
    forest = nx.DiGraph()
    next_offset = 1
    for i in range(n_trees):
        main_length = max_depth #np.random.choice(max_depth)
        tree = generate_fixed_length_river(main_length=main_length,
                                           branch_prob=branch_prob,
                                           branch_factor_options=branch_factor_options,
                                           branch_factor_probs=branch_factor_probs,
                                           tau_mean=tau_mean,
                                           tau_sigma=tau_sigma)
        # Relabel nodes to ensure uniqueness across trees
        mapping = {node: node + next_offset for node in tree.nodes()}
        tree = nx.relabel_nodes(tree, mapping)
        forest = nx.disjoint_union(forest, tree)
        next_offset = max(forest.nodes()) + 1
    # Assign each node a random tau between 1 and 100.
    for node in forest.nodes():
        forest.nodes[node]["tau"] = np.random.uniform(1, 100)
    return forest

def generate_river_tree(max_depth=10, 
                        branch_prob=0.7, 
                        branch_factor_options=[1, 2], 
                        branch_factor_probs=[0.6, 0.4],
                        tau_mean=3.0, 
                        tau_sigma=0.5):
    """
        Generates a single river network (a directed tree) with a more realistic,
        unbalanced structure.
        
        Parameters:
        - max_depth: maximum recursion depth (limits how far upstream to grow).
        - branch_prob: probability of generating tributaries at a node.
        - branch_factor_options & branch_factor_probs: defines the possible number 
          of tributaries and their probabilities.
        - tau_mean, tau_sigma: parameters for the lognormal distribution for delays.
        
        Returns:
        - G: a directed graph (DiGraph) where edges point from upstream (tributaries)
             to downstream (confluence) nodes.
    """
    G = nx.DiGraph()
    # Use a mutable counter stored in a list
    node_counter = [0]
    
    # Create the outlet node (the downstream-most node)
    outlet = node_counter[0]
    G.add_node(outlet)
    # Assign a delay to the outlet as well
    G.nodes[outlet]['tau'] = np.random.lognormal(mean=tau_mean, sigma=tau_sigma)
    
    def add_tributaries(current_node, depth):
        # Stop recursion at max_depth
        if depth >= max_depth:
            return
        
        # With probability branch_prob, add tributaries; otherwise, this is a headwater.
        if np.random.rand() > branch_prob:
            return
        
        # Sample the number of tributaries for the current node
        num_tributaries = np.random.choice(branch_factor_options, p=branch_factor_probs)
        for _ in range(num_tributaries):
            # Create a new upstream node
            node_counter[0] += 1
            new_node = node_counter[0]
            G.add_node(new_node)
            # Assign tau using a lognormal distribution for heterogeneity
            G.nodes[new_node]['tau'] = np.random.lognormal(mean=tau_mean, sigma=tau_sigma)
            # Add a directed edge from the new (upstream) node to the current (downstream) node
            G.add_edge(new_node, current_node)
            # Recursively add tributaries upstream of the new node
            add_tributaries(new_node, depth + 1)
    
    # Start the recursive generation from the outlet
    add_tributaries(outlet, depth=0)
    return G