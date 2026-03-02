"""
Generate network reconfiguration scenarios: (G, G') network pair data

This module accomplishes the following tasks:
1. Uses Latin Hypercube Sampling (LHS) to generate base scenario OD matrices and G edge attributes
2. Implements three types of network mutations (add edge, delete edge, change attributes)
3. Follows a 40%/30%/30% distribution for mutation types to construct (G, G') network pairs

Strict Constraints:
- G and G' share exactly the same OD matrix (OD is only used for SUE computation, never as node features)
- G and G' have identical node sets (24 nodes in Sioux Falls); only edges and edge attributes may vary
- Mutation type distribution: 40% topology change only, 30% attribute change only, 30% both changes
"""

import numpy as np
import networkx as nx
from scipy.stats import qmc
from copy import deepcopy
from tqdm import tqdm


# ============================================================
# Part 1: Base Scenario Generation (LHS Sampling)
# ============================================================

def generate_lhs_base_scenarios(
    num_samples: int = 2000,
    num_centroids: int = 11,
    num_edges: int = 76,
    seed: int = 42
) -> tuple:
    """
    Use Latin Hypercube Sampling (LHS) to generate base scenario OD matrices and edge attributes for G.

    Parameter ranges (from paper specifications):
    - OD demand: 0 - 1500 vehicles/OD pair
    - Capacity: 4000 - 26000
    - Speed: 45 - 80 km/h

    Args:
        num_samples:   Number of scenarios to generate (default 2000)
        num_centroids: Number of centroid nodes (11 for Sioux Falls)
        num_edges:     Number of edges (76 for Sioux Falls)
        seed:          Random seed

    Returns:
        od_matrices: np.ndarray [num_samples, num_centroids, num_centroids]
        capacities:  np.ndarray [num_samples, num_edges]   -- G's capacities
        speeds:      np.ndarray [num_samples, num_edges]   -- G's speeds
    """
    print(f"\n{'='*60}")
    print(f"Generating {num_samples} base scenarios with LHS sampling")
    print(f"{'='*60}")

    num_od_pairs = num_centroids * num_centroids          # 11*11 = 121
    num_dims = num_od_pairs + num_edges * 2               # 121 + 76 + 76 = 273

    sampler = qmc.LatinHypercube(d=num_dims, seed=seed)
    samples = sampler.random(n=num_samples)               # [num_samples, 273]

    # Split by dimension
    od_raw  = samples[:, :num_od_pairs]                   # [N, 121]
    cap_raw = samples[:, num_od_pairs : num_od_pairs + num_edges]   # [N, 76]
    spd_raw = samples[:, num_od_pairs + num_edges:]       # [N, 76]

    # Map to actual range
    od_matrices = (od_raw * 1500.0).reshape(num_samples, num_centroids, num_centroids)
    capacities  = cap_raw * (26000 - 4000) + 4000
    speeds      = spd_raw * (80 - 45) + 45

    print(f"  OD matrix shape:  {od_matrices.shape}")
    print(f"  Capacity matrix shape: {capacities.shape}")
    print(f"  Speed matrix shape: {speeds.shape}")
    print(f"  OD range: [{od_matrices.min():.1f}, {od_matrices.max():.1f}]")
    print(f"  Capacity range: [{capacities.min():.1f}, {capacities.max():.1f}]")
    print(f"  Speed range: [{speeds.min():.1f}, {speeds.max():.1f}]")
    print(f"  LHS sampling complete!")

    return od_matrices, capacities, speeds


# ============================================================
# Part 2: Three Types of Topology Mutation Algorithms
# ============================================================

def mutate_add_edges(
    G: nx.DiGraph,
    flows_old: np.ndarray,
    rng: np.random.Generator,
    top_k_range: tuple = (5, 10),
    edges_per_node_range: tuple = (1, 3)
) -> tuple:
    """
    Mutation Operation 1: Add Edges
    Add new edges to the top_k nodes with highest total flow in G.

    Algorithm logic:
    1. Compute node flow importance: sum up flows of all edges (in/out) related to each node, yielding node_flow_sum[v].
    2. Randomly choose top_k (uniform in [top_k_range[0], top_k_range[1]]), take top_k nodes with highest node_flow_sum as candidate source nodes for adding new edges.
    3. For each candidate node, randomly add 1-3 out-edges:
       - Target node: uniform random sample from all nodes in G, excluding self and already connected targets
       - New edge attributes: uniformly sample within [min, max] of each attribute from existing edges; ensures reasonable values
       - Automatically calculate free_flow_time = (length / speed) * 60

    Args:
        G:                  Current scenario NetworkX DiGraph (with full edge attributes)
        flows_old:          np.ndarray [num_edges_G], indexed by list(G.edges()) order
        rng:                numpy random generator
        top_k_range:        Range for randomly selecting top_k nodes (inclusive)
        edges_per_node_range: Range for number of edges to add per node (inclusive)

    Returns:
        G_new:       DiGraph with new edges added
        added_edges: list of (u, v) newly added directed edges
    """
    edges    = list(G.edges())
    node_ids = list(G.nodes())   # Sioux Falls: nodes 1-24

    # --- Step 1: Compute node flow strength ---
    # For each node, accumulate flows of all in- and out- edges
    node_flow_sum = {n: 0.0 for n in node_ids}
    for idx, (u, v) in enumerate(edges):
        node_flow_sum[u] += flows_old[idx]
        node_flow_sum[v] += flows_old[idx]

    # --- Step 2: Pick top_k highest-flow nodes ---
    top_k = int(rng.integers(top_k_range[0], top_k_range[1] + 1))
    sorted_nodes = sorted(node_ids, key=lambda n: node_flow_sum[n], reverse=True)
    top_nodes = sorted_nodes[:top_k]

    # --- Step 3: For each high-flow node, add 1-3 new out-edges ---
    G_new = deepcopy(G)
    added_edges = []

    # Compute attribute ranges from existing edges; new edges uniformly sample in-range
    all_caps = [G[u][v]['capacity']      for u, v in edges]
    all_spds = [G[u][v]['speed']         for u, v in edges]
    all_lens = [G[u][v]['length']        for u, v in edges]
    cap_min, cap_max = min(all_caps), max(all_caps)
    spd_min, spd_max = min(all_spds), max(all_spds)
    len_min, len_max = min(all_lens), max(all_lens)

    for node in top_nodes:
        # Exclude self and already existing out-edge targets (no repeats)
        existing_targets = set(G_new.successors(node))
        candidates = [n for n in node_ids if n != node and n not in existing_targets]

        if not candidates:
            # Node already connected to all other nodes; skip
            continue

        num_to_add = int(rng.integers(edges_per_node_range[0], edges_per_node_range[1] + 1))
        num_to_add = min(num_to_add, len(candidates))

        # Randomly pick target nodes from candidates
        targets = rng.choice(candidates, size=num_to_add, replace=False)

        for target in targets:
            # Edge attributes: sample uniformly within [min, max] from existing
            new_cap = float(rng.uniform(cap_min, cap_max))
            new_spd = float(rng.uniform(spd_min, spd_max))
            new_len = float(rng.uniform(len_min, len_max))
            # free_flow_time (minutes) = length (km) / speed (km/h) * 60
            new_fft = (new_len / new_spd) * 60.0

            G_new.add_edge(
                int(node), int(target),
                capacity=new_cap,
                speed=new_spd,
                length=new_len,
                free_flow_time=new_fft
            )
            added_edges.append((int(node), int(target)))

    return G_new, added_edges


def mutate_delete_edges(
    G: nx.DiGraph,
    rng: np.random.Generator,
    num_delete_range: tuple = (5, 10)
) -> tuple:
    """
    Mutation Operation 2: Delete Edges
    Randomly delete 5-10 directed edges from G, always maintaining strong connectivity.

    Algorithm logic:
    1. Randomly determine target number of deletions num_delete ∈ [5, 10]
    2. Randomly permute the edge order, and try deleting each edge in turn:
       - Temporarily remove the edge
       - Check if the graph is still strongly connected
         * Strong connectivity: for every ordered pair of nodes, there is a directed path from one to the other
         * This is necessary for SUE to be well-posed
       - If remains connected: confirm deletion and add to deleted list
       - If not connected: restore the edge (with all its original attributes)
    3. Stop as soon as num_delete deletions confirmed

    Note: If after going through all edges, the target is not reached, just keep what you deleted so far (no hard error)

    Args:
        G:                Current scenario NetworkX DiGraph
        rng:              numpy random generator
        num_delete_range: Range for number of edges to delete (inclusive)

    Returns:
        G_new:         DiGraph after deletions
        deleted_edges: list of (u, v) deleted directed edges
    """
    edges = list(G.edges())
    num_delete = int(rng.integers(num_delete_range[0], num_delete_range[1] + 1))

    G_new = deepcopy(G)
    deleted_edges = []

    # Shuffle edge order to randomize deletion results
    shuffled_indices = rng.permutation(len(edges))

    for idx in shuffled_indices:
        if len(deleted_edges) >= num_delete:
            break

        u, v = edges[idx]

        # Temporarily remove this edge
        edge_data = dict(G_new[u][v])  # Save full attributes for possible restoration
        G_new.remove_edge(u, v)

        # Check strong connectivity: all pairs have a directed path
        # Must enforce is_strongly_connected, not is_weakly_connected!
        # Weak only checks "connected ignoring direction" and can create dead-ends.
        # SUE solver (Frank-Wolfe) depends on directed shortest path being available;
        # if not, nx.shortest_path throws NetworkXNoPath, crashing the whole batch!
        # Sioux Falls base network is strongly connected; must preserve this.
        if nx.is_strongly_connected(G_new):
            # Valid deletion, keep it
            deleted_edges.append((u, v))
        else:
            # Would break strong connectivity; restore full original edge attributes
            G_new.add_edge(u, v, **edge_data)

    return G_new, deleted_edges


def mutate_attributes(
    G: nx.DiGraph,
    rng: np.random.Generator,
    cap_scale_range: tuple = (0.3, 2.0),
    spd_scale_range: tuple = (0.3, 2.0)
) -> tuple:
    """
    Mutation Operation 3: Change Attributes
    Independently randomly scale the capacity and speed of each edge in G.

    Algorithm logic:
    1. Iterate over every directed edge (u, v) in G
    2. Independently sample capacity scale λ_cap ~ Uniform(0.3, 2.0)
       Independently sample speed scale λ_spd ~ Uniform(0.3, 2.0)
    3. Update attributes:
       new_capacity      = old_capacity * λ_cap
       new_speed         = old_speed    * λ_spd
       new_free_flow_time = (length / new_speed) * 60   (recalculated with updated speed)
    4. Return the modified graph and per-edge scaling logs

    Note: length is unchanged (physical distance is not affected by reconfiguration);
    free_flow_time must be recalculated based on updated speed.

    Args:
        G:               Current scenario NetworkX DiGraph
        rng:             numpy random generator
        cap_scale_range: Range for capacity scale factor (inclusive)
        spd_scale_range: Range for speed scale factor (inclusive)

    Returns:
        G_new:       DiGraph with attributes mutated
        attr_changes: dict {(u, v): {'cap_scale': float, 'spd_scale': float}}
    """
    G_new = deepcopy(G)
    attr_changes = {}

    for u, v in list(G_new.edges()):
        cap_scale = float(rng.uniform(cap_scale_range[0], cap_scale_range[1]))
        spd_scale = float(rng.uniform(spd_scale_range[0], spd_scale_range[1]))

        old_cap = G_new[u][v]['capacity']
        old_spd = G_new[u][v]['speed']
        old_len = G_new[u][v]['length']   # Physical distance remains unchanged

        new_cap = old_cap * cap_scale
        new_spd = old_spd * spd_scale
        # free_flow_time (minutes) = length (km) / speed (km/h) * 60
        new_fft = (old_len / new_spd) * 60.0

        G_new[u][v]['capacity']       = new_cap
        G_new[u][v]['speed']          = new_spd
        G_new[u][v]['free_flow_time'] = new_fft

        attr_changes[(u, v)] = {'cap_scale': cap_scale, 'spd_scale': spd_scale}

    return G_new, attr_changes


# ============================================================
# Part 3: Build Concrete Scenario G and Generate G'
# ============================================================

def build_scenario_graph(G_topo: nx.DiGraph, capacities_i: np.ndarray, speeds_i: np.ndarray) -> nx.DiGraph:
    """
    Assign Sioux Falls base topology (G_topo) with the edge attributes for the i-th scenario,
    and return the specific G_i for that scenario (can be passed directly to SUE solver).

    Edge order matches exactly list(G_topo.edges()),
    thus capacities_i[j] corresponds to list(G_topo.edges())[j].

    Args:
        G_topo:       Sioux Falls base directed graph (with length attribute)
        capacities_i: np.ndarray [num_edges_G], edge capacities for scenario i
        speeds_i:     np.ndarray [num_edges_G], edge speeds for scenario i

    Returns:
        G_i: NetworkX DiGraph with assigned attributes
    """
    G_i = deepcopy(G_topo)
    edges = list(G_topo.edges())

    for j, (u, v) in enumerate(edges):
        length = G_topo[u][v]['length']         # Physical distance remains unchanged
        spd    = float(speeds_i[j])
        cap    = float(capacities_i[j])
        fft    = (length / spd) * 60.0          # Minutes

        G_i[u][v]['capacity']       = cap
        G_i[u][v]['speed']          = spd
        G_i[u][v]['free_flow_time'] = fft

    return G_i


def _apply_topology_mutation(
    G: nx.DiGraph,
    flows_old: np.ndarray,
    rng: np.random.Generator,
    mutation_info: dict
) -> nx.DiGraph:
    """
    Internal helper: apply topology mutation (add, delete, or both) to G.

    Sub-operations for topology mutation are randomly chosen with equal probability:
    - 'add'   (33%): add edges only
    - 'delete'(33%): delete edges only
    - 'both'  (34%): add edges first, then delete edges

    Args:
        G:           The current graph
        flows_old:   Historical flows on G (used for identifying top-flow nodes during edge addition)
        rng:         numpy random generator
        mutation_info: dict for recording mutation details (modified in-place)

    Returns:
        G_mutated: Graph after topology mutation
    """
    # Note: Here, flows_old indexing relies on the order of list(G.edges());
    # this function must only be called on the unmutated topology G (i.e. G_scenario), not repeatedly on mutated graphs.
    topo_op = rng.choice(['add', 'delete', 'both'])

    G_mut = G
    if topo_op in ('add', 'both'):
        G_mut, added_edges = mutate_add_edges(G_mut, flows_old, rng)
        mutation_info['added_edges'] = added_edges

    if topo_op in ('delete', 'both'):
        # For deletion, flows_old already refers to the original G; edge deletion doesn't change meaning of flows_old indices
        G_mut, deleted_edges = mutate_delete_edges(G_mut, rng)
        mutation_info['deleted_edges'] = deleted_edges

    mutation_info['topo_op'] = str(topo_op)
    return G_mut


# ============================================================
# Part 4: Main Function for Network Pair Generation
# ============================================================

def generate_network_pairs(
    G_topo: nx.DiGraph,
    od_matrices: np.ndarray,
    capacities: np.ndarray,
    speeds: np.ndarray,
    flows_old: np.ndarray,
    seed: int = 42
) -> list:
    """
    Main function: for each base scenario, generate the corresponding mutated network G', yielding a full list of (G, G') pairs.

    Mutation type distribution (strictly following .cursorrules specification):
    - 40%  topology_only  : topology change only (add/delete edges)
    - 30%  attribute_only : attribute change only (capacity/speed scaled 0.3x-2.0x)
    - 30%  both           : topology change first, then attribute change

    Returned structure for each pair (scenario_pair dict):
    ```
    {
        'od_matrix'     : np.ndarray [11, 11],  # shared OD matrix (for SUE only, never model input!)
        'G'             : nx.DiGraph,           # scenario G (fixed topology + LHS attributes)
        'G_prime'       : nx.DiGraph,           # mutated network G' (may have different edge set and attributes)
        'mutation_type' : str,                  # 'topology_only'|'attribute_only'|'both'
        'mutation_info' : dict,                 # details: added/deleted edges, attribute scaling info
    }
    ```

    Node set consistency guarantee:
    - G and G' always have identical node sets (24 nodes in Sioux Falls)
    - Nodes are never added or deleted; only edge set and edge attributes are changed

    Args:
        G_topo:      Sioux Falls base directed graph (read from .tntp file, contains length attribute)
        od_matrices: np.ndarray [N, 11, 11], OD matrices generated by LHS
        capacities:  np.ndarray [N, 76],     G's capacities generated by LHS
        speeds:      np.ndarray [N, 76],     G's speeds generated by LHS
        flows_old:   np.ndarray [N, 76],     historical flows solved on G (used to locate top-flow nodes for edge add mutations)
        seed:        Random seed

    Returns:
        scenario_pairs: list of dict, length N
    """
    num_samples = od_matrices.shape[0]
    rng = np.random.default_rng(seed)

    # Strictly allocate mutation types by 40/30/30 ratio, using "precise counting + shuffling" strategy.
    # Do NOT use probability sampling (rng.choice with p=[...]): this is subject to random drift,
    # and cannot guarantee the strict 40/30/30 ratio stated in paper (for perfect reproducibility).
    # Strategy: first build an array with exact counts, then shuffle it in-place to randomize ordering.
    # num_both absorbs any int() rounding error.
    num_topo = int(num_samples * 0.40)
    num_attr = int(num_samples * 0.30)
    num_both = num_samples - num_topo - num_attr   # Assign residual to both; handles int truncation

    mutation_types = np.array(
        ['topology_only'] * num_topo +
        ['attribute_only'] * num_attr +
        ['both'] * num_both,
        dtype=object
    )
    rng.shuffle(mutation_types)  # Shuffle in-place so order is randomized (no systematic bias)

    print(f"\n{'='*60}")
    print(f"Generating {num_samples} (G, G') network pairs")
    print(f"{'='*60}")
    count = {t: int(np.sum(mutation_types == t)) for t in ['topology_only', 'attribute_only', 'both']}
    print(f"  Mutation type distribution (exact counts):")
    print(f"    Topology only (topology_only) : {count['topology_only']} ({count['topology_only']/num_samples*100:.1f}%)")
    print(f"    Attribute only (attribute_only): {count['attribute_only']} ({count['attribute_only']/num_samples*100:.1f}%)")
    print(f"    Topology+attribute (both)     : {count['both']} ({count['both']/num_samples*100:.1f}%)")

    scenario_pairs = []
    num_nodes = G_topo.number_of_nodes()

    for i in tqdm(range(num_samples), desc="  Generating network pairs"):
        mutation_type = mutation_types[i]
        mutation_info = {'type': mutation_type}

        # --- Step 1: Build G_i for scenario i (fixed topology + LHS attributes) ---
        G_i = build_scenario_graph(G_topo, capacities[i], speeds[i])

        # --- Step 2: Starting from G_i, build G' ---
        G_prime = deepcopy(G_i)

        if mutation_type == 'topology_only':
            # Apply topology mutation (add/delete edges) only; do not change attributes
            G_prime = _apply_topology_mutation(G_prime, flows_old[i], rng, mutation_info)

        elif mutation_type == 'attribute_only':
            # Apply attribute mutation only; topology is unchanged
            G_prime, attr_changes = mutate_attributes(G_prime, rng)
            mutation_info['attr_changes_count'] = len(attr_changes)  # Don't store all to avoid memory explosion

        elif mutation_type == 'both':
            # Topology change first, then mutate attributes on the changed graph
            # Note: topology first, then attribute, so attribute mutation covers all (including newly added) edges
            G_prime = _apply_topology_mutation(G_prime, flows_old[i], rng, mutation_info)
            G_prime, attr_changes = mutate_attributes(G_prime, rng)
            mutation_info['attr_changes_count'] = len(attr_changes)

        # --- Step 3: Node set consistency check ---
        # G and G' must have identical node sets; only edge set may change
        assert set(G_i.nodes()) == set(G_prime.nodes()), (
            f"Scenario {i}: Node set mismatch between G and G'! G: {set(G_i.nodes())}, G': {set(G_prime.nodes())}"
        )
        assert G_prime.number_of_nodes() == num_nodes, (
            f"Scenario {i}: G' node count {G_prime.number_of_nodes()} does not match expected {num_nodes}!"
        )

        scenario_pairs.append({
            'od_matrix'    : od_matrices[i].copy(),   # [11, 11], OD matrix (for SUE only)
            'G'            : G_i,                     # Scenario G (NetworkX DiGraph)
            'G_prime'      : G_prime,                 # Mutated network G' (NetworkX DiGraph)
            'mutation_type': mutation_type,
            'mutation_info': mutation_info,
        })

    # --- Collate stats ---
    edge_counts_G      = [len(list(p['G'].edges()))       for p in scenario_pairs]
    edge_counts_Gprime = [len(list(p['G_prime'].edges())) for p in scenario_pairs]

    print(f"\n  Generation complete!")
    print(f"  G  edge count: fixed at {edge_counts_G[0]} (same in all scenarios)")
    print(f"  G' edge count: min={min(edge_counts_Gprime)}, "
          f"max={max(edge_counts_Gprime)}, "
          f"mean={np.mean(edge_counts_Gprime):.1f}")

    return scenario_pairs


# ============================================================
# Part 5: Utilities for Saving and Loading
# ============================================================

def save_scenarios(od_matrices, capacities, speeds, save_path='processed_data/raw/base_scenarios.npz'):
    """
    Save LHS base scenario data (numpy format, compact and compressed).

    Args:
        od_matrices: [N, 11, 11]
        capacities:  [N, 76]
        speeds:      [N, 76]
        save_path:   Target file path (.npz)
    """
    import os
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    np.savez_compressed(save_path, od_matrices=od_matrices, capacities=capacities, speeds=speeds)
    print(f"  Base scenarios saved: {save_path}")


def load_scenarios(load_path='processed_data/raw/base_scenarios.npz'):
    """
    Load LHS base scenario data.

    Returns:
        od_matrices, capacities, speeds
    """
    data = np.load(load_path)
    print(f"  Base scenarios loaded: {load_path}")
    return data['od_matrices'], data['capacities'], data['speeds']


def save_scenario_pairs(scenario_pairs: list, save_path='processed_data/raw/scenario_pairs.pkl'):
    """
    Save (G, G') pair list (includes NetworkX graph objects, serialized with pickle).

    Note: Each scenario_pair contains two NetworkX DiGraph objects, memory intensive.
    For large datasets (> 5000 samples), consider saving in batches.

    Args:
        scenario_pairs: list of dict, length = N
        save_path:      Target file path (.pkl)
    """
    import os
    import pickle
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, 'wb') as f:
        pickle.dump(scenario_pairs, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"  Network pair data saved: {save_path} ({len(scenario_pairs)} pairs)")


def load_scenario_pairs(load_path='processed_data/raw/scenario_pairs.pkl') -> list:
    """
    Load (G, G') pair list.

    Returns:
        scenario_pairs: list of dict
    """
    import pickle
    with open(load_path, 'rb') as f:
        scenario_pairs = pickle.load(f)
    print(f"  Network pair data loaded: {load_path} ({len(scenario_pairs)} pairs)")
    return scenario_pairs
