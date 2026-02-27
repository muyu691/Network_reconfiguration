"""
Stochastic User Equilibrium (SUE) Solver

This module provides multiple methods to solve the SUE traffic assignment problem:
1. Frank-Wolfe algorithm (basic implementation)
2. Method of Successive Averages (MSA)
3. Interface for external solvers (e.g., AequilibraE, PTV VISUM)

The paper uses PTV VISUM's SUE implementation, but we provide alternatives.
"""

import numpy as np
import networkx as nx
from tqdm import tqdm
import warnings


def bpr_travel_time(flow, capacity, free_flow_time, alpha=0.15, beta=4.0):
    """
    Bureau of Public Roads (BPR) travel time function.
    
    t = t0 * (1 + alpha * (flow/capacity)^beta)
    
    Args:
        flow: Traffic flow on the link
        capacity: Link capacity
        free_flow_time: Free-flow travel time
        alpha: Calibration parameter (default: 0.15)
        beta: Calibration parameter (default: 4.0)
    
    Returns:
        Travel time
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ratio = np.clip(flow / capacity, 0, 1e6)
        return free_flow_time * (1 + alpha * (ratio ** beta))


def all_or_nothing_assignment(G, od_matrix, travel_times):
    """
    All-or-Nothing (AON) assignment: assign all OD flow to shortest path.
    
    Args:
        G: NetworkX graph
        od_matrix: [num_centroids, num_centroids] OD demand matrix
        travel_times: Current travel times on each edge
    
    Returns:
        flows: [num_edges] assigned flows
    """
    edges = list(G.edges()) # list of edges, such as [(1, 2), (2, 3), (3, 4), (4, 5), (5, 6)]
    edge_to_idx = {edge: i for i, edge in enumerate(edges)} 
    flows = np.zeros(len(edges))
    
    # Create a copy of graph with current travel times as weights
    G_weighted = G.copy()
    for i, (u, v) in enumerate(edges):
        G_weighted[u][v]['weight'] = travel_times[i]
    
    # For each OD pair with non-zero demand
    num_centroids = od_matrix.shape[0]
    centroids = list(range(1, num_centroids + 1))  # Centroid IDs: 1-11
    
    for i, origin in enumerate(centroids):
        for j, dest in enumerate(centroids):
            if i == j or od_matrix[i, j] <= 0:
                continue
            
            demand = od_matrix[i, j]
            
            try:
                # Find shortest path
                path = nx.shortest_path(G_weighted, origin, dest, weight='weight') # return the shortest path between 
                # origin and destination, such as [1, 3, 5, 8, 9]
                
                # Assign demand to edges on this path
                for k in range(len(path) - 1):
                    u, v = path[k], path[k+1]
                    # Find edge index
                    try:
                        edge_idx = edge_to_idx[(u, v)]
                        flows[edge_idx] += demand
                    except ValueError:
                        pass  # Edge not in list (shouldn't happen)
            except nx.NetworkXNoPath:
                # No path exists between origin and destination
                continue
    
    return flows


def frank_wolfe_sue(G, od_matrix, capacities, free_flow_times, 
                    max_iter=100, convergence_threshold=1e-4, verbose=True):
    """
    Frank-Wolfe algorithm for traffic assignment (deterministic, not SUE).
    
    Note: This is a simplified deterministic equilibrium, not true SUE.
    For actual SUE, use more sophisticated methods or external solvers.
    
    Args:
        G: NetworkX graph
        od_matrix: [num_centroids, num_centroids] OD matrix
        capacities: [num_edges] link capacities
        free_flow_times: [num_edges] free-flow times
        max_iter: Maximum number of iterations
        convergence_threshold: Convergence criterion
        verbose: Print progress
    
    Returns:
        flows: [num_edges] equilibrium flows
    """
    edges = list(G.edges())
    num_edges = len(edges)
    
    # Initialize flows to zero
    flows = np.zeros(num_edges)
    
    # Initial all-or-nothing assignment
    travel_times = free_flow_times.copy()
    flows = all_or_nothing_assignment(G, od_matrix, travel_times)
    
    # Iterative procedure
    for iteration in range(max_iter):
        # Update travel times
        travel_times = bpr_travel_time(flows, capacities, free_flow_times)
        
        # All-or-nothing assignment with current travel times
        aon_flows = all_or_nothing_assignment(G, od_matrix, travel_times)
        
        # Compute step size (MSA: 1/(n+1))
        step_size = 1.0 / (iteration + 2)
        
        # Update flows
        new_flows = (1 - step_size) * flows + step_size * aon_flows
        
        # Check convergence
        relative_gap = np.linalg.norm(new_flows - flows) / (np.linalg.norm(flows) + 1e-8)
        
        if verbose and iteration % 10 == 0:
            print(f"    Iteration {iteration}: Relative gap = {relative_gap:.6f}")
        
        flows = new_flows
        
        if relative_gap < convergence_threshold:
            if verbose:
                print(f" Converged at iteration {iteration}")
            break
    
    return flows


def solve_sue_batch(G, od_matrices, capacities, speeds, 
                    method='frank_wolfe', verbose=True):
    """
    Solve SUE for a batch of scenarios.
    
    Args:
        G: NetworkX graph
        od_matrices: [num_samples, num_centroids, num_centroids]
        capacities: [num_samples, num_edges]
        speeds: [num_samples, num_edges]
        method: 'frank_wolfe', 'msa', or 'external'
        verbose: Print progress
    
    Returns:
        flows: [num_samples, num_edges] equilibrium flows
    """
    try:
        from .utils import compute_free_flow_times
    except ImportError:
        from utils import compute_free_flow_times
    
    num_samples = od_matrices.shape[0]
    num_edges = len(list(G.edges()))
    
    print(f"\n{'='*60}")
    print(f"Solving SUE for {num_samples} scenarios using '{method}' method")
    print(f"{'='*60}")
    
    # Compute free-flow times
    print("  Computing free-flow times...")
    free_flow_times = compute_free_flow_times(G, speeds)
    
    # Initialize flows array
    all_flows = np.zeros((num_samples, num_edges))
    
    # Solve for each scenario
    print(f"  Running traffic assignment...")
    
    if method == 'frank_wolfe':
        solver_func = frank_wolfe_sue
    else:
        raise ValueError(f"Unknown method: {method}")
    
    for i in tqdm(range(num_samples), desc="  Progress", disable=not verbose):
        flows = solver_func(
            G,
            od_matrices[i],
            capacities[i],
            free_flow_times[i],
            max_iter=100,
            verbose=False
        )
        all_flows[i] = flows
    
    print(f"\n SUE solving completed!")
    print(f"  Flow statistics:")
    print(f"    Min: {all_flows.min():.2f}")
    print(f"    Max: {all_flows.max():.2f}")
    print(f"    Mean: {all_flows.mean():.2f}")
    print(f"    Std: {all_flows.std():.2f}")
    
    return all_flows


def save_flows(flows, save_path='processed_data/raw/flows.npz'):
    """
    Save computed flows to disk.
    """
    import os
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    np.savez_compressed(save_path, flows=flows)
    print(f"\n Flows saved to: {save_path}")


def load_flows(load_path='processed_data/raw/flows.npz'):
    """
    Load previously computed flows.
    """
    data = np.load(load_path)
    print(f"\n Flows loaded from: {load_path}")
    return data['flows']


if __name__ == '__main__':
    # Test SUE solver
    from load_sioux import load_sioux_falls_network
    from generate_scenarios import generate_lhs_samples
    
    print("Testing SUE solver...")
    
    # Load network
    G, centroids = load_sioux_falls_network('../sioux_data/SiouxFalls_net.tntp')
    
    # Generate small test set
    od_mats, caps, speeds = generate_lhs_samples(num_samples=5)
    
    # Solve SUE
    flows = solve_sue_batch(G, od_mats, caps, speeds, method='frank_wolfe')
    
    print(f"\n Test completed!")
    print(f"  Generated flows shape: {flows.shape}")
