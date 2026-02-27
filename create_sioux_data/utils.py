"""
Utility functions for Sioux Falls dataset creation
"""

import numpy as np
import torch


def compute_free_flow_times(G, speeds):
    """
    Compute free-flow travel times for each scenario.
    
    Formula: time (minutes) = length (km) / speed (km/h) * 60
    
    Args:
        G: NetworkX graph with 'length' edge attribute
        speeds: [num_samples, num_edges] - speeds for each edge in each scenario
    
    Returns:
        free_flow_times: [num_samples, num_edges] - in minutes
    """
    num_samples = speeds.shape[0]
    edges = list(G.edges())
    num_edges = len(edges)
    
    free_flow_times = np.zeros((num_samples, num_edges))
    
    for i, (u, v) in enumerate(edges):
        length = G[u][v]['length']  # km
        # Time = length / speed * 60 (convert to minutes)
        free_flow_times[:, i] = (length / speeds[:, i]) * 60
    
    return free_flow_times


def get_edge_index_from_graph(G):
    """
    Convert NetworkX graph to PyTorch Geometric edge_index format.
    
    Args:
        G: NetworkX DiGraph
    
    Returns:
        edge_index: [2, num_edges] numpy array
    """
    edges = list(G.edges())
    # Convert node IDs (1-indexed) to 0-indexed
    edge_index = np.array([[u-1, v-1] for u, v in edges]).T
    return edge_index


def validate_data_shapes(od_matrices, capacities, speeds, flows=None):
    """
    Validate that all data arrays have consistent shapes.
    """
    num_samples = od_matrices.shape[0]
    
    print(f"\n{'='*60}")
    print("Validating data shapes")
    print(f"{'='*60}")
    
    print(f"  Number of samples: {num_samples}")
    print(f"  OD matrices shape: {od_matrices.shape} (expected: [{num_samples}, 11, 11])")
    print(f"  Capacities shape: {capacities.shape} (expected: [{num_samples}, 76])")
    print(f"  Speeds shape: {speeds.shape} (expected: [{num_samples}, 76])")
    
    if flows is not None:
        print(f"  Flows shape: {flows.shape} (expected: [{num_samples}, 76])")
    
    # Check shapes
    assert od_matrices.shape == (num_samples, 11, 11), "OD matrices shape mismatch"
    assert capacities.shape == (num_samples, 76), "Capacities shape mismatch"
    assert speeds.shape == (num_samples, 76), "Speeds shape mismatch"
    
    if flows is not None:
        assert flows.shape == (num_samples, 76), "Flows shape mismatch"
    
    print(f"\n All shapes validated successfully!")


def check_for_nans_and_infs(data_dict):
    """
    Check for NaN and Inf values in data arrays.
    
    Args:
        data_dict: Dictionary of {name: array}
    """
    print(f"\n{'='*60}")
    print("Checking for NaN and Inf values")
    print(f"{'='*60}")
    
    has_issues = False
    for name, data in data_dict.items():
        if isinstance(data, torch.Tensor):
            data = data.numpy()
        
        num_nans = np.isnan(data).sum()
        num_infs = np.isinf(data).sum()
        
        if num_nans > 0 or num_infs > 0:
            print(f" {name}: {num_nans} NaNs, {num_infs} Infs")
            has_issues = True
        else:
            print(f"  {name}: No issues")
    
    if not has_issues:
        print(f"\nAll data clean!")
    else:
        print(f"\n Warning: Some data contains NaN or Inf values!")
    
    return not has_issues


def compute_statistics(data_dict):
    """
    Compute and print statistics for data arrays.
    
    Args:
        data_dict: Dictionary of {name: array}
    """
    print(f"\n{'='*60}")
    print("Data Statistics")
    print(f"{'='*60}")
    
    for name, data in data_dict.items():
        if isinstance(data, torch.Tensor):
            data = data.numpy()
        
        print(f"\n  {name}:")
        print(f"    Shape: {data.shape}")
        print(f"    Min: {data.min():.4f}")
        print(f"    Max: {data.max():.4f}")
        print(f"    Mean: {data.mean():.4f}")
        print(f"    Std: {data.std():.4f}")
        print(f"    Median: {np.median(data):.4f}")


def create_data_directories():
    """
    Create necessary directories for storing processed data.
    """
    import os
    
    directories = [
        'processed_data',
        'processed_data/raw',
        'processed_data/processed',
        'processed_data/ood',
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        print(f"  Created/verified directory: {directory}")

