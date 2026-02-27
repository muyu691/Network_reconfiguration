import pandas as pd
import numpy as np
import networkx as nx

def load_sioux_falls_network(net_file_path):
    """
    Loading the Sioux Falls network

    Returns:
        G: NetworkX directed graph
        centroids: List of centroid nodes
        num_nodes: Total number of nodes
        num_edges: Total number of edges
    """
    df = pd.read_csv(
        net_file_path, 
        skiprows=9,  # Skip metadata AND header row
        sep='\t',
        names=['marker', 'init_node', 'term_node', 'capacity', 'length', 
               'free_flow_time', 'b', 'power', 'speed', 'toll', 'type', 'semicolon'],
        usecols=['init_node', 'term_node', 'capacity', 'length', 
                 'free_flow_time', 'speed']  
    )
    
    G = nx.DiGraph()
    
    for _, row in df.iterrows():
        G.add_edge(
            int(row['init_node']),
            int(row['term_node']),
            capacity=row['capacity'],
            free_flow_time=row['free_flow_time'],
            speed=row['speed'],
            length=row['length']
        )
    
    centroids = list(range(1, 12))  
    
    print(f"Network statistics:")
    print(f"  Number of nodes: {G.number_of_nodes()}")  
    print(f"  Number of edges: {G.number_of_edges()}")    
    print(f"  Centroids: {centroids}")
    
    return G, centroids