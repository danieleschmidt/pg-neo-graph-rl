"""Graph data fixtures for testing."""

import jax
import jax.numpy as jnp
import networkx as nx
import numpy as np
from typing import Dict, Tuple, Any


def create_synthetic_traffic_graph(num_intersections: int = 50, seed: int = 42) -> Dict[str, Any]:
    """Create a synthetic traffic network graph.
    
    Args:
        num_intersections: Number of traffic intersections
        seed: Random seed for reproducibility
        
    Returns:
        Dictionary containing graph structure and features
    """
    rng = np.random.RandomState(seed)
    
    # Create street network topology
    graph = nx.grid_2d_graph(int(np.sqrt(num_intersections)), int(np.sqrt(num_intersections)))
    graph = nx.convert_node_labels_to_integers(graph)
    
    # Add random additional connections
    for _ in range(num_intersections // 4):
        u, v = rng.choice(num_intersections, 2, replace=False)
        if not graph.has_edge(u, v):
            graph.add_edge(u, v)
    
    # Extract edge information
    edges = list(graph.edges())
    edge_index = jnp.array([[u, v] for u, v in edges]).T
    
    # Node features: [traffic_flow, queue_length, signal_phase, capacity]
    node_features = jnp.array(rng.random((num_intersections, 4)))
    
    # Edge features: [travel_time, capacity, congestion_level]
    edge_features = jnp.array(rng.random((len(edges), 3)))
    
    return {
        "node_features": node_features,
        "edge_index": edge_index,
        "edge_features": edge_features,
        "num_nodes": num_intersections,
        "num_edges": len(edges),
        "graph_type": "traffic",
        "nx_graph": graph
    }


def create_synthetic_power_grid(num_buses: int = 30, seed: int = 42) -> Dict[str, Any]:
    """Create a synthetic power grid graph.
    
    Args:
        num_buses: Number of power buses
        seed: Random seed for reproducibility
        
    Returns:
        Dictionary containing graph structure and features
    """
    rng = np.random.RandomState(seed)
    
    # Create power grid topology (tree + some cycles for redundancy)
    graph = nx.random_tree(num_buses, seed=seed)
    
    # Add some cycles for grid stability
    num_additional_edges = num_buses // 5
    for _ in range(num_additional_edges):
        u, v = rng.choice(num_buses, 2, replace=False)
        if not graph.has_edge(u, v):
            graph.add_edge(u, v)
    
    edges = list(graph.edges())
    edge_index = jnp.array([[u, v] for u, v in edges]).T
    
    # Node features: [voltage_magnitude, voltage_angle, active_power, reactive_power]
    node_features = jnp.array(rng.random((num_buses, 4)))
    
    # Edge features: [resistance, reactance, thermal_limit]
    edge_features = jnp.array(rng.random((len(edges), 3)))
    
    return {
        "node_features": node_features,
        "edge_index": edge_index,
        "edge_features": edge_features,
        "num_nodes": num_buses,
        "num_edges": len(edges),
        "graph_type": "power_grid",
        "nx_graph": graph
    }


def create_synthetic_swarm_graph(num_agents: int = 25, seed: int = 42) -> Dict[str, Any]:
    """Create a synthetic swarm communication graph.
    
    Args:
        num_agents: Number of swarm agents
        seed: Random seed for reproducibility
        
    Returns:
        Dictionary containing graph structure and features
    """
    rng = np.random.RandomState(seed)
    
    # Create proximity-based communication graph
    positions = rng.random((num_agents, 2)) * 100  # 100x100 area
    communication_range = 25.0
    
    edges = []
    for i in range(num_agents):
        for j in range(i + 1, num_agents):
            distance = np.linalg.norm(positions[i] - positions[j])
            if distance <= communication_range:
                edges.append((i, j))
    
    edge_index = jnp.array([[u, v] for u, v in edges]).T
    
    # Node features: [x_pos, y_pos, velocity_x, velocity_y, battery_level]
    node_features = jnp.concatenate([
        jnp.array(positions),
        jnp.array(rng.random((num_agents, 3)))  # velocity + battery
    ], axis=1)
    
    # Edge features: [distance, signal_strength]
    edge_distances = jnp.array([
        np.linalg.norm(positions[u] - positions[v])
        for u, v in edges
    ])
    edge_features = jnp.column_stack([
        edge_distances,
        jnp.maximum(0, 1 - edge_distances / communication_range)  # signal strength
    ])
    
    return {
        "node_features": node_features,
        "edge_index": edge_index,
        "edge_features": edge_features,
        "num_nodes": num_agents,
        "num_edges": len(edges),
        "graph_type": "swarm",
        "positions": positions,
        "communication_range": communication_range
    }


def create_dynamic_graph_sequence(
    base_graph: Dict[str, Any], 
    num_timesteps: int = 10,
    seed: int = 42
) -> Dict[str, Any]:
    """Create a sequence of dynamic graphs.
    
    Args:
        base_graph: Base graph structure
        num_timesteps: Number of time steps
        seed: Random seed
        
    Returns:
        Dictionary containing temporal graph sequence
    """
    rng = np.random.RandomState(seed)
    num_nodes = base_graph["num_nodes"]
    feature_dim = base_graph["node_features"].shape[1]
    
    # Generate temporal node features
    temporal_features = []
    for t in range(num_timesteps):
        # Add temporal drift to features
        drift = rng.normal(0, 0.1, (num_nodes, feature_dim))
        features = base_graph["node_features"] + drift
        temporal_features.append(features)
    
    temporal_features = jnp.stack(temporal_features)
    
    # Generate timestamps
    timestamps = jnp.arange(num_timesteps, dtype=jnp.float32)
    
    return {
        **base_graph,
        "temporal_node_features": temporal_features,
        "timestamps": timestamps,
        "num_timesteps": num_timesteps
    }


def create_federated_graph_partition(
    graph: Dict[str, Any], 
    num_agents: int = 4,
    overlap_ratio: float = 0.1,
    seed: int = 42
) -> Dict[int, Dict[str, Any]]:
    """Partition a graph for federated learning.
    
    Args:
        graph: Graph to partition
        num_agents: Number of federated agents
        overlap_ratio: Ratio of overlapping nodes between partitions
        seed: Random seed
        
    Returns:
        Dictionary mapping agent_id to subgraph
    """
    rng = np.random.RandomState(seed)
    num_nodes = graph["num_nodes"]
    
    # Simple node assignment with overlap
    agent_nodes = {}
    nodes_per_agent = num_nodes // num_agents
    overlap_nodes = int(nodes_per_agent * overlap_ratio)
    
    for agent_id in range(num_agents):
        start_idx = agent_id * nodes_per_agent
        end_idx = min(start_idx + nodes_per_agent + overlap_nodes, num_nodes)
        agent_nodes[agent_id] = list(range(start_idx, end_idx))
    
    # Create subgraphs
    agent_subgraphs = {}
    for agent_id, node_list in agent_nodes.items():
        # Extract subgraph edges
        edge_mask = jnp.isin(graph["edge_index"][0], jnp.array(node_list)) & \
                   jnp.isin(graph["edge_index"][1], jnp.array(node_list))
        
        subgraph_edges = graph["edge_index"][:, edge_mask]
        
        # Remap node indices to be contiguous
        node_mapping = {old_id: new_id for new_id, old_id in enumerate(node_list)}
        remapped_edges = jnp.array([
            [node_mapping[edge[0]], node_mapping[edge[1]]]
            for edge in subgraph_edges.T
        ]).T
        
        agent_subgraphs[agent_id] = {
            "node_features": graph["node_features"][node_list],
            "edge_index": remapped_edges,
            "edge_features": graph["edge_features"][edge_mask] if "edge_features" in graph else None,
            "num_nodes": len(node_list),
            "num_edges": subgraph_edges.shape[1],
            "original_node_ids": jnp.array(node_list),
            "agent_id": agent_id
        }
    
    return agent_subgraphs


def create_benchmark_graphs() -> Dict[str, Dict[str, Any]]:
    """Create a suite of benchmark graphs for testing.
    
    Returns:
        Dictionary of named benchmark graphs
    """
    benchmarks = {}
    
    # Small graphs for unit tests
    benchmarks["small_traffic"] = create_synthetic_traffic_graph(16, seed=42)
    benchmarks["small_power"] = create_synthetic_power_grid(12, seed=42)
    benchmarks["small_swarm"] = create_synthetic_swarm_graph(10, seed=42)
    
    # Medium graphs for integration tests
    benchmarks["medium_traffic"] = create_synthetic_traffic_graph(100, seed=42)
    benchmarks["medium_power"] = create_synthetic_power_grid(50, seed=42)
    benchmarks["medium_swarm"] = create_synthetic_swarm_graph(50, seed=42)
    
    # Large graphs for performance tests
    benchmarks["large_traffic"] = create_synthetic_traffic_graph(400, seed=42)
    benchmarks["large_power"] = create_synthetic_power_grid(200, seed=42)
    benchmarks["large_swarm"] = create_synthetic_swarm_graph(200, seed=42)
    
    # Dynamic graph sequences
    benchmarks["dynamic_traffic"] = create_dynamic_graph_sequence(
        benchmarks["medium_traffic"], num_timesteps=20, seed=42
    )
    
    return benchmarks


# Utility functions for test data validation
def validate_graph_structure(graph: Dict[str, Any]) -> bool:
    """Validate that a graph has the required structure."""
    required_keys = ["node_features", "edge_index", "num_nodes", "num_edges"]
    
    if not all(key in graph for key in required_keys):
        return False
    
    # Check dimensions
    if graph["node_features"].shape[0] != graph["num_nodes"]:
        return False
    
    if graph["edge_index"].shape[1] != graph["num_edges"]:
        return False
    
    # Check edge indices are valid
    max_node_id = jnp.max(graph["edge_index"])
    if max_node_id >= graph["num_nodes"]:
        return False
    
    return True


def compute_graph_statistics(graph: Dict[str, Any]) -> Dict[str, float]:
    """Compute basic statistics for a graph."""
    num_nodes = graph["num_nodes"]
    num_edges = graph["num_edges"]
    
    # Basic connectivity stats
    density = 2 * num_edges / (num_nodes * (num_nodes - 1))
    avg_degree = 2 * num_edges / num_nodes
    
    # Feature statistics
    node_feature_mean = float(jnp.mean(graph["node_features"]))
    node_feature_std = float(jnp.std(graph["node_features"]))
    
    stats = {
        "num_nodes": num_nodes,
        "num_edges": num_edges,
        "density": density,
        "avg_degree": avg_degree,
        "node_feature_mean": node_feature_mean,
        "node_feature_std": node_feature_std
    }
    
    if "edge_features" in graph and graph["edge_features"] is not None:
        stats["edge_feature_mean"] = float(jnp.mean(graph["edge_features"]))
        stats["edge_feature_std"] = float(jnp.std(graph["edge_features"]))
    
    return stats