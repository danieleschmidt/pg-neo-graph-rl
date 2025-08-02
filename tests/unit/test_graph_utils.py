"""Unit tests for graph utility functions."""

import pytest
import jax.numpy as jnp
import numpy as np
from tests.fixtures.graph_data import (
    create_synthetic_traffic_graph,
    create_synthetic_power_grid,
    create_synthetic_swarm_graph,
    validate_graph_structure,
    compute_graph_statistics
)
from tests.utils import assert_arrays_close, assert_graph_properties


@pytest.mark.unit
class TestGraphDataFixtures:
    """Test graph data generation fixtures."""
    
    def test_traffic_graph_creation(self):
        """Test synthetic traffic graph creation."""
        graph = create_synthetic_traffic_graph(num_intersections=25, seed=42)
        
        # Validate structure
        assert validate_graph_structure(graph)
        assert graph["graph_type"] == "traffic"
        assert graph["num_nodes"] == 25
        
        # Check feature dimensions
        assert graph["node_features"].shape == (25, 4)
        assert graph["edge_features"].shape[1] == 3
        
        # Check reproducibility
        graph2 = create_synthetic_traffic_graph(num_intersections=25, seed=42)
        assert_arrays_close(graph["node_features"], graph2["node_features"])
    
    def test_power_grid_creation(self):
        """Test synthetic power grid creation."""
        graph = create_synthetic_power_grid(num_buses=20, seed=42)
        
        assert validate_graph_structure(graph)
        assert graph["graph_type"] == "power_grid"
        assert graph["num_nodes"] == 20
        assert graph["node_features"].shape == (20, 4)
        assert graph["edge_features"].shape[1] == 3
    
    def test_swarm_graph_creation(self):
        """Test synthetic swarm graph creation."""
        graph = create_synthetic_swarm_graph(num_agents=15, seed=42)
        
        assert validate_graph_structure(graph)
        assert graph["graph_type"] == "swarm"
        assert graph["num_nodes"] == 15
        assert graph["node_features"].shape == (15, 5)  # includes position + velocity + battery
        
        # Check position consistency
        positions = graph["positions"]
        assert positions.shape == (15, 2)
        
        # Verify edges respect communication range
        for i in range(graph["edge_index"].shape[1]):
            u, v = graph["edge_index"][:, i]
            distance = np.linalg.norm(positions[u] - positions[v])
            assert distance <= graph["communication_range"]
    
    def test_graph_validation(self):
        """Test graph structure validation."""
        # Valid graph
        valid_graph = {
            "node_features": jnp.array([[1, 2], [3, 4]]),
            "edge_index": jnp.array([[0, 1], [1, 0]]).T,
            "num_nodes": 2,
            "num_edges": 2
        }
        assert validate_graph_structure(valid_graph)
        
        # Invalid: wrong node count
        invalid_graph = valid_graph.copy()
        invalid_graph["num_nodes"] = 3
        assert not validate_graph_structure(invalid_graph)
        
        # Invalid: edge indices out of bounds
        invalid_graph = valid_graph.copy()
        invalid_graph["edge_index"] = jnp.array([[0, 2], [1, 0]]).T
        assert not validate_graph_structure(invalid_graph)
    
    def test_graph_statistics(self):
        """Test graph statistics computation."""
        graph = create_synthetic_traffic_graph(num_intersections=16, seed=42)
        stats = compute_graph_statistics(graph)
        
        # Check required statistics
        required_stats = ["num_nodes", "num_edges", "density", "avg_degree", 
                         "node_feature_mean", "node_feature_std"]
        for stat in required_stats:
            assert stat in stats
            assert isinstance(stats[stat], (int, float))
        
        # Validate ranges
        assert 0 <= stats["density"] <= 1
        assert stats["avg_degree"] >= 0
        assert stats["num_nodes"] == 16


@pytest.mark.unit
class TestGraphTransformations:
    """Test graph transformation utilities."""
    
    def test_graph_normalization(self):
        """Test graph feature normalization."""
        graph = create_synthetic_traffic_graph(num_intersections=10, seed=42)
        
        # Test node feature normalization
        normalized_features = (graph["node_features"] - jnp.mean(graph["node_features"], axis=0)) / \
                             jnp.std(graph["node_features"], axis=0)
        
        # Check mean is approximately zero
        feature_means = jnp.mean(normalized_features, axis=0)
        assert_arrays_close(feature_means, jnp.zeros_like(feature_means), atol=1e-6)
        
        # Check std is approximately one
        feature_stds = jnp.std(normalized_features, axis=0)
        assert_arrays_close(feature_stds, jnp.ones_like(feature_stds), atol=1e-6)
    
    def test_edge_index_validation(self):
        """Test edge index format validation."""
        # Test correct format (2, num_edges)
        edge_index = jnp.array([[0, 1, 2], [1, 2, 0]])
        assert edge_index.shape[0] == 2
        
        # Test all indices are valid
        num_nodes = 3
        assert jnp.all(edge_index >= 0)
        assert jnp.all(edge_index < num_nodes)
    
    def test_adjacency_matrix_conversion(self):
        """Test conversion between edge_index and adjacency matrix."""
        num_nodes = 4
        edge_index = jnp.array([[0, 1, 2, 3], [1, 2, 3, 0]])
        
        # Convert to adjacency matrix
        adjacency = jnp.zeros((num_nodes, num_nodes))
        adjacency = adjacency.at[edge_index[0], edge_index[1]].set(1)
        
        # Check symmetric for undirected graph
        symmetric_adjacency = adjacency + adjacency.T
        assert jnp.all(symmetric_adjacency >= adjacency)
        
        # Check diagonal is zero (no self-loops)
        assert jnp.all(jnp.diag(adjacency) == 0)


@pytest.mark.unit
class TestGraphSampling:
    """Test graph sampling and subgraph extraction."""
    
    def test_node_sampling(self):
        """Test random node sampling."""
        graph = create_synthetic_traffic_graph(num_intersections=20, seed=42)
        
        # Sample half the nodes
        num_sample = 10
        rng = np.random.RandomState(42)
        sampled_nodes = rng.choice(graph["num_nodes"], num_sample, replace=False)
        
        # Extract subgraph
        sampled_features = graph["node_features"][sampled_nodes]
        
        assert sampled_features.shape == (num_sample, 4)
        assert len(np.unique(sampled_nodes)) == num_sample
    
    def test_edge_sampling(self):
        """Test edge sampling for large graphs."""
        graph = create_synthetic_traffic_graph(num_intersections=25, seed=42)
        
        # Sample edges
        num_sample_edges = min(graph["num_edges"] // 2, 10)
        rng = np.random.RandomState(42)
        edge_indices = rng.choice(graph["num_edges"], num_sample_edges, replace=False)
        
        sampled_edges = graph["edge_index"][:, edge_indices]
        sampled_edge_features = graph["edge_features"][edge_indices]
        
        assert sampled_edges.shape == (2, num_sample_edges)
        assert sampled_edge_features.shape == (num_sample_edges, 3)
    
    def test_neighborhood_sampling(self):
        """Test k-hop neighborhood sampling."""
        graph = create_synthetic_traffic_graph(num_intersections=16, seed=42)
        
        # Build adjacency list for efficient neighborhood queries
        adjacency_list = {}
        for i in range(graph["num_nodes"]):
            adjacency_list[i] = []
        
        for i in range(graph["edge_index"].shape[1]):
            u, v = graph["edge_index"][:, i]
            adjacency_list[int(u)].append(int(v))
            adjacency_list[int(v)].append(int(u))  # Undirected
        
        # Test 1-hop neighborhood
        center_node = 0
        neighbors = set(adjacency_list[center_node])
        neighbors.add(center_node)  # Include center node
        
        assert center_node in neighbors
        assert len(neighbors) >= 1  # At least the center node


@pytest.mark.unit 
@pytest.mark.parametrize("graph_type", ["traffic", "power_grid", "swarm"])
def test_graph_type_properties(graph_type):
    """Test properties specific to each graph type."""
    if graph_type == "traffic":
        graph = create_synthetic_traffic_graph(num_intersections=16, seed=42)
        expected_node_features = 4  # traffic_flow, queue_length, signal_phase, capacity
        expected_edge_features = 3  # travel_time, capacity, congestion_level
    elif graph_type == "power_grid":
        graph = create_synthetic_power_grid(num_buses=16, seed=42)
        expected_node_features = 4  # voltage_magnitude, voltage_angle, active_power, reactive_power
        expected_edge_features = 3  # resistance, reactance, thermal_limit
    else:  # swarm
        graph = create_synthetic_swarm_graph(num_agents=16, seed=42)
        expected_node_features = 5  # x_pos, y_pos, velocity_x, velocity_y, battery_level
        expected_edge_features = 2  # distance, signal_strength
    
    assert graph["graph_type"] == graph_type
    assert graph["node_features"].shape[1] == expected_node_features
    assert graph["edge_features"].shape[1] == expected_edge_features
    assert validate_graph_structure(graph)


@pytest.mark.unit
def test_reproducibility():
    """Test that graph generation is reproducible with same seed."""
    seed = 12345
    
    # Generate graphs multiple times with same seed
    graph1 = create_synthetic_traffic_graph(num_intersections=10, seed=seed)
    graph2 = create_synthetic_traffic_graph(num_intersections=10, seed=seed)
    
    # Should be identical
    assert_arrays_close(graph1["node_features"], graph2["node_features"])
    assert_arrays_close(graph1["edge_index"], graph2["edge_index"])
    assert_arrays_close(graph1["edge_features"], graph2["edge_features"])
    
    # Different seed should produce different results
    graph3 = create_synthetic_traffic_graph(num_intersections=10, seed=seed+1)
    
    # Should be different (with high probability)
    with pytest.raises(AssertionError):
        assert_arrays_close(graph1["node_features"], graph3["node_features"])


@pytest.mark.unit
def test_edge_cases():
    """Test edge cases in graph generation."""
    # Minimum graph size
    small_graph = create_synthetic_traffic_graph(num_intersections=4, seed=42)
    assert validate_graph_structure(small_graph)
    assert small_graph["num_nodes"] == 4
    
    # Single node (degenerate case)
    single_node_graph = {
        "node_features": jnp.array([[1, 2, 3, 4]]),
        "edge_index": jnp.array([[], []]).astype(int),
        "edge_features": jnp.array([]).reshape(0, 3),
        "num_nodes": 1,
        "num_edges": 0
    }
    assert validate_graph_structure(single_node_graph)
    
    # Empty graph (pathological case)
    empty_graph = {
        "node_features": jnp.array([]).reshape(0, 4),
        "edge_index": jnp.array([[], []]).astype(int),
        "edge_features": jnp.array([]).reshape(0, 3),
        "num_nodes": 0,
        "num_edges": 0
    }
    # Note: This should probably fail validation
    assert not validate_graph_structure(empty_graph)