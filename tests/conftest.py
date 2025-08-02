"""Shared pytest configuration and fixtures."""

import pytest
import jax
import jax.numpy as jnp
import numpy as np
import tempfile
import shutil
import os
from pathlib import Path
from typing import Dict, Any, Generator, Optional
from unittest.mock import Mock, MagicMock
import logging


@pytest.fixture(scope="session")
def jax_config():
    """Configure JAX for testing."""
    # Use CPU for tests by default
    jax.config.update("jax_platform_name", "cpu")
    # Disable JIT compilation for faster test execution
    jax.config.update("jax_disable_jit", True)


@pytest.fixture
def sample_graph():
    """Create a small sample graph for testing."""
    num_nodes = 10
    num_edges = 15
    
    # Random edge indices
    edges = np.random.randint(0, num_nodes, size=(num_edges, 2))
    edge_index = jnp.array(edges.T)
    
    # Node features
    node_features = jnp.array(np.random.randn(num_nodes, 4))
    
    # Edge features
    edge_features = jnp.array(np.random.randn(num_edges, 2))
    
    return {
        "node_features": node_features,
        "edge_index": edge_index,
        "edge_features": edge_features,
        "num_nodes": num_nodes,
        "num_edges": num_edges
    }


@pytest.fixture
def federated_config():
    """Standard federated learning configuration for testing."""
    return {
        "num_agents": 4,
        "communication_rounds": 5,
        "aggregation_method": "fedavg",
        "learning_rate": 0.01,
        "batch_size": 32
    }


@pytest.fixture
def graph_rl_config():
    """Standard Graph RL configuration for testing."""
    return {
        "gnn_layers": 2,
        "hidden_dim": 64,
        "attention_heads": 4,
        "learning_rate": 3e-4,
        "gamma": 0.99,
        "gae_lambda": 0.95
    }


@pytest.fixture(scope="function")
def temp_model_dir(tmp_path):
    """Temporary directory for model checkpoints."""
    return tmp_path / "models"


@pytest.fixture
def mock_environment_config():
    """Mock environment configuration."""
    return {
        "num_nodes": 20,
        "edge_prob": 0.3,
        "node_features": 4,
        "edge_features": 2,
        "max_steps": 100
    }


# =============================================================================
# ADVANCED TESTING FIXTURES
# =============================================================================

@pytest.fixture(scope="session")
def test_data_dir():
    """Directory containing test data files."""
    return Path(__file__).parent / "data"


@pytest.fixture
def large_graph():
    """Create a larger graph for performance testing."""
    num_nodes = 1000
    num_edges = 2000
    
    # Create a more realistic graph structure
    edges = []
    for i in range(num_nodes):
        # Connect to a few random neighbors
        neighbors = np.random.choice(
            [j for j in range(num_nodes) if j != i], 
            size=min(5, np.random.poisson(2)), 
            replace=False
        )
        for neighbor in neighbors:
            edges.append([i, neighbor])
    
    edges = np.array(edges[:num_edges])
    edge_index = jnp.array(edges.T)
    
    # More realistic node features
    node_features = jnp.array(np.random.randn(num_nodes, 8))
    edge_features = jnp.array(np.random.randn(len(edges), 4))
    
    return {
        "node_features": node_features,
        "edge_index": edge_index,
        "edge_features": edge_features,
        "num_nodes": num_nodes,
        "num_edges": len(edges)
    }


@pytest.fixture
def traffic_network_graph():
    """Create a traffic network-like graph."""
    # Grid topology like a city street network
    grid_size = 10
    num_nodes = grid_size * grid_size
    
    # Create grid connections
    edges = []
    for i in range(grid_size):
        for j in range(grid_size):
            node_id = i * grid_size + j
            
            # Connect to right neighbor
            if j < grid_size - 1:
                edges.append([node_id, node_id + 1])
                edges.append([node_id + 1, node_id])  # Bidirectional
            
            # Connect to bottom neighbor
            if i < grid_size - 1:
                edges.append([node_id, node_id + grid_size])
                edges.append([node_id + grid_size, node_id])  # Bidirectional
    
    edge_index = jnp.array(np.array(edges).T)
    
    # Traffic-specific features
    node_features = jnp.array(np.random.uniform(0, 1, (num_nodes, 6)))  # Traffic state
    edge_features = jnp.array(np.random.uniform(0, 1, (len(edges), 3)))  # Flow, density, speed
    
    return {
        "node_features": node_features,
        "edge_index": edge_index,
        "edge_features": edge_features,
        "num_nodes": num_nodes,
        "num_edges": len(edges),
        "topology": "grid"
    }


@pytest.fixture
def power_grid_graph():
    """Create a power grid-like graph."""
    # Power grids have hub-like topology
    num_hubs = 5
    nodes_per_hub = 10
    num_nodes = num_hubs * nodes_per_hub
    
    edges = []
    
    # Connect nodes within each hub
    for hub in range(num_hubs):
        hub_start = hub * nodes_per_hub
        for i in range(nodes_per_hub):
            for j in range(i + 1, nodes_per_hub):
                node1 = hub_start + i
                node2 = hub_start + j
                if np.random.random() < 0.4:  # Sparse within-hub connections
                    edges.append([node1, node2])
                    edges.append([node2, node1])
    
    # Connect hubs to each other
    hub_centers = [h * nodes_per_hub for h in range(num_hubs)]
    for i in range(num_hubs):
        for j in range(i + 1, num_hubs):
            if np.random.random() < 0.8:  # High inter-hub connectivity
                edges.append([hub_centers[i], hub_centers[j]])
                edges.append([hub_centers[j], hub_centers[i]])
    
    edge_index = jnp.array(np.array(edges).T)
    
    # Power grid features: voltage, frequency, load, generation
    node_features = jnp.array(np.random.uniform(0.9, 1.1, (num_nodes, 4)))
    edge_features = jnp.array(np.random.uniform(0, 1, (len(edges), 2)))  # Power flow, capacity
    
    return {
        "node_features": node_features,
        "edge_index": edge_index,
        "edge_features": edge_features,
        "num_nodes": num_nodes,
        "num_edges": len(edges),
        "topology": "hub"
    }


@pytest.fixture
def swarm_graph():
    """Create a dynamic swarm-like graph."""
    num_agents = 50
    
    # Random positions in 2D space
    positions = np.random.uniform(-10, 10, (num_agents, 2))
    
    # Connect agents within communication range
    comm_range = 3.0
    edges = []
    
    for i in range(num_agents):
        for j in range(i + 1, num_agents):
            distance = np.linalg.norm(positions[i] - positions[j])
            if distance <= comm_range:
                edges.append([i, j])
                edges.append([j, i])
    
    edge_index = jnp.array(np.array(edges).T) if edges else jnp.zeros((2, 0), dtype=int)
    
    # Swarm features: position, velocity, goal
    node_features = jnp.array(np.concatenate([
        positions,
        np.random.uniform(-1, 1, (num_agents, 2)),  # velocity
        np.random.uniform(-5, 5, (num_agents, 2))   # goal position
    ], axis=1))
    
    edge_features = jnp.array(np.random.uniform(0, 1, (len(edges), 1))) if edges else jnp.zeros((0, 1))
    
    return {
        "node_features": node_features,
        "edge_index": edge_index,
        "edge_features": edge_features,
        "num_nodes": num_agents,
        "num_edges": len(edges),
        "positions": positions,
        "topology": "proximity"
    }


@pytest.fixture
def mock_federated_system():
    """Mock federated learning system for testing."""
    mock_system = Mock()
    mock_system.num_agents = 4
    mock_system.current_round = 0
    mock_system.aggregation_method = "fedavg"
    
    # Mock methods
    mock_system.aggregate_parameters = Mock(return_value={"params": "aggregated"})
    mock_system.distribute_parameters = Mock()
    mock_system.collect_updates = Mock(return_value=[{"update": i} for i in range(4)])
    
    return mock_system


@pytest.fixture
def mock_graph_environment():
    """Mock graph environment for testing."""
    mock_env = Mock()
    mock_env.num_nodes = 20
    mock_env.num_edges = 30
    mock_env.observation_space = Mock()
    mock_env.action_space = Mock()
    
    # Mock methods
    mock_env.reset = Mock(return_value=np.zeros((20, 4)))
    mock_env.step = Mock(return_value=(
        np.zeros((20, 4)),  # observation
        0.5,  # reward
        False,  # done
        {}  # info
    ))
    mock_env.render = Mock()
    
    return mock_env


@pytest.fixture
def privacy_config():
    """Privacy configuration for testing."""
    return {
        "differential_privacy": True,
        "epsilon": 1.0,
        "delta": 1e-5,
        "noise_multiplier": 1.1,
        "max_grad_norm": 1.0,
        "secure_aggregation": True,
        "min_participants": 3
    }


@pytest.fixture
def communication_config():
    """Communication configuration for testing."""
    return {
        "backend": "mock",
        "timeout": 30,
        "max_retries": 3,
        "gossip_fanout": 3,
        "gossip_interval": 10,
        "compression": False,
        "encryption": False
    }


@pytest.fixture
def benchmark_config():
    """Benchmark configuration for testing."""
    return {
        "num_episodes": 10,
        "max_steps_per_episode": 100,
        "num_runs": 3,
        "metrics": ["reward", "convergence", "communication_cost"],
        "baseline_algorithms": ["random", "greedy"],
        "statistical_significance": 0.05
    }


# =============================================================================
# TEMPORARY DIRECTORIES AND FILES
# =============================================================================

@pytest.fixture
def temp_checkpoint_dir(tmp_path):
    """Temporary directory for model checkpoints."""
    checkpoint_dir = tmp_path / "checkpoints"
    checkpoint_dir.mkdir()
    return checkpoint_dir


@pytest.fixture
def temp_log_dir(tmp_path):
    """Temporary directory for logs."""
    log_dir = tmp_path / "logs"
    log_dir.mkdir()
    return log_dir


@pytest.fixture
def temp_config_file(tmp_path):
    """Temporary configuration file."""
    config_file = tmp_path / "test_config.yaml"
    config_content = """
    system:
      log_level: DEBUG
      random_seed: 42
    
    federated_learning:
      num_agents: 4
      communication_rounds: 5
      aggregation_method: fedavg
    
    training:
      learning_rate: 0.001
      batch_size: 32
      max_episodes: 100
    """
    config_file.write_text(config_content)
    return config_file


# =============================================================================
# PERFORMANCE AND BENCHMARKING
# =============================================================================

@pytest.fixture
def performance_monitor():
    """Performance monitoring utilities."""
    import time
    import psutil
    import tracemalloc
    
    class PerformanceMonitor:
        def __init__(self):
            self.start_time = None
            self.start_memory = None
            
        def start(self):
            tracemalloc.start()
            self.start_time = time.time()
            self.start_memory = psutil.Process().memory_info().rss
            
        def stop(self):
            end_time = time.time()
            end_memory = psutil.Process().memory_info().rss
            current, peak = tracemalloc.get_traced_memory()
            tracemalloc.stop()
            
            return {
                "wall_time": end_time - self.start_time,
                "memory_used": end_memory - self.start_memory,
                "peak_memory": peak,
                "current_memory": current
            }
    
    return PerformanceMonitor()


# =============================================================================
# LOGGING AND DEBUGGING
# =============================================================================

@pytest.fixture
def test_logger():
    """Test logger configuration."""
    logger = logging.getLogger("test")
    logger.setLevel(logging.DEBUG)
    
    # Remove existing handlers
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
    
    # Add console handler
    handler = logging.StreamHandler()
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    
    return logger


@pytest.fixture
def capture_logs():
    """Capture logs during test execution."""
    import io
    log_capture = io.StringIO()
    handler = logging.StreamHandler(log_capture)
    
    # Add handler to root logger
    root_logger = logging.getLogger()
    root_logger.addHandler(handler)
    
    yield log_capture
    
    # Clean up
    root_logger.removeHandler(handler)


# =============================================================================
# PYTEST CONFIGURATION
# =============================================================================

def pytest_configure(config):
    """Configure pytest with custom markers."""
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )
    config.addinivalue_line(
        "markers", "integration: marks tests as integration tests"
    )
    config.addinivalue_line(
        "markers", "unit: marks tests as unit tests"
    )
    config.addinivalue_line(
        "markers", "gpu: marks tests that require GPU"
    )
    config.addinivalue_line(
        "markers", "distributed: marks tests that require multiple processes"
    )
    config.addinivalue_line(
        "markers", "benchmark: marks performance benchmark tests"
    )


def pytest_collection_modifyitems(config, items):
    """Modify test collection to add markers based on test location."""
    for item in items:
        # Add markers based on test location
        if "integration" in str(item.fspath):
            item.add_marker(pytest.mark.integration)
        elif "unit" in str(item.fspath):
            item.add_marker(pytest.mark.unit)
        elif "benchmarks" in str(item.fspath):
            item.add_marker(pytest.mark.benchmark)
        
        # Add slow marker for tests that might take a while
        if any(keyword in item.name.lower() for keyword in ["large", "performance", "benchmark"]):
            item.add_marker(pytest.mark.slow)


@pytest.fixture(autouse=True)
def setup_test_environment():
    """Automatically setup test environment for each test."""
    # Set random seeds
    np.random.seed(42)
    
    # Configure JAX
    jax.config.update("jax_platform_name", "cpu")
    jax.config.update("jax_enable_x64", False)
    
    # Set environment variables
    os.environ["ENVIRONMENT"] = "testing"
    os.environ["LOG_LEVEL"] = "WARNING"  # Reduce log noise during tests
    
    yield
    
    # Cleanup after test
    # Reset JAX compilation cache
    jax.clear_caches()