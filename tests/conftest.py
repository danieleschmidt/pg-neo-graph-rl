"""Shared pytest configuration and fixtures."""

import pytest
import jax
import jax.numpy as jnp
import numpy as np
from typing import Dict, Any


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