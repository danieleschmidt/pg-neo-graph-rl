"""Shared test configuration and fixtures."""

import pytest
import numpy as np
import jax
import jax.numpy as jnp


@pytest.fixture
def rng_key():
    """Random key for JAX operations."""
    return jax.random.PRNGKey(42)


@pytest.fixture
def sample_graph():
    """Sample graph data for testing."""
    return {
        "nodes": jnp.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]),
        "edges": jnp.array([[0, 1], [1, 2], [2, 0]]),
        "edge_attr": jnp.array([[0.5], [0.8], [0.3]]),
    }


@pytest.fixture
def mock_environment():
    """Mock environment for testing."""
    class MockEnv:
        def __init__(self):
            self.num_nodes = 10
            self.num_agents = 3
            
        def reset(self):
            return jnp.ones((self.num_nodes, 4))
            
        def step(self, actions):
            obs = jnp.ones((self.num_nodes, 4))
            reward = jnp.array(1.0)
            done = False
            info = {}
            return obs, reward, done, info
    
    return MockEnv()


@pytest.fixture(scope="session")
def gpu_available():
    """Check if GPU is available for testing."""
    try:
        devices = jax.devices("gpu")
        return len(devices) > 0
    except RuntimeError:
        return False