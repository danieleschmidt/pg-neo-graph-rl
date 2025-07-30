"""Sample test to verify testing infrastructure."""

import jax.numpy as jnp
import pytest


def test_jax_setup(jax_config):
    """Test that JAX is properly configured for testing."""
    x = jnp.array([1, 2, 3])
    y = jnp.sum(x)
    assert y == 6


def test_sample_graph_fixture(sample_graph):
    """Test the sample graph fixture."""
    assert "node_features" in sample_graph
    assert "edge_index" in sample_graph
    assert sample_graph["num_nodes"] == 10
    assert sample_graph["node_features"].shape == (10, 4)


def test_federated_config_fixture(federated_config):
    """Test the federated config fixture."""
    assert federated_config["num_agents"] == 4
    assert "learning_rate" in federated_config


@pytest.mark.unit
def test_basic_computation():
    """Basic computation test."""
    result = jnp.dot(jnp.array([1, 2]), jnp.array([3, 4]))
    assert result == 11


@pytest.mark.integration
def test_integration_placeholder():
    """Placeholder for integration tests."""
    assert True