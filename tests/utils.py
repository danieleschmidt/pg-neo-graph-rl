"""Testing utilities and helper functions."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import time
import functools
from typing import Any, Callable, Dict, List, Optional, Tuple
from unittest.mock import Mock, patch


def with_jax_config(**config_kwargs):
    """Decorator to temporarily set JAX configuration for a test.
    
    Args:
        **config_kwargs: JAX configuration options to set
    
    Example:
        @with_jax_config(jax_disable_jit=True, jax_platform_name="cpu")
        def test_function():
            pass
    """
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Save original config
            original_config = {}
            for key, value in config_kwargs.items():
                try:
                    original_config[key] = jax.config.read(key)
                except:
                    original_config[key] = None
                jax.config.update(key, value)
            
            try:
                return func(*args, **kwargs)
            finally:
                # Restore original config
                for key, value in original_config.items():
                    if value is not None:
                        jax.config.update(key, value)
        
        return wrapper
    return decorator


def assert_arrays_close(a: jnp.ndarray, b: jnp.ndarray, rtol: float = 1e-5, atol: float = 1e-8):
    """Assert that two JAX arrays are close within tolerance.
    
    Args:
        a, b: Arrays to compare
        rtol: Relative tolerance
        atol: Absolute tolerance
    """
    np.testing.assert_allclose(np.array(a), np.array(b), rtol=rtol, atol=atol)


def assert_graph_properties(graph: Dict[str, Any], expected_properties: Dict[str, Any]):
    """Assert that a graph has expected properties.
    
    Args:
        graph: Graph dictionary
        expected_properties: Dictionary of expected property values
    """
    for prop, expected_value in expected_properties.items():
        if prop in ["num_nodes", "num_edges"]:
            assert graph[prop] == expected_value, f"{prop}: got {graph[prop]}, expected {expected_value}"
        elif prop in ["node_features_shape", "edge_features_shape"]:
            feature_key = prop.replace("_shape", "")
            if feature_key in graph:
                actual_shape = graph[feature_key].shape
                assert actual_shape == expected_value, f"{prop}: got {actual_shape}, expected {expected_value}"


def mock_federated_agents(num_agents: int, agent_class: Any = None) -> List[Mock]:
    """Create mock federated learning agents.
    
    Args:
        num_agents: Number of agents to create
        agent_class: Optional agent class to mock
        
    Returns:
        List of mock agent objects
    """
    agents = []
    for i in range(num_agents):
        agent = Mock()
        agent.agent_id = i
        agent.get_parameters.return_value = jnp.random.normal(
            key=jax.random.PRNGKey(i), shape=(100,)
        )
        agent.set_parameters = Mock()
        agent.train_step = Mock(return_value={"loss": 0.5, "accuracy": 0.8})
        agents.append(agent)
    
    return agents


def create_mock_environment(env_type: str = "traffic", **kwargs) -> Mock:
    """Create a mock graph environment.
    
    Args:
        env_type: Type of environment (traffic, power_grid, swarm)
        **kwargs: Additional environment parameters
        
    Returns:
        Mock environment object
    """
    env = Mock()
    
    # Default parameters based on environment type
    if env_type == "traffic":
        num_nodes = kwargs.get("num_nodes", 50)
        node_features_dim = 4
        edge_features_dim = 3
    elif env_type == "power_grid":
        num_nodes = kwargs.get("num_nodes", 30)
        node_features_dim = 4
        edge_features_dim = 3
    elif env_type == "swarm":
        num_nodes = kwargs.get("num_nodes", 25)
        node_features_dim = 5
        edge_features_dim = 2
    else:
        num_nodes = kwargs.get("num_nodes", 20)
        node_features_dim = 4
        edge_features_dim = 2
    
    num_edges = kwargs.get("num_edges", num_nodes * 2)
    
    # Mock environment methods
    env.reset.return_value = {
        "node_features": jnp.random.normal(key=jax.random.PRNGKey(42), shape=(num_nodes, node_features_dim)),
        "edge_index": jnp.random.randint(key=jax.random.PRNGKey(43), minval=0, maxval=num_nodes, shape=(2, num_edges)),
        "edge_features": jnp.random.normal(key=jax.random.PRNGKey(44), shape=(num_edges, edge_features_dim)),
    }
    
    env.step.return_value = (
        env.reset.return_value,  # next_state
        jnp.random.normal(key=jax.random.PRNGKey(45), shape=(num_nodes,)),  # rewards
        False,  # done
        {"step": 1}  # info
    )
    
    env.num_nodes = num_nodes
    env.num_edges = num_edges
    env.action_space = Mock()
    env.observation_space = Mock()
    
    return env


class Timer:
    """Context manager for timing code execution."""
    
    def __init__(self, name: str = "Operation"):
        self.name = name
        self.start_time = None
        self.end_time = None
        self.elapsed = None
    
    def __enter__(self):
        self.start_time = time.perf_counter()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.end_time = time.perf_counter()
        self.elapsed = self.end_time - self.start_time
        print(f"{self.name}: {self.elapsed:.4f}s")


def memory_usage_test(func: Callable, *args, **kwargs) -> Dict[str, Any]:
    """Test memory usage of a function (simplified version).
    
    Args:
        func: Function to test
        *args, **kwargs: Arguments for the function
        
    Returns:
        Dictionary with memory usage information
    """
    # This is a simplified version - in production you'd use memory_profiler
    import psutil
    import os
    
    process = psutil.Process(os.getpid())
    
    # Memory before
    memory_before = process.memory_info().rss / 1024 / 1024  # MB
    
    # Execute function
    start_time = time.perf_counter()
    result = func(*args, **kwargs)
    end_time = time.perf_counter()
    
    # Memory after
    memory_after = process.memory_info().rss / 1024 / 1024  # MB
    
    return {
        "result": result,
        "memory_before_mb": memory_before,
        "memory_after_mb": memory_after,
        "memory_delta_mb": memory_after - memory_before,
        "execution_time_s": end_time - start_time
    }


def parametrize_graph_types(graph_types: List[str] = None):
    """Pytest parametrize decorator for different graph types.
    
    Args:
        graph_types: List of graph types to test
    """
    if graph_types is None:
        graph_types = ["traffic", "power_grid", "swarm"]
    
    return pytest.mark.parametrize("graph_type", graph_types)


def parametrize_agent_counts(agent_counts: List[int] = None):
    """Pytest parametrize decorator for different agent counts.
    
    Args:
        agent_counts: List of agent counts to test
    """
    if agent_counts is None:
        agent_counts = [1, 4, 8, 16]
    
    return pytest.mark.parametrize("num_agents", agent_counts)


def skip_if_no_gpu():
    """Skip test if no GPU is available."""
    try:
        devices = jax.devices("gpu")
        return len(devices) == 0
    except:
        return True


def requires_gpu(func):
    """Decorator to skip test if no GPU is available."""
    return pytest.mark.skipif(skip_if_no_gpu(), reason="No GPU available")(func)


def slow_test(func):
    """Decorator to mark a test as slow."""
    return pytest.mark.slow(func)


def integration_test(func):
    """Decorator to mark a test as integration test."""
    return pytest.mark.integration(func)


def benchmark_test(func):
    """Decorator to mark a test as benchmark test."""
    return pytest.mark.benchmark(func)


class MockCommunicationChannel:
    """Mock communication channel for federated learning tests."""
    
    def __init__(self, agents: List[int], latency: float = 0.01):
        self.agents = agents
        self.latency = latency
        self.message_log = []
        self.failure_rate = 0.0
    
    def send_message(self, sender: int, receiver: int, message: Any) -> bool:
        """Send a message between agents."""
        # Simulate network latency
        time.sleep(self.latency)
        
        # Simulate message failures
        if np.random.random() < self.failure_rate:
            return False
        
        self.message_log.append({
            "sender": sender,
            "receiver": receiver,
            "message": message,
            "timestamp": time.time()
        })
        
        return True
    
    def broadcast(self, sender: int, message: Any) -> List[bool]:
        """Broadcast a message to all other agents."""
        results = []
        for agent in self.agents:
            if agent != sender:
                success = self.send_message(sender, agent, message)
                results.append(success)
        return results
    
    def get_message_log(self) -> List[Dict]:
        """Get the communication log."""
        return self.message_log.copy()
    
    def clear_log(self):
        """Clear the communication log."""
        self.message_log.clear()
    
    def set_failure_rate(self, rate: float):
        """Set the message failure rate."""
        self.failure_rate = max(0.0, min(1.0, rate))


def assert_federated_convergence(
    loss_history: List[float],
    tolerance: float = 0.01,
    min_improvement: float = 0.1
) -> bool:
    """Assert that federated learning is converging.
    
    Args:
        loss_history: List of loss values over time
        tolerance: Tolerance for convergence check
        min_improvement: Minimum improvement required
        
    Returns:
        True if converging, raises AssertionError otherwise
    """
    if len(loss_history) < 2:
        raise AssertionError("Need at least 2 loss values to check convergence")
    
    # Check that loss is generally decreasing
    initial_loss = loss_history[0]
    final_loss = loss_history[-1]
    improvement = (initial_loss - final_loss) / initial_loss
    
    assert improvement >= min_improvement, f"Insufficient improvement: {improvement:.3f} < {min_improvement}"
    
    # Check that recent losses are stable
    recent_losses = loss_history[-5:] if len(loss_history) >= 5 else loss_history
    loss_std = np.std(recent_losses)
    loss_mean = np.mean(recent_losses)
    
    if loss_mean > 0:
        relative_std = loss_std / loss_mean
        assert relative_std <= tolerance, f"Loss not stable: relative_std={relative_std:.3f} > {tolerance}"
    
    return True


def create_test_config(overrides: Dict[str, Any] = None) -> Dict[str, Any]:
    """Create a standard test configuration.
    
    Args:
        overrides: Configuration values to override
        
    Returns:
        Test configuration dictionary
    """
    config = {
        # Model configuration
        "gnn_layers": 2,
        "hidden_dim": 64,
        "attention_heads": 4,
        "dropout_rate": 0.1,
        
        # Training configuration
        "learning_rate": 3e-4,
        "batch_size": 32,
        "num_epochs": 10,
        "gamma": 0.99,
        "gae_lambda": 0.95,
        
        # Federated learning configuration
        "num_agents": 4,
        "communication_rounds": 5,
        "aggregation_method": "fedavg",
        "local_steps": 10,
        
        # Environment configuration
        "max_episode_steps": 100,
        "reward_scale": 1.0,
        
        # Testing configuration
        "test_episodes": 5,
        "random_seed": 42,
        "device": "cpu"
    }
    
    if overrides:
        config.update(overrides)
    
    return config