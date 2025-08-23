"""Basic integration tests for pg-neo-graph-rl."""
import pytest
import numpy as np
import jax.numpy as jnp
from pg_neo_graph_rl import FederatedGraphRL, TrafficEnvironment, GraphPPO
from pg_neo_graph_rl.core.types import GraphState


def test_federated_system_initialization():
    """Test basic federated system initialization."""
    fed_system = FederatedGraphRL(num_agents=2)
    assert fed_system.config.num_agents == 2
    # Note: agents are created lazily, so we test the configuration


def test_traffic_environment():
    """Test traffic environment basic operations."""
    env = TrafficEnvironment(num_intersections=4)  # Fixed: environment creates 4 nodes by default
    state = env.reset()
    
    assert isinstance(state, GraphState)
    assert state.nodes.shape[0] == 4  # 4 intersections
    assert state.nodes.shape[1] > 0   # Has features
    

def test_agent_creation():
    """Test agent creation and basic operations."""
    agent = GraphPPO(agent_id=0, action_dim=3, node_dim=4)  # Fixed: added required args
    assert agent.agent_id == 0
    assert hasattr(agent, 'actor')  # Fixed: correct attribute names
    assert hasattr(agent, 'critic')


def test_basic_federated_workflow():
    """Test basic federated learning workflow."""
    # Create small system for testing
    fed_system = FederatedGraphRL(num_agents=2)
    env = TrafficEnvironment(num_intersections=4)
    
    # Reset environment
    state = env.reset()
    
    # Get subgraphs for agents
    subgraph_0 = fed_system.get_subgraph(0, state)
    subgraph_1 = fed_system.get_subgraph(1, state)
    
    assert isinstance(subgraph_0, GraphState)
    assert isinstance(subgraph_1, GraphState)
    
    # Test that subgraphs have reasonable sizes
    assert subgraph_0.nodes.shape[0] > 0
    assert subgraph_1.nodes.shape[0] > 0


def test_environment_step():
    """Test environment step functionality."""
    env = TrafficEnvironment(num_intersections=4)
    state = env.reset()
    
    # Create JAX array actions (fixed)
    actions = jnp.array([0, 1, 2, 1])  # Fixed: use JAX array
    
    # Take environment step
    next_state, rewards, done, info = env.step(actions)
    
    assert isinstance(next_state, GraphState)
    assert isinstance(rewards, jnp.ndarray)  # Fixed: returns JAX array
    assert rewards.shape[0] == 4
    assert isinstance(done, bool)
    assert isinstance(info, dict)


def test_federated_aggregation():
    """Test basic federated aggregation."""
    fed_system = FederatedGraphRL(num_agents=2)
    
    # Create dummy gradients - single gradient per agent
    dummy_grad_0 = np.random.randn(10)
    dummy_grad_1 = np.random.randn(10)
    
    # Test gossip aggregation (fixed method signature)
    result_0 = fed_system.gossip_aggregate(0, dummy_grad_0, {})
    result_1 = fed_system.gossip_aggregate(1, dummy_grad_1, {})
    
    assert isinstance(result_0, np.ndarray)
    assert isinstance(result_1, np.ndarray)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])