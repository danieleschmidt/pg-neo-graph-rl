#!/usr/bin/env python3
"""
Comprehensive test suite for pg-neo-graph-rl.
"""
import pytest
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import jax
import jax.numpy as jnp
import numpy as np
from pg_neo_graph_rl import FederatedGraphRL, GraphPPO, TrafficEnvironment
from pg_neo_graph_rl.core.types import GraphState
from pg_neo_graph_rl.utils.exceptions import ValidationError, FederatedLearningError


class TestEnvironments:
    """Test environment functionality."""
    
    def test_traffic_environment_initialization(self):
        """Test traffic environment initialization."""
        env = TrafficEnvironment(city="test", num_intersections=9, time_resolution=1.0)
        assert env.city == "test"
        assert env.time_resolution == 1.0
        assert len(env.graph.nodes()) > 0
    
    def test_traffic_environment_reset(self):
        """Test environment reset."""
        env = TrafficEnvironment(city="test", num_intersections=9, time_resolution=1.0)
        state = env.reset()
        assert isinstance(state, GraphState)
        assert state.nodes.shape[1] == 4  # node features
        assert state.adjacency.shape[0] == state.adjacency.shape[1]
    
    def test_traffic_environment_step(self):
        """Test environment step function."""
        env = TrafficEnvironment(city="test", num_intersections=9, time_resolution=1.0)
        state = env.reset()
        num_nodes = state.nodes.shape[0]
        
        actions = jnp.zeros(num_nodes, dtype=int)
        next_state, rewards, done, info = env.step(actions)
        
        assert isinstance(next_state, GraphState)
        assert len(rewards) == num_nodes
        assert isinstance(done, bool)
        assert isinstance(info, dict)
    
    def test_invalid_actions(self):
        """Test environment handles invalid actions gracefully."""
        env = TrafficEnvironment(city="test", num_intersections=4, time_resolution=1.0)
        state = env.reset()
        
        # Test with wrong action dimension
        with pytest.raises((ValueError, TypeError, AssertionError)):
            wrong_actions = jnp.zeros(100)  # Wrong size
            env.step(wrong_actions)


class TestFederatedSystem:
    """Test federated learning system."""
    
    def test_federated_initialization(self):
        """Test federated system initialization."""
        fed_system = FederatedGraphRL(
            num_agents=3,
            aggregation="gossip",
            communication_rounds=2
        )
        assert fed_system.config.num_agents == 3
        assert fed_system.config.aggregation == "gossip"
        assert fed_system.config.communication_rounds == 2
    
    def test_federated_agent_registration(self):
        """Test agent registration."""
        fed_system = FederatedGraphRL(num_agents=2)
        agent = GraphPPO(agent_id=0, action_dim=3, node_dim=4)
        
        fed_system.register_agent(agent)
        assert len(fed_system.agents) == 1
    
    def test_subgraph_extraction(self):
        """Test subgraph extraction."""
        env = TrafficEnvironment(city="test", num_intersections=4, time_resolution=1.0)
        fed_system = FederatedGraphRL(num_agents=2)
        
        state = env.reset()
        subgraph = fed_system.get_subgraph(0, state)
        
        assert isinstance(subgraph, GraphState)
        assert subgraph.nodes.shape[0] > 0


class TestAlgorithms:
    """Test RL algorithms."""
    
    def test_graph_ppo_initialization(self):
        """Test GraphPPO initialization."""
        from pg_neo_graph_rl.algorithms.graph_ppo import PPOConfig
        config = PPOConfig(learning_rate=3e-4)
        agent = GraphPPO(
            agent_id=0,
            action_dim=3,
            node_dim=4,
            config=config
        )
        assert agent.agent_id == 0
        assert agent.action_dim == 3
        assert agent.node_dim == 4
        assert agent.config.learning_rate == 3e-4
    
    def test_graph_ppo_action_selection(self):
        """Test action selection."""
        agent = GraphPPO(agent_id=0, action_dim=3, node_dim=4)
        
        # Create dummy graph state
        nodes = jnp.ones((5, 4))  # 5 nodes, 4 features each
        edges = jnp.array([[0, 1], [1, 2], [2, 3]], dtype=int)
        edge_attr = jnp.ones((3, 2))
        adjacency = jnp.eye(5)
        timestamps = jnp.zeros(5)
        
        state = GraphState(
            nodes=nodes,
            edges=edges,
            edge_attr=edge_attr,
            adjacency=adjacency,
            timestamps=timestamps
        )
        
        actions, info = agent.act(state, training=False)
        assert actions.shape[0] == 5  # One action per node
        assert jnp.all(actions >= 0) and jnp.all(actions < 3)  # Valid action range


class TestRobustness:
    """Test system robustness and error handling."""
    
    def test_invalid_configuration(self):
        """Test system handles invalid configurations."""
        from pg_neo_graph_rl.utils.exceptions import FederatedLearningError
        with pytest.raises((ValueError, ValidationError, FederatedLearningError)):
            FederatedGraphRL(num_agents=-1)  # Invalid number of agents
        
        with pytest.raises((ValueError, ValidationError)):
            GraphPPO(agent_id=0, action_dim=0, node_dim=4)  # Invalid action dim
    
    def test_memory_usage(self):
        """Test system doesn't leak excessive memory."""
        import psutil
        import gc
        
        initial_memory = psutil.Process().memory_info().rss
        
        # Create and destroy multiple environments
        for _ in range(10):
            env = TrafficEnvironment(city="test", num_intersections=4)
            state = env.reset()
            del env
            gc.collect()
        
        final_memory = psutil.Process().memory_info().rss
        memory_increase = final_memory - initial_memory
        
        # Allow reasonable memory increase (< 100MB)
        assert memory_increase < 100 * 1024 * 1024
    
    def test_nan_handling(self):
        """Test system handles NaN values gracefully."""
        agent = GraphPPO(agent_id=0, action_dim=3, node_dim=4)
        
        # Create state with NaN values
        nodes = jnp.array([[1.0, 2.0, jnp.nan, 4.0]])
        edges = jnp.array([[0, 0]], dtype=int)
        edge_attr = jnp.ones((1, 2))
        adjacency = jnp.eye(1)
        timestamps = jnp.zeros(1)
        
        state = GraphState(
            nodes=nodes,
            edges=edges,
            edge_attr=edge_attr,
            adjacency=adjacency,
            timestamps=timestamps
        )
        
        # Should handle NaN gracefully or raise appropriate error
        try:
            actions, info = agent.act(state, training=False)
            # If it doesn't error, actions should not be NaN
            assert not jnp.any(jnp.isnan(actions))
        except (ValueError, ValidationError):
            # This is acceptable - system detected and rejected invalid input
            pass


class TestSecurity:
    """Test security measures."""
    
    def test_input_validation(self):
        """Test input validation prevents malicious inputs."""
        env = TrafficEnvironment(city="test", num_intersections=4)
        state = env.reset()
        
        # Test extremely large actions
        with pytest.raises((ValueError, AssertionError)):
            large_actions = jnp.full(state.nodes.shape[0], 1e10)
            env.step(large_actions)
    
    def test_gradient_explosion_protection(self):
        """Test protection against gradient explosion."""
        agent = GraphPPO(agent_id=0, action_dim=3, node_dim=4)
        
        # Create dummy trajectories that might cause gradient explosion
        nodes = jnp.ones((2, 4)) * 1e6  # Very large values
        edges = jnp.array([[0, 1]], dtype=int)
        edge_attr = jnp.ones((1, 2))
        adjacency = jnp.eye(2)
        timestamps = jnp.zeros(2)
        
        state = GraphState(
            nodes=nodes,
            edges=edges,
            edge_attr=edge_attr,
            adjacency=adjacency,
            timestamps=timestamps
        )
        
        # Create proper trajectory data
        actions_array = jnp.array([0, 1])
        trajectories = {
            'states': [state],
            'actions': [actions_array],
            'rewards': [jnp.array([1.0, 1.0])],
            'log_probs': [jnp.array([-1.0, -1.0])],
            'values': [jnp.array([0.5, 0.5])],
            'dones': [jnp.array([False, False])]
        }
        
        # Should not crash or produce invalid gradients
        try:
            gradients = agent.compute_gradients(trajectories)
            # Check gradients are finite
            for grad in jax.tree.leaves(gradients):
                assert jnp.all(jnp.isfinite(grad))
        except (ValueError, ValidationError):
            # Acceptable - system detected problematic input
            pass


def run_comprehensive_tests():
    """Run all tests and return results."""
    test_results = {
        'passed': 0,
        'failed': 0,
        'errors': []
    }
    
    test_classes = [TestEnvironments, TestFederatedSystem, TestAlgorithms, TestRobustness, TestSecurity]
    
    for test_class in test_classes:
        instance = test_class()
        for method_name in dir(instance):
            if method_name.startswith('test_'):
                try:
                    print(f"Running {test_class.__name__}.{method_name}...")
                    method = getattr(instance, method_name)
                    method()
                    test_results['passed'] += 1
                    print(f"  ✅ PASSED")
                except Exception as e:
                    test_results['failed'] += 1
                    test_results['errors'].append(f"{test_class.__name__}.{method_name}: {str(e)}")
                    print(f"  ❌ FAILED: {str(e)}")
    
    return test_results


if __name__ == "__main__":
    print("🧪 Running Comprehensive Test Suite...")
    print("=" * 60)
    
    # Install required dependencies
    try:
        import psutil
    except ImportError:
        print("Installing psutil for memory tests...")
        import subprocess
        subprocess.check_call([sys.executable, "-m", "pip", "install", "--break-system-packages", "psutil"])
        import psutil
    
    results = run_comprehensive_tests()
    
    print("\n" + "=" * 60)
    print("📊 Test Results:")
    print(f"  ✅ Passed: {results['passed']}")
    print(f"  ❌ Failed: {results['failed']}")
    
    if results['errors']:
        print("\n🔍 Error Details:")
        for error in results['errors']:
            print(f"  • {error}")
    
    if results['failed'] == 0:
        print("\n🎉 All tests passed! System is robust!")
        print("✅ GENERATION 2 SUCCESS: System is robust and reliable!")
    else:
        print(f"\n⚠️  {results['failed']} tests failed. Addressing issues...")
    
    exit(0 if results['failed'] == 0 else 1)