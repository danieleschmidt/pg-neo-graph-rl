"""End-to-end integration tests."""

import pytest
import jax
import jax.numpy as jnp
import time
from unittest.mock import Mock, patch

from tests.fixtures.graph_data import (
    create_synthetic_traffic_graph,
    create_federated_graph_partition,
    create_benchmark_graphs
)
from tests.utils import (
    mock_federated_agents,
    create_mock_environment,
    assert_federated_convergence,
    Timer,
    create_test_config
)


@pytest.mark.integration
class TestFederatedLearningPipeline:
    """Test complete federated learning pipeline."""
    
    def test_basic_federated_training_loop(self, federated_config):
        """Test basic federated training loop."""
        num_agents = federated_config["num_agents"]
        communication_rounds = federated_config["communication_rounds"]
        
        # Create mock agents
        agents = mock_federated_agents(num_agents)
        
        # Mock parameter aggregation
        def mock_aggregate(gradients):
            return jnp.mean(jnp.stack(gradients), axis=0)
        
        # Training loop
        loss_history = []
        for round_num in range(communication_rounds):
            # Local training
            round_losses = []
            gradients = []
            
            for agent in agents:
                # Mock local training
                loss = 1.0 / (round_num + 1) + jnp.random.normal(key=jax.random.PRNGKey(round_num)) * 0.1
                round_losses.append(float(loss))
                
                # Mock gradient computation
                gradient = jnp.random.normal(key=jax.random.PRNGKey(agent.agent_id + round_num), shape=(100,))
                gradients.append(gradient)
            
            # Aggregate gradients
            aggregated_gradient = mock_aggregate(gradients)
            
            # Update all agents
            for agent in agents:
                agent.set_parameters(aggregated_gradient)
            
            avg_loss = jnp.mean(jnp.array(round_losses))
            loss_history.append(float(avg_loss))
        
        # Verify convergence trend
        assert len(loss_history) == communication_rounds
        assert loss_history[-1] < loss_history[0]  # Loss should decrease
    
    def test_graph_environment_integration(self):
        """Test integration with graph environments."""
        # Create test environment
        env = create_mock_environment("traffic", num_nodes=20)
        
        # Reset environment
        initial_state = env.reset()
        
        # Verify state structure
        assert "node_features" in initial_state
        assert "edge_index" in initial_state
        assert initial_state["node_features"].shape[0] == 20
        
        # Take random actions
        num_steps = 5
        for step in range(num_steps):
            # Mock actions (random values)
            actions = jnp.random.normal(key=jax.random.PRNGKey(step), shape=(20,))
            
            # Environment step
            next_state, rewards, done, info = env.step(actions)
            
            # Verify step results
            assert "node_features" in next_state
            assert rewards.shape == (20,)
            assert isinstance(done, bool)
            assert isinstance(info, dict)
    
    def test_distributed_graph_partitioning(self):
        """Test graph partitioning for distributed training."""
        # Create base graph
        base_graph = create_synthetic_traffic_graph(num_intersections=36, seed=42)
        
        # Partition for federated learning
        num_agents = 4
        agent_subgraphs = create_federated_graph_partition(
            base_graph, num_agents=num_agents, overlap_ratio=0.1, seed=42
        )
        
        # Verify partitioning
        assert len(agent_subgraphs) == num_agents
        
        total_unique_nodes = set()
        for agent_id, subgraph in agent_subgraphs.items():
            assert "node_features" in subgraph
            assert "agent_id" in subgraph
            assert subgraph["agent_id"] == agent_id
            
            # Track unique nodes
            original_nodes = subgraph["original_node_ids"]
            total_unique_nodes.update(original_nodes)
        
        # Should cover most original nodes
        coverage = len(total_unique_nodes) / base_graph["num_nodes"]
        assert coverage >= 0.8  # At least 80% coverage
    
    @pytest.mark.slow
    def test_scalability_with_many_agents(self):
        """Test scalability with increasing number of agents."""
        agent_counts = [4, 8, 16]
        base_graph = create_synthetic_traffic_graph(num_intersections=64, seed=42)
        
        timing_results = {}
        
        for num_agents in agent_counts:
            with Timer(f"Partitioning for {num_agents} agents") as timer:
                agent_subgraphs = create_federated_graph_partition(
                    base_graph, num_agents=num_agents, seed=42
                )
                
                # Mock communication round
                gradients = []
                for agent_id in range(num_agents):
                    gradient = jnp.random.normal(
                        key=jax.random.PRNGKey(agent_id), shape=(100,)
                    )
                    gradients.append(gradient)
                
                # Aggregate
                aggregated = jnp.mean(jnp.stack(gradients), axis=0)
            
            timing_results[num_agents] = timer.elapsed
            
            # Verify all partitions created
            assert len(agent_subgraphs) == num_agents
        
        # Check timing doesn't grow too quickly
        assert timing_results[16] < timing_results[4] * 10  # Less than 10x slower


@pytest.mark.integration
class TestEndToEndScenarios:
    """Test complete end-to-end scenarios."""
    
    def test_traffic_control_scenario(self):
        """Test complete traffic control scenario."""
        config = create_test_config({
            "num_agents": 4,
            "communication_rounds": 10,
            "max_episode_steps": 50
        })
        
        # Create traffic environment
        env = create_mock_environment("traffic", num_nodes=config["max_episode_steps"])
        
        # Create mock federated learning system
        agents = mock_federated_agents(config["num_agents"])
        
        # Training episode
        state = env.reset()
        episode_rewards = []
        
        for step in range(config["max_episode_steps"]):
            # Each agent acts on a subset of nodes
            nodes_per_agent = state["node_features"].shape[0] // config["num_agents"]
            
            all_actions = []
            for i, agent in enumerate(agents):
                start_idx = i * nodes_per_agent
                end_idx = min(start_idx + nodes_per_agent, state["node_features"].shape[0])
                
                # Mock action selection
                agent_actions = jnp.random.normal(
                    key=jax.random.PRNGKey(step * 100 + i), 
                    shape=(end_idx - start_idx,)
                )
                all_actions.append(agent_actions)
            
            # Combine actions
            combined_actions = jnp.concatenate(all_actions)
            
            # Environment step
            next_state, rewards, done, info = env.step(combined_actions)
            episode_rewards.append(jnp.mean(rewards))
            
            state = next_state
            if done:
                break
        
        # Verify episode completed successfully
        assert len(episode_rewards) > 0
        assert all(isinstance(r, (int, float, jnp.ndarray)) for r in episode_rewards)
    
    def test_power_grid_control_scenario(self):
        """Test power grid control scenario."""
        config = create_test_config({
            "num_agents": 3,
            "communication_rounds": 8
        })
        
        # Create power grid environment
        env = create_mock_environment("power_grid", num_nodes=30)
        
        # Mock hierarchical federated structure
        agents = mock_federated_agents(config["num_agents"])
        
        # Simulate control scenario
        state = env.reset()
        stability_metrics = []
        
        for episode in range(5):  # Short test
            # Mock voltage/frequency control
            for agent in agents:
                # Mock safety-constrained actions
                raw_actions = jnp.random.normal(key=jax.random.PRNGKey(episode), shape=(10,))
                # Apply safety constraints (clipping)
                safe_actions = jnp.clip(raw_actions, -1.0, 1.0)
                
                # Mock stability check
                stability_score = jnp.mean(jnp.abs(safe_actions))
                stability_metrics.append(float(stability_score))
        
        # Verify stability metrics are reasonable
        assert len(stability_metrics) > 0
        assert all(0 <= metric <= 2.0 for metric in stability_metrics)
    
    def test_swarm_coordination_scenario(self):
        """Test swarm coordination scenario."""
        config = create_test_config({
            "num_agents": 5,
            "communication_rounds": 6
        })
        
        # Create swarm environment
        swarm_graph = create_synthetic_swarm_graph(num_agents=25, seed=42)
        
        # Mock distributed formation control
        agents = mock_federated_agents(config["num_agents"])
        
        formation_errors = []
        for timestep in range(10):
            # Mock formation control computation
            for agent in agents:
                # Mock local neighborhood processing
                agent_error = abs(jnp.random.normal(key=jax.random.PRNGKey(timestep + agent.agent_id)))
                formation_errors.append(float(agent_error))
            
            # Mock communication and coordination
            if timestep % 2 == 0:  # Every other timestep
                # Mock parameter exchange
                for agent in agents:
                    agent.get_parameters()
        
        # Verify coordination metrics
        assert len(formation_errors) > 0
        avg_error = jnp.mean(jnp.array(formation_errors))
        assert avg_error >= 0  # Errors should be non-negative


@pytest.mark.integration
@pytest.mark.slow
class TestLargeScaleIntegration:
    """Test large-scale integration scenarios."""
    
    def test_large_graph_processing(self):
        """Test processing of large graphs."""
        large_graph = create_synthetic_traffic_graph(num_intersections=225, seed=42)  # 15x15 grid
        
        # Verify graph creation succeeded
        assert large_graph["num_nodes"] == 225
        assert large_graph["node_features"].shape == (225, 4)
        
        # Test basic graph operations
        with Timer("Large graph statistics") as timer:
            from tests.fixtures.graph_data import compute_graph_statistics
            stats = compute_graph_statistics(large_graph)
        
        assert timer.elapsed < 1.0  # Should complete within 1 second
        assert stats["num_nodes"] == 225
        assert stats["density"] > 0
    
    def test_many_agent_coordination(self):
        """Test coordination with many agents."""
        num_agents = 20
        agents = mock_federated_agents(num_agents)
        
        # Mock large-scale coordination
        coordination_times = []
        
        for round_num in range(5):
            start_time = time.time()
            
            # Mock parameter collection
            all_parameters = []
            for agent in agents:
                params = agent.get_parameters()
                all_parameters.append(params)
            
            # Mock aggregation
            aggregated = jnp.mean(jnp.stack(all_parameters), axis=0)
            
            # Mock parameter distribution
            for agent in agents:
                agent.set_parameters(aggregated)
            
            round_time = time.time() - start_time
            coordination_times.append(round_time)
        
        # Verify reasonable performance
        avg_time = jnp.mean(jnp.array(coordination_times))
        assert avg_time < 0.1  # Should be fast for mock operations
    
    @pytest.mark.benchmark
    def test_end_to_end_performance(self):
        """Benchmark end-to-end performance."""
        benchmarks = create_benchmark_graphs()
        
        performance_results = {}
        
        for graph_name, graph in benchmarks.items():
            if graph_name.startswith("large_"):  # Only test large graphs
                with Timer(f"Processing {graph_name}") as timer:
                    # Mock end-to-end processing
                    features = graph["node_features"]
                    
                    # Mock GNN forward pass
                    processed = jnp.tanh(jnp.dot(features, features.T))
                    result = jax.block_until_ready(processed)
                
                performance_results[graph_name] = {
                    "time": timer.elapsed,
                    "nodes": graph["num_nodes"],
                    "edges": graph["num_edges"]
                }
        
        # Log performance results
        for graph_name, results in performance_results.items():
            print(f"{graph_name}: {results['time']:.4f}s for {results['nodes']} nodes")
        
        # Verify reasonable performance
        for results in performance_results.values():
            assert results["time"] < 5.0  # Should complete within 5 seconds


@pytest.mark.integration
def test_configuration_validation():
    """Test that configurations are properly validated."""
    # Valid configuration
    valid_config = create_test_config()
    assert valid_config["num_agents"] > 0
    assert valid_config["learning_rate"] > 0
    assert valid_config["communication_rounds"] > 0
    
    # Test configuration override
    custom_config = create_test_config({"num_agents": 8, "learning_rate": 1e-3})
    assert custom_config["num_agents"] == 8
    assert custom_config["learning_rate"] == 1e-3
    
    # Other values should remain default
    default_config = create_test_config()
    assert custom_config["batch_size"] == default_config["batch_size"]


@pytest.mark.integration 
def test_error_handling():
    """Test error handling in integration scenarios."""
    # Test with invalid graph
    invalid_graph = {
        "node_features": jnp.array([[1, 2]]),
        "edge_index": jnp.array([[0, 2], [1, 0]]).T,  # Invalid edge (node 2 doesn't exist)
        "num_nodes": 2,
        "num_edges": 2
    }
    
    from tests.fixtures.graph_data import validate_graph_structure
    assert not validate_graph_structure(invalid_graph)
    
    # Test with empty environment
    with pytest.raises((ValueError, AssertionError)):
        create_mock_environment("invalid_type")
    
    # Test with zero agents
    with pytest.raises((ValueError, AssertionError)):
        mock_federated_agents(0)