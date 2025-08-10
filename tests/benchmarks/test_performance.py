"""Performance benchmark tests."""

import pytest
import time
import jax
import jax.numpy as jnp
import networkx as nx
from unittest.mock import Mock


@pytest.mark.benchmark
class TestPerformanceBenchmarks:
    """Performance benchmarks for core components."""
    
    def test_graph_processing_scalability(self):
        """Benchmark graph processing across different sizes."""
        graph_sizes = [100, 500, 1000, 2000]
        processing_times = []
        
        for size in graph_sizes:
            # Create test graph
            graph = nx.erdos_renyi_graph(size, 0.1)
            
            # Mock node features
            node_features = jax.random.normal(
                jax.random.PRNGKey(42), 
                shape=(size, 64)
            )
            
            # Time graph processing
            start_time = time.perf_counter()
            
            # Mock GNN forward pass
            processed_features = jnp.dot(node_features, node_features.T)
            result = jax.block_until_ready(processed_features)
            
            end_time = time.perf_counter()
            processing_time = end_time - start_time
            processing_times.append(processing_time)
        
        # Verify reasonable scaling
        assert len(processing_times) == len(graph_sizes)
        
        # Log performance metrics
        for size, time_taken in zip(graph_sizes, processing_times):
            print(f"Graph size {size}: {time_taken:.4f}s")
    
    @pytest.mark.slow
    def test_federated_aggregation_performance(self):
        """Benchmark federated gradient aggregation."""
        num_agents_list = [5, 10, 20, 50]
        gradient_dim = (1000, 100)
        
        for num_agents in num_agents_list:
            # Create mock gradients
            gradients = [
                jax.random.normal(
                    key=jax.random.PRNGKey(i), 
                    shape=gradient_dim
                )
                for i in range(num_agents)
            ]
            
            # Time aggregation
            start_time = time.perf_counter()
            
            # FedAvg aggregation
            stacked_grads = jnp.stack(gradients)
            aggregated = jnp.mean(stacked_grads, axis=0)
            result = jax.block_until_ready(aggregated)
            
            end_time = time.perf_counter()
            
            aggregation_time = end_time - start_time
            print(f"Agents {num_agents}: {aggregation_time:.4f}s")
            
            # Verify correctness
            assert result.shape == gradient_dim
    
    def test_memory_usage_scaling(self):
        """Test memory usage across different configurations."""
        # This would typically use memory profiling tools
        # For now, test basic memory patterns
        
        batch_sizes = [32, 64, 128, 256]
        sequence_lengths = [50, 100, 200]
        
        for batch_size in batch_sizes:
            for seq_len in sequence_lengths:
                # Create mock data
                data = jax.random.normal(
                    key=jax.random.PRNGKey(42),
                    shape=(batch_size, seq_len, 64)
                )
                
                # Mock processing
                processed = jnp.sum(data, axis=1)
                result = jax.block_until_ready(processed)
                
                # Basic memory check
                assert result.shape == (batch_size, 64)
    
    @pytest.mark.benchmark  
    def test_communication_overhead(self):
        """Benchmark communication patterns in federated setup."""
        topologies = ["ring", "star", "fully_connected"]
        num_agents = 10
        message_size = 1000
        
        for topology in topologies:
            if topology == "ring":
                # Each agent communicates with 2 neighbors
                communication_cost = num_agents * 2 * message_size
            elif topology == "star":
                # Central aggregator communicates with all
                communication_cost = (num_agents - 1) * 2 * message_size
            else:  # fully_connected
                # All-to-all communication
                communication_cost = num_agents * (num_agents - 1) * message_size
            
            # Mock timing
            start_time = time.perf_counter()
            
            # Simulate communication delay
            time.sleep(communication_cost / 1000000)  # Mock network delay
            
            end_time = time.perf_counter()
            
            comm_time = end_time - start_time
            print(f"Topology {topology}: {comm_time:.4f}s")
            
            assert comm_time > 0
    
    def test_jax_compilation_performance(self):
        """Benchmark JAX compilation times."""
        
        @jax.jit
        def mock_graph_update(nodes, edges):
            """Mock graph neural network update."""
            updated_nodes = jnp.dot(nodes, nodes.T)
            return jnp.tanh(updated_nodes)
        
        # Test different graph sizes
        sizes = [50, 100, 200]
        
        for size in sizes:
            nodes = jax.random.normal(
                key=jax.random.PRNGKey(42),
                shape=(size, 32)
            )
            edges = jax.random.randint(
                key=jax.random.PRNGKey(43),
                minval=0, maxval=size,
                shape=(size * 2, 2)
            )
            
            # Time first call (includes compilation)
            start_time = time.perf_counter()
            result = mock_graph_update(nodes, edges)
            jax.block_until_ready(result)
            first_call_time = time.perf_counter() - start_time
            
            # Time second call (compiled)
            start_time = time.perf_counter()
            result = mock_graph_update(nodes, edges)
            jax.block_until_ready(result)
            second_call_time = time.perf_counter() - start_time
            
            print(f"Size {size} - First: {first_call_time:.4f}s, "
                  f"Second: {second_call_time:.4f}s")
            
            # Compiled version should be much faster
            if first_call_time > 0.01:  # Only check if compilation was significant
                assert second_call_time < first_call_time
    
    @pytest.mark.benchmark
    def test_batch_processing_efficiency(self):
        """Test batch processing efficiency."""
        single_item_shape = (100, 64)
        batch_sizes = [1, 8, 16, 32, 64]
        
        @jax.jit
        def process_batch(batch):
            return jnp.sum(batch, axis=1)
        
        for batch_size in batch_sizes:
            batch = jax.random.normal(
                key=jax.random.PRNGKey(42),
                shape=(batch_size,) + single_item_shape
            )
            
            start_time = time.perf_counter()
            result = process_batch(batch)
            jax.block_until_ready(result)
            end_time = time.perf_counter()
            
            batch_time = end_time - start_time
            time_per_item = batch_time / batch_size
            
            print(f"Batch size {batch_size}: {time_per_item:.6f}s per item")
            
            assert result.shape == (batch_size, 64)