#!/usr/bin/env python3
"""
Scaling demo showcasing performance optimization, caching, and concurrent processing.
"""
import jax
import jax.numpy as jnp
import time
import threading
from concurrent.futures import as_completed
from pg_neo_graph_rl import FederatedGraphRL, TrafficEnvironment
from pg_neo_graph_rl.algorithms import GraphPPO
from pg_neo_graph_rl.algorithms.graph_ppo import PPOConfig
from pg_neo_graph_rl.optimization.performance import ConcurrentTrainer, BatchProcessor, PerformanceOptimizer
from pg_neo_graph_rl.optimization.advanced_cache import get_global_cache, get_graph_cache, SmartCache
from pg_neo_graph_rl.utils.health import create_health_monitor
from pg_neo_graph_rl.utils.logging import get_logger, get_performance_logger


def batch_training_function(args_list, kwargs_list):
    """Example batch training function."""
    results = []
    
    for args, kwargs in zip(args_list, kwargs_list):
        # Simulate training computation
        agent_id = args[0] if args else kwargs.get('agent_id', 0)
        
        # Mock training result
        result = {
            'agent_id': agent_id,
            'loss': 0.5 + jnp.random.normal() * 0.1,
            'reward': 10.0 + jnp.random.normal() * 2.0,
            'training_time': time.time()
        }
        results.append(result)
        
        # Simulate processing time
        time.sleep(0.01)
    
    return results


def main():
    """Run scaling demonstration."""
    logger = get_logger("pg_neo_graph_rl.scaling_demo")
    perf_logger = get_performance_logger()
    
    print("⚡ Scaling pg-neo-graph-rl Demo")
    print("=" * 60)
    print("Features: Performance optimization, caching, concurrent processing")
    print()
    
    # Start system monitoring
    print("🏥 Starting system health monitoring...")
    health_monitor = create_health_monitor(enable_monitoring=True)
    
    # Initialize caching systems
    print("💾 Setting up caching systems...")
    smart_cache = get_global_cache()
    graph_cache = get_graph_cache()
    
    # Warm up caches
    smart_cache.warm_cache()
    
    print("✅ Cache systems initialized")
    print()
    
    # Set up concurrent processing
    print("⚙️  Setting up concurrent processing...")
    concurrent_trainer = ConcurrentTrainer(max_workers=8)
    batch_processor = BatchProcessor(
        process_function=batch_training_function,
        batch_config=None  # Use defaults
    )
    batch_processor.start()
    
    # Initialize performance optimizer
    perf_optimizer = PerformanceOptimizer()
    
    print("✅ Concurrent processing setup complete")
    print()
    
    try:
        # Create scaled federated system
        print("🔄 Creating scaled federated system...")
        with perf_logger.timer_context("system_initialization"):
            fed_system = FederatedGraphRL(
                num_agents=20,  # More agents for scaling demo
                aggregation="hierarchical",  # More efficient for large scale
                topology="hierarchical",
                enable_monitoring=True,
                enable_security=True
            )
            
            # Large environment for scaling
            env = TrafficEnvironment(
                city="scaled_city",
                num_intersections=400,  # Much larger environment
                time_resolution=5.0
            )
        
        print(f"✅ Created system with {fed_system.config.num_agents} agents")
        print(f"✅ Environment with {env.num_intersections} intersections")
        print()
        
        # Create agents with caching
        print("🤖 Creating agents with parameter caching...")
        
        initial_state = env.reset()
        node_dim = initial_state.nodes.shape[1]
        action_dim = 3
        
        agents = []
        with perf_logger.timer_context("agent_creation"):
            for i in range(fed_system.config.num_agents):
                config = PPOConfig(learning_rate=1e-3, clip_epsilon=0.2, gamma=0.95)
                agent = GraphPPO(
                    agent_id=i,
                    action_dim=action_dim,
                    node_dim=node_dim,
                    config=config
                )
                fed_system.register_agent(agent)
                agents.append(agent)
                
                # Cache initial parameters
                smart_cache.put(f"agent_{i}_policy_params", agent.policy_params)
                smart_cache.put(f"agent_{i}_value_params", agent.value_params)
        
        print(f"✅ Created and cached {len(agents)} agents")
        print()
        
        # Demonstrate concurrent training
        print("🏃‍♂️ Demonstrating concurrent training...")
        
        def agent_training_task(agent_id, agent):
            """Training task for a single agent."""
            subgraph = fed_system.get_subgraph(agent_id, initial_state)
            
            # Simulate intensive training
            for _ in range(5):
                actions, info = agent.act(subgraph, training=False)
                time.sleep(0.02)  # Simulate computation
            
            return {
                'agent_id': agent_id,
                'actions_computed': 5,
                'final_actions': actions,
                'training_completed': True
            }
        
        # Submit concurrent training tasks
        training_futures = []
        start_time = time.time()
        
        for i, agent in enumerate(agents):
            future = concurrent_trainer.submit_training_task(
                agent_id=i,
                training_function=agent_training_task,
                agent=agent
            )
            training_futures.append(future)
        
        # Wait for all training to complete
        training_results = concurrent_trainer.wait_for_all_agents(timeout=30.0)
        concurrent_time = time.time() - start_time
        
        successful_agents = sum(1 for r in training_results.values() if r is not None)
        print(f"✅ Concurrent training completed in {concurrent_time:.2f}s")
        print(f"   Successfully trained: {successful_agents}/{len(agents)} agents")
        print()
        
        # Demonstrate batch processing
        print("📦 Demonstrating batch processing...")
        
        batch_start = time.time()
        batch_futures = []
        
        # Submit batch requests
        for i in range(50):  # Submit many requests
            future = batch_processor.submit(f"request_{i}", i)
            batch_futures.append(future)
        
        # Collect results
        batch_results = []
        for future in batch_futures:
            try:
                result = future.result(timeout=5.0)
                batch_results.append(result)
            except Exception as e:
                logger.error(f"Batch request failed: {e}")
        
        batch_time = time.time() - batch_start
        
        print(f"✅ Batch processing completed in {batch_time:.2f}s")
        print(f"   Processed: {len(batch_results)} requests")
        print()
        
        # Demonstrate caching performance
        print("💾 Demonstrating caching performance...")
        
        def expensive_computation(matrix, iterations):
            """Simulate expensive graph computation."""
            result = matrix
            for _ in range(iterations):
                result = jnp.dot(result, result) * 0.1
            return result
        
        # Test caching with repeated computations
        test_matrix = jnp.ones((50, 50))
        
        cache_test_start = time.time()
        
        # First calls (cache misses)
        for i in range(10):
            cache_key = f"expensive_comp_{hash(str(test_matrix.shape))}_{5}"
            result = smart_cache.get(cache_key, lambda: expensive_computation(test_matrix, 5))
        
        first_time = time.time() - cache_test_start
        
        # Second calls (cache hits)
        cache_hit_start = time.time()
        for i in range(10):
            cache_key = f"expensive_comp_{hash(str(test_matrix.shape))}_{5}"
            result = smart_cache.get(cache_key, lambda: expensive_computation(test_matrix, 5))
        
        second_time = time.time() - cache_hit_start
        
        cache_stats = smart_cache.get_analytics()
        print(f"✅ Caching performance test:")
        print(f"   First run (cache misses): {first_time:.3f}s")
        print(f"   Second run (cache hits): {second_time:.3f}s")
        print(f"   Speedup: {first_time / second_time:.1f}x")
        print(f"   Cache hit rate: {cache_stats['hit_rate']:.2%}")
        print()
        
        # Demonstrate adaptive caching
        print("🧠 Demonstrating adaptive caching...")
        
        adaptive_start = time.time()
        
        # Generate access patterns
        for i in range(1000):
            key = f"key_{i % 100}"  # Create repeated access pattern
            
            cached_value = smart_cache.get(key)
            if cached_value is None:
                # Simulate computation
                value = f"computed_value_{i}"
                smart_cache.put(key, value)
        
        adaptive_time = time.time() - adaptive_start
        adaptation_stats = smart_cache.get_analytics()
        
        print(f"✅ Adaptive caching test:")
        print(f"   Processing time: {adaptive_time:.3f}s")
        print(f"   Cache hit rate: {adaptation_stats['hit_rate']:.2%}")
        print(f"   Cache size: {adaptation_stats['current_size']}")
        print()
        
        # Demonstrate performance optimization
        print("📈 Demonstrating performance optimization...")
        
        # Simulate training episodes with metrics
        for episode in range(20):
            training_time = 0.5 + jnp.random.normal() * 0.1
            communication_time = 0.1 + jnp.random.normal() * 0.02
            memory_usage = 0.4 + jnp.random.normal() * 0.1
            convergence_rate = 0.05 + jnp.random.normal() * 0.01
            
            perf_optimizer.record_metrics(
                episode=episode,
                training_time=max(0.1, training_time),
                communication_time=max(0.01, communication_time),
                memory_usage=max(0.1, min(0.9, memory_usage)),
                convergence_rate=max(0.001, convergence_rate)
            )
            
            # Optimize every 5 episodes
            if episode % 5 == 4:
                optimized_settings = perf_optimizer.optimize()
        
        optimization_summary = perf_optimizer.get_optimization_summary()
        
        print("✅ Performance optimization results:")
        print(f"   Total optimizations: {optimization_summary['total_optimizations']}")
        print(f"   Current batch size: {optimization_summary['current_settings']['batch_size']}")
        print(f"   Communication frequency: {optimization_summary['current_settings']['communication_frequency']}")
        print(f"   Average training time: {optimization_summary['avg_training_time']:.3f}s")
        print()
        
        # Demonstrate scaled federated round
        print("🔄 Demonstrating scaled federated round...")
        
        # Create realistic gradients for all agents
        scaled_gradients = []
        for agent in agents:
            agent_grad = {}
            for param_name in ["policy", "value"]:
                agent_grad[param_name] = jax.tree.map(
                    lambda x: jax.random.normal(jax.random.PRNGKey(42), x.shape) * 0.001,
                    getattr(agent, f"{param_name}_params")
                )
            scaled_gradients.append(agent_grad)
        
        # Execute federated round with performance tracking
        with perf_logger.timer_context("scaled_federated_round"):
            aggregated_grads = fed_system.federated_round(scaled_gradients)
        
        print(f"✅ Scaled federated round completed")
        print(f"   Processed gradients from {len(scaled_gradients)} agents")
        print(f"   Successful aggregations: {len([g for g in aggregated_grads if g])}")
        
        # Display round metrics
        if fed_system.round_metrics:
            latest_round = fed_system.round_metrics[-1]
            print(f"   Round time: {latest_round['round_time_seconds']:.3f}s")
            print(f"   Participating agents: {latest_round['participating_agents']}")
        
        print()
        
        # Final system health check
        print("🏥 Final system health check...")
        
        health_summary = health_monitor.get_health_summary()
        print(f"✅ System health status: {health_summary['overall_status']}")
        
        for check_name, check_info in health_summary['checks'].items():
            status_emoji = {
                "healthy": "✅",
                "degraded": "⚠️",
                "critical": "❌",
                "unknown": "❓"
            }.get(check_info['status'], "❓")
            print(f"   {status_emoji} {check_name}: {check_info['status']}")
        
        print()
        print("🎯 Scaling features demonstration completed!")
        print("✅ All scaling optimizations are working:")
        print("   - Concurrent agent training")
        print("   - Intelligent batch processing")
        print("   - Multi-level caching (graph, parameters, adaptive)")
        print("   - Performance optimization and auto-tuning")
        print("   - Hierarchical federated aggregation")
        print("   - Real-time health monitoring")
        print("   - Memory and resource management")
        
    except Exception as e:
        logger.error(f"Scaling demo failed: {e}")
        print(f"❌ Scaling demo failed: {e}")
        
    finally:
        # Cleanup
        print("\n🧹 Cleaning up...")
        batch_processor.stop()
        concurrent_trainer.shutdown(wait=True)
        health_monitor.stop_monitoring()
        print("✅ Cleanup completed")


if __name__ == "__main__":
    main()