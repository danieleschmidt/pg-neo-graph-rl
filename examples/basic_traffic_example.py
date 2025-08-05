#!/usr/bin/env python3
"""
Basic traffic control example using pg-neo-graph-rl.
"""
import jax
import jax.numpy as jnp
from pg_neo_graph_rl import FederatedGraphRL, GraphPPO, TrafficEnvironment
from pg_neo_graph_rl.monitoring import MetricsCollector


def main():
    """Run basic traffic control example."""
    print("🚦 Starting Basic Traffic Control Example")
    print("=" * 50)
    
    # Initialize traffic environment
    env = TrafficEnvironment(
        city="manhattan",
        num_intersections=25,  # Smaller for demo
        time_resolution=5.0
    )
    
    # Initialize federated learning system
    fed_system = FederatedGraphRL(
        num_agents=5,
        aggregation="gossip",
        communication_rounds=3
    )
    
    # Create agents
    agents = []
    for i in range(fed_system.config.num_agents):
        agent = GraphPPO(
            agent_id=i,
            action_dim=3,  # Traffic light actions: 0=short, 1=medium, 2=long
            node_dim=4     # Node features: flow, density, speed, queue
        )
        fed_system.register_agent(agent)
        agents.append(agent)
    
    # Initialize metrics collection
    metrics_collector = MetricsCollector(log_interval=10)
    
    print(f"✅ Environment: {env.num_intersections} intersections")
    print(f"✅ Federated system: {fed_system.config.num_agents} agents")
    print(f"✅ Communication: {fed_system.config.aggregation} topology")
    print()
    
    # Training loop
    num_episodes = 50
    
    for episode in range(num_episodes):
        # Reset environment
        initial_state = env.reset()
        
        # Initialize episode tracking
        episode_rewards = []
        agent_rewards = [0.0] * fed_system.config.num_agents
        agent_losses = [0.0] * fed_system.config.num_agents
        
        # Run episode
        for step in range(100):  # 100 steps per episode
            
            # Each agent acts on their subgraph
            all_actions = []
            all_trajectories = []
            
            for agent_id, agent in enumerate(agents):
                # Get agent's subgraph
                subgraph = fed_system.get_subgraph(agent_id, initial_state)
                
                # Agent selects actions
                actions, info = agent.act(subgraph, training=True)
                all_actions.append(actions)
                
                # Collect dummy trajectories for this demo
                trajectories = agent.collect_trajectories(subgraph, num_steps=10)
                all_trajectories.append(trajectories)
            
            # Combine actions (simple concatenation for demo)
            if all_actions:
                combined_actions = jnp.concatenate(all_actions)[:env.num_intersections]
                # Ensure we have the right number of actions
                if len(combined_actions) < env.num_intersections:
                    # Pad with zeros
                    padding = jnp.zeros(env.num_intersections - len(combined_actions))
                    combined_actions = jnp.concatenate([combined_actions, padding])
                elif len(combined_actions) > env.num_intersections:
                    # Truncate
                    combined_actions = combined_actions[:env.num_intersections]
            else:
                combined_actions = jnp.zeros(env.num_intersections)
            
            # Convert to integer actions for traffic lights
            combined_actions = jnp.clip(combined_actions.astype(int), 0, 2)
            
            # Environment step
            next_state, rewards, done, info = env.step(combined_actions)
            
            episode_rewards.append(jnp.mean(rewards))
            
            if done:
                break
        
        # Federated learning update
        if episode % 5 == 0:  # Update every 5 episodes
            # Compute gradients from each agent
            all_gradients = []
            for agent_id, agent in enumerate(agents):
                trajectories = all_trajectories[agent_id] if agent_id < len(all_trajectories) else agent.collect_trajectories(initial_state)
                gradients = agent.compute_gradients(trajectories)
                all_gradients.append(gradients)
            
            # Federated aggregation
            aggregated_gradients = fed_system.federated_round(all_gradients)
            
            # Apply aggregated gradients
            for agent_id, agent in enumerate(agents):
                if agent_id < len(aggregated_gradients):
                    agent.apply_gradients(aggregated_gradients[agent_id])
        
        # Compute episode metrics
        global_reward = jnp.mean(jnp.array(episode_rewards))
        
        # Get environment performance
        env_metrics = env.evaluate_global_performance()
        
        # Get communication stats
        comm_stats = fed_system.get_communication_stats()
        
        # Update agent rewards (dummy values for demo)
        for i in range(len(agents)):
            agent_rewards[i] = float(global_reward + jnp.random.normal() * 0.1)
            agent_losses[i] = float(1.0 / (1.0 + global_reward + 1.0))  # Dummy loss
        
        # Log metrics
        metrics_collector.log_episode(
            episode=episode,
            global_reward=float(global_reward),
            agent_rewards=agent_rewards,
            agent_losses=agent_losses,
            communication_rounds=fed_system.config.communication_rounds,
            graph_connectivity=comm_stats["average_degree"] / fed_system.config.num_agents,
            graph_size=env.num_intersections,
            environment_metrics=env_metrics
        )
    
    print("\n🎯 Training Complete!")
    print("=" * 50)
    
    # Print final results
    summary = metrics_collector.get_metrics_summary()
    print("📊 Final Results:")
    print(f"  Total Episodes: {summary['total_episodes']}")
    print(f"  Final Reward: {summary['final_reward']:.4f}")
    print(f"  Best Reward: {summary['best_reward']:.4f}")
    print(f"  Average Reward: {summary['avg_reward']:.4f}")
    print(f"  Training Time: {summary['total_training_time']:.2f}s")
    
    # Export metrics
    metrics_collector.export_metrics("traffic_metrics.json")
    print("📄 Metrics exported to traffic_metrics.json")
    
    # Final environment evaluation
    final_performance = env.evaluate_global_performance()
    print("\n🚦 Final Traffic Performance:")
    for key, value in final_performance.items():
        print(f"  {key}: {value:.4f}")
    
    print("\n✅ Example completed successfully!")


if __name__ == "__main__":
    main()