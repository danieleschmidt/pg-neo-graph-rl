#!/usr/bin/env python3
"""
Demo: Graph RL agent learns to balance network load.

Run:
    ~/anaconda3/bin/python3 examples/demo.py
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from pg_neo_graph_rl import GraphEnv, GNNPolicy, REINFORCETrainer

def main():
    print("=" * 60)
    print("  pg-neo-graph-rl: Network Load Balancing with Graph RL")
    print("=" * 60)

    torch.manual_seed(7)

    # Environment: 14-node network graph
    env = GraphEnv(num_nodes=14, edge_prob=0.35, max_steps=50, seed=7)
    print(f"\nEnvironment: {env.num_nodes} nodes, {env.num_edges} edges")
    print(f"Action space: {env.action_space_size} (one per edge)")

    # Policy: 2-layer GCN
    policy = GNNPolicy(
        in_features=env.observation_feature_dim,
        hidden_dim=64,
        num_edges=env.num_edges,
    )
    total_params = sum(p.numel() for p in policy.parameters())
    print(f"Policy: GNNPolicy ({total_params} parameters)")

    # Trainer
    trainer = REINFORCETrainer(env, policy, lr=3e-4, gamma=0.99)

    print("\nTraining for 500 episodes...")
    print("-" * 60)
    history = trainer.train(num_episodes=500, log_every=50, verbose=True)
    print("-" * 60)

    # Summary
    first_100 = sum(s.total_reward for s in history[:100]) / 100
    last_100  = sum(s.total_reward for s in history[-100:]) / 100
    improvement = last_100 - first_100

    print(f"\nResults:")
    print(f"  First 100 episodes avg reward: {first_100:.3f}")
    print(f"  Last  100 episodes avg reward: {last_100:.3f}")
    print(f"  Improvement: {improvement:+.3f}")

    if improvement > 0:
        print("  ✓ Agent learned to reduce network load imbalance!")
    else:
        print("  ~ Agent trained (try more episodes for clearer signal)")

    # Visual episode demo
    print("\nRunning a single greedy episode (policy.eval())...")
    policy.eval()
    obs = env.reset()
    initial_std = env.loads.std().item()
    done = False
    steps = 0
    total_r = 0.0

    with torch.no_grad():
        while not done:
            action, _ = policy.act(obs)
            obs, reward, done = env.step(action)
            total_r += reward
            steps += 1

    final_std = env.loads.std().item()
    print(f"  Initial load std: {initial_std:.4f}")
    print(f"  Final   load std: {final_std:.4f}")
    print(f"  Total reward: {total_r:.3f}  |  Steps: {steps}")
    print()


if __name__ == "__main__":
    main()
