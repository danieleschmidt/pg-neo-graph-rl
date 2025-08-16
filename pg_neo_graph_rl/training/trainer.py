"""
Federated trainer for graph reinforcement learning.
"""
import pickle
import time
from pathlib import Path
from typing import Any, Dict, Optional

import jax.numpy as jnp

from ..algorithms import GraphPPO, GraphSAC
from ..core import FederatedGraphRL
from ..environments import PowerGridEnvironment, SwarmEnvironment, TrafficEnvironment
from ..monitoring.metrics import MetricsCollector
from ..utils.exceptions import TrainingError
from ..utils.logging import get_logger, get_performance_logger
from .config import TrainingConfig


class FederatedTrainer:
    """Federated trainer for graph reinforcement learning."""

    def __init__(self, config: TrainingConfig):
        """Initialize federated trainer."""
        self.config = config
        self.logger = get_logger("pg_neo_graph_rl.trainer")
        self.perf_logger = get_performance_logger()

        # Create directories
        config.create_directories()

        # Initialize components
        self.env = None
        self.fed_system = None
        self.agents = []
        self.metrics_collector = None

        # Training state
        self.episode = 0
        self.global_step = 0
        self.best_reward = float('-inf')

        self._initialize_components()

    def _initialize_components(self) -> None:
        """Initialize environment, federated system, and agents."""

        # Create environment
        self.logger.info(f"Creating {self.config.environment_type} environment")
        if self.config.environment_type == "traffic":
            self.env = TrafficEnvironment(**self.config.environment_params)
        elif self.config.environment_type == "power":
            self.env = PowerGridEnvironment(**self.config.environment_params)
        elif self.config.environment_type == "swarm":
            self.env = SwarmEnvironment(**self.config.environment_params)
        else:
            raise TrainingError(f"Unknown environment type: {self.config.environment_type}")

        # Create federated system
        self.logger.info("Creating federated learning system")
        self.fed_system = FederatedGraphRL(
            num_agents=self.config.num_agents,
            aggregation=self.config.aggregation_method,
            communication_rounds=self.config.communication_rounds,
            topology=self.config.topology,
            privacy_noise=self.config.privacy_noise
        )

        # Create agents
        self.logger.info("Creating federated agents")
        initial_state = self.env.reset()
        node_dim = initial_state.nodes.shape[1]
        action_dim = 3  # Default action dimension

        for agent_id in range(self.config.num_agents):
            if self.config.algorithm.lower() == "ppo":
                from ..algorithms.graph_ppo import PPOConfig
                ppo_config = PPOConfig(
                    learning_rate=self.config.learning_rate,
                    clip_epsilon=self.config.clip_epsilon,
                    entropy_coeff=self.config.entropy_coeff,
                    value_coeff=self.config.value_coeff
                )
                agent = GraphPPO(
                    agent_id=agent_id,
                    action_dim=action_dim,
                    node_dim=node_dim,
                    config=ppo_config
                )
            elif self.config.algorithm.lower() == "sac":
                agent = GraphSAC(
                    agent_id=agent_id,
                    action_dim=action_dim,
                    node_dim=node_dim
                )
            else:
                raise TrainingError(f"Unknown algorithm: {self.config.algorithm}")

            self.agents.append(agent)
            self.fed_system.register_agent(agent)

        # Initialize metrics collector
        if self.config.enable_monitoring:
            self.metrics_collector = MetricsCollector(
                output_file=self.config.log_dir / "metrics.json"
            )

        self.logger.info(f"Trainer initialized with {len(self.agents)} agents")

    def train(self) -> Dict[str, Any]:
        """Execute federated training."""
        self.logger.info("Starting federated training")
        training_start_time = time.time()

        try:
            for episode in range(self.config.total_episodes):
                self.episode = episode

                with self.perf_logger.timer_context(f"episode_{episode}"):
                    episode_metrics = self._train_episode()

                # Log episode metrics
                if episode % self.config.log_frequency == 0:
                    self._log_episode_metrics(episode, episode_metrics)

                # Evaluate periodically
                if episode % self.config.eval_frequency == 0:
                    eval_metrics = self._evaluate()
                    self._log_evaluation_metrics(episode, eval_metrics)

                    # Save best model
                    avg_reward = eval_metrics.get("avg_episode_reward", float('-inf'))
                    if avg_reward > self.best_reward:
                        self.best_reward = avg_reward
                        self._save_checkpoint(episode, "best_model")

                # Save checkpoint periodically
                if episode % self.config.save_frequency == 0:
                    self._save_checkpoint(episode)

        except KeyboardInterrupt:
            self.logger.info("Training interrupted by user")
        except Exception as e:
            self.logger.error(f"Training failed: {e}")
            raise TrainingError(f"Training failed: {e}")

        total_training_time = time.time() - training_start_time

        # Final checkpoint
        final_metrics = self._save_final_checkpoint(total_training_time)

        self.logger.info(f"Training completed in {total_training_time:.2f} seconds")
        return final_metrics

    def _train_episode(self) -> Dict[str, Any]:
        """Train a single episode."""

        # Reset environment
        state = self.env.reset()
        episode_rewards = []
        episode_losses = []

        for step in range(self.config.max_episode_length):
            self.global_step += 1

            # Collect agent actions and experiences
            agent_experiences = []

            for agent_id, agent in enumerate(self.agents):
                # Get agent's subgraph
                subgraph = self.fed_system.get_subgraph(agent_id, state)

                # Agent acts
                actions, info = agent.act(subgraph, training=True)

                # Store experience (simplified for demo)
                experience = {
                    "state": subgraph,
                    "actions": actions,
                    "info": info
                }
                agent_experiences.append(experience)

            # Environment step (using first agent's actions as example)
            env_actions = agent_experiences[0]["actions"]
            # Ensure we have the right number of actions
            if len(env_actions) != len(state.nodes):
                # Pad or truncate actions to match number of nodes
                if len(env_actions) < len(state.nodes):
                    env_actions = jnp.concatenate([
                        env_actions,
                        jnp.zeros(len(state.nodes) - len(env_actions), dtype=int)
                    ])
                else:
                    env_actions = env_actions[:len(state.nodes)]

            next_state, rewards, done, info = self.env.step(env_actions)

            episode_rewards.append(float(jnp.mean(rewards)))

            if done:
                break

            state = next_state

        # Federated learning update (simplified)
        if self.global_step % self.config.communication_rounds == 0:
            try:
                # Create dummy gradients for demonstration
                dummy_gradients = []
                for agent in self.agents:
                    dummy_grad = {
                        "policy": {"dummy": jnp.ones((10, 10))},
                        "value": {"dummy": jnp.ones((5, 5))}
                    }
                    dummy_gradients.append(dummy_grad)

                # Federated aggregation
                aggregated_grads = self.fed_system.federated_round(dummy_gradients)

                self.fed_system.step()

            except Exception as e:
                self.logger.warning(f"Federated learning update failed: {e}")

        return {
            "episode_reward": sum(episode_rewards),
            "episode_length": len(episode_rewards),
            "avg_reward": sum(episode_rewards) / len(episode_rewards) if episode_rewards else 0.0
        }

    def _evaluate(self) -> Dict[str, Any]:
        """Evaluate current policy."""
        eval_episodes = 5
        eval_rewards = []

        for _ in range(eval_episodes):
            state = self.env.reset()
            episode_reward = 0.0

            for step in range(self.config.max_episode_length):
                # Use first agent for evaluation
                agent = self.agents[0]
                subgraph = self.fed_system.get_subgraph(0, state)

                actions, _ = agent.act(subgraph, training=False)

                # Ensure correct action shape
                if len(actions) != len(state.nodes):
                    if len(actions) < len(state.nodes):
                        actions = jnp.concatenate([
                            actions,
                            jnp.zeros(len(state.nodes) - len(actions), dtype=int)
                        ])
                    else:
                        actions = actions[:len(state.nodes)]

                next_state, rewards, done, _ = self.env.step(actions)
                episode_reward += float(jnp.mean(rewards))

                if done:
                    break

                state = next_state

            eval_rewards.append(episode_reward)

        return {
            "avg_episode_reward": sum(eval_rewards) / len(eval_rewards),
            "min_episode_reward": min(eval_rewards),
            "max_episode_reward": max(eval_rewards),
            "eval_episodes": len(eval_rewards)
        }

    def _log_episode_metrics(self, episode: int, metrics: Dict[str, Any]) -> None:
        """Log episode training metrics."""
        self.logger.info(
            f"Episode {episode}: "
            f"reward={metrics['episode_reward']:.3f}, "
            f"length={metrics['episode_length']}, "
            f"avg_reward={metrics['avg_reward']:.3f}"
        )

        if self.metrics_collector:
            self.metrics_collector.log_scalar("train/episode_reward", metrics["episode_reward"], episode)
            self.metrics_collector.log_scalar("train/episode_length", metrics["episode_length"], episode)
            self.metrics_collector.log_scalar("train/avg_reward", metrics["avg_reward"], episode)

    def _log_evaluation_metrics(self, episode: int, metrics: Dict[str, Any]) -> None:
        """Log evaluation metrics."""
        self.logger.info(
            f"Eval at episode {episode}: "
            f"avg_reward={metrics['avg_episode_reward']:.3f}, "
            f"min={metrics['min_episode_reward']:.3f}, "
            f"max={metrics['max_episode_reward']:.3f}"
        )

        if self.metrics_collector:
            self.metrics_collector.log_scalar("eval/avg_episode_reward", metrics["avg_episode_reward"], episode)
            self.metrics_collector.log_scalar("eval/min_episode_reward", metrics["min_episode_reward"], episode)
            self.metrics_collector.log_scalar("eval/max_episode_reward", metrics["max_episode_reward"], episode)

    def _save_checkpoint(self, episode: int, name: Optional[str] = None) -> None:
        """Save training checkpoint."""
        checkpoint_name = name or f"checkpoint_episode_{episode}"
        checkpoint_path = self.config.checkpoint_dir / f"{checkpoint_name}.pkl"

        checkpoint_data = {
            "episode": episode,
            "global_step": self.global_step,
            "config": self.config,
            "best_reward": self.best_reward,
            "agent_params": [
                {
                    "policy_params": agent.policy_params,
                    "value_params": agent.value_params,
                    "policy_opt_state": agent.policy_opt_state,
                    "value_opt_state": agent.value_opt_state
                }
                for agent in self.agents
            ]
        }

        with open(checkpoint_path, 'wb') as f:
            pickle.dump(checkpoint_data, f)

        self.logger.info(f"Checkpoint saved: {checkpoint_path}")

    def _save_final_checkpoint(self, training_time: float) -> Dict[str, Any]:
        """Save final checkpoint and return training summary."""

        # Save final checkpoint
        self._save_checkpoint(self.episode, "final_model")

        # Training summary
        summary = {
            "total_episodes": self.episode + 1,
            "total_steps": self.global_step,
            "training_time_seconds": training_time,
            "best_reward": self.best_reward,
            "config": self.config
        }

        # Save summary
        summary_path = self.config.output_dir / "training_summary.json"
        import json
        with open(summary_path, 'w') as f:
            # Convert config to dict for JSON serialization
            summary_for_json = summary.copy()
            summary_for_json["config"] = summary_for_json["config"].__dict__.copy()
            # Convert Path objects to strings
            for key, value in summary_for_json["config"].items():
                if isinstance(value, Path):
                    summary_for_json["config"][key] = str(value)

            json.dump(summary_for_json, f, indent=2, default=str)

        return summary

    def load_checkpoint(self, checkpoint_path: Path) -> None:
        """Load training checkpoint."""
        with open(checkpoint_path, 'rb') as f:
            checkpoint_data = pickle.load(f)

        self.episode = checkpoint_data["episode"]
        self.global_step = checkpoint_data["global_step"]
        self.best_reward = checkpoint_data["best_reward"]

        # Restore agent parameters
        for i, (agent, agent_data) in enumerate(zip(self.agents, checkpoint_data["agent_params"])):
            agent.policy_params = agent_data["policy_params"]
            agent.value_params = agent_data["value_params"]
            agent.policy_opt_state = agent_data["policy_opt_state"]
            agent.value_opt_state = agent_data["value_opt_state"]

        self.logger.info(f"Checkpoint loaded from {checkpoint_path}")
        self.logger.info(f"Resumed at episode {self.episode}, step {self.global_step}")
