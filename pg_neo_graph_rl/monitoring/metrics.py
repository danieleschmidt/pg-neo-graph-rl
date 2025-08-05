"""
Metrics collection for federated graph RL.
"""
import jax.numpy as jnp
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
import time


@dataclass
class FederatedMetrics:
    """Container for federated learning metrics."""
    episode: int = 0
    global_reward: float = 0.0
    convergence_rate: float = 0.0
    communication_rounds: int = 0
    training_time: float = 0.0
    
    # Agent-specific metrics
    agent_rewards: List[float] = None
    agent_losses: List[float] = None
    
    # Graph-specific metrics
    graph_connectivity: float = 0.0
    graph_size: int = 0
    
    # Environment-specific metrics
    environment_metrics: Dict[str, float] = None
    
    def __post_init__(self):
        if self.agent_rewards is None:
            self.agent_rewards = []
        if self.agent_losses is None:
            self.agent_losses = []
        if self.environment_metrics is None:
            self.environment_metrics = {}


class MetricsCollector:
    """Collects and tracks metrics during federated training."""
    
    def __init__(self, log_interval: int = 10):
        self.log_interval = log_interval
        self.metrics_history: List[FederatedMetrics] = []
        self.start_time = time.time()
        
    def log_episode(self, 
                   episode: int,
                   global_reward: float,
                   agent_rewards: List[float],
                   agent_losses: List[float],
                   communication_rounds: int,
                   graph_connectivity: float,
                   graph_size: int,
                   environment_metrics: Optional[Dict[str, float]] = None) -> None:
        """Log metrics for an episode."""
        
        current_time = time.time()
        training_time = current_time - self.start_time
        
        # Compute convergence rate (simple gradient-based estimate)
        convergence_rate = 0.0
        if len(self.metrics_history) > 0:
            prev_reward = self.metrics_history[-1].global_reward
            convergence_rate = abs(global_reward - prev_reward) / max(abs(prev_reward), 1e-8)
        
        metrics = FederatedMetrics(
            episode=episode,
            global_reward=global_reward,
            convergence_rate=convergence_rate,
            communication_rounds=communication_rounds,
            training_time=training_time,
            agent_rewards=agent_rewards.copy(),
            agent_losses=agent_losses.copy(),
            graph_connectivity=graph_connectivity,
            graph_size=graph_size,
            environment_metrics=environment_metrics.copy() if environment_metrics else {}
        )
        
        self.metrics_history.append(metrics)
        
        # Print metrics periodically
        if episode % self.log_interval == 0:
            self._print_metrics(metrics)
    
    def _print_metrics(self, metrics: FederatedMetrics) -> None:
        """Print current metrics."""
        print(f"Episode {metrics.episode}:")
        print(f"  Global Reward: {metrics.global_reward:.4f}")
        print(f"  Convergence Rate: {metrics.convergence_rate:.6f}")
        print(f"  Communication Rounds: {metrics.communication_rounds}")
        print(f"  Graph Connectivity: {metrics.graph_connectivity:.4f}")
        print(f"  Training Time: {metrics.training_time:.2f}s")
        
        if metrics.agent_rewards:
            avg_agent_reward = sum(metrics.agent_rewards) / len(metrics.agent_rewards)
            print(f"  Avg Agent Reward: {avg_agent_reward:.4f}")
            
        if metrics.environment_metrics:
            print("  Environment Metrics:")
            for key, value in metrics.environment_metrics.items():
                print(f"    {key}: {value:.4f}")
        print()
    
    def get_latest_metrics(self) -> Optional[FederatedMetrics]:
        """Get the most recent metrics."""
        return self.metrics_history[-1] if self.metrics_history else None
    
    def get_metrics_summary(self) -> Dict[str, Any]:
        """Get summary statistics over all episodes."""
        if not self.metrics_history:
            return {}
        
        global_rewards = [m.global_reward for m in self.metrics_history]
        convergence_rates = [m.convergence_rate for m in self.metrics_history]
        communication_rounds = [m.communication_rounds for m in self.metrics_history]
        
        return {
            "total_episodes": len(self.metrics_history),
            "final_reward": global_rewards[-1],
            "best_reward": max(global_rewards),
            "avg_reward": sum(global_rewards) / len(global_rewards),
            "avg_convergence_rate": sum(convergence_rates) / len(convergence_rates),
            "total_communication_rounds": sum(communication_rounds),
            "total_training_time": self.metrics_history[-1].training_time
        }
    
    def export_metrics(self, filename: str) -> None:
        """Export metrics to file (simplified JSON format)."""
        import json
        
        # Convert metrics to serializable format
        serializable_metrics = []
        for m in self.metrics_history:
            metric_dict = {
                "episode": m.episode,
                "global_reward": float(m.global_reward),
                "convergence_rate": float(m.convergence_rate),
                "communication_rounds": m.communication_rounds,
                "training_time": m.training_time,
                "agent_rewards": [float(r) for r in m.agent_rewards],
                "agent_losses": [float(l) for l in m.agent_losses],
                "graph_connectivity": float(m.graph_connectivity),
                "graph_size": m.graph_size,
                "environment_metrics": {k: float(v) for k, v in m.environment_metrics.items()}
            }
            serializable_metrics.append(metric_dict)
        
        with open(filename, 'w') as f:
            json.dump(serializable_metrics, f, indent=2)
    
    def compute_learning_curve(self, window_size: int = 10) -> List[float]:
        """Compute smoothed learning curve."""
        if len(self.metrics_history) < window_size:
            return [m.global_reward for m in self.metrics_history]
        
        smoothed_rewards = []
        for i in range(len(self.metrics_history)):
            start_idx = max(0, i - window_size + 1)
            end_idx = i + 1
            window_rewards = [self.metrics_history[j].global_reward for j in range(start_idx, end_idx)]
            smoothed_rewards.append(sum(window_rewards) / len(window_rewards))
        
        return smoothed_rewards
