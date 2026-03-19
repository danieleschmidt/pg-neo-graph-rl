"""
REINFORCE trainer with moving-average baseline.

Algorithm:
  For each episode:
    1. Roll out policy to get trajectory (obs, action, log_prob, reward)
    2. Compute discounted returns G_t
    3. Subtract baseline b (exponential moving average of mean returns)
    4. Loss = -sum( log_prob_t * (G_t - b) )
    5. Gradient step on policy
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import List

import torch
import torch.optim as optim

from .env import GraphEnv
from .policy import GNNPolicy


@dataclass
class TrainStats:
    episode: int
    total_reward: float
    mean_return: float
    baseline: float
    loss: float


@dataclass
class Trajectory:
    log_probs: List[torch.Tensor] = field(default_factory=list)
    rewards: List[float] = field(default_factory=list)

    def compute_returns(self, gamma: float = 0.99) -> torch.Tensor:
        """Compute discounted returns for each timestep."""
        G = 0.0
        returns = []
        for r in reversed(self.rewards):
            G = r + gamma * G
            returns.insert(0, G)
        return torch.tensor(returns, dtype=torch.float32)


class REINFORCETrainer:
    """
    REINFORCE with moving-average baseline.

    Parameters
    ----------
    env : GraphEnv
    policy : GNNPolicy
    lr : float
        Learning rate.
    gamma : float
        Discount factor.
    baseline_alpha : float
        EMA coefficient for baseline update (higher = faster adaptation).
    """

    def __init__(
        self,
        env: GraphEnv,
        policy: GNNPolicy,
        lr: float = 3e-4,
        gamma: float = 0.99,
        baseline_alpha: float = 0.05,
    ) -> None:
        self.env = env
        self.policy = policy
        self.optimizer = optim.Adam(policy.parameters(), lr=lr)
        self.gamma = gamma
        self.baseline_alpha = baseline_alpha
        self.baseline = 0.0  # running baseline

    def run_episode(self) -> tuple[Trajectory, float]:
        """Roll out one full episode."""
        obs = self.env.reset()
        traj = Trajectory()
        total_reward = 0.0
        done = False

        while not done:
            action, log_prob = self.policy.act(obs)
            obs, reward, done = self.env.step(action)
            traj.log_probs.append(log_prob)
            traj.rewards.append(reward)
            total_reward += reward

        return traj, total_reward

    def train_step(self) -> TrainStats:
        """One episode of training."""
        self.policy.train()
        traj, total_reward = self.run_episode()

        returns = traj.compute_returns(self.gamma)
        mean_return = returns.mean().item()

        # Update baseline (EMA)
        self.baseline = (
            (1 - self.baseline_alpha) * self.baseline
            + self.baseline_alpha * mean_return
        )

        # Normalize advantages
        advantages = returns - self.baseline

        # REINFORCE loss
        log_probs = torch.stack(traj.log_probs)
        loss = -(log_probs * advantages).sum()

        self.optimizer.zero_grad()
        loss.backward()
        # Gradient clipping for stability
        torch.nn.utils.clip_grad_norm_(self.policy.parameters(), max_norm=1.0)
        self.optimizer.step()

        return TrainStats(
            episode=0,  # caller fills this in
            total_reward=total_reward,
            mean_return=mean_return,
            baseline=self.baseline,
            loss=loss.item(),
        )

    def train(
        self,
        num_episodes: int = 500,
        log_every: int = 50,
        verbose: bool = True,
    ) -> List[TrainStats]:
        """
        Train for `num_episodes` episodes.

        Returns list of TrainStats per episode.
        """
        history: List[TrainStats] = []

        for ep in range(1, num_episodes + 1):
            stats = self.train_step()
            stats.episode = ep
            history.append(stats)

            if verbose and ep % log_every == 0:
                # Compute smoothed reward over last log_every episodes
                recent = history[-log_every:]
                avg_reward = sum(s.total_reward for s in recent) / len(recent)
                print(
                    f"Episode {ep:4d} | "
                    f"Avg reward (last {log_every}): {avg_reward:7.3f} | "
                    f"Baseline: {stats.baseline:7.3f} | "
                    f"Loss: {stats.loss:8.4f}"
                )

        return history
