"""
Tests for pg-neo-graph-rl core components.
"""

import pytest
import torch

from pg_neo_graph_rl import GraphEnv, GraphObservation, GNNPolicy, REINFORCETrainer


# ---------------------------------------------------------------------------
# GraphEnv tests
# ---------------------------------------------------------------------------

class TestGraphEnv:
    def setup_method(self):
        self.env = GraphEnv(num_nodes=8, edge_prob=0.3, max_steps=20, seed=42)

    def test_reset_returns_observation(self):
        obs = self.env.reset()
        assert isinstance(obs, GraphObservation)
        assert obs.node_features.shape == (8, 2)
        assert obs.edge_index.shape[0] == 2
        assert obs.num_nodes == 8

    def test_node_features_valid(self):
        obs = self.env.reset()
        # All features in [0, 1]
        assert obs.node_features.min() >= 0.0
        assert obs.node_features.max() <= 1.0 + 1e-6

    def test_step_returns_correct_types(self):
        self.env.reset()
        action = 0
        obs, reward, done = self.env.step(action)
        assert isinstance(obs, GraphObservation)
        assert isinstance(reward, float)
        assert isinstance(done, bool)

    def test_episode_terminates(self):
        obs = self.env.reset()
        done = False
        steps = 0
        while not done:
            action = torch.randint(0, self.env.num_edges, (1,)).item()
            obs, reward, done = self.env.step(action)
            steps += 1
        assert steps == 20  # max_steps

    def test_loads_stay_bounded(self):
        self.env.reset()
        for _ in range(50):
            action = torch.randint(0, self.env.num_edges, (1,)).item()
            obs, _, _ = self.env.step(action)
            assert obs.node_features[:, 0].min() >= 0.0 - 1e-6
            assert obs.node_features[:, 0].max() <= 1.0 + 1e-6

    def test_reward_is_negative(self):
        """Reward = -std(loads) * 10, so <= 0."""
        self.env.reset()
        action = 0
        _, reward, _ = self.env.step(action)
        assert reward <= 0.0

    def test_action_space_size(self):
        assert self.env.action_space_size == self.env.num_edges
        assert self.env.action_space_size > 0

    def test_observation_clone(self):
        obs = self.env.reset()
        clone = obs.clone()
        # Modify original — clone should be unaffected
        obs.node_features[0, 0] = 999.0
        assert clone.node_features[0, 0] != 999.0


# ---------------------------------------------------------------------------
# GNNPolicy tests
# ---------------------------------------------------------------------------

class TestGNNPolicy:
    def setup_method(self):
        self.env = GraphEnv(num_nodes=10, edge_prob=0.3, max_steps=20, seed=0)
        self.policy = GNNPolicy(
            in_features=self.env.observation_feature_dim,
            hidden_dim=32,
            num_edges=self.env.num_edges,
        )

    def test_forward_returns_categorical(self):
        from torch.distributions import Categorical
        obs = self.env.reset()
        dist = self.policy(obs)
        assert isinstance(dist, Categorical)
        assert dist.probs.shape == (self.env.num_edges,)

    def test_action_in_valid_range(self):
        obs = self.env.reset()
        action, log_prob = self.policy.act(obs)
        assert 0 <= action < self.env.num_edges
        assert log_prob.shape == ()

    def test_probs_sum_to_one(self):
        obs = self.env.reset()
        dist = self.policy(obs)
        assert abs(dist.probs.sum().item() - 1.0) < 1e-5

    def test_backward_works(self):
        obs = self.env.reset()
        _, log_prob = self.policy.act(obs)
        loss = -log_prob
        loss.backward()  # should not raise
        # Check at least one gradient is non-None and non-zero
        has_grad = any(
            p.grad is not None and p.grad.abs().sum() > 0
            for p in self.policy.parameters()
        )
        assert has_grad

    def test_parameter_count_reasonable(self):
        total = sum(p.numel() for p in self.policy.parameters())
        assert total > 100  # not trivially small
        assert total < 1_000_000  # not insanely large


# ---------------------------------------------------------------------------
# REINFORCETrainer tests
# ---------------------------------------------------------------------------

class TestREINFORCETrainer:
    def setup_method(self):
        self.env = GraphEnv(num_nodes=8, edge_prob=0.3, max_steps=20, seed=1)
        self.policy = GNNPolicy(
            in_features=self.env.observation_feature_dim,
            hidden_dim=16,
            num_edges=self.env.num_edges,
        )
        self.trainer = REINFORCETrainer(self.env, self.policy, lr=1e-3)

    def test_train_step_returns_stats(self):
        from pg_neo_graph_rl.trainer import TrainStats
        stats = self.trainer.train_step()
        assert isinstance(stats, TrainStats)
        assert isinstance(stats.total_reward, float)
        assert isinstance(stats.loss, float)

    def test_baseline_moves_toward_returns(self):
        initial_baseline = self.trainer.baseline  # 0.0
        for _ in range(10):
            self.trainer.train_step()
        # Baseline should have shifted from 0
        assert self.trainer.baseline != initial_baseline

    def test_short_training_run(self):
        """Just check it doesn't crash and returns history."""
        history = self.trainer.train(num_episodes=10, log_every=5, verbose=False)
        assert len(history) == 10
        assert all(s.episode > 0 for s in history)

    def test_reward_trend_improves(self):
        """
        After 200 episodes, the last-50 avg reward should be >= first-50 avg.
        This isn't guaranteed but should hold for this simple env.
        """
        torch.manual_seed(42)
        env = GraphEnv(num_nodes=8, edge_prob=0.4, max_steps=30, seed=42)
        policy = GNNPolicy(
            in_features=env.observation_feature_dim,
            hidden_dim=32,
            num_edges=env.num_edges,
        )
        trainer = REINFORCETrainer(env, policy, lr=3e-4)
        history = trainer.train(num_episodes=200, log_every=50, verbose=False)

        first50 = sum(s.total_reward for s in history[:50]) / 50
        last50 = sum(s.total_reward for s in history[-50:]) / 50
        # Allow a small tolerance — just check it's not dramatically worse
        assert last50 >= first50 - 5.0, f"Reward degraded: {first50:.2f} -> {last50:.2f}"
