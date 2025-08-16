"""
Graph Soft Actor-Critic (SAC) for federated learning.
"""
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import jax
import jax.numpy as jnp
import optax
from flax import linen as nn

from ..core.types import GraphState
from ..networks.graph_networks import GraphEncoder


@dataclass
class SACConfig:
    """Configuration for GraphSAC algorithm."""
    learning_rate: float = 3e-4
    tau: float = 0.005  # Soft update coefficient
    alpha: float = 0.2  # Temperature parameter
    gamma: float = 0.99
    target_update_freq: int = 1


class GraphActor(nn.Module):
    """Actor network for SAC."""
    action_dim: int
    hidden_dims: tuple = (128, 64)

    def setup(self):
        """Initialize actor components."""
        self.encoder = GraphEncoder(hidden_dims=self.hidden_dims, output_dim=self.hidden_dims[-1])
        self.actor_layers = [nn.Dense(dim) for dim in self.hidden_dims]
        self.mean_head = nn.Dense(self.action_dim)
        self.log_std_head = nn.Dense(self.action_dim)

    def __call__(self, nodes, edges, adjacency, training=True):
        # Graph encoding
        embeddings = self.encoder(nodes, edges, adjacency, training=training)

        # Actor head - output mean and log_std
        for layer in self.actor_layers:
            embeddings = layer(embeddings)
            embeddings = jax.nn.relu(embeddings)

        mean = self.mean_head(embeddings)
        log_std = self.log_std_head(embeddings)
        log_std = jnp.clip(log_std, -20, 2)  # Bound log_std

        return mean, log_std


class GraphCritic(nn.Module):
    """Critic network for SAC."""
    hidden_dims: tuple = (128, 64)

    def setup(self):
        """Initialize critic components."""
        self.encoder = GraphEncoder(hidden_dims=self.hidden_dims, output_dim=self.hidden_dims[-1])
        self.critic_layers = [nn.Dense(dim) for dim in self.hidden_dims]
        self.value_head = nn.Dense(1)

    def __call__(self, nodes, edges, adjacency, actions, training=True):
        # Graph encoding
        embeddings = self.encoder(nodes, edges, adjacency, training=training)

        # Concatenate with actions
        x = jnp.concatenate([embeddings, actions], axis=-1)

        # Critic head
        for layer in self.critic_layers:
            x = layer(x)
            x = jax.nn.relu(x)

        q_values = self.value_head(x)
        return q_values.squeeze(-1)


class GraphSAC:
    """Graph Soft Actor-Critic algorithm."""

    def __init__(self, agent_id: int, action_dim: int, node_dim: int, config: Optional[SACConfig] = None):
        self.agent_id = agent_id
        self.action_dim = action_dim
        self.node_dim = node_dim
        self.config = config or SACConfig()

        # Networks
        self.actor = GraphActor(action_dim=action_dim)
        self.critic1 = GraphCritic()
        self.critic2 = GraphCritic()

        # Optimizers
        self.actor_optimizer = optax.adam(self.config.learning_rate)
        self.critic_optimizer = optax.adam(self.config.learning_rate)

        self.rng_key = jax.random.PRNGKey(agent_id)
        self._init_networks()

    def _init_networks(self):
        """Initialize network parameters."""
        dummy_nodes = jnp.ones((10, self.node_dim))
        dummy_edges = jnp.array([[0, 1], [1, 2]])
        dummy_adjacency = jnp.eye(10)
        dummy_actions = jnp.ones((10, self.action_dim))

        keys = jax.random.split(self.rng_key, 4)

        # Initialize actor
        self.actor_params = self.actor.init(keys[0], dummy_nodes, dummy_edges, dummy_adjacency)
        self.actor_opt_state = self.actor_optimizer.init(self.actor_params)

        # Initialize critics
        self.critic1_params = self.critic1.init(keys[1], dummy_nodes, dummy_edges, dummy_adjacency, dummy_actions)
        self.critic2_params = self.critic2.init(keys[2], dummy_nodes, dummy_edges, dummy_adjacency, dummy_actions)

        # Target networks
        self.critic1_target_params = self.critic1_params.copy()
        self.critic2_target_params = self.critic2_params.copy()

        self.critic_opt_state = self.critic_optimizer.init({
            'critic1': self.critic1_params,
            'critic2': self.critic2_params
        })

    def act(self, graph_state: GraphState, training: bool = True) -> Tuple[jnp.ndarray, Dict[str, Any]]:
        """Select actions using the current policy."""
        mean, log_std = self.actor.apply(self.actor_params, graph_state.nodes, graph_state.edges, graph_state.adjacency)

        if training:
            # Sample from policy
            key, self.rng_key = jax.random.split(self.rng_key)
            noise = jax.random.normal(key, mean.shape)
            actions = mean + jnp.exp(log_std) * noise
        else:
            # Deterministic action
            actions = mean

        return actions, {'mean': mean, 'log_std': log_std}

    def compute_gradients(self, trajectories: Dict[str, Any]) -> Dict[str, jnp.ndarray]:
        """Compute SAC gradients."""
        # Simplified gradient computation
        dummy_grads = {
            'actor': jax.tree.map(lambda x: jnp.zeros_like(x), self.actor_params),
            'critic1': jax.tree.map(lambda x: jnp.zeros_like(x), self.critic1_params),
            'critic2': jax.tree.map(lambda x: jnp.zeros_like(x), self.critic2_params)
        }
        return dummy_grads

    def apply_gradients(self, gradients: Dict[str, jnp.ndarray]) -> None:
        """Apply gradients to update parameters."""
        # Update actor
        actor_updates, self.actor_opt_state = self.actor_optimizer.update(
            gradients['actor'], self.actor_opt_state
        )
        self.actor_params = optax.apply_updates(self.actor_params, actor_updates)

        # Update critics (simplified)
        critic_grads = {'critic1': gradients['critic1'], 'critic2': gradients['critic2']}
        critic_updates, self.critic_opt_state = self.critic_optimizer.update(
            critic_grads, self.critic_opt_state
        )

        self.critic1_params = optax.apply_updates(self.critic1_params, critic_updates['critic1'])
        self.critic2_params = optax.apply_updates(self.critic2_params, critic_updates['critic2'])
