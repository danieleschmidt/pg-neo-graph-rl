"""
Graph Proximal Policy Optimization (PPO) for federated learning.
"""
import jax
import jax.numpy as jnp
from flax import linen as nn
import optax
from typing import Dict, Tuple, Any, Optional
from dataclasses import dataclass

from ..core.federated import GraphState
from ..networks.graph_networks import GraphEncoder


@dataclass
class PPOConfig:
    """Configuration for GraphPPO algorithm."""
    learning_rate: float = 3e-4
    clip_epsilon: float = 0.2
    value_coeff: float = 0.5
    entropy_coeff: float = 0.01
    gamma: float = 0.99
    gae_lambda: float = 0.95
    max_grad_norm: float = 1.0
    

class GraphPolicyNetwork(nn.Module):
    """
    Policy network for graph-based reinforcement learning.
    Outputs action logits for each node in the graph.
    """
    action_dim: int
    hidden_dims: tuple = (128, 64)
    gnn_architecture: str = "gcn"
    
    @nn.compact
    def __call__(self, 
                 nodes: jnp.ndarray,
                 edges: jnp.ndarray,
                 adjacency: jnp.ndarray,
                 training: bool = True) -> jnp.ndarray:
        """
        Forward pass through policy network.
        
        Args:
            nodes: Node features [num_nodes, node_dim]
            edges: Edge indices [num_edges, 2]
            adjacency: Adjacency matrix [num_nodes, num_nodes]
            training: Whether in training mode
            
        Returns:
            Action logits [num_nodes, action_dim]
        """
        # Graph encoding
        graph_encoder = GraphEncoder(
            architecture=self.gnn_architecture,
            hidden_dims=self.hidden_dims,
            output_dim=self.hidden_dims[-1]
        )
        
        embeddings = graph_encoder(nodes, edges, adjacency, training=training)
        
        # Policy head
        for hidden_dim in self.hidden_dims:
            embeddings = nn.Dense(hidden_dim)(embeddings)
            embeddings = jax.nn.relu(embeddings)
            if training:
                embeddings = nn.Dropout(0.1)(embeddings)
        
        # Output action logits
        action_logits = nn.Dense(self.action_dim)(embeddings)
        
        return action_logits


class GraphValueNetwork(nn.Module):
    """
    Value network for graph-based reinforcement learning.
    Outputs state values for each node.
    """
    hidden_dims: tuple = (128, 64)
    gnn_architecture: str = "gcn"
    
    @nn.compact
    def __call__(self,
                 nodes: jnp.ndarray,
                 edges: jnp.ndarray, 
                 adjacency: jnp.ndarray,
                 training: bool = True) -> jnp.ndarray:
        """
        Forward pass through value network.
        
        Args:
            nodes: Node features [num_nodes, node_dim]
            edges: Edge indices [num_edges, 2]
            adjacency: Adjacency matrix [num_nodes, num_nodes]
            training: Whether in training mode
            
        Returns:
            State values [num_nodes, 1]
        """
        # Graph encoding
        graph_encoder = GraphEncoder(
            architecture=self.gnn_architecture,
            hidden_dims=self.hidden_dims,
            output_dim=self.hidden_dims[-1]
        )
        
        embeddings = graph_encoder(nodes, edges, adjacency, training=training)
        
        # Value head
        for hidden_dim in self.hidden_dims:
            embeddings = nn.Dense(hidden_dim)(embeddings)
            embeddings = jax.nn.relu(embeddings)
            if training:
                embeddings = nn.Dropout(0.1)(embeddings)
        
        # Output state values
        values = nn.Dense(1)(embeddings)
        
        return values.squeeze(-1)  # [num_nodes]


class GraphPPO:
    """
    Graph Proximal Policy Optimization algorithm for federated learning.
    
    Implements PPO with graph neural networks for distributed control problems.
    """
    
    def __init__(self, 
                 agent_id: int,
                 action_dim: int,
                 node_dim: int,
                 config: Optional[PPOConfig] = None):
        """
        Initialize GraphPPO agent.
        
        Args:
            agent_id: Unique identifier for this agent
            action_dim: Dimension of action space
            node_dim: Dimension of node features
            config: PPO configuration
        """
        self.agent_id = agent_id
        self.action_dim = action_dim
        self.node_dim = node_dim
        self.config = config or PPOConfig()
        
        # Initialize networks
        self.policy_network = GraphPolicyNetwork(action_dim=action_dim)
        self.value_network = GraphValueNetwork()
        
        # Initialize optimizers
        self.policy_optimizer = optax.adam(self.config.learning_rate)
        self.value_optimizer = optax.adam(self.config.learning_rate)
        
        # Initialize network parameters and optimizer states
        self.rng_key = jax.random.PRNGKey(agent_id)
        self._init_networks()
        
    def _init_networks(self):
        """Initialize network parameters."""
        # Dummy input for initialization
        dummy_nodes = jnp.ones((10, self.node_dim))
        dummy_edges = jnp.array([[0, 1], [1, 2]])
        dummy_adjacency = jnp.eye(10)
        
        # Initialize policy network
        policy_key, value_key = jax.random.split(self.rng_key, 2)
        
        self.policy_params = self.policy_network.init(
            policy_key, dummy_nodes, dummy_edges, dummy_adjacency, training=False
        )
        self.policy_opt_state = self.policy_optimizer.init(self.policy_params)
        
        # Initialize value network  
        self.value_params = self.value_network.init(
            value_key, dummy_nodes, dummy_edges, dummy_adjacency, training=False
        )
        self.value_opt_state = self.value_optimizer.init(self.value_params)
        
    def act(self, graph_state: GraphState, training: bool = True) -> Tuple[jnp.ndarray, Dict[str, Any]]:
        """
        Select actions for the current graph state.
        
        Args:
            graph_state: Current graph state
            training: Whether in training mode
            
        Returns:
            actions: Selected actions [num_nodes]
            info: Additional information (log_probs, values, etc.)
        """
        # Get action logits from policy network
        action_logits = self.policy_network.apply(
            self.policy_params,
            graph_state.nodes,
            graph_state.edges,
            graph_state.adjacency,
            training=training
        )
        
        # Sample actions
        key, self.rng_key = jax.random.split(self.rng_key)
        actions = jax.random.categorical(key, action_logits)
        
        # Compute log probabilities
        log_probs = jax.nn.log_softmax(action_logits)
        action_log_probs = log_probs[jnp.arange(len(actions)), actions]
        
        # Get state values
        values = self.value_network.apply(
            self.value_params,
            graph_state.nodes,
            graph_state.edges, 
            graph_state.adjacency,
            training=training
        )
        
        info = {
            "log_probs": action_log_probs,
            "values": values,
            "action_logits": action_logits
        }
        
        return actions, info
    
    def compute_advantages(self, 
                          rewards: jnp.ndarray,
                          values: jnp.ndarray,
                          dones: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """
        Compute advantages using Generalized Advantage Estimation (GAE).
        
        Args:
            rewards: Rewards [seq_len, num_nodes]
            values: State values [seq_len, num_nodes]  
            dones: Done flags [seq_len, num_nodes]
            
        Returns:
            advantages: Computed advantages [seq_len, num_nodes]
            returns: Discounted returns [seq_len, num_nodes]
        """
        seq_len = rewards.shape[0]
        advantages = jnp.zeros_like(rewards)
        
        # Compute advantages using GAE
        gae = 0
        for t in reversed(range(seq_len)):
            if t == seq_len - 1:
                next_value = 0
            else:
                next_value = values[t + 1] * (1 - dones[t])
            
            delta = rewards[t] + self.config.gamma * next_value - values[t]
            gae = delta + self.config.gamma * self.config.gae_lambda * (1 - dones[t]) * gae
            advantages = advantages.at[t].set(gae)
        
        # Compute returns
        returns = advantages + values
        
        return advantages, returns
    
    def compute_loss(self,
                     params: Dict,
                     graph_states: GraphState,
                     actions: jnp.ndarray,
                     old_log_probs: jnp.ndarray,
                     advantages: jnp.ndarray,
                     returns: jnp.ndarray) -> Tuple[jnp.ndarray, Dict[str, Any]]:
        """
        Compute PPO loss.
        
        Args:
            params: Network parameters
            graph_states: Batch of graph states
            actions: Actions taken [batch_size, num_nodes]
            old_log_probs: Old action log probabilities [batch_size, num_nodes] 
            advantages: Advantage estimates [batch_size, num_nodes]
            returns: Target returns [batch_size, num_nodes]
            
        Returns:
            loss: Total loss
            info: Loss breakdown
        """
        # Get current action logits and values
        action_logits = self.policy_network.apply(
            params['policy'],
            graph_states.nodes,
            graph_states.edges,
            graph_states.adjacency,
            training=True
        )
        
        values = self.value_network.apply(
            params['value'],
            graph_states.nodes,
            graph_states.edges,
            graph_states.adjacency,
            training=True
        )
        
        # Compute new log probabilities
        log_probs = jax.nn.log_softmax(action_logits)
        new_log_probs = log_probs[jnp.arange(len(actions)), actions]
        
        # Compute probability ratio
        ratio = jnp.exp(new_log_probs - old_log_probs)
        
        # Compute policy loss (PPO clipped objective)
        surr1 = ratio * advantages
        surr2 = jnp.clip(ratio, 1 - self.config.clip_epsilon, 1 + self.config.clip_epsilon) * advantages
        policy_loss = -jnp.mean(jnp.minimum(surr1, surr2))
        
        # Compute value loss
        value_loss = jnp.mean((values - returns) ** 2)
        
        # Compute entropy bonus
        entropy = -jnp.sum(jax.nn.softmax(action_logits) * log_probs, axis=-1)
        entropy_loss = -jnp.mean(entropy)
        
        # Total loss
        total_loss = (policy_loss + 
                     self.config.value_coeff * value_loss +
                     self.config.entropy_coeff * entropy_loss)
        
        info = {
            "policy_loss": policy_loss,
            "value_loss": value_loss,
            "entropy_loss": entropy_loss,
            "total_loss": total_loss
        }
        
        return total_loss, info
    
    def update(self, trajectories: Dict[str, Any]) -> Dict[str, Any]:
        """
        Update policy and value networks using collected trajectories.
        
        Args:
            trajectories: Dictionary containing trajectory data
            
        Returns:
            Training metrics
        """
        # Extract trajectory data
        graph_states = trajectories["states"]
        actions = trajectories["actions"]
        rewards = trajectories["rewards"]
        old_log_probs = trajectories["log_probs"]
        values = trajectories["values"]
        dones = trajectories["dones"]
        
        # Compute advantages and returns
        advantages, returns = self.compute_advantages(rewards, values, dones)
        
        # Normalize advantages
        advantages = (advantages - jnp.mean(advantages)) / (jnp.std(advantages) + 1e-8)
        
        # Update policy network
        def policy_loss_fn(params):
            loss, info = self.compute_loss(
                {"policy": params, "value": self.value_params},
                graph_states, actions, old_log_probs, advantages, returns
            )
            return loss, info
        
        (policy_loss, policy_info), policy_grads = jax.value_and_grad(
            policy_loss_fn, has_aux=True
        )(self.policy_params)
        
        # Clip gradients
        policy_grads = optax.clip_by_global_norm(self.config.max_grad_norm)(policy_grads)[0]
        
        # Apply policy updates
        policy_updates, self.policy_opt_state = self.policy_optimizer.update(
            policy_grads, self.policy_opt_state
        )
        self.policy_params = optax.apply_updates(self.policy_params, policy_updates)
        
        # Update value network
        def value_loss_fn(params):
            values_pred = self.value_network.apply(
                params, graph_states.nodes, graph_states.edges, graph_states.adjacency, training=True
            )
            return jnp.mean((values_pred - returns) ** 2)
        
        value_loss, value_grads = jax.value_and_grad(value_loss_fn)(self.value_params)
        
        value_grads = optax.clip_by_global_norm(self.config.max_grad_norm)(value_grads)[0]
        
        value_updates, self.value_opt_state = self.value_optimizer.update(
            value_grads, self.value_opt_state
        )
        self.value_params = optax.apply_updates(self.value_params, value_updates)
        
        return {
            "policy_loss": policy_info["policy_loss"],
            "value_loss": value_loss,
            "entropy_loss": policy_info["entropy_loss"],
            "total_loss": policy_info["total_loss"]
        }
    
    def collect_trajectories(self, 
                           graph_state: GraphState,
                           num_steps: int = 100) -> Dict[str, Any]:
        """
        Collect trajectories by interacting with the environment.
        
        Args:
            graph_state: Initial graph state
            num_steps: Number of steps to collect
            
        Returns:
            Dictionary containing trajectory data
        """
        # For now, return dummy trajectory data
        # In practice, this would interact with an actual environment
        
        num_nodes = graph_state.nodes.shape[0]
        
        dummy_trajectories = {
            "states": graph_state,
            "actions": jax.random.randint(self.rng_key, (num_steps, num_nodes), 0, self.action_dim),
            "rewards": jax.random.normal(self.rng_key, (num_steps, num_nodes)),
            "log_probs": jax.random.normal(self.rng_key, (num_steps, num_nodes)),
            "values": jax.random.normal(self.rng_key, (num_steps, num_nodes)),
            "dones": jnp.zeros((num_steps, num_nodes))
        }
        
        return dummy_trajectories
    
    def compute_gradients(self, trajectories: Dict[str, Any]) -> Dict[str, jnp.ndarray]:
        """
        Compute gradients for federated learning.
        
        Args:
            trajectories: Trajectory data
            
        Returns:
            Gradients dictionary
        """
        # Extract trajectory data
        graph_states = trajectories["states"] 
        actions = trajectories["actions"]
        rewards = trajectories["rewards"]
        old_log_probs = trajectories["log_probs"]
        values = trajectories["values"]
        dones = trajectories["dones"]
        
        # Compute advantages and returns
        advantages, returns = self.compute_advantages(rewards, values, dones)
        advantages = (advantages - jnp.mean(advantages)) / (jnp.std(advantages) + 1e-8)
        
        # Compute gradients
        def loss_fn(params):
            loss, _ = self.compute_loss(
                params, graph_states, actions, old_log_probs, advantages, returns
            )
            return loss
        
        params = {"policy": self.policy_params, "value": self.value_params}
        gradients = jax.grad(loss_fn)(params)
        
        return gradients
    
    def apply_gradients(self, gradients: Dict[str, jnp.ndarray]) -> None:
        """
        Apply gradients to update network parameters.
        
        Args:
            gradients: Gradients to apply
        """
        # Apply policy gradients
        policy_updates, self.policy_opt_state = self.policy_optimizer.update(
            gradients["policy"], self.policy_opt_state
        )
        self.policy_params = optax.apply_updates(self.policy_params, policy_updates)
        
        # Apply value gradients
        value_updates, self.value_opt_state = self.value_optimizer.update(
            gradients["value"], self.value_opt_state
        )
        self.value_params = optax.apply_updates(self.value_params, value_updates)