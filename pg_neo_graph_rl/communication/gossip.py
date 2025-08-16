"""
Gossip-based communication protocol for federated learning.
"""
from typing import Dict, List

import jax
import jax.numpy as jnp
import networkx as nx


class GossipProtocol:
    """Implements gossip-based parameter sharing."""

    def __init__(self, num_agents: int, topology: str = "random", gossip_probability: float = 0.3):
        self.num_agents = num_agents
        self.topology = topology
        self.gossip_probability = gossip_probability
        self.communication_graph = self._build_topology()

    def _build_topology(self) -> nx.Graph:
        """Build communication topology."""
        G = nx.Graph()
        G.add_nodes_from(range(self.num_agents))

        if self.topology == "random":
            for i in range(self.num_agents):
                for j in range(i + 1, self.num_agents):
                    if jax.random.uniform(jax.random.PRNGKey(i * j)) < self.gossip_probability:
                        G.add_edge(i, j)
        elif self.topology == "ring":
            for i in range(self.num_agents):
                G.add_edge(i, (i + 1) % self.num_agents)
        elif self.topology == "complete":
            for i in range(self.num_agents):
                for j in range(i + 1, self.num_agents):
                    G.add_edge(i, j)

        return G

    def get_neighbors(self, agent_id: int) -> List[int]:
        """Get neighbors for gossip communication."""
        return list(self.communication_graph.neighbors(agent_id))

    def gossip_round(self, agent_params: Dict[int, Dict[str, jnp.ndarray]]) -> Dict[int, Dict[str, jnp.ndarray]]:
        """Execute one round of gossip communication."""
        updated_params = agent_params.copy()

        for agent_id in range(self.num_agents):
            neighbors = self.get_neighbors(agent_id)
            if not neighbors:
                continue

            # Average parameters with neighbors
            neighbor_params = [agent_params[neighbor] for neighbor in neighbors]
            neighbor_params.append(agent_params[agent_id])  # Include self

            # Average all parameters
            averaged_params = {}
            for key in agent_params[agent_id].keys():
                param_stack = jnp.stack([params[key] for params in neighbor_params])
                averaged_params[key] = jnp.mean(param_stack, axis=0)

            updated_params[agent_id] = averaged_params

        return updated_params

    def add_privacy_noise(self, params: Dict[str, jnp.ndarray], noise_scale: float = 0.01) -> Dict[str, jnp.ndarray]:
        """Add differential privacy noise to parameters."""
        noisy_params = {}
        for key, param in params.items():
            noise = jax.random.normal(jax.random.PRNGKey(42), param.shape) * noise_scale
            noisy_params[key] = param + noise
        return noisy_params
