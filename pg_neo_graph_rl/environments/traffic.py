"""
Traffic network environment for federated RL.
"""
from typing import Any, Dict, Tuple

import jax.numpy as jnp
import networkx as nx

from ..core.types import GraphState


class TrafficEnvironment:
    """Simple traffic network environment."""

    def __init__(self, city: str = "manhattan", num_intersections: int = 100, time_resolution: float = 5.0):
        self.city = city
        self.num_intersections = num_intersections
        self.time_resolution = time_resolution

        # Create traffic network
        self.graph = self._create_traffic_network()
        self.reset()

    def _create_traffic_network(self) -> nx.Graph:
        """Create a simple grid-like traffic network."""
        G = nx.grid_2d_graph(int(jnp.sqrt(self.num_intersections)), int(jnp.sqrt(self.num_intersections)))
        # Convert to standard graph with integer nodes
        mapping = {node: i for i, node in enumerate(G.nodes())}
        G = nx.relabel_nodes(G, mapping)
        return G

    def reset(self) -> GraphState:
        """Reset environment to initial state."""
        num_nodes = len(self.graph.nodes())

        # Initialize node features: [flow, density, speed, queue_length]
        self.node_features = jnp.array([
            [1.0, 0.5, 30.0, 0.0] for _ in range(num_nodes)  # flow, density, speed, queue
        ])

        # Create edge features and adjacency
        edges = list(self.graph.edges())
        self.edges = jnp.array(edges) if edges else jnp.zeros((0, 2), dtype=int)
        self.edge_features = jnp.ones((len(edges), 3))  # capacity, travel_time, congestion

        # Adjacency matrix
        self.adjacency = jnp.array(nx.adjacency_matrix(self.graph).todense())

        self.timestep = 0

        return self.get_state()

    def get_state(self) -> GraphState:
        """Get current graph state."""
        return GraphState(
            nodes=self.node_features,
            edges=self.edges,
            edge_attr=self.edge_features,
            adjacency=self.adjacency,
            timestamps=jnp.full(len(self.node_features), self.timestep)
        )

    def step(self, actions: jnp.ndarray) -> Tuple[GraphState, jnp.ndarray, bool, Dict[str, Any]]:
        """Take environment step with traffic light actions."""
        # Validate actions
        if not isinstance(actions, jnp.ndarray):
            raise ValueError("Actions must be JAX array")
        
        if len(actions) != len(self.graph.nodes()):
            raise ValueError(f"Expected {len(self.graph.nodes())} actions, got {len(actions)}")
        
        # Check for invalid action values
        if jnp.any(jnp.isnan(actions)) or jnp.any(jnp.isinf(actions)):
            raise ValueError("Actions contain NaN or infinite values")
        
        if jnp.any(actions < 0) or jnp.any(actions > 100):  # Reasonable bounds
            raise ValueError("Actions must be in range [0, 100]")
        
        # Actions: traffic light timing (0=short, 1=medium, 2=long green)
        self.timestep += 1

        # Simple traffic dynamics
        # Update flow based on actions (longer green = more flow)
        flow_multiplier = 1.0 + 0.2 * actions  # actions in [0, 1, 2]
        new_flow = self.node_features[:, 0] * flow_multiplier

        # Update density (inverse of flow)
        new_density = jnp.maximum(0.1, self.node_features[:, 1] - 0.1 * (new_flow - 1.0))

        # Update speed (inverse of density)
        new_speed = jnp.maximum(5.0, 40.0 - 30.0 * new_density)

        # Update queue length based on congestion
        queue_change = new_density - self.node_features[:, 1]
        new_queue = jnp.maximum(0.0, self.node_features[:, 3] + queue_change)

        # Update node features
        self.node_features = jnp.column_stack([new_flow, new_density, new_speed, new_queue])

        # Compute rewards (negative of average delay)
        avg_delay = jnp.mean(60.0 / new_speed)  # delay in minutes
        rewards = -avg_delay * jnp.ones(len(actions))  # Same reward for all nodes

        # Episode ends after 1000 timesteps
        done = self.timestep >= 1000

        info = {
            "avg_delay": avg_delay,
            "avg_speed": jnp.mean(new_speed),
            "avg_density": jnp.mean(new_density)
        }

        return self.get_state(), rewards, done, info

    def get_subgraph(self, agent_id: int) -> GraphState:
        """Get subgraph for specific agent."""
        # Simple partitioning: assign nodes to agents in round-robin
        total_agents = 10  # Assume 10 agents for now
        agent_nodes = jnp.arange(agent_id, len(self.node_features), total_agents)

        sub_nodes = self.node_features[agent_nodes]

        # Find edges within subgraph
        edge_mask = jnp.isin(self.edges[:, 0], agent_nodes) & jnp.isin(self.edges[:, 1], agent_nodes)
        sub_edges = self.edges[edge_mask] if len(self.edges) > 0 else jnp.zeros((0, 2), dtype=int)
        sub_edge_attr = self.edge_features[edge_mask] if len(self.edge_features) > 0 else jnp.zeros((0, 3))

        # Create subgraph adjacency
        sub_adjacency = jnp.zeros((len(agent_nodes), len(agent_nodes)))

        return GraphState(
            nodes=sub_nodes,
            edges=sub_edges,
            edge_attr=sub_edge_attr,
            adjacency=sub_adjacency,
            timestamps=jnp.full(len(sub_nodes), self.timestep)
        )

    def evaluate_global_performance(self) -> Dict[str, float]:
        """Evaluate global traffic performance."""
        avg_delay = jnp.mean(60.0 / self.node_features[:, 2])  # minutes
        avg_speed = jnp.mean(self.node_features[:, 2])  # km/h
        avg_queue = jnp.mean(self.node_features[:, 3])

        return {
            "avg_delay": float(avg_delay),
            "avg_speed": float(avg_speed),
            "avg_queue_length": float(avg_queue),
            "total_throughput": float(jnp.sum(self.node_features[:, 0]))
        }
