"""
Core federated graph reinforcement learning orchestration.
"""
import jax
import jax.numpy as jnp
from typing import Dict, List, Optional, Tuple, Any, Callable
from dataclasses import dataclass
import networkx as nx
from flax import linen as nn
from pydantic import BaseModel, Field


@dataclass
class GraphState:
    """Graph state representation for federated learning."""
    nodes: jnp.ndarray  # Node features [num_nodes, node_dim]
    edges: jnp.ndarray  # Edge indices [num_edges, 2] 
    edge_attr: jnp.ndarray  # Edge features [num_edges, edge_dim]
    adjacency: jnp.ndarray  # Adjacency matrix [num_nodes, num_nodes]
    timestamps: Optional[jnp.ndarray] = None  # Node timestamps


class FederatedConfig(BaseModel):
    """Configuration for federated learning system."""
    num_agents: int = Field(default=10, description="Number of federated agents")
    aggregation: str = Field(default="gossip", description="Aggregation method: gossip, hierarchical, ring")
    communication_rounds: int = Field(default=10, description="Communication rounds per episode")
    privacy_noise: float = Field(default=0.0, description="Differential privacy noise level")
    topology: str = Field(default="random", description="Communication topology")


class FederatedGraphRL:
    """
    Main orchestration class for federated graph reinforcement learning.
    
    Coordinates distributed learning across multiple agents with different
    communication topologies and aggregation methods.
    """
    
    def __init__(self, 
                 num_agents: int = 10,
                 aggregation: str = "gossip",
                 communication_rounds: int = 10,
                 privacy_noise: float = 0.0,
                 topology: str = "random"):
        """
        Initialize federated learning system.
        
        Args:
            num_agents: Number of distributed agents
            aggregation: Aggregation method ("gossip", "hierarchical", "ring")
            communication_rounds: Communication rounds per training step
            privacy_noise: Differential privacy noise level
            topology: Communication graph topology
        """
        self.config = FederatedConfig(
            num_agents=num_agents,
            aggregation=aggregation,
            communication_rounds=communication_rounds,
            privacy_noise=privacy_noise,
            topology=topology
        )
        
        self.agents = []
        self.communication_graph = self._build_communication_graph()
        self.global_step = 0
        
        # Initialize random key for JAX
        self.rng_key = jax.random.PRNGKey(42)
        
    def _build_communication_graph(self) -> nx.Graph:
        """Build communication graph between agents."""
        G = nx.Graph()
        G.add_nodes_from(range(self.config.num_agents))
        
        if self.config.topology == "random":
            # Random graph with average degree 4
            for i in range(self.config.num_agents):
                for j in range(i + 1, self.config.num_agents):
                    if jax.random.uniform(self.rng_key) < 0.3:
                        G.add_edge(i, j)
        elif self.config.topology == "ring":
            # Ring topology
            for i in range(self.config.num_agents):
                G.add_edge(i, (i + 1) % self.config.num_agents)
        elif self.config.topology == "hierarchical":
            # Tree-like hierarchy
            for i in range(1, self.config.num_agents):
                parent = (i - 1) // 2
                G.add_edge(i, parent)
        
        return G
    
    def register_agent(self, agent: Any) -> int:
        """Register a new agent in the federated system."""
        agent_id = len(self.agents)
        self.agents.append(agent)
        return agent_id
    
    def get_neighbors(self, agent_id: int) -> List[int]:
        """Get neighbors of an agent in communication graph."""
        return list(self.communication_graph.neighbors(agent_id))
    
    def get_subgraph(self, agent_id: int, graph_state: GraphState) -> GraphState:
        """
        Extract subgraph for a specific agent.
        
        For simplicity, this assigns nodes to agents in round-robin fashion.
        In practice, this would be based on geographical or logical partitioning.
        """
        num_nodes = graph_state.nodes.shape[0]
        agent_nodes = jnp.arange(agent_id, num_nodes, self.config.num_agents)
        
        # Extract subgraph nodes
        sub_nodes = graph_state.nodes[agent_nodes]
        
        # Find edges within this subgraph
        edge_mask = jnp.isin(graph_state.edges[:, 0], agent_nodes) & \
                   jnp.isin(graph_state.edges[:, 1], agent_nodes)
        sub_edges = graph_state.edges[edge_mask]
        sub_edge_attr = graph_state.edge_attr[edge_mask]
        
        # Remap edge indices to local subgraph
        node_mapping = {old_idx: new_idx for new_idx, old_idx in enumerate(agent_nodes)}
        remapped_edges = jnp.array([[node_mapping.get(e[0], 0), node_mapping.get(e[1], 0)] 
                                   for e in sub_edges])
        
        # Create subgraph adjacency
        sub_adjacency = jnp.zeros((len(agent_nodes), len(agent_nodes)))
        if len(remapped_edges) > 0:
            sub_adjacency = sub_adjacency.at[remapped_edges[:, 0], remapped_edges[:, 1]].set(1.0)
            sub_adjacency = sub_adjacency.at[remapped_edges[:, 1], remapped_edges[:, 0]].set(1.0)
        
        return GraphState(
            nodes=sub_nodes,
            edges=remapped_edges,
            edge_attr=sub_edge_attr,
            adjacency=sub_adjacency,
            timestamps=graph_state.timestamps[agent_nodes] if graph_state.timestamps is not None else None
        )
    
    def gossip_aggregate(self, 
                        agent_id: int, 
                        local_gradients: Dict[str, jnp.ndarray],
                        neighbors: Optional[List[int]] = None) -> Dict[str, jnp.ndarray]:
        """
        Perform gossip-based gradient aggregation.
        
        Args:
            agent_id: ID of the requesting agent
            local_gradients: Local gradients to aggregate
            neighbors: Optional list of neighbors (uses communication graph if None)
            
        Returns:
            Aggregated gradients
        """
        if neighbors is None:
            neighbors = self.get_neighbors(agent_id)
        
        if not neighbors:
            return local_gradients
            
        # Simple averaging for now (in practice would collect from actual neighbors)
        aggregated = {}
        for key, grad in local_gradients.items():
            # Add privacy noise if configured
            if self.config.privacy_noise > 0:
                noise = jax.random.normal(self.rng_key, grad.shape) * self.config.privacy_noise
                grad = grad + noise
            
            # For now just return local gradients (would aggregate in real implementation)
            aggregated[key] = grad
            
        return aggregated
    
    def hierarchical_aggregate(self, 
                              agent_gradients: List[Dict[str, jnp.ndarray]]) -> Dict[str, jnp.ndarray]:
        """Hierarchical gradient aggregation."""
        if not agent_gradients:
            return {}
            
        # Simple averaging across all agents
        aggregated = {}
        for key in agent_gradients[0].keys():
            grads = [agent_grads[key] for agent_grads in agent_gradients]
            aggregated[key] = jnp.mean(jnp.stack(grads), axis=0)
            
        return aggregated
    
    def federated_round(self, 
                       agent_gradients: List[Dict[str, jnp.ndarray]]) -> List[Dict[str, jnp.ndarray]]:
        """
        Execute one round of federated learning.
        
        Args:
            agent_gradients: List of local gradients from each agent
            
        Returns:
            List of aggregated gradients for each agent
        """
        if self.config.aggregation == "gossip":
            # Each agent gossips with neighbors
            aggregated_gradients = []
            for agent_id, local_grads in enumerate(agent_gradients):
                agg_grads = self.gossip_aggregate(agent_id, local_grads)
                aggregated_gradients.append(agg_grads)
            return aggregated_gradients
            
        elif self.config.aggregation == "hierarchical":
            # Central aggregation
            global_grads = self.hierarchical_aggregate(agent_gradients)
            return [global_grads] * len(agent_gradients)
            
        else:
            # No aggregation, return local gradients
            return agent_gradients
    
    def get_communication_stats(self) -> Dict[str, Any]:
        """Get statistics about the communication graph."""
        G = self.communication_graph
        return {
            "num_agents": G.number_of_nodes(),
            "num_connections": G.number_of_edges(),
            "average_degree": sum(dict(G.degree()).values()) / G.number_of_nodes(),
            "is_connected": nx.is_connected(G),
            "clustering_coefficient": nx.average_clustering(G),
            "diameter": nx.diameter(G) if nx.is_connected(G) else float('inf')
        }
    
    def step(self) -> None:
        """Increment global training step."""
        self.global_step += 1
        
    def reset(self) -> None:
        """Reset the federated learning system."""
        self.global_step = 0
        self.agents.clear()
        self.rng_key = jax.random.PRNGKey(42)