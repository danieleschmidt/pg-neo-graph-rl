"""
Core data types for pg-neo-graph-rl.
"""
import jax.numpy as jnp
from typing import Optional
from dataclasses import dataclass


@dataclass
class GraphState:
    """Graph state representation for federated learning."""
    nodes: jnp.ndarray  # Node features [num_nodes, node_dim]
    edges: jnp.ndarray  # Edge indices [num_edges, 2] 
    edge_attr: jnp.ndarray  # Edge features [num_edges, edge_dim]
    adjacency: jnp.ndarray  # Adjacency matrix [num_nodes, num_nodes]
    timestamps: Optional[jnp.ndarray] = None  # Node timestamps