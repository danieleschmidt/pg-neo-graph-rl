"""
Core data types for pg-neo-graph-rl.
"""
from dataclasses import dataclass
from typing import Optional

import jax.numpy as jnp


@dataclass
class GraphState:
    """Graph state representation for federated learning."""
    nodes: jnp.ndarray  # Node features [num_nodes, node_dim]
    edges: jnp.ndarray  # Edge indices [num_edges, 2]
    edge_attr: jnp.ndarray  # Edge features [num_edges, edge_dim]
    adjacency: jnp.ndarray  # Adjacency matrix [num_nodes, num_nodes]
    timestamps: Optional[jnp.ndarray] = None  # Node timestamps
