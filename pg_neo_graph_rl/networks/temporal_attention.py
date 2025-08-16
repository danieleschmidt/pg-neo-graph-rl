"""
Temporal graph attention networks for handling dynamic graphs.
"""
from typing import Optional

import jax
import jax.numpy as jnp
from flax import linen as nn


class TemporalGraphAttention(nn.Module):
    """
    Temporal Graph Attention that handles time-varying graphs.
    Includes time embeddings and temporal attention mechanisms.
    """
    hidden_dim: int = 64
    num_heads: int = 4
    time_embedding_dim: int = 16
    output_dim: int = 32

    def time_encoding(self, timestamps: jnp.ndarray) -> jnp.ndarray:
        """
        Create learnable time embeddings using sinusoidal encoding.
        
        Args:
            timestamps: Node timestamps [num_nodes]
            
        Returns:
            Time embeddings [num_nodes, time_embedding_dim]
        """
        # Learnable frequency parameters
        frequencies = self.param(
            'time_frequencies',
            nn.initializers.normal(0.1),
            (self.time_embedding_dim // 2,)
        )

        # Create sinusoidal embeddings
        angles = timestamps[:, None] * frequencies[None, :]

        embeddings = jnp.concatenate([
            jnp.sin(angles),
            jnp.cos(angles)
        ], axis=-1)

        return embeddings

    def setup(self):
        """Initialize attention components."""
        self.query_projections = [
            nn.Dense(self.hidden_dim // self.num_heads, name=f'query_{i}')
            for i in range(self.num_heads)
        ]
        self.key_projections = [
            nn.Dense(self.hidden_dim // self.num_heads, name=f'key_{i}')
            for i in range(self.num_heads)
        ]
        self.value_projections = [
            nn.Dense(self.hidden_dim // self.num_heads, name=f'value_{i}')
            for i in range(self.num_heads)
        ]
        self.output_projection = nn.Dense(self.output_dim)
        self.layer_norm = nn.LayerNorm()

    def __call__(self,
                 nodes: jnp.ndarray,
                 edges: jnp.ndarray,
                 adjacency: jnp.ndarray,
                 timestamps: Optional[jnp.ndarray] = None) -> jnp.ndarray:
        """
        Forward pass with temporal attention.
        
        Args:
            nodes: Node features [num_nodes, node_dim]
            edges: Edge indices [num_edges, 2]
            adjacency: Adjacency matrix [num_nodes, num_nodes]
            timestamps: Optional node timestamps [num_nodes]
            
        Returns:
            Updated node embeddings [num_nodes, output_dim]
        """
        num_nodes = nodes.shape[0]

        # Add time embeddings if timestamps provided
        if timestamps is not None:
            time_embeds = self.time_encoding(timestamps)
            # Combine node features with time embeddings
            enhanced_nodes = jnp.concatenate([nodes, time_embeds], axis=-1)
        else:
            enhanced_nodes = nodes

        # Multi-head temporal attention
        attention_outputs = []

        for head in range(self.num_heads):
            head_dim = self.hidden_dim // self.num_heads

            # Separate transformations for Q, K, V
            query = nn.Dense(head_dim, name=f'temporal_query_{head}')(enhanced_nodes)
            key = nn.Dense(head_dim, name=f'temporal_key_{head}')(enhanced_nodes)
            value = nn.Dense(head_dim, name=f'temporal_value_{head}')(enhanced_nodes)

            # Compute attention scores
            scores = jnp.dot(query, key.T) / jnp.sqrt(head_dim)

            # Apply temporal decay if timestamps available
            if timestamps is not None:
                # Create temporal decay matrix
                time_diff = jnp.abs(timestamps[:, None] - timestamps[None, :])
                temporal_decay = jnp.exp(-time_diff / 10.0)  # Decay parameter
                scores = scores * temporal_decay

            # Mask to only connected nodes (plus self-connections)
            mask = adjacency + jnp.eye(num_nodes)
            scores = jnp.where(mask > 0, scores, -jnp.inf)

            # Softmax attention weights
            attention_weights = jax.nn.softmax(scores, axis=-1)

            # Apply attention to values
            attended = jnp.dot(attention_weights, value)
            attention_outputs.append(attended)

        # Concatenate multi-head outputs
        multi_head_output = jnp.concatenate(attention_outputs, axis=-1)

        # Residual connection and layer norm
        if multi_head_output.shape[-1] == enhanced_nodes.shape[-1]:
            multi_head_output = multi_head_output + enhanced_nodes

        multi_head_output = nn.LayerNorm()(multi_head_output)

        # Feed-forward with temporal processing
        ff_hidden = nn.Dense(self.hidden_dim * 2)(multi_head_output)
        ff_hidden = jax.nn.gelu(ff_hidden)  # GELU for better gradient flow
        ff_output = nn.Dense(self.output_dim)(ff_hidden)

        return ff_output


class TemporalGRUCell(nn.Module):
    """
    GRU cell for processing temporal sequences in graphs.
    """
    hidden_size: int

    def setup(self):
        """Initialize GRU cell components."""
        self.reset_gate = nn.Dense(self.hidden_size, use_bias=True, name='reset_gate')
        self.update_gate = nn.Dense(self.hidden_size, use_bias=True, name='update_gate')
        self.new_gate = nn.Dense(self.hidden_size, use_bias=True, name='new_gate')

    def __call__(self,
                 hidden_state: jnp.ndarray,
                 input_features: jnp.ndarray) -> jnp.ndarray:
        """
        GRU cell forward pass.
        
        Args:
            hidden_state: Previous hidden state [num_nodes, hidden_size]
            input_features: Current input features [num_nodes, input_dim]
            
        Returns:
            New hidden state [num_nodes, hidden_size]
        """
        # Reset gate
        reset_gate = nn.Dense(self.hidden_size, name='reset_gate')
        r = jax.nn.sigmoid(reset_gate(jnp.concatenate([hidden_state, input_features], axis=-1)))

        # Update gate
        update_gate = nn.Dense(self.hidden_size, name='update_gate')
        z = jax.nn.sigmoid(update_gate(jnp.concatenate([hidden_state, input_features], axis=-1)))

        # New gate
        new_gate = nn.Dense(self.hidden_size, name='new_gate')
        reset_hidden = r * hidden_state
        n = jax.nn.tanh(new_gate(jnp.concatenate([reset_hidden, input_features], axis=-1)))

        # Update hidden state
        new_hidden = (1 - z) * n + z * hidden_state

        return new_hidden


class DynamicGraphNetwork(nn.Module):
    """
    Network for processing dynamic graphs with temporal evolution.
    """
    hidden_dim: int = 64
    num_layers: int = 2
    time_embedding_dim: int = 16
    output_dim: int = 32

    def setup(self):
        """Initialize dynamic graph network components."""
        self.temporal_attention = TemporalGraphAttention(
            hidden_dim=self.hidden_dim,
            time_embedding_dim=self.time_embedding_dim,
            output_dim=self.hidden_dim
        )
        self.gru_cell = TemporalGRUCell(hidden_size=self.hidden_dim)
        self.output_projection = nn.Dense(self.output_dim)

    def __call__(self,
                 node_sequence: jnp.ndarray,  # [seq_len, num_nodes, node_dim]
                 edge_sequence: jnp.ndarray,  # [seq_len, num_edges, 2]
                 adjacency_sequence: jnp.ndarray,  # [seq_len, num_nodes, num_nodes]
                 timestamps: Optional[jnp.ndarray] = None) -> jnp.ndarray:
        """
        Process a sequence of dynamic graphs.
        
        Args:
            node_sequence: Sequence of node features
            edge_sequence: Sequence of edge indices
            adjacency_sequence: Sequence of adjacency matrices
            timestamps: Optional timestamps for each time step
            
        Returns:
            Final node embeddings [num_nodes, output_dim]
        """
        seq_len, num_nodes, node_dim = node_sequence.shape

        # Initialize hidden states
        hidden_states = jnp.zeros((num_nodes, self.hidden_dim))

        # Process each time step
        for t in range(seq_len):
            current_nodes = node_sequence[t]
            current_edges = edge_sequence[t]
            current_adjacency = adjacency_sequence[t]
            current_timestamps = timestamps[t] if timestamps is not None else None

            # Apply temporal graph attention
            temporal_attention = TemporalGraphAttention(
                hidden_dim=self.hidden_dim,
                time_embedding_dim=self.time_embedding_dim,
                output_dim=self.hidden_dim
            )

            attended_features = temporal_attention(
                current_nodes,
                current_edges,
                current_adjacency,
                current_timestamps
            )

            # Update hidden states with GRU
            gru_cell = TemporalGRUCell(hidden_size=self.hidden_dim)
            hidden_states = gru_cell(hidden_states, attended_features)

        # Final output projection
        output = nn.Dense(self.output_dim)(hidden_states)

        return output
