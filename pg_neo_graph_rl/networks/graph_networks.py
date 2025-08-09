"""
Basic graph neural networks for federated RL.
"""
import jax
import jax.numpy as jnp
from flax import linen as nn
from typing import Optional, Callable


def graph_conv_layer(nodes: jnp.ndarray, 
                    edges: jnp.ndarray, 
                    adjacency: jnp.ndarray,
                    weight: jnp.ndarray,
                    bias: Optional[jnp.ndarray] = None,
                    activation: Callable = jax.nn.relu) -> jnp.ndarray:
    """
    Basic graph convolution operation.
    
    Args:
        nodes: Node features [num_nodes, in_features]
        edges: Edge indices [num_edges, 2]
        adjacency: Adjacency matrix [num_nodes, num_nodes]  
        weight: Transformation weights [in_features, out_features]
        bias: Optional bias term [out_features]
        activation: Activation function
        
    Returns:
        Updated node features [num_nodes, out_features]
    """
    # Transform node features
    transformed = jnp.dot(nodes, weight)
    if bias is not None:
        transformed = transformed + bias
    
    # Aggregate neighbor information
    # Simple approach: average neighbor features
    degree = jnp.sum(adjacency, axis=1, keepdims=True)
    degree = jnp.maximum(degree, 1.0)  # Avoid division by zero
    
    aggregated = jnp.dot(adjacency, transformed) / degree
    
    # Combine self and neighbor information
    output = transformed + aggregated
    
    return activation(output)


class GraphConvNetwork(nn.Module):
    """
    Simple Graph Convolutional Network.
    """
    hidden_dims: tuple = (64, 64)
    output_dim: int = 32
    activation: Callable = jax.nn.relu
    
    @nn.compact
    def __call__(self, 
                 nodes: jnp.ndarray,
                 edges: jnp.ndarray, 
                 adjacency: jnp.ndarray) -> jnp.ndarray:
        """
        Forward pass through graph conv network.
        
        Args:
            nodes: Node features [num_nodes, node_dim]
            edges: Edge indices [num_edges, 2]
            adjacency: Adjacency matrix [num_nodes, num_nodes]
            
        Returns:
            Node embeddings [num_nodes, output_dim]
        """
        x = nodes
        
        # Apply graph convolution layers
        for i, hidden_dim in enumerate(self.hidden_dims):
            weight = self.param(f'conv_weight_{i}', 
                              nn.initializers.xavier_uniform(),
                              (x.shape[-1], hidden_dim))
            bias = self.param(f'conv_bias_{i}',
                            nn.initializers.zeros,
                            (hidden_dim,))
            
            x = graph_conv_layer(x, edges, adjacency, weight, bias, self.activation)
        
        # Final output layer
        weight_out = self.param('out_weight',
                              nn.initializers.xavier_uniform(), 
                              (x.shape[-1], self.output_dim))
        bias_out = self.param('out_bias',
                            nn.initializers.zeros,
                            (self.output_dim,))
        
        output = jnp.dot(x, weight_out) + bias_out
        
        return output


class GraphAttentionNetwork(nn.Module):
    """
    Graph Attention Network with multi-head attention.
    """
    hidden_dim: int = 64
    num_heads: int = 4
    output_dim: int = 32
    dropout_rate: float = 0.1
    
    @nn.compact  
    def __call__(self, 
                 nodes: jnp.ndarray,
                 edges: jnp.ndarray,
                 adjacency: jnp.ndarray,
                 training: bool = True) -> jnp.ndarray:
        """
        Forward pass through graph attention network.
        
        Args:
            nodes: Node features [num_nodes, node_dim]
            edges: Edge indices [num_edges, 2]  
            adjacency: Adjacency matrix [num_nodes, num_nodes]
            training: Whether in training mode
            
        Returns:
            Node embeddings [num_nodes, output_dim]
        """
        num_nodes = nodes.shape[0]
        
        # Multi-head attention computation
        attention_outputs = []
        
        for head in range(self.num_heads):
            # Linear transformations for Q, K, V
            query = nn.Dense(self.hidden_dim // self.num_heads, name=f'query_{head}')(nodes)
            key = nn.Dense(self.hidden_dim // self.num_heads, name=f'key_{head}')(nodes)
            value = nn.Dense(self.hidden_dim // self.num_heads, name=f'value_{head}')(nodes)
            
            # Compute attention scores
            scores = jnp.dot(query, key.T) / jnp.sqrt(self.hidden_dim // self.num_heads)
            
            # Mask attention to only connected nodes
            mask = adjacency + jnp.eye(num_nodes)  # Include self-connections
            scores = jnp.where(mask > 0, scores, -jnp.inf)
            
            # Apply softmax to get attention weights
            attention_weights = jax.nn.softmax(scores, axis=-1)
            
            # Apply dropout during training
            if training:
                attention_weights = nn.Dropout(self.dropout_rate, deterministic=not training)(attention_weights)
            
            # Apply attention to values
            attended = jnp.dot(attention_weights, value)
            attention_outputs.append(attended)
        
        # Concatenate multi-head outputs
        multi_head_output = jnp.concatenate(attention_outputs, axis=-1)
        
        # Add residual connection and layer norm
        if multi_head_output.shape[-1] == nodes.shape[-1]:
            multi_head_output = multi_head_output + nodes
        
        multi_head_output = nn.LayerNorm()(multi_head_output)
        
        # Feed-forward network
        ff_output = nn.Dense(self.hidden_dim * 2)(multi_head_output)
        ff_output = jax.nn.relu(ff_output)
        ff_output = nn.Dense(self.hidden_dim)(ff_output)
        
        if training:
            ff_output = nn.Dropout(self.dropout_rate, deterministic=not training)(ff_output)
        
        # Another residual connection and layer norm
        if ff_output.shape[-1] == multi_head_output.shape[-1]:
            ff_output = ff_output + multi_head_output
        
        ff_output = nn.LayerNorm()(ff_output)
        
        # Final output projection
        output = nn.Dense(self.output_dim)(ff_output)
        
        return output


class GraphEncoder(nn.Module):
    """
    General graph encoder that can use different GNN architectures.
    """
    architecture: str = "gcn"  # "gcn" or "gat"
    hidden_dims: tuple = (64, 64)
    output_dim: int = 32
    num_heads: int = 4
    
    @nn.compact
    def __call__(self,
                 nodes: jnp.ndarray,
                 edges: jnp.ndarray, 
                 adjacency: jnp.ndarray,
                 training: bool = True) -> jnp.ndarray:
        """
        Encode graph into node embeddings.
        
        Args:
            nodes: Node features [num_nodes, node_dim]
            edges: Edge indices [num_edges, 2]
            adjacency: Adjacency matrix [num_nodes, num_nodes] 
            training: Whether in training mode
            
        Returns:
            Node embeddings [num_nodes, output_dim]
        """
        if self.architecture == "gcn":
            encoder = GraphConvNetwork(
                hidden_dims=self.hidden_dims,
                output_dim=self.output_dim
            )
            return encoder(nodes, edges, adjacency)
            
        elif self.architecture == "gat":
            encoder = GraphAttentionNetwork(
                hidden_dim=max(self.hidden_dims) if self.hidden_dims else 64,
                num_heads=self.num_heads,
                output_dim=self.output_dim
            )
            return encoder(nodes, edges, adjacency, training=training)
            
        else:
            raise ValueError(f"Unknown architecture: {self.architecture}")


def create_graph_encoder(architecture: str = "gcn", **kwargs) -> GraphEncoder:
    """
    Factory function to create graph encoders.
    
    Args:
        architecture: Type of GNN ("gcn" or "gat")
        **kwargs: Additional arguments for the encoder
        
    Returns:
        Configured graph encoder
    """
    return GraphEncoder(architecture=architecture, **kwargs)