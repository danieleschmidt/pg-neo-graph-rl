"""
Input validation utilities for pg-neo-graph-rl.
"""
import re
import jax.numpy as jnp
from typing import Any, Dict, List, Optional, Union
from ..core.types import GraphState
from .exceptions import ValidationError, SecurityError
from .security import SecurityValidator


def validate_graph_state(state: GraphState) -> None:
    """
    Validate GraphState object for consistency and correctness.
    
    Args:
        state: GraphState to validate
        
    Raises:
        ValidationError: If validation fails
    """
    if not isinstance(state, GraphState):
        raise ValidationError("Expected GraphState object", expected_type="GraphState")
    
    # Validate nodes
    if not isinstance(state.nodes, jnp.ndarray):
        raise ValidationError("Nodes must be JAX array", field_name="nodes")
    
    if state.nodes.ndim != 2:
        raise ValidationError(
            f"Nodes must be 2D array, got {state.nodes.ndim}D", 
            field_name="nodes"
        )
    
    num_nodes = state.nodes.shape[0]
    if num_nodes == 0:
        raise ValidationError("Graph must have at least one node", field_name="nodes")
    
    # Validate edges
    if not isinstance(state.edges, jnp.ndarray):
        raise ValidationError("Edges must be JAX array", field_name="edges")
    
    if state.edges.ndim != 2 or (state.edges.size > 0 and state.edges.shape[1] != 2):
        raise ValidationError("Edges must be [num_edges, 2] array", field_name="edges")
    
    # Check edge indices are valid
    if state.edges.size > 0:
        max_edge_idx = jnp.max(state.edges)
        if max_edge_idx >= num_nodes:
            raise ValidationError(
                f"Edge index {max_edge_idx} exceeds number of nodes {num_nodes}",
                field_name="edges"
            )
        
        min_edge_idx = jnp.min(state.edges)
        if min_edge_idx < 0:
            raise ValidationError("Edge indices must be non-negative", field_name="edges")
    
    # Validate adjacency matrix
    if not isinstance(state.adjacency, jnp.ndarray):
        raise ValidationError("Adjacency must be JAX array", field_name="adjacency")
    
    if state.adjacency.shape != (num_nodes, num_nodes):
        raise ValidationError(
            f"Adjacency matrix must be [{num_nodes}, {num_nodes}], got {state.adjacency.shape}",
            field_name="adjacency"
        )
    
    # Check adjacency is symmetric (for undirected graphs)
    if not jnp.allclose(state.adjacency, state.adjacency.T, atol=1e-6):
        raise ValidationError("Adjacency matrix must be symmetric", field_name="adjacency")
    
    # Validate edge attributes
    if state.edges.size > 0 and state.edge_attr is not None:
        if not isinstance(state.edge_attr, jnp.ndarray):
            raise ValidationError("Edge attributes must be JAX array", field_name="edge_attr")
        
        if state.edge_attr.shape[0] != state.edges.shape[0]:
            raise ValidationError(
                f"Edge attributes length {state.edge_attr.shape[0]} != edges length {state.edges.shape[0]}",
                field_name="edge_attr"
            )
    
    # Validate timestamps
    if state.timestamps is not None:
        if not isinstance(state.timestamps, jnp.ndarray):
            raise ValidationError("Timestamps must be JAX array", field_name="timestamps")
        
        if state.timestamps.shape[0] != num_nodes:
            raise ValidationError(
                f"Timestamps length {state.timestamps.shape[0]} != nodes length {num_nodes}",
                field_name="timestamps"
            )


def validate_agent_config(config: Dict[str, Any], agent_type: str = "ppo") -> None:
    """
    Validate agent configuration.
    
    Args:
        config: Configuration dictionary
        agent_type: Type of agent ("ppo" or "sac")
        
    Raises:
        ValidationError: If validation fails
    """
    if not isinstance(config, dict):
        raise ValidationError("Config must be dictionary", expected_type="dict")
    
    required_fields = ["learning_rate", "action_dim", "node_dim"]
    
    for field in required_fields:
        if field not in config:
            raise ValidationError(f"Missing required field", field_name=field)
    
    # Validate learning rate
    lr = config["learning_rate"]
    if not isinstance(lr, (int, float)) or lr <= 0:
        raise ValidationError("Learning rate must be positive number", field_name="learning_rate")
    
    if lr > 1.0:
        raise ValidationError("Learning rate seems too high (>1.0)", field_name="learning_rate")
    
    # Validate dimensions
    for dim_field in ["action_dim", "node_dim"]:
        dim = config[dim_field]
        if not isinstance(dim, int) or dim <= 0:
            raise ValidationError(f"{dim_field} must be positive integer", field_name=dim_field)
        
        if dim > 10000:
            raise ValidationError(f"{dim_field} seems too large (>{10000})", field_name=dim_field)
    
    # Agent-specific validation
    if agent_type == "ppo":
        ppo_fields = ["clip_epsilon", "gamma", "gae_lambda"]
        for field in ppo_fields:
            if field in config:
                value = config[field]
                if not isinstance(value, (int, float)) or not (0 <= value <= 1):
                    raise ValidationError(
                        f"{field} must be between 0 and 1", 
                        field_name=field
                    )
    
    elif agent_type == "sac":
        sac_fields = ["tau", "alpha", "gamma"]
        for field in sac_fields:
            if field in config:
                value = config[field]
                if not isinstance(value, (int, float)) or value < 0:
                    raise ValidationError(
                        f"{field} must be non-negative", 
                        field_name=field
                    )


def validate_environment_config(config: Dict[str, Any], env_type: str) -> None:
    """
    Validate environment configuration.
    
    Args:
        config: Configuration dictionary
        env_type: Type of environment ("traffic", "power_grid", "swarm")
        
    Raises:
        ValidationError: If validation fails
    """
    if not isinstance(config, dict):
        raise ValidationError("Config must be dictionary", expected_type="dict")
    
    # Common validation
    if env_type == "traffic":
        required_fields = ["num_intersections"]
        
        for field in required_fields:
            if field not in config:
                raise ValidationError(f"Missing required field", field_name=field)
        
        num_intersections = config["num_intersections"]
        if not isinstance(num_intersections, int) or num_intersections <= 0:
            raise ValidationError(
                "num_intersections must be positive integer", 
                field_name="num_intersections"
            )
        
        if num_intersections > 100000:
            raise ValidationError(
                "num_intersections too large (>100000)", 
                field_name="num_intersections"
            )
    
    elif env_type == "power_grid":
        required_fields = ["num_buses"]
        
        for field in required_fields:
            if field not in config:
                raise ValidationError(f"Missing required field", field_name=field)
        
        num_buses = config["num_buses"]
        if not isinstance(num_buses, int) or num_buses <= 0:
            raise ValidationError(
                "num_buses must be positive integer", 
                field_name="num_buses"
            )
    
    elif env_type == "swarm":
        required_fields = ["num_drones", "communication_range"]
        
        for field in required_fields:
            if field not in config:
                raise ValidationError(f"Missing required field", field_name=field)
        
        num_drones = config["num_drones"]
        if not isinstance(num_drones, int) or num_drones <= 0:
            raise ValidationError(
                "num_drones must be positive integer", 
                field_name="num_drones"
            )
        
        comm_range = config["communication_range"]
        if not isinstance(comm_range, (int, float)) or comm_range <= 0:
            raise ValidationError(
                "communication_range must be positive number", 
                field_name="communication_range"
            )


def validate_federated_config(config: Dict[str, Any]) -> None:
    """
    Validate federated learning configuration.
    
    Args:
        config: Configuration dictionary
        
    Raises:
        ValidationError: If validation fails
    """
    if not isinstance(config, dict):
        raise ValidationError("Config must be dictionary", expected_type="dict")
    
    required_fields = ["num_agents"]
    
    for field in required_fields:
        if field not in config:
            raise ValidationError(f"Missing required field", field_name=field)
    
    num_agents = config["num_agents"]
    if not isinstance(num_agents, int) or num_agents <= 0:
        raise ValidationError(
            "num_agents must be positive integer", 
            field_name="num_agents"
        )
    
    if num_agents > 10000:
        raise ValidationError(
            "num_agents too large (>10000)", 
            field_name="num_agents"
        )
    
    # Validate aggregation method
    if "aggregation" in config:
        valid_methods = ["gossip", "hierarchical", "ring", "fedavg"]
        if config["aggregation"] not in valid_methods:
            raise ValidationError(
                f"Invalid aggregation method. Valid: {valid_methods}",
                field_name="aggregation"
            )
    
    # Validate communication rounds
    if "communication_rounds" in config:
        comm_rounds = config["communication_rounds"]
        if not isinstance(comm_rounds, int) or comm_rounds <= 0:
            raise ValidationError(
                "communication_rounds must be positive integer",
                field_name="communication_rounds"
            )


def validate_array_shape(array: jnp.ndarray, 
                        expected_shape: Union[tuple, List[tuple]], 
                        name: str) -> None:
    """
    Validate array has expected shape(s).
    
    Args:
        array: Array to validate
        expected_shape: Expected shape or list of valid shapes
        name: Name of array for error messages
        
    Raises:
        ValidationError: If shape is invalid
    """
    if not isinstance(array, jnp.ndarray):
        raise ValidationError(f"{name} must be JAX array")
    
    if isinstance(expected_shape, tuple):
        expected_shapes = [expected_shape]
    else:
        expected_shapes = expected_shape
    
    for shape in expected_shapes:
        if len(shape) != len(array.shape):
            continue
            
        match = True
        for i, (expected, actual) in enumerate(zip(shape, array.shape)):
            if expected is not None and expected != actual:
                match = False
                break
        
        if match:
            return
    
    raise ValidationError(
        f"{name} has shape {array.shape}, expected one of {expected_shapes}",
        field_name=name
    )


def validate_numeric_range(value: Union[int, float], 
                          min_val: Optional[float] = None,
                          max_val: Optional[float] = None,
                          name: str = "value") -> None:
    """
    Validate numeric value is in expected range.
    
    Args:
        value: Value to validate
        min_val: Minimum allowed value
        max_val: Maximum allowed value  
        name: Name for error messages
        
    Raises:
        ValidationError: If value is out of range
    """
    if not isinstance(value, (int, float)):
        raise ValidationError(f"{name} must be numeric", field_name=name)
    
    if min_val is not None and value < min_val:
        raise ValidationError(
            f"{name} must be >= {min_val}, got {value}",
            field_name=name
        )
    
    if max_val is not None and value > max_val:
        raise ValidationError(
            f"{name} must be <= {max_val}, got {value}",
            field_name=name
        )


def validate_file_path(file_path: str, 
                      allowed_extensions: Optional[List[str]] = None,
                      max_length: int = 4096) -> None:
    """
    Validate file path for security and correctness.
    
    Args:
        file_path: File path to validate
        allowed_extensions: List of allowed file extensions
        max_length: Maximum path length
        
    Raises:
        ValidationError: If path is invalid
        SecurityError: If path poses security risk
    """
    if not isinstance(file_path, str):
        raise ValidationError("File path must be string", field_name="file_path")
    
    if len(file_path) > max_length:
        raise ValidationError(
            f"File path too long (>{max_length} chars)",
            field_name="file_path"
        )
    
    # Security checks
    SecurityValidator.validate_file_path(file_path)
    
    # Extension validation
    if allowed_extensions:
        extension = file_path.lower().split('.')[-1] if '.' in file_path else ''
        if extension not in [ext.lower().lstrip('.') for ext in allowed_extensions]:
            raise ValidationError(
                f"File extension '{extension}' not allowed. Allowed: {allowed_extensions}",
                field_name="file_path"
            )


def validate_string_input(text: str,
                         field_name: str = "input",
                         min_length: int = 0,
                         max_length: int = 10000,
                         allow_special_chars: bool = True,
                         pattern: Optional[str] = None) -> None:
    """
    Validate string input with security checks.
    
    Args:
        text: Text to validate
        field_name: Field name for error messages
        min_length: Minimum text length
        max_length: Maximum text length
        allow_special_chars: Whether to allow special characters
        pattern: Regex pattern that text must match
        
    Raises:
        ValidationError: If validation fails
        SecurityError: If input poses security risk
    """
    if not isinstance(text, str):
        raise ValidationError(f"{field_name} must be string", field_name=field_name)
    
    if len(text) < min_length:
        raise ValidationError(
            f"{field_name} too short (min {min_length} chars)",
            field_name=field_name
        )
    
    if len(text) > max_length:
        raise ValidationError(
            f"{field_name} too long (max {max_length} chars)",
            field_name=field_name
        )
    
    # Security validation
    SecurityValidator.validate_string_input(text, field_name)
    
    # Special character check
    if not allow_special_chars:
        if not text.replace(' ', '').replace('-', '').replace('_', '').isalnum():
            raise ValidationError(
                f"{field_name} contains disallowed special characters",
                field_name=field_name
            )
    
    # Pattern matching
    if pattern and not re.match(pattern, text):
        raise ValidationError(
            f"{field_name} does not match required pattern",
            field_name=field_name
        )