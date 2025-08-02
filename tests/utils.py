"""Testing utilities and helper functions."""

import numpy as np
import jax.numpy as jnp
from typing import Dict, Any, List, Optional, Tuple, Union
import networkx as nx
from pathlib import Path
import json
import yaml
import tempfile
import shutil


class GraphTestUtils:
    """Utilities for testing graph-related functionality."""
    
    @staticmethod
    def create_random_graph(
        num_nodes: int, 
        num_edges: int, 
        node_features: int = 4, 
        edge_features: int = 2,
        seed: int = 42
    ) -> Dict[str, Any]:
        """Create a random graph for testing."""
        np.random.seed(seed)
        
        # Generate random edges
        edges = []
        for _ in range(num_edges):
            source = np.random.randint(0, num_nodes)
            target = np.random.randint(0, num_nodes)
            if source != target:  # No self-loops
                edges.append([source, target])
        
        edges = np.array(edges)
        edge_index = jnp.array(edges.T)
        
        # Generate random features
        node_features_array = jnp.array(np.random.randn(num_nodes, node_features))
        edge_features_array = jnp.array(np.random.randn(len(edges), edge_features))
        
        return {
            "node_features": node_features_array,
            "edge_index": edge_index,
            "edge_features": edge_features_array,
            "num_nodes": num_nodes,
            "num_edges": len(edges)
        }
    
    @staticmethod
    def create_grid_graph(grid_size: int, bidirectional: bool = True) -> Dict[str, Any]:
        """Create a grid graph for testing."""
        num_nodes = grid_size * grid_size
        edges = []
        
        for i in range(grid_size):
            for j in range(grid_size):
                node_id = i * grid_size + j
                
                # Right neighbor
                if j < grid_size - 1:
                    edges.append([node_id, node_id + 1])
                    if bidirectional:
                        edges.append([node_id + 1, node_id])
                
                # Bottom neighbor
                if i < grid_size - 1:
                    edges.append([node_id, node_id + grid_size])
                    if bidirectional:
                        edges.append([node_id + grid_size, node_id])
        
        edge_index = jnp.array(np.array(edges).T)
        node_features = jnp.array(np.ones((num_nodes, 2)))  # Simple features
        edge_features = jnp.array(np.ones((len(edges), 1)))
        
        return {
            "node_features": node_features,
            "edge_index": edge_index,
            "edge_features": edge_features,
            "num_nodes": num_nodes,
            "num_edges": len(edges),
            "topology": "grid"
        }
    
    @staticmethod
    def create_star_graph(num_nodes: int) -> Dict[str, Any]:
        """Create a star graph (one central node connected to all others)."""
        edges = []
        
        # Connect center (node 0) to all other nodes
        for i in range(1, num_nodes):
            edges.append([0, i])
            edges.append([i, 0])  # Bidirectional
        
        edge_index = jnp.array(np.array(edges).T)
        node_features = jnp.array(np.ones((num_nodes, 2)))
        edge_features = jnp.array(np.ones((len(edges), 1)))
        
        return {
            "node_features": node_features,
            "edge_index": edge_index,
            "edge_features": edge_features,
            "num_nodes": num_nodes,
            "num_edges": len(edges),
            "topology": "star"
        }
    
    @staticmethod
    def validate_graph_structure(graph_data: Dict[str, Any]) -> bool:
        """Validate that graph data has the correct structure."""
        required_keys = ["node_features", "edge_index", "edge_features", "num_nodes", "num_edges"]
        
        # Check required keys
        for key in required_keys:
            if key not in graph_data:
                return False
        
        # Check dimensions
        node_features = graph_data["node_features"]
        edge_index = graph_data["edge_index"]
        edge_features = graph_data["edge_features"]
        
        if node_features.shape[0] != graph_data["num_nodes"]:
            return False
        
        if edge_index.shape[0] != 2:
            return False
        
        if edge_features.shape[0] != graph_data["num_edges"]:
            return False
        
        # Check edge indices are valid
        max_node_id = jnp.max(edge_index)
        if max_node_id >= graph_data["num_nodes"]:
            return False
        
        return True


class FederatedTestUtils:
    """Utilities for testing federated learning functionality."""
    
    @staticmethod
    def create_mock_agents(
        num_agents: int, 
        model_params: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """Create mock agents for federated learning tests."""
        if model_params is None:
            model_params = {
                "weights": np.random.randn(10, 5),
                "bias": np.random.randn(5)
            }
        
        agents = []
        for i in range(num_agents):
            agent = {
                "agent_id": i,
                "model_params": {
                    key: value + np.random.randn(*value.shape) * 0.1
                    for key, value in model_params.items()
                },
                "local_data_size": np.random.randint(50, 200),
                "training_rounds": 0
            }
            agents.append(agent)
        
        return agents
    
    @staticmethod
    def simulate_federated_round(
        agents: List[Dict[str, Any]], 
        aggregation_method: str = "fedavg"
    ) -> Dict[str, Any]:
        """Simulate a federated learning round."""
        if aggregation_method == "fedavg":
            return FederatedTestUtils._federated_averaging(agents)
        else:
            raise NotImplementedError(f"Aggregation method {aggregation_method} not implemented")
    
    @staticmethod
    def _federated_averaging(agents: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Perform federated averaging."""
        total_data_size = sum(agent["local_data_size"] for agent in agents)
        
        # Get parameter names from first agent
        param_names = list(agents[0]["model_params"].keys())
        
        # Weighted average of parameters
        aggregated_params = {}
        for param_name in param_names:
            weighted_sum = np.zeros_like(agents[0]["model_params"][param_name])
            
            for agent in agents:
                weight = agent["local_data_size"] / total_data_size
                weighted_sum += weight * agent["model_params"][param_name]
            
            aggregated_params[param_name] = weighted_sum
        
        return aggregated_params


class EnvironmentTestUtils:
    """Utilities for testing environment functionality."""
    
    @staticmethod
    def create_mock_trajectory(
        num_steps: int, 
        observation_dim: int, 
        action_dim: int,
        reward_range: Tuple[float, float] = (-1.0, 1.0)
    ) -> Dict[str, Any]:
        """Create a mock RL trajectory for testing."""
        observations = np.random.randn(num_steps, observation_dim)
        actions = np.random.randn(num_steps, action_dim)
        rewards = np.random.uniform(*reward_range, size=num_steps)
        dones = np.zeros(num_steps, dtype=bool)
        dones[-1] = True  # Terminal state
        
        return {
            "observations": observations,
            "actions": actions,
            "rewards": rewards,
            "dones": dones,
            "episode_length": num_steps
        }
    
    @staticmethod
    def validate_trajectory(trajectory: Dict[str, Any]) -> bool:
        """Validate that a trajectory has the correct structure."""
        required_keys = ["observations", "actions", "rewards", "dones"]
        
        for key in required_keys:
            if key not in trajectory:
                return False
        
        # Check that all arrays have the same length
        lengths = [len(trajectory[key]) for key in required_keys]
        if len(set(lengths)) != 1:
            return False
        
        # Check that dones is boolean
        if trajectory["dones"].dtype != bool:
            return False
        
        return True


class ConfigTestUtils:
    """Utilities for testing configuration management."""
    
    @staticmethod
    def create_temp_config(
        config_data: Dict[str, Any], 
        format: str = "yaml",
        temp_dir: Optional[Path] = None
    ) -> Path:
        """Create a temporary configuration file."""
        if temp_dir is None:
            temp_dir = Path(tempfile.mkdtemp())
        
        if format == "yaml":
            config_file = temp_dir / "config.yaml"
            with open(config_file, 'w') as f:
                yaml.dump(config_data, f, default_flow_style=False)
        elif format == "json":
            config_file = temp_dir / "config.json"
            with open(config_file, 'w') as f:
                json.dump(config_data, f, indent=2)
        else:
            raise ValueError(f"Unsupported format: {format}")
        
        return config_file
    
    @staticmethod
    def cleanup_temp_files(*file_paths: Path):
        """Clean up temporary files and directories."""
        for file_path in file_paths:
            if file_path.is_file():
                file_path.unlink()
            elif file_path.is_dir():
                shutil.rmtree(file_path)


class MetricsTestUtils:
    """Utilities for testing metrics and monitoring functionality."""
    
    @staticmethod
    def create_mock_metrics(
        num_episodes: int = 10,
        metrics_names: Optional[List[str]] = None
    ) -> Dict[str, List[float]]:
        """Create mock training metrics."""
        if metrics_names is None:
            metrics_names = ["reward", "loss", "accuracy", "episode_length"]
        
        metrics = {}
        for name in metrics_names:
            if name == "reward":
                # Gradually improving rewards
                values = [np.random.normal(i * 0.1, 0.2) for i in range(num_episodes)]
            elif name == "loss":
                # Gradually decreasing loss
                values = [np.random.normal(1.0 - i * 0.05, 0.1) for i in range(num_episodes)]
            elif name == "accuracy":
                # Gradually improving accuracy
                base_acc = 0.5
                values = [min(0.99, base_acc + i * 0.03 + np.random.normal(0, 0.02)) 
                         for i in range(num_episodes)]
            else:
                # Random values for other metrics
                values = [np.random.normal(0, 1) for _ in range(num_episodes)]
            
            metrics[name] = values
        
        return metrics
    
    @staticmethod
    def assert_metrics_improvement(
        metrics: Dict[str, List[float]], 
        metric_name: str,
        improvement_threshold: float = 0.1
    ) -> bool:
        """Assert that a metric shows improvement over time."""
        if metric_name not in metrics:
            return False
        
        values = metrics[metric_name]
        if len(values) < 2:
            return False
        
        # Check if final value is better than initial value
        improvement = values[-1] - values[0]
        return improvement >= improvement_threshold


class PerformanceTestUtils:
    """Utilities for performance testing."""
    
    @staticmethod
    def measure_execution_time(func, *args, **kwargs) -> Tuple[Any, float]:
        """Measure execution time of a function."""
        import time
        
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        
        execution_time = end_time - start_time
        return result, execution_time
    
    @staticmethod
    def measure_memory_usage(func, *args, **kwargs) -> Tuple[Any, int]:
        """Measure memory usage of a function."""
        import tracemalloc
        
        tracemalloc.start()
        result = func(*args, **kwargs)
        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        
        return result, peak
    
    @staticmethod
    def assert_performance_bounds(
        execution_time: float,
        max_time: float,
        memory_usage: int,
        max_memory: int
    ) -> bool:
        """Assert that performance is within acceptable bounds."""
        time_ok = execution_time <= max_time
        memory_ok = memory_usage <= max_memory
        
        if not time_ok:
            print(f"Execution time {execution_time:.3f}s exceeds maximum {max_time:.3f}s")
        
        if not memory_ok:
            print(f"Memory usage {memory_usage} bytes exceeds maximum {max_memory} bytes")
        
        return time_ok and memory_ok


# Convenience functions for common test scenarios
def create_simple_test_graph() -> Dict[str, Any]:
    """Create a simple graph for basic testing."""
    return GraphTestUtils.create_random_graph(
        num_nodes=10, 
        num_edges=15, 
        node_features=4, 
        edge_features=2
    )


def create_federated_test_setup(num_agents: int = 4) -> Dict[str, Any]:
    """Create a complete federated learning test setup."""
    agents = FederatedTestUtils.create_mock_agents(num_agents)
    graph = create_simple_test_graph()
    
    return {
        "agents": agents,
        "graph": graph,
        "config": {
            "num_agents": num_agents,
            "communication_rounds": 5,
            "aggregation_method": "fedavg"
        }
    }


def assert_graph_valid(graph_data: Dict[str, Any], test_name: str = ""):
    """Assert that graph data is valid."""
    assert GraphTestUtils.validate_graph_structure(graph_data), \
        f"Invalid graph structure in test: {test_name}"


def assert_trajectory_valid(trajectory: Dict[str, Any], test_name: str = ""):
    """Assert that trajectory data is valid."""
    assert EnvironmentTestUtils.validate_trajectory(trajectory), \
        f"Invalid trajectory structure in test: {test_name}"
