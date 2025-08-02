"""Test data fixtures and datasets."""

import numpy as np
import jax.numpy as jnp
from pathlib import Path
from typing import Dict, Any, List, Tuple
import json


class TestDatasets:
    """Collection of test datasets for various scenarios."""
    
    @staticmethod
    def manhattan_traffic_small() -> Dict[str, Any]:
        """Small version of Manhattan traffic network for testing."""
        # 5x5 grid representing a small Manhattan section
        grid_size = 5
        num_nodes = grid_size * grid_size
        
        # Create grid connections
        edges = []
        for i in range(grid_size):
            for j in range(grid_size):
                node_id = i * grid_size + j
                
                # East-West streets
                if j < grid_size - 1:
                    edges.append([node_id, node_id + 1])
                    edges.append([node_id + 1, node_id])
                
                # North-South avenues
                if i < grid_size - 1:
                    edges.append([node_id, node_id + grid_size])
                    edges.append([node_id + grid_size, node_id])
        
        edge_index = jnp.array(np.array(edges).T)
        
        # Traffic features: [flow, density, speed, congestion, signal_phase, queue_length]
        node_features = jnp.array([
            [0.5, 0.4, 0.8, 0.3, 1.0, 0.2],  # Low congestion intersection
            [0.8, 0.7, 0.4, 0.8, 0.0, 0.9],  # High congestion intersection
            [0.3, 0.2, 0.9, 0.1, 1.0, 0.1],  # Free flow intersection
            [0.6, 0.5, 0.6, 0.5, 0.5, 0.4],  # Moderate traffic
            [0.9, 0.8, 0.3, 0.9, 0.0, 1.0],  # Heavy congestion
        ] * 5)[:num_nodes]  # Repeat pattern and trim to size
        
        # Edge features: [flow_rate, capacity_utilization, travel_time]
        edge_features = jnp.array(np.random.uniform(0.1, 0.9, (len(edges), 3)))
        
        return {
            "node_features": node_features,
            "edge_index": edge_index,
            "edge_features": edge_features,
            "num_nodes": num_nodes,
            "num_edges": len(edges),
            "scenario": "manhattan_traffic",
            "topology": "grid",
            "real_world": True
        }
    
    @staticmethod
    def texas_power_grid_small() -> Dict[str, Any]:
        """Small version of Texas power grid for testing."""
        # Simplified power grid with generators, transmission lines, and loads
        generators = [0, 1, 2]  # Generator buses
        loads = [3, 4, 5, 6, 7]  # Load buses
        transmission = [8, 9]  # Transmission substations
        
        num_nodes = len(generators) + len(loads) + len(transmission)
        
        # Power grid connections
        edges = [
            # Generators to transmission
            [0, 8], [1, 8], [2, 9],
            # Transmission to loads
            [8, 3], [8, 4], [8, 5],
            [9, 5], [9, 6], [9, 7],
            # Inter-transmission
            [8, 9],
            # Some load interconnections
            [3, 4], [6, 7]
        ]
        
        # Make bidirectional
        bidirectional_edges = edges + [[b, a] for a, b in edges]
        edge_index = jnp.array(np.array(bidirectional_edges).T)
        
        # Power grid features: [voltage_magnitude, voltage_angle, active_power, reactive_power]
        node_features = jnp.array([
            # Generators (higher voltage, power generation)
            [1.05, 0.0, 0.8, 0.2],
            [1.04, -0.1, 0.9, 0.1],
            [1.06, 0.05, 0.7, 0.3],
            # Loads (lower voltage, power consumption)
            [0.98, -0.2, -0.3, -0.1],
            [0.97, -0.25, -0.4, -0.15],
            [0.99, -0.15, -0.2, -0.05],
            [0.96, -0.3, -0.5, -0.2],
            [0.98, -0.22, -0.35, -0.12],
            # Transmission (intermediate voltage)
            [1.02, -0.1, 0.0, 0.0],
            [1.01, -0.12, 0.0, 0.0]
        ])
        
        # Edge features: [power_flow, capacity_utilization]
        edge_features = jnp.array(np.random.uniform(0.1, 0.8, (len(bidirectional_edges), 2)))
        
        return {
            "node_features": node_features,
            "edge_index": edge_index,
            "edge_features": edge_features,
            "num_nodes": num_nodes,
            "num_edges": len(bidirectional_edges),
            "scenario": "texas_power_grid",
            "topology": "hub",
            "real_world": True,
            "node_types": {
                "generators": generators,
                "loads": loads,
                "transmission": transmission
            }
        }
    
    @staticmethod
    def drone_swarm_formation() -> Dict[str, Any]:
        """Drone swarm in formation for testing."""
        num_drones = 12
        
        # Initial positions in a diamond formation
        positions = np.array([
            [0, 0], [1, 1], [-1, 1], [1, -1], [-1, -1],  # Center and corners
            [2, 0], [-2, 0], [0, 2], [0, -2],  # Mid edges
            [2, 2], [-2, 2], [2, -2]  # Outer corners (partial)
        ])
        
        # Communication range
        comm_range = 2.5
        edges = []
        
        for i in range(num_drones):
            for j in range(i + 1, num_drones):
                distance = np.linalg.norm(positions[i] - positions[j])
                if distance <= comm_range:
                    edges.append([i, j])
                    edges.append([j, i])
        
        edge_index = jnp.array(np.array(edges).T)
        
        # Drone features: [x_pos, y_pos, x_vel, y_vel, battery, status]
        velocities = np.random.uniform(-0.5, 0.5, (num_drones, 2))
        batteries = np.random.uniform(0.7, 1.0, (num_drones, 1))
        statuses = np.ones((num_drones, 1))  # All operational
        
        node_features = jnp.array(np.concatenate([
            positions, velocities, batteries, statuses
        ], axis=1))
        
        # Edge features: [communication_strength]
        edge_features = jnp.array(np.random.uniform(0.5, 1.0, (len(edges), 1)))
        
        return {
            "node_features": node_features,
            "edge_index": edge_index,
            "edge_features": edge_features,
            "num_nodes": num_drones,
            "num_edges": len(edges),
            "scenario": "drone_swarm",
            "topology": "proximity",
            "real_world": True,
            "positions": positions,
            "formation": "diamond"
        }
    
    @staticmethod
    def water_distribution_network() -> Dict[str, Any]:
        """Water distribution network for testing."""
        # Water network components
        reservoirs = [0]  # Water source
        pumps = [1, 2]  # Pumping stations
        junctions = [3, 4, 5, 6, 7, 8]  # Network junctions
        consumers = [9, 10, 11, 12]  # End consumers
        
        num_nodes = len(reservoirs) + len(pumps) + len(junctions) + len(consumers)
        
        # Network connections
        edges = [
            # Reservoir to pumps
            [0, 1], [0, 2],
            # Pumps to distribution network
            [1, 3], [1, 4], [2, 5], [2, 6],
            # Junction interconnections
            [3, 4], [4, 7], [5, 6], [6, 8], [7, 8],
            # Junctions to consumers
            [7, 9], [7, 10], [8, 11], [8, 12],
            # Some backup connections
            [3, 5], [4, 6]
        ]
        
        edge_index = jnp.array(np.array(edges).T)
        
        # Water network features: [pressure, flow_rate, demand, water_quality]
        node_features = jnp.array([
            # Reservoir (high pressure, source)
            [100.0, 0.0, 0.0, 0.95],
            # Pumps (boost pressure)
            [80.0, 50.0, 0.0, 0.93],
            [85.0, 45.0, 0.0, 0.94],
            # Junctions (intermediate pressure)
            [70.0, 35.0, 10.0, 0.92],
            [68.0, 30.0, 15.0, 0.91],
            [75.0, 40.0, 12.0, 0.93],
            [72.0, 38.0, 8.0, 0.92],
            [65.0, 25.0, 20.0, 0.90],
            [67.0, 28.0, 18.0, 0.91],
            # Consumers (low pressure, high demand)
            [60.0, 15.0, 25.0, 0.89],
            [58.0, 12.0, 30.0, 0.88],
            [62.0, 18.0, 22.0, 0.90],
            [59.0, 14.0, 28.0, 0.87]
        ])
        
        # Edge features: [pipe_diameter, flow_capacity, age]
        edge_features = jnp.array(np.random.uniform(0.3, 1.0, (len(edges), 3)))
        
        return {
            "node_features": node_features,
            "edge_index": edge_index,
            "edge_features": edge_features,
            "num_nodes": num_nodes,
            "num_edges": len(edges),
            "scenario": "water_distribution",
            "topology": "tree_with_loops",
            "real_world": True,
            "node_types": {
                "reservoirs": reservoirs,
                "pumps": pumps,
                "junctions": junctions,
                "consumers": consumers
            }
        }


class TestScenarios:
    """Complete test scenarios with multiple components."""
    
    @staticmethod
    def basic_federated_learning() -> Dict[str, Any]:
        """Basic federated learning scenario."""
        return {
            "num_agents": 4,
            "graph": TestDatasets.manhattan_traffic_small(),
            "config": {
                "communication_rounds": 5,
                "local_epochs": 3,
                "learning_rate": 0.01,
                "batch_size": 32,
                "aggregation_method": "fedavg"
            },
            "expected_metrics": {
                "convergence_rounds": 5,
                "final_accuracy": 0.8,
                "communication_cost": 1000
            }
        }
    
    @staticmethod
    def privacy_preserving_scenario() -> Dict[str, Any]:
        """Privacy-preserving federated learning scenario."""
        scenario = TestScenarios.basic_federated_learning()
        scenario["config"].update({
            "differential_privacy": True,
            "epsilon": 1.0,
            "delta": 1e-5,
            "noise_multiplier": 1.1,
            "secure_aggregation": True
        })
        # Lower expected accuracy due to privacy noise
        scenario["expected_metrics"]["final_accuracy"] = 0.7
        return scenario
    
    @staticmethod
    def multi_environment_scenario() -> Dict[str, Any]:
        """Scenario with multiple environment types."""
        return {
            "environments": {
                "traffic": TestDatasets.manhattan_traffic_small(),
                "power_grid": TestDatasets.texas_power_grid_small(),
                "swarm": TestDatasets.drone_swarm_formation(),
                "water": TestDatasets.water_distribution_network()
            },
            "agents_per_env": 2,
            "cross_domain_learning": True,
            "transfer_learning": True
        }
    
    @staticmethod
    def benchmark_scenario() -> Dict[str, Any]:
        """Comprehensive benchmarking scenario."""
        return {
            "algorithms": ["GraphPPO", "GraphSAC", "RandomPolicy"],
            "environments": ["traffic", "power_grid", "swarm"],
            "metrics": [
                "episode_reward", "convergence_time", "sample_efficiency",
                "communication_cost", "computational_cost", "scalability"
            ],
            "num_runs": 5,
            "max_episodes": 100,
            "statistical_tests": ["t_test", "mann_whitney", "friedman"]
        }


def get_test_dataset(name: str) -> Dict[str, Any]:
    """Get a test dataset by name."""
    datasets = {
        "manhattan_traffic": TestDatasets.manhattan_traffic_small,
        "texas_power_grid": TestDatasets.texas_power_grid_small,
        "drone_swarm": TestDatasets.drone_swarm_formation,
        "water_distribution": TestDatasets.water_distribution_network
    }
    
    if name not in datasets:
        raise ValueError(f"Unknown dataset: {name}. Available: {list(datasets.keys())}")
    
    return datasets[name]()


def get_test_scenario(name: str) -> Dict[str, Any]:
    """Get a test scenario by name."""
    scenarios = {
        "basic_federated": TestScenarios.basic_federated_learning,
        "privacy_preserving": TestScenarios.privacy_preserving_scenario,
        "multi_environment": TestScenarios.multi_environment_scenario,
        "benchmark": TestScenarios.benchmark_scenario
    }
    
    if name not in scenarios:
        raise ValueError(f"Unknown scenario: {name}. Available: {list(scenarios.keys())}")
    
    return scenarios[name]()


# Export commonly used datasets
MANHATTAN_TRAFFIC = TestDatasets.manhattan_traffic_small()
TEXAS_POWER_GRID = TestDatasets.texas_power_grid_small()
DRONE_SWARM = TestDatasets.drone_swarm_formation()
WATER_DISTRIBUTION = TestDatasets.water_distribution_network()

# Export common scenarios
BASIC_FEDERATED = TestScenarios.basic_federated_learning()
PRIVACY_SCENARIO = TestScenarios.privacy_preserving_scenario()
MULTI_ENV_SCENARIO = TestScenarios.multi_environment_scenario()
BENCHMARK_SCENARIO = TestScenarios.benchmark_scenario()
