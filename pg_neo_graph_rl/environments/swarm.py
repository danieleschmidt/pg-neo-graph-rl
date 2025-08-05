"""
Swarm environment for federated RL.
"""
import jax
import jax.numpy as jnp
import networkx as nx
from typing import Dict, Tuple, Any, Optional
from ..core.federated import GraphState


class SwarmEnvironment:
    """Simple drone swarm environment."""
    
    def __init__(self, num_drones: int = 100, communication_range: float = 50.0, dynamics: str = "quadrotor"):
        self.num_drones = num_drones
        self.communication_range = communication_range
        self.dynamics = dynamics
        
        # Environment bounds
        self.bounds = jnp.array([[-500, 500], [-500, 500], [0, 100]])  # x, y, z bounds
        
        self.reset()
        
    def reset(self) -> GraphState:
        """Reset swarm to initial configuration."""
        # Initialize drone positions randomly
        key = jax.random.PRNGKey(42)
        
        # Positions: [x, y, z, vx, vy, vz]
        positions = jax.random.uniform(
            key, (self.num_drones, 3), 
            minval=jnp.array([-100, -100, 10]),
            maxval=jnp.array([100, 100, 50])
        )
        velocities = jnp.zeros((self.num_drones, 3))
        
        self.drone_states = jnp.concatenate([positions, velocities], axis=1)
        
        # Node features: [x, y, z, vx, vy, vz, battery, task_progress]
        battery_levels = jnp.ones(self.num_drones) * 100.0  # 100% battery
        task_progress = jnp.zeros(self.num_drones)
        
        self.node_features = jnp.column_stack([
            self.drone_states, battery_levels, task_progress
        ])
        
        # Build proximity graph
        self.adjacency = self._build_proximity_graph(positions)
        self.edges, self.edge_features = self._get_edges_from_adjacency()
        
        self.timestep = 0
        
        return self.get_state()
    
    def _build_proximity_graph(self, positions: jnp.ndarray) -> jnp.ndarray:
        """Build communication graph based on proximity."""
        # Compute pairwise distances
        diff = positions[:, None, :] - positions[None, :, :]
        distances = jnp.linalg.norm(diff, axis=2)
        
        # Create adjacency matrix based on communication range
        adjacency = (distances <= self.communication_range).astype(float)
        # Remove self-connections
        adjacency = adjacency.at[jnp.diag_indices(len(positions))].set(0.0)
        
        return adjacency
    
    def _get_edges_from_adjacency(self) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """Extract edge list and features from adjacency matrix."""
        edges_list = []
        edge_features_list = []
        
        for i in range(self.num_drones):
            for j in range(i + 1, self.num_drones):
                if self.adjacency[i, j] > 0:
                    edges_list.append([i, j])
                    # Edge features: [distance, signal_strength, bandwidth]
                    distance = jnp.linalg.norm(self.drone_states[i, :3] - self.drone_states[j, :3])
                    signal_strength = 1.0 - (distance / self.communication_range)
                    edge_features_list.append([distance, signal_strength, 1.0])
        
        if edges_list:
            edges = jnp.array(edges_list)
            edge_features = jnp.array(edge_features_list)
        else:
            edges = jnp.zeros((0, 2), dtype=int)
            edge_features = jnp.zeros((0, 3))
            
        return edges, edge_features
    
    def get_state(self) -> GraphState:
        """Get current swarm state."""
        return GraphState(
            nodes=self.node_features,
            edges=self.edges,
            edge_attr=self.edge_features,
            adjacency=self.adjacency,
            timestamps=jnp.full(self.num_drones, self.timestep)
        )
    
    def step(self, actions: jnp.ndarray) -> Tuple[GraphState, jnp.ndarray, bool, Dict[str, Any]]:
        """Take step with drone control actions."""
        self.timestep += 1
        
        # Actions: [thrust_x, thrust_y, thrust_z] for each drone
        if actions.ndim == 1:
            actions = actions.reshape(self.num_drones, -1)
        
        # Simple quadrotor dynamics
        dt = 0.1  # Time step
        
        # Update velocities
        new_velocities = self.drone_states[:, 3:6] + actions * dt
        
        # Apply drag
        drag_coeff = 0.1
        new_velocities = new_velocities * (1 - drag_coeff * dt)
        
        # Update positions
        new_positions = self.drone_states[:, :3] + new_velocities * dt
        
        # Apply boundary constraints
        new_positions = jnp.clip(new_positions, self.bounds[:, 0], self.bounds[:, 1])
        
        # Update drone states
        self.drone_states = jnp.concatenate([new_positions, new_velocities], axis=1)
        
        # Update battery (decreases with action magnitude)
        action_magnitude = jnp.linalg.norm(actions, axis=1)
        battery_drain = 0.1 + 0.05 * action_magnitude
        new_battery = jnp.maximum(0.0, self.node_features[:, 6] - battery_drain)
        
        # Update task progress (dummy: increases with time)
        new_task_progress = jnp.minimum(100.0, self.node_features[:, 7] + 0.5)
        
        # Update node features
        self.node_features = jnp.column_stack([
            self.drone_states, new_battery, new_task_progress
        ])
        
        # Rebuild proximity graph
        self.adjacency = self._build_proximity_graph(new_positions)
        self.edges, self.edge_features = self._get_edges_from_adjacency()
        
        # Compute rewards
        # Reward for maintaining formation and connectivity
        connectivity_reward = jnp.sum(self.adjacency) / (self.num_drones * (self.num_drones - 1))
        
        # Penalty for collisions (drones too close)
        min_distance = 5.0
        distances = jnp.linalg.norm(
            new_positions[:, None, :] - new_positions[None, :, :], axis=2
        )
        collision_penalty = jnp.sum((distances < min_distance) & (distances > 0))
        
        # Task completion reward
        task_reward = jnp.mean(new_task_progress) / 100.0
        
        total_reward = connectivity_reward - 0.1 * collision_penalty + task_reward
        rewards = total_reward * jnp.ones(self.num_drones)
        
        # Episode termination
        done = (jnp.mean(new_battery) < 10.0) or (self.timestep >= 1000)
        
        info = {
            "connectivity": float(connectivity_reward),
            "avg_battery": float(jnp.mean(new_battery)),
            "avg_task_progress": float(jnp.mean(new_task_progress)),
            "num_collisions": int(collision_penalty),
            "formation_quality": float(1.0 - jnp.std(distances[distances > 0]) / jnp.mean(distances[distances > 0]))
        }
        
        return self.get_state(), rewards, done, info
    
    def build_proximity_graph(self, positions: jnp.ndarray, radius: float) -> jnp.ndarray:
        """Build proximity graph for external use."""
        return self._build_proximity_graph(positions)
    
    def evaluate_global_performance(self) -> Dict[str, float]:
        """Evaluate global swarm performance."""
        positions = self.drone_states[:, :3]
        
        # Compute coverage area (convex hull approximation)
        coverage = float(jnp.var(positions[:, 0]) + jnp.var(positions[:, 1]))
        
        # Connectivity measure
        connectivity = float(jnp.sum(self.adjacency) / (self.num_drones * (self.num_drones - 1)))
        
        return {
            "coverage_area": coverage,
            "connectivity": connectivity,
            "avg_battery": float(jnp.mean(self.node_features[:, 6])),
            "task_completion": float(jnp.mean(self.node_features[:, 7]))
        }
