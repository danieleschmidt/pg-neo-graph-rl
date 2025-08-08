"""
Power grid environment for federated RL.
"""
import jax
import jax.numpy as jnp
import networkx as nx
from typing import Dict, Tuple, Any, Optional
from ..core.federated import GraphState


class PowerGridEnvironment:
    """Simple power grid environment."""
    
    def __init__(self, grid_file: Optional[str] = None, num_buses: int = 50):
        self.grid_file = grid_file
        self.num_buses = num_buses
        
        # Create power grid network
        self.graph = self._create_power_grid()
        self.reset()
        
    def _create_power_grid(self) -> nx.Graph:
        """Create a simple power grid network."""
        # Create a scale-free network (typical for power grids)
        G = nx.barabasi_albert_graph(self.num_buses, 3)
        return G
    
    def reset(self) -> GraphState:
        """Reset power grid to initial state."""
        num_nodes = len(self.graph.nodes())
        
        # Node features: [voltage, frequency, active_power, reactive_power, load]
        self.node_features = jnp.array([
            [1.0, 50.0, 0.8, 0.3, 0.5] for _ in range(num_nodes)
        ])
        
        # Edge features: [impedance, capacity, power_flow]
        edges = list(self.graph.edges())
        self.edges = jnp.array(edges) if edges else jnp.zeros((0, 2), dtype=int)
        self.edge_features = jnp.array([[0.1, 100.0, 0.0] for _ in range(len(edges))])
        
        # Adjacency matrix
        self.adjacency = jnp.array(nx.adjacency_matrix(self.graph).todense())
        
        self.timestep = 0
        
        return self.get_state()
    
    def get_state(self) -> GraphState:
        """Get current power grid state."""
        return GraphState(
            nodes=self.node_features,
            edges=self.edges,
            edge_attr=self.edge_features,
            adjacency=self.adjacency,
            timestamps=jnp.full(len(self.node_features), self.timestep)
        )
    
    def step(self, actions: jnp.ndarray) -> Tuple[GraphState, jnp.ndarray, bool, Dict[str, Any]]:
        """Take step with control actions (voltage/reactive power control)."""
        self.timestep += 1
        
        # Actions: [voltage_control, reactive_power_control]
        voltage_action = actions[:, 0] if actions.ndim > 1 else actions
        
        # Update voltage based on control actions
        new_voltage = jnp.clip(
            self.node_features[:, 0] + 0.05 * (voltage_action - 0.5),
            0.95, 1.05  # Voltage limits
        )
        
        # Update frequency (simple dynamics)
        frequency_deviation = jnp.random.normal(0, 0.1, size=len(new_voltage))
        new_frequency = jnp.clip(
            self.node_features[:, 1] + frequency_deviation,
            49.5, 50.5  # Frequency limits
        )
        
        # Update power flows (simplified)
        power_change = jnp.random.normal(0, 0.05, size=len(new_voltage))
        new_active_power = jnp.clip(
            self.node_features[:, 2] + power_change,
            0.0, 2.0
        )
        
        # Keep reactive power and load similar
        new_reactive_power = self.node_features[:, 3]
        new_load = self.node_features[:, 4]
        
        # Update node features
        self.node_features = jnp.column_stack([
            new_voltage, new_frequency, new_active_power, new_reactive_power, new_load
        ])
        
        # Compute rewards (stability-based)
        voltage_penalty = jnp.sum((new_voltage - 1.0) ** 2)  # Penalty for voltage deviation
        frequency_penalty = jnp.sum((new_frequency - 50.0) ** 2)  # Penalty for frequency deviation
        
        stability_reward = -(voltage_penalty + frequency_penalty)
        rewards = stability_reward * jnp.ones(len(actions))
        
        # Check for instability
        voltage_violation = jnp.any((new_voltage < 0.9) | (new_voltage > 1.1))
        frequency_violation = jnp.any((new_frequency < 49.0) | (new_frequency > 51.0))
        
        done = voltage_violation or frequency_violation or self.timestep >= 1000
        
        info = {
            "voltage_deviation": float(jnp.max(jnp.abs(new_voltage - 1.0))),
            "frequency_deviation": float(jnp.max(jnp.abs(new_frequency - 50.0))),
            "stability": float(jnp.mean((new_voltage > 0.95) & (new_voltage < 1.05))),
            "grid_stable": not (voltage_violation or frequency_violation)
        }
        
        return self.get_state(), rewards, done, info
    
    def safety_projection(self, actions: jnp.ndarray) -> jnp.ndarray:
        """Apply safety constraints to actions."""
        # Clip actions to safe ranges
        return jnp.clip(actions, -1.0, 1.0)
    
    def evaluate_global_performance(self) -> Dict[str, float]:
        """Evaluate global power grid performance."""
        voltage_stability = float(jnp.mean(
            (self.node_features[:, 0] > 0.95) & (self.node_features[:, 0] < 1.05)
        ))
        
        frequency_stability = float(jnp.mean(
            (self.node_features[:, 1] > 49.8) & (self.node_features[:, 1] < 50.2)
        ))
        
        return {
            "voltage_stability": voltage_stability,
            "frequency_stability": frequency_stability,
            "avg_voltage": float(jnp.mean(self.node_features[:, 0])),
            "avg_frequency": float(jnp.mean(self.node_features[:, 1]))
        }
