"""
Core federated graph reinforcement learning orchestration.
"""
import time
from typing import Any, Dict, List, Optional

import jax
import jax.numpy as jnp
import networkx as nx
from pydantic import BaseModel, Field

from ..utils.exceptions import (
    FederatedLearningError,
    ValidationError,
)
from ..utils.health import FederatedHealthMonitor
from ..utils.logging import get_logger, get_performance_logger, log_exception
from ..utils.security import SecurityAudit, check_gradient_norm, validate_input_data
from ..utils.validation import (
    validate_federated_config,
    validate_graph_state,
    validate_numeric_range,
)
from .types import GraphState


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
                 topology: str = "random",
                 enable_monitoring: bool = True,
                 enable_security: bool = True):
        """
        Initialize federated learning system.
        
        Args:
            num_agents: Number of distributed agents
            aggregation: Aggregation method ("gossip", "hierarchical", "ring")
            communication_rounds: Communication rounds per training step
            privacy_noise: Differential privacy noise level
            topology: Communication graph topology
            enable_monitoring: Whether to enable health monitoring
            enable_security: Whether to enable security features
        """
        # Validate configuration
        config_dict = {
            "num_agents": num_agents,
            "aggregation": aggregation,
            "communication_rounds": communication_rounds,
            "privacy_noise": privacy_noise,
            "topology": topology
        }

        try:
            validate_federated_config(config_dict)
            validate_numeric_range(privacy_noise, min_val=0.0, max_val=10.0, name="privacy_noise")
        except ValidationError as e:
            raise FederatedLearningError(f"Invalid configuration: {str(e)}")

        self.config = FederatedConfig(
            num_agents=num_agents,
            aggregation=aggregation,
            communication_rounds=communication_rounds,
            privacy_noise=privacy_noise,
            topology=topology
        )

        # Initialize logging and monitoring
        self.logger = get_logger(f"pg_neo_graph_rl.federated_{id(self)}")
        self.perf_logger = get_performance_logger()

        # Initialize security and health monitoring
        self.security_audit = SecurityAudit() if enable_security else None
        self.health_monitor = FederatedHealthMonitor() if enable_monitoring else None
        self.enable_security = enable_security

        self.agents = []
        self.global_step = 0
        self.failed_agents = set()  # Track failed agents
        self.round_metrics = []  # Track round performance

        # Initialize random key for JAX with security if enabled
        if enable_security:
            from ..utils.security import secure_random_key
            self.rng_key = secure_random_key()
            if self.security_audit:
                self.security_audit.log_security_event(
                    "initialization",
                    "Secure random key initialized",
                    "INFO"
                )
        else:
            self.rng_key = jax.random.PRNGKey(42)

        # Build communication graph after initializing rng_key
        try:
            with self.perf_logger.timer_context("communication_graph_build"):
                self.communication_graph = self._build_communication_graph()
        except Exception as e:
            error_msg = f"Failed to build communication graph: {str(e)}"
            log_exception(self.logger, e, {"operation": "graph_building"})
            raise FederatedLearningError(error_msg)

        self.logger.info(f"Federated system initialized with {num_agents} agents")

        if self.security_audit:
            self.security_audit.log_security_event(
                "system_initialization",
                f"Federated system initialized with {num_agents} agents",
                "INFO",
                {"topology": topology, "aggregation": aggregation}
            )

    def _build_communication_graph(self) -> nx.Graph:
        """Build communication graph between agents."""
        G = nx.Graph()
        G.add_nodes_from(range(self.config.num_agents))

        if self.config.topology == "random":
            # Random graph with average degree 4
            key = self.rng_key
            for i in range(self.config.num_agents):
                for j in range(i + 1, self.config.num_agents):
                    key, subkey = jax.random.split(key)
                    if jax.random.uniform(subkey) < 0.3:
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
        Extract subgraph for a specific agent with robust error handling.
        
        For simplicity, this assigns nodes to agents in round-robin fashion.
        In practice, this would be based on geographical or logical partitioning.
        """
        try:
            # Validate inputs
            if agent_id < 0 or agent_id >= self.config.num_agents:
                raise ValidationError(f"Invalid agent_id {agent_id}, must be in [0, {self.config.num_agents})")

            validate_graph_state(graph_state)

            if self.enable_security:
                validate_input_data(graph_state.nodes)
                validate_input_data(graph_state.edges)

            with self.perf_logger.timer_context(f"subgraph_extraction_agent_{agent_id}"):
                num_nodes = graph_state.nodes.shape[0]

                # Check if we have enough nodes for all agents
                if num_nodes < self.config.num_agents:
                    self.logger.warning(
                        f"Fewer nodes ({num_nodes}) than agents ({self.config.num_agents})"
                    )

                agent_nodes = jnp.arange(agent_id, num_nodes, self.config.num_agents)

                if len(agent_nodes) == 0:
                    # Create minimal subgraph if no nodes assigned
                    self.logger.warning(f"No nodes assigned to agent {agent_id}")
                    return GraphState(
                        nodes=jnp.zeros((1, graph_state.nodes.shape[1])),
                        edges=jnp.zeros((0, 2), dtype=int),
                        edge_attr=jnp.zeros((0, graph_state.edge_attr.shape[1])) if graph_state.edge_attr.size > 0 else jnp.zeros((0, 1)),
                        adjacency=jnp.zeros((1, 1)),
                        timestamps=None
                    )

                # Extract subgraph nodes
                sub_nodes = graph_state.nodes[agent_nodes]

                # Find edges within this subgraph
                if graph_state.edges.size > 0:
                    edge_mask = jnp.isin(graph_state.edges[:, 0], agent_nodes) & \
                               jnp.isin(graph_state.edges[:, 1], agent_nodes)
                    sub_edges = graph_state.edges[edge_mask]
                    sub_edge_attr = graph_state.edge_attr[edge_mask] if graph_state.edge_attr.size > 0 else jnp.zeros((len(sub_edges), 1))
                else:
                    sub_edges = jnp.zeros((0, 2), dtype=int)
                    sub_edge_attr = jnp.zeros((0, graph_state.edge_attr.shape[1])) if graph_state.edge_attr.size > 0 else jnp.zeros((0, 1))

                # Remap edge indices to local subgraph
                agent_nodes_list = agent_nodes.tolist()
                node_mapping = {old_idx: new_idx for new_idx, old_idx in enumerate(agent_nodes_list)}

                if len(sub_edges) > 0:
                    try:
                        remapped_edges = jnp.array([[node_mapping.get(int(e[0]), 0), node_mapping.get(int(e[1]), 0)]
                                                   for e in sub_edges])
                    except (KeyError, ValueError) as e:
                        self.logger.error(f"Edge remapping failed for agent {agent_id}: {e}")
                        remapped_edges = jnp.zeros((0, 2), dtype=int)
                        sub_edge_attr = jnp.zeros((0, sub_edge_attr.shape[1])) if sub_edge_attr.size > 0 else jnp.zeros((0, 1))
                else:
                    remapped_edges = jnp.zeros((0, 2), dtype=int)

                # Create subgraph adjacency
                num_subgraph_nodes = len(agent_nodes)
                sub_adjacency = jnp.zeros((num_subgraph_nodes, num_subgraph_nodes))

                if len(remapped_edges) > 0:
                    # Validate edge indices before setting adjacency
                    valid_indices = (remapped_edges[:, 0] < num_subgraph_nodes) & \
                                   (remapped_edges[:, 1] < num_subgraph_nodes) & \
                                   (remapped_edges[:, 0] >= 0) & \
                                   (remapped_edges[:, 1] >= 0)

                    valid_edges = remapped_edges[valid_indices]

                    if len(valid_edges) > 0:
                        sub_adjacency = sub_adjacency.at[valid_edges[:, 0], valid_edges[:, 1]].set(1.0)
                        sub_adjacency = sub_adjacency.at[valid_edges[:, 1], valid_edges[:, 0]].set(1.0)

                    if len(valid_edges) < len(remapped_edges):
                        self.logger.warning(f"Agent {agent_id}: {len(remapped_edges) - len(valid_edges)} invalid edges filtered out")

                # Create subgraph with validated components
                result = GraphState(
                    nodes=sub_nodes,
                    edges=remapped_edges,
                    edge_attr=sub_edge_attr,
                    adjacency=sub_adjacency,
                    timestamps=graph_state.timestamps[agent_nodes] if graph_state.timestamps is not None else None
                )

                # Validate result
                validate_graph_state(result)

                self.logger.debug(
                    f"Agent {agent_id} subgraph: {len(agent_nodes)} nodes, {len(remapped_edges)} edges"
                )

                return result

        except Exception as e:
            error_msg = f"Failed to extract subgraph for agent {agent_id}: {str(e)}"
            log_exception(self.logger, e, {"agent_id": agent_id, "operation": "subgraph_extraction"})

            if self.security_audit:
                self.security_audit.log_security_event(
                    "subgraph_extraction_failure",
                    error_msg,
                    "HIGH",
                    {"agent_id": agent_id}
                )

            # Add to failed agents
            self.failed_agents.add(agent_id)

            raise FederatedLearningError(error_msg, agent_id=agent_id)

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
        Execute one round of federated learning with comprehensive error handling.
        
        Args:
            agent_gradients: List of local gradients from each agent
            
        Returns:
            List of aggregated gradients for each agent
        """
        round_start_time = time.time()

        try:
            # Validate inputs
            if not agent_gradients:
                raise ValidationError("Empty gradient list provided")

            if len(agent_gradients) != self.config.num_agents:
                self.logger.warning(
                    f"Expected {self.config.num_agents} gradients, got {len(agent_gradients)}"
                )

            # Security validation for gradients
            valid_gradients = []
            failed_agent_ids = []

            for agent_id, gradients in enumerate(agent_gradients):
                try:
                    if self.enable_security:
                        # Check gradient integrity
                        grad_norm = check_gradient_norm(gradients, max_norm=10.0)

                        # Validate each gradient tensor
                        for key, grad in gradients.items():
                            validate_input_data(grad)

                    valid_gradients.append(gradients)

                except Exception as e:
                    self.logger.error(f"Agent {agent_id} gradients failed validation: {e}")
                    failed_agent_ids.append(agent_id)
                    self.failed_agents.add(agent_id)

                    if self.security_audit:
                        self.security_audit.log_security_event(
                            "gradient_validation_failure",
                            f"Agent {agent_id} gradients invalid: {str(e)}",
                            "HIGH",
                            {"agent_id": agent_id}
                        )

                    # Use zero gradients for failed agents
                    if agent_gradients:
                        zero_grads = {}
                        for key, grad in agent_gradients[0].items():
                            zero_grads[key] = jnp.zeros_like(grad)
                        valid_gradients.append(zero_grads)
                    else:
                        valid_gradients.append({})

            with self.perf_logger.timer_context(f"federated_aggregation_{self.config.aggregation}"):

                if self.config.aggregation == "gossip":
                    # Each agent gossips with neighbors
                    aggregated_gradients = []
                    successful_updates = 0

                    for agent_id, local_grads in enumerate(valid_gradients):
                        try:
                            if agent_id in self.failed_agents:
                                # Skip failed agents
                                aggregated_gradients.append(local_grads)
                                continue

                            agg_grads = self.gossip_aggregate(agent_id, local_grads)
                            aggregated_gradients.append(agg_grads)
                            successful_updates += 1

                        except Exception as e:
                            self.logger.error(f"Gossip aggregation failed for agent {agent_id}: {e}")
                            aggregated_gradients.append(local_grads)  # Fallback to local gradients

                elif self.config.aggregation == "hierarchical":
                    # Central aggregation with fault tolerance
                    try:
                        global_grads = self.hierarchical_aggregate(valid_gradients)
                        aggregated_gradients = [global_grads] * len(valid_gradients)
                        successful_updates = len(valid_gradients) - len(failed_agent_ids)

                    except Exception as e:
                        self.logger.error(f"Hierarchical aggregation failed: {e}")
                        # Fallback to local gradients
                        aggregated_gradients = valid_gradients
                        successful_updates = 0

                else:
                    # No aggregation, return local gradients
                    aggregated_gradients = valid_gradients
                    successful_updates = len(valid_gradients) - len(failed_agent_ids)

            # Record round metrics
            round_time = time.time() - round_start_time

            round_metrics = {
                "round_number": self.global_step,
                "aggregation_method": self.config.aggregation,
                "total_agents": self.config.num_agents,
                "participating_agents": len(valid_gradients),
                "successful_updates": successful_updates,
                "failed_agents": list(failed_agent_ids),
                "round_time_seconds": round_time
            }

            self.round_metrics.append(round_metrics)

            # Health monitoring
            if self.health_monitor:
                comm_status = self.health_monitor.check_communication_health(
                    self.global_step,
                    self.config.num_agents,
                    successful_updates,
                    round_time
                )

                if comm_status.value != "healthy":
                    self.logger.warning(f"Communication health status: {comm_status.value}")

            # Log federated round completion
            from ..utils.logging import log_federated_round
            log_federated_round(
                self.logger,
                self.global_step,
                self.config.num_agents,
                self.config.aggregation,
                round_metrics
            )

            return aggregated_gradients

        except Exception as e:
            error_msg = f"Federated round {self.global_step} failed: {str(e)}"
            log_exception(self.logger, e, {
                "round_number": self.global_step,
                "operation": "federated_round"
            })

            if self.security_audit:
                self.security_audit.log_security_event(
                    "federated_round_failure",
                    error_msg,
                    "CRITICAL"
                )

            raise FederatedLearningError(error_msg, round_number=self.global_step)

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
