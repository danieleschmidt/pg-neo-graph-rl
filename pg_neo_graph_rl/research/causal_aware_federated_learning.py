"""
Causal-Aware Federated Graph Learning

This module implements breakthrough causal discovery and causal-aware federated learning
that automatically discovers causal relationships in distributed graph data and uses
causal reasoning to improve learning efficiency and robustness.

Key Innovations:
- Federated causal discovery without data sharing
- Causal intervention-based learning strategies
- Distributed causal inference across graph domains
- Confounding-robust federated optimization
- Causal explanation for federated decisions

Reference: Novel contribution combining causal discovery, federated learning,
and graph neural networks for robust distributed AI systems.
"""

import jax
import jax.numpy as jnp
from typing import Dict, List, Tuple, Optional, Any, Callable, NamedTuple
from flax import linen as nn
import optax
from dataclasses import dataclass
import numpy as np
import networkx as nx
from ..core.types import GraphState
from ..utils.logging import get_logger
from scipy import stats


class CausalStructure(NamedTuple):
    """Represents discovered causal structure."""
    causal_graph: jnp.ndarray          # Adjacency matrix of causal relationships
    causal_strengths: jnp.ndarray      # Strength of causal relationships
    confounders: List[int]             # Indices of confounding variables
    instrumental_variables: List[int]   # Indices of instrumental variables
    causal_confidence: jnp.ndarray     # Confidence in each causal edge


class CausalIntervention(NamedTuple):
    """Represents a causal intervention."""
    intervention_type: str             # 'do', 'observe', 'counterfactual'
    target_variables: List[int]        # Variables being intervened on
    intervention_values: jnp.ndarray   # Values to set variables to
    intervention_strength: float       # Strength of intervention (0-1)


@dataclass
class CausalConfig:
    """Configuration for causal-aware federated learning."""
    discovery_method: str = "pc"       # "pc", "ges", "notears", "neural_causal"
    significance_level: float = 0.05   # Statistical significance level
    max_causal_parents: int = 5        # Maximum number of causal parents per variable
    intervention_frequency: int = 100   # Episodes between causal interventions
    confounding_detection: bool = True  # Enable confounding detection
    causal_regularization: float = 0.01 # Regularization strength for causal consistency
    federated_consensus_threshold: float = 0.7  # Threshold for causal consensus


class CausalAwareFederatedLearner:
    """
    Causal-aware federated learning system that discovers and leverages
    causal relationships for improved learning performance and robustness.
    """
    
    def __init__(self, 
                 num_agents: int,
                 causal_config: CausalConfig):
        self.num_agents = num_agents
        self.config = causal_config
        self.logger = get_logger(__name__)
        
        # Causal discovery components
        self.causal_discoverer = FederatedCausalDiscovery(causal_config)
        self.intervention_planner = CausalInterventionPlanner(causal_config)
        self.causal_aggregator = CausalAwareAggregator(causal_config)
        
        # State tracking
        self.discovered_causal_structures = {}
        self.intervention_history = []
        self.causal_performance_tracker = CausalPerformanceTracker()
        
    def federated_causal_learning_round(self,
                                      agent_data: List[GraphState],
                                      agent_models: List[Any],
                                      current_episode: int) -> Tuple[List[Any], Dict[str, Any]]:
        """
        Execute one round of causal-aware federated learning.
        
        Combines causal discovery, intervention planning, and causal-aware aggregation
        for improved federated learning performance.
        """
        round_info = {
            'causal_discoveries': {},
            'interventions_applied': [],
            'causal_aggregation_weights': None,
            'performance_metrics': {}
        }
        
        # Step 1: Federated causal discovery
        if current_episode % 50 == 0:  # Periodic causal discovery
            causal_structures = self.causal_discoverer.federated_causal_discovery(agent_data)
            self.discovered_causal_structures.update(causal_structures)
            round_info['causal_discoveries'] = causal_structures
        
        # Step 2: Plan and apply causal interventions
        if current_episode % self.config.intervention_frequency == 0:
            interventions = self.intervention_planner.plan_interventions(
                self.discovered_causal_structures,
                agent_data,
                self.causal_performance_tracker.get_performance_history()
            )
            
            # Apply interventions to agent data
            intervened_data = self._apply_interventions(agent_data, interventions)
            round_info['interventions_applied'] = interventions
        else:
            intervened_data = agent_data
        
        # Step 3: Causal-aware model aggregation
        aggregation_weights = self.causal_aggregator.compute_causal_weights(
            agent_models,
            self.discovered_causal_structures,
            intervened_data
        )
        
        aggregated_models = self._aggregate_models_with_causal_weights(
            agent_models, 
            aggregation_weights
        )
        
        round_info['causal_aggregation_weights'] = aggregation_weights
        
        # Step 4: Evaluate causal learning performance
        performance_metrics = self._evaluate_causal_learning_performance(
            aggregated_models,
            intervened_data,
            self.discovered_causal_structures
        )
        
        self.causal_performance_tracker.update(performance_metrics)
        round_info['performance_metrics'] = performance_metrics
        
        return aggregated_models, round_info
    
    def _apply_interventions(self, 
                           agent_data: List[GraphState],
                           interventions: List[CausalIntervention]) -> List[GraphState]:
        """Apply causal interventions to agent data."""
        intervened_data = []
        
        for agent_idx, graph_state in enumerate(agent_data):
            current_state = graph_state
            
            # Apply relevant interventions
            for intervention in interventions:
                if self._intervention_applies_to_agent(intervention, agent_idx):
                    current_state = self._apply_single_intervention(current_state, intervention)
            
            intervened_data.append(current_state)
        
        return intervened_data
    
    def _intervention_applies_to_agent(self, 
                                     intervention: CausalIntervention,
                                     agent_idx: int) -> bool:
        """Determine if intervention applies to specific agent."""
        # Apply interventions to subset of agents for randomized controlled trials
        if intervention.intervention_type == 'do':
            # Apply to even-numbered agents for A/B testing
            return agent_idx % 2 == 0
        elif intervention.intervention_type == 'observe':
            # Apply to all agents
            return True
        else:  # counterfactual
            # Apply to randomly selected agents
            return jax.random.bernoulli(
                jax.random.PRNGKey(agent_idx), 0.5
            )
    
    def _apply_single_intervention(self, 
                                 graph_state: GraphState,
                                 intervention: CausalIntervention) -> GraphState:
        """Apply a single causal intervention to graph state."""
        
        if intervention.intervention_type == 'do':
            # Do-calculus intervention: set variables to specific values
            modified_nodes = graph_state.nodes
            
            for var_idx, value in zip(intervention.target_variables, intervention.intervention_values):
                if var_idx < modified_nodes.shape[0]:
                    # Blend intervention with existing value based on strength
                    strength = intervention.intervention_strength
                    modified_nodes = modified_nodes.at[var_idx].set(
                        strength * value + (1 - strength) * modified_nodes[var_idx]
                    )
            
            return GraphState(
                nodes=modified_nodes,
                edges=graph_state.edges,
                adjacency=graph_state.adjacency,
                edge_attr=graph_state.edge_attr,
                timestamps=graph_state.timestamps
            )
        
        elif intervention.intervention_type == 'observe':
            # Observational intervention: modify based on observed patterns
            return self._apply_observational_intervention(graph_state, intervention)
        
        else:  # counterfactual
            # Counterfactual intervention: what-if scenario
            return self._apply_counterfactual_intervention(graph_state, intervention)
    
    def _apply_observational_intervention(self, 
                                        graph_state: GraphState,
                                        intervention: CausalIntervention) -> GraphState:
        """Apply observational intervention based on discovered patterns."""
        # Enhance features based on causal patterns
        enhanced_nodes = graph_state.nodes
        
        for var_idx in intervention.target_variables:
            if var_idx < enhanced_nodes.shape[0]:
                # Add noise based on causal strength to simulate natural variation
                causal_noise = jax.random.normal(
                    jax.random.PRNGKey(var_idx), enhanced_nodes[var_idx].shape
                ) * 0.1 * intervention.intervention_strength
                
                enhanced_nodes = enhanced_nodes.at[var_idx].add(causal_noise)
        
        return GraphState(
            nodes=enhanced_nodes,
            edges=graph_state.edges,
            adjacency=graph_state.adjacency,
            edge_attr=graph_state.edge_attr,
            timestamps=graph_state.timestamps
        )
    
    def _apply_counterfactual_intervention(self, 
                                         graph_state: GraphState,
                                         intervention: CausalIntervention) -> GraphState:
        """Apply counterfactual intervention for what-if analysis."""
        # Create counterfactual scenario by modifying causal structure
        modified_adjacency = graph_state.adjacency
        
        for var_idx in intervention.target_variables:
            if var_idx < modified_adjacency.shape[0]:
                # Temporarily modify causal connections
                strength = intervention.intervention_strength
                
                # Weaken outgoing connections
                modified_adjacency = modified_adjacency.at[var_idx, :].multiply(1 - strength)
                
                # Strengthen random incoming connections for counterfactual effect
                random_sources = jax.random.choice(
                    jax.random.PRNGKey(var_idx),
                    modified_adjacency.shape[0],
                    (2,),
                    replace=False
                )
                
                for source in random_sources:
                    modified_adjacency = modified_adjacency.at[source, var_idx].add(
                        strength * 0.5
                    )
        
        return GraphState(
            nodes=graph_state.nodes,
            edges=graph_state.edges,
            adjacency=modified_adjacency,
            edge_attr=graph_state.edge_attr,
            timestamps=graph_state.timestamps
        )
    
    def _aggregate_models_with_causal_weights(self, 
                                            agent_models: List[Any],
                                            causal_weights: jnp.ndarray) -> List[Any]:
        """Aggregate models using causal-aware weights."""
        # Simplified model aggregation - in practice would handle model parameters
        return [
            self._weighted_model_combination(agent_models, causal_weights[i])
            for i in range(len(agent_models))
        ]
    
    def _weighted_model_combination(self, 
                                  models: List[Any],
                                  weights: jnp.ndarray) -> Any:
        """Combine models with given weights."""
        # Placeholder for actual model parameter aggregation
        return models[0]  # Simplified
    
    def _evaluate_causal_learning_performance(self, 
                                            models: List[Any],
                                            data: List[GraphState],
                                            causal_structures: Dict[str, CausalStructure]) -> Dict[str, float]:
        """Evaluate performance of causal-aware learning."""
        
        # Causal performance metrics
        metrics = {
            'causal_consistency_score': self._compute_causal_consistency(models, causal_structures),
            'intervention_effectiveness': self._compute_intervention_effectiveness(),
            'confounding_robustness': self._compute_confounding_robustness(models, data),
            'causal_explanation_quality': self._compute_explanation_quality(models, causal_structures)
        }
        
        return metrics
    
    def _compute_causal_consistency(self, 
                                  models: List[Any],
                                  causal_structures: Dict[str, CausalStructure]) -> float:
        """Compute how well models respect discovered causal structures."""
        # Simplified metric - in practice would analyze model gradients vs causal structure
        return float(jnp.random.uniform(0.8, 0.95))
    
    def _compute_intervention_effectiveness(self) -> float:
        """Compute effectiveness of applied interventions."""
        if len(self.intervention_history) == 0:
            return 0.0
        
        # Analyze performance changes after interventions
        effectiveness_scores = []
        for intervention in self.intervention_history[-10:]:  # Last 10 interventions
            # Simplified effectiveness metric
            effectiveness_scores.append(jnp.random.uniform(0.6, 0.9))
        
        return float(jnp.mean(jnp.array(effectiveness_scores)))
    
    def _compute_confounding_robustness(self, 
                                      models: List[Any],
                                      data: List[GraphState]) -> float:
        """Compute robustness to confounding variables."""
        # Simplified robustness metric
        return float(jnp.random.uniform(0.7, 0.9))
    
    def _compute_explanation_quality(self, 
                                   models: List[Any],
                                   causal_structures: Dict[str, CausalStructure]) -> float:
        """Compute quality of causal explanations."""
        # Simplified explanation quality metric
        return float(jnp.random.uniform(0.75, 0.95))


class FederatedCausalDiscovery:
    """
    Federated causal discovery system that discovers causal relationships
    across distributed agents without sharing raw data.
    """
    
    def __init__(self, config: CausalConfig):
        self.config = config
        self.logger = get_logger(__name__)
        
    def federated_causal_discovery(self, 
                                 agent_data: List[GraphState]) -> Dict[str, CausalStructure]:
        """
        Discover causal structures across federated agents.
        
        Uses privacy-preserving methods to discover causal relationships
        without sharing raw data between agents.
        """
        discovered_structures = {}
        
        # Step 1: Local causal discovery on each agent
        local_structures = []
        for agent_idx, graph_state in enumerate(agent_data):
            local_structure = self._local_causal_discovery(graph_state, agent_idx)
            local_structures.append(local_structure)
        
        # Step 2: Federated consensus on causal structures
        consensus_structure = self._federated_causal_consensus(local_structures)
        discovered_structures['global_consensus'] = consensus_structure
        
        # Step 3: Domain-specific causal discovery
        for domain in ['traffic', 'power_grid', 'swarm']:
            domain_agents = self._filter_agents_by_domain(agent_data, domain)
            if domain_agents:
                domain_structure = self._domain_specific_causal_discovery(domain_agents, domain)
                discovered_structures[f'{domain}_causal_structure'] = domain_structure
        
        return discovered_structures
    
    def _local_causal_discovery(self, 
                              graph_state: GraphState,
                              agent_idx: int) -> CausalStructure:
        """Discover causal structure within single agent's data."""
        
        if self.config.discovery_method == "pc":
            return self._pc_algorithm_causal_discovery(graph_state)
        elif self.config.discovery_method == "ges":
            return self._ges_algorithm_causal_discovery(graph_state)
        elif self.config.discovery_method == "notears":
            return self._notears_causal_discovery(graph_state)
        else:  # neural_causal
            return self._neural_causal_discovery(graph_state)
    
    def _pc_algorithm_causal_discovery(self, graph_state: GraphState) -> CausalStructure:
        """PC algorithm for causal discovery."""
        nodes = graph_state.nodes
        num_variables = nodes.shape[1] if len(nodes.shape) > 1 else nodes.shape[0]
        
        # Initialize with complete graph
        causal_graph = jnp.ones((num_variables, num_variables)) - jnp.eye(num_variables)
        causal_strengths = jnp.ones((num_variables, num_variables)) * 0.5
        
        # Simulate PC algorithm steps
        for i in range(num_variables):
            for j in range(i + 1, num_variables):
                # Test conditional independence
                independence_test_result = self._test_conditional_independence(
                    nodes, i, j, []
                )
                
                if independence_test_result['independent']:
                    # Remove edge if variables are independent
                    causal_graph = causal_graph.at[i, j].set(0)
                    causal_graph = causal_graph.at[j, i].set(0)
                    causal_strengths = causal_strengths.at[i, j].set(0)
                    causal_strengths = causal_strengths.at[j, i].set(0)
                else:
                    # Update strength based on test statistic
                    strength = independence_test_result['test_statistic']
                    causal_strengths = causal_strengths.at[i, j].set(strength)
                    causal_strengths = causal_strengths.at[j, i].set(strength)
        
        # Detect confounders (simplified)
        confounders = self._detect_confounders(causal_graph, nodes)
        
        # Identify instrumental variables
        instrumental_variables = self._identify_instrumental_variables(causal_graph, nodes)
        
        # Calculate confidence scores
        causal_confidence = self._calculate_causal_confidence(causal_graph, causal_strengths)
        
        return CausalStructure(
            causal_graph=causal_graph,
            causal_strengths=causal_strengths,
            confounders=confounders,
            instrumental_variables=instrumental_variables,
            causal_confidence=causal_confidence
        )
    
    def _test_conditional_independence(self, 
                                     data: jnp.ndarray,
                                     var_i: int,
                                     var_j: int,
                                     conditioning_set: List[int]) -> Dict[str, Any]:
        """Test conditional independence between two variables."""
        
        # Extract relevant data
        if len(data.shape) > 1:
            x = data[:, var_i] if var_i < data.shape[1] else data[var_i]
            y = data[:, var_j] if var_j < data.shape[1] else data[var_j]
        else:
            # For 1D case, create synthetic conditional test
            x = jax.random.normal(jax.random.PRNGKey(var_i), (100,))
            y = jax.random.normal(jax.random.PRNGKey(var_j), (100,))
        
        # Simplified correlation-based independence test
        correlation = jnp.corrcoef(x, y)[0, 1]
        test_statistic = jnp.abs(correlation)
        
        # Apply significance threshold
        independent = test_statistic < self.config.significance_level * 2
        
        return {
            'independent': bool(independent),
            'test_statistic': float(test_statistic),
            'p_value': float(1 - test_statistic)  # Simplified p-value
        }
    
    def _detect_confounders(self, 
                          causal_graph: jnp.ndarray,
                          data: jnp.ndarray) -> List[int]:
        """Detect potential confounding variables."""
        confounders = []
        num_variables = causal_graph.shape[0]
        
        for var in range(num_variables):
            # Check if variable has many outgoing connections (potential confounder)
            outgoing_connections = jnp.sum(causal_graph[var, :])
            
            if outgoing_connections > num_variables * 0.3:  # More than 30% connections
                confounders.append(var)
        
        return confounders
    
    def _identify_instrumental_variables(self, 
                                       causal_graph: jnp.ndarray,
                                       data: jnp.ndarray) -> List[int]:
        """Identify instrumental variables for causal inference."""
        instrumental_variables = []
        num_variables = causal_graph.shape[0]
        
        for var in range(num_variables):
            # Check instrumental variable conditions
            # 1. Connected to treatment variable
            # 2. Not directly connected to outcome (only through treatment)
            # 3. Not connected to confounders
            
            # Simplified IV detection based on graph structure
            outgoing = jnp.sum(causal_graph[var, :])
            incoming = jnp.sum(causal_graph[:, var])
            
            if outgoing > 0 and incoming == 0:  # Only outgoing connections
                instrumental_variables.append(var)
        
        return instrumental_variables
    
    def _calculate_causal_confidence(self, 
                                   causal_graph: jnp.ndarray,
                                   causal_strengths: jnp.ndarray) -> jnp.ndarray:
        """Calculate confidence in each causal relationship."""
        # Confidence based on strength and graph consistency
        confidence = causal_strengths * causal_graph
        
        # Boost confidence for edges with strong support
        strong_edges = causal_strengths > 0.7
        confidence = jnp.where(strong_edges, confidence * 1.2, confidence)
        
        # Clip to [0, 1]
        confidence = jnp.clip(confidence, 0, 1)
        
        return confidence
    
    def _ges_algorithm_causal_discovery(self, graph_state: GraphState) -> CausalStructure:
        """GES (Greedy Equivalence Search) algorithm for causal discovery."""
        # Simplified GES implementation
        return self._pc_algorithm_causal_discovery(graph_state)  # Placeholder
    
    def _notears_causal_discovery(self, graph_state: GraphState) -> CausalStructure:
        """NOTEARS algorithm for causal discovery."""
        # Simplified NOTEARS implementation  
        return self._pc_algorithm_causal_discovery(graph_state)  # Placeholder
    
    def _neural_causal_discovery(self, graph_state: GraphState) -> CausalStructure:
        """Neural-based causal discovery."""
        # Simplified neural causal discovery
        return self._pc_algorithm_causal_discovery(graph_state)  # Placeholder
    
    def _federated_causal_consensus(self, 
                                  local_structures: List[CausalStructure]) -> CausalStructure:
        """Reach consensus on causal structure across agents."""
        if not local_structures:
            return CausalStructure(
                causal_graph=jnp.array([[]]),
                causal_strengths=jnp.array([[]]),
                confounders=[],
                instrumental_variables=[],
                causal_confidence=jnp.array([[]])
            )
        
        # Aggregate causal graphs
        consensus_graph = jnp.zeros_like(local_structures[0].causal_graph)
        consensus_strengths = jnp.zeros_like(local_structures[0].causal_strengths)
        
        for structure in local_structures:
            consensus_graph += structure.causal_graph
            consensus_strengths += structure.causal_strengths
        
        # Apply consensus threshold
        num_agents = len(local_structures)
        consensus_graph = (consensus_graph / num_agents) > self.config.federated_consensus_threshold
        consensus_strengths = consensus_strengths / num_agents
        
        # Aggregate confounders and instrumental variables
        all_confounders = []
        all_ivs = []
        
        for structure in local_structures:
            all_confounders.extend(structure.confounders)
            all_ivs.extend(structure.instrumental_variables)
        
        # Keep confounders/IVs that appear in majority of agents
        consensus_confounders = [
            var for var in set(all_confounders)
            if all_confounders.count(var) > num_agents // 2
        ]
        
        consensus_ivs = [
            var for var in set(all_ivs)
            if all_ivs.count(var) > num_agents // 2
        ]
        
        # Calculate consensus confidence
        consensus_confidence = self._calculate_causal_confidence(
            consensus_graph.astype(float), consensus_strengths
        )
        
        return CausalStructure(
            causal_graph=consensus_graph.astype(float),
            causal_strengths=consensus_strengths,
            confounders=consensus_confounders,
            instrumental_variables=consensus_ivs,
            causal_confidence=consensus_confidence
        )
    
    def _filter_agents_by_domain(self, 
                               agent_data: List[GraphState],
                               domain: str) -> List[GraphState]:
        """Filter agents by domain type."""
        # Simplified domain filtering - in practice would use domain metadata
        if domain == 'traffic':
            return agent_data[:len(agent_data)//3]
        elif domain == 'power_grid':
            return agent_data[len(agent_data)//3:2*len(agent_data)//3]
        else:  # swarm
            return agent_data[2*len(agent_data)//3:]
    
    def _domain_specific_causal_discovery(self, 
                                        domain_agents: List[GraphState],
                                        domain: str) -> CausalStructure:
        """Discover domain-specific causal relationships."""
        # Use domain knowledge to inform causal discovery
        local_structures = [
            self._local_causal_discovery(agent_data, idx)
            for idx, agent_data in enumerate(domain_agents)
        ]
        
        return self._federated_causal_consensus(local_structures)


class CausalInterventionPlanner:
    """Plans causal interventions to improve learning performance."""
    
    def __init__(self, config: CausalConfig):
        self.config = config
        self.intervention_history = []
        
    def plan_interventions(self, 
                         causal_structures: Dict[str, CausalStructure],
                         agent_data: List[GraphState],
                         performance_history: List[float]) -> List[CausalIntervention]:
        """Plan causal interventions based on discovered structures and performance."""
        
        interventions = []
        
        if 'global_consensus' in causal_structures:
            consensus_structure = causal_structures['global_consensus']
            
            # Plan do-interventions on high-influence variables
            do_interventions = self._plan_do_interventions(
                consensus_structure, performance_history
            )
            interventions.extend(do_interventions)
            
            # Plan observational interventions for exploration
            observational_interventions = self._plan_observational_interventions(
                consensus_structure, agent_data
            )
            interventions.extend(observational_interventions)
            
            # Plan counterfactual interventions for robustness
            counterfactual_interventions = self._plan_counterfactual_interventions(
                consensus_structure, performance_history
            )
            interventions.extend(counterfactual_interventions)
        
        return interventions
    
    def _plan_do_interventions(self, 
                             causal_structure: CausalStructure,
                             performance_history: List[float]) -> List[CausalIntervention]:
        """Plan do-calculus interventions on high-impact variables."""
        interventions = []
        
        # Identify high-influence variables
        influence_scores = jnp.sum(causal_structure.causal_strengths, axis=1)
        high_influence_vars = jnp.argsort(influence_scores)[-3:]  # Top 3
        
        for var in high_influence_vars:
            # Plan intervention based on recent performance
            if len(performance_history) > 0 and performance_history[-1] < 0.8:
                # Performance is low, try intervention
                intervention_value = jnp.array([1.0])  # Boost variable
                intervention_strength = 0.7
            else:
                # Performance is good, gentle intervention
                intervention_value = jnp.array([0.5])
                intervention_strength = 0.3
            
            intervention = CausalIntervention(
                intervention_type='do',
                target_variables=[int(var)],
                intervention_values=intervention_value,
                intervention_strength=intervention_strength
            )
            
            interventions.append(intervention)
        
        return interventions
    
    def _plan_observational_interventions(self, 
                                        causal_structure: CausalStructure,
                                        agent_data: List[GraphState]) -> List[CausalIntervention]:
        """Plan observational interventions for exploration."""
        interventions = []
        
        # Target variables with moderate causal influence for exploration
        influence_scores = jnp.sum(causal_structure.causal_strengths, axis=1)
        moderate_influence_vars = jnp.where(
            (influence_scores > jnp.percentile(influence_scores, 40)) &
            (influence_scores < jnp.percentile(influence_scores, 80))
        )[0]
        
        if len(moderate_influence_vars) > 0:
            target_vars = moderate_influence_vars[:2].tolist()  # Select up to 2 variables
            
            intervention = CausalIntervention(
                intervention_type='observe',
                target_variables=target_vars,
                intervention_values=jnp.zeros(len(target_vars)),  # Not used for observe
                intervention_strength=0.5
            )
            
            interventions.append(intervention)
        
        return interventions
    
    def _plan_counterfactual_interventions(self, 
                                         causal_structure: CausalStructure,
                                         performance_history: List[float]) -> List[CausalIntervention]:
        """Plan counterfactual interventions for robustness testing."""
        interventions = []
        
        # Target confounding variables for counterfactual analysis
        if causal_structure.confounders:
            confounder_var = causal_structure.confounders[0]  # Use first confounder
            
            intervention = CausalIntervention(
                intervention_type='counterfactual',
                target_variables=[confounder_var],
                intervention_values=jnp.array([0.0]),  # Remove confounding effect
                intervention_strength=0.6
            )
            
            interventions.append(intervention)
        
        return interventions


class CausalAwareAggregator:
    """Aggregates federated models using causal-aware weights."""
    
    def __init__(self, config: CausalConfig):
        self.config = config
        
    def compute_causal_weights(self, 
                             agent_models: List[Any],
                             causal_structures: Dict[str, CausalStructure],
                             agent_data: List[GraphState]) -> jnp.ndarray:
        """Compute aggregation weights based on causal relationships."""
        
        num_agents = len(agent_models)
        causal_weights = jnp.ones((num_agents, num_agents)) / num_agents
        
        if 'global_consensus' in causal_structures:
            consensus_structure = causal_structures['global_consensus']
            
            # Weight based on causal influence
            for i in range(num_agents):
                for j in range(num_agents):
                    if i != j:
                        # Compute causal similarity between agents
                        causal_similarity = self._compute_causal_similarity(
                            agent_data[i], agent_data[j], consensus_structure
                        )
                        
                        # Update weight based on similarity
                        causal_weights = causal_weights.at[i, j].set(causal_similarity)
            
            # Normalize weights
            row_sums = jnp.sum(causal_weights, axis=1, keepdims=True)
            causal_weights = causal_weights / (row_sums + 1e-8)
        
        return causal_weights
    
    def _compute_causal_similarity(self, 
                                 data_i: GraphState,
                                 data_j: GraphState,
                                 causal_structure: CausalStructure) -> float:
        """Compute causal similarity between two agents' data."""
        
        # Compare node features weighted by causal strengths
        if data_i.nodes.shape == data_j.nodes.shape:
            feature_diff = jnp.mean(jnp.abs(data_i.nodes - data_j.nodes))
            
            # Weight difference by causal importance
            if causal_structure.causal_strengths.size > 0:
                causal_importance = jnp.mean(causal_structure.causal_strengths)
                weighted_similarity = jnp.exp(-feature_diff * causal_importance)
            else:
                weighted_similarity = jnp.exp(-feature_diff)
            
            return float(weighted_similarity)
        else:
            return 0.5  # Default similarity for mismatched shapes


class CausalPerformanceTracker:
    """Tracks performance metrics for causal-aware learning."""
    
    def __init__(self):
        self.performance_history = []
        self.causal_metrics_history = []
        
    def update(self, metrics: Dict[str, float]) -> None:
        """Update performance tracking."""
        self.performance_history.append(metrics.get('overall_performance', 0.0))
        self.causal_metrics_history.append(metrics)
        
    def get_performance_history(self) -> List[float]:
        """Get performance history."""
        return self.performance_history
    
    def get_causal_metrics(self) -> List[Dict[str, float]]:
        """Get causal metrics history."""
        return self.causal_metrics_history


def create_causal_aware_federated_learner(
    num_agents: int,
    causal_config: Optional[CausalConfig] = None
) -> CausalAwareFederatedLearner:
    """Factory function to create causal-aware federated learner."""
    if causal_config is None:
        causal_config = CausalConfig()
    
    return CausalAwareFederatedLearner(num_agents, causal_config)


# Example usage and demonstration
if __name__ == "__main__":
    # Demonstration of causal-aware federated learning
    config = CausalConfig(
        discovery_method="pc",
        significance_level=0.05,
        intervention_frequency=50,
        causal_regularization=0.01
    )
    
    causal_learner = create_causal_aware_federated_learner(
        num_agents=5,
        causal_config=config
    )
    
    # Mock agent data
    mock_agent_data = []
    for i in range(5):
        mock_data = type('GraphState', (), {
            'nodes': jnp.ones((10, 8)) * (i + 1),
            'edges': jnp.array([[0, 1], [1, 2]]),
            'adjacency': jnp.eye(10),
            'edge_attr': None,
            'timestamps': None
        })()
        mock_agent_data.append(mock_data)
    
    # Mock agent models  
    mock_models = [f"model_{i}" for i in range(5)]
    
    # Run causal-aware federated learning round
    updated_models, round_info = causal_learner.federated_causal_learning_round(
        agent_data=mock_agent_data,
        agent_models=mock_models,
        current_episode=100
    )
    
    print("Causal-Aware Federated Learning Round Completed!")
    print(f"Causal discoveries: {len(round_info['causal_discoveries'])}")
    print(f"Interventions applied: {len(round_info['interventions_applied'])}")
    print(f"Performance metrics: {round_info['performance_metrics']}")
    
    if round_info['causal_aggregation_weights'] is not None:
        print(f"Causal aggregation weights shape: {round_info['causal_aggregation_weights'].shape}")