"""
Research Module: Breakthrough Algorithms and Experimental Frameworks

This module contains cutting-edge research implementations including:
- Self-organizing communication topologies
- Hierarchical graph attention with temporal memory  
- Quantum-inspired optimization algorithms
- Neuromorphic computing for energy efficiency
- Federated causal discovery
- Performance optimization suite
- Advanced experimental frameworks for research validation
"""

from .adaptive_topology import SelfOrganizingFederatedRL
from .autonomous_meta_learning import AutonomousMetaLearner, NeuralArchitectureSearch
from .causal_aware_federated_learning import CausalAwareFederatedLearner
from .causal_aware_federated_learning import (
    FederatedCausalDiscovery as CausalDiscoveryEngine,
)
from .causal_discovery import FederatedCausalDiscovery
from .experimental_framework import ResearchBenchmarkSuite
from .multimodal_fusion import MultiModalFederatedLearner, MultiModalFusionNetwork
from .neuromorphic_computing import NeuromorphicFederatedRL
from .performance_optimization import AutoScalingFederatedRL
from .quantum_optimization import (
    QuantumClassicalHybridOptimizer,
    QuantumInspiredFederatedRL,
)
from .self_evolving_architectures import EvolutionController, SelfEvolvingGraphNetwork
from .temporal_memory import HierarchicalTemporalGraphAttention

__all__ = [
    "SelfOrganizingFederatedRL",
    "HierarchicalTemporalGraphAttention",
    "QuantumInspiredFederatedRL",
    "QuantumClassicalHybridOptimizer",
    "NeuromorphicFederatedRL",
    "FederatedCausalDiscovery",
    "AutoScalingFederatedRL",
    "ResearchBenchmarkSuite",
    "AutonomousMetaLearner",
    "NeuralArchitectureSearch",
    "SelfEvolvingGraphNetwork",
    "EvolutionController",
    "CausalAwareFederatedLearner",
    "CausalDiscoveryEngine",
    "MultiModalFederatedLearner",
    "MultiModalFusionNetwork"
]
