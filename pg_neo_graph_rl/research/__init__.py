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
from .temporal_memory import HierarchicalTemporalGraphAttention
from .quantum_optimization import QuantumInspiredFederatedRL
from .neuromorphic_computing import NeuromorphicFederatedRL
from .causal_discovery import FederatedCausalDiscovery
from .performance_optimization import AutoScalingFederatedRL
from .experimental_framework import ResearchBenchmarkSuite

__all__ = [
    "SelfOrganizingFederatedRL",
    "HierarchicalTemporalGraphAttention", 
    "QuantumInspiredFederatedRL",
    "NeuromorphicFederatedRL",
    "FederatedCausalDiscovery",
    "AutoScalingFederatedRL", 
    "ResearchBenchmarkSuite"
]