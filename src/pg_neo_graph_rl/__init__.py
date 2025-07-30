"""
pg-neo-graph-rl: Federated Graph-Neural Reinforcement Learning toolkit.

This package provides tools for distributed control of city-scale infrastructure
using federated learning and graph neural networks.
"""

__version__ = "0.1.0"

from .core import FederatedGraphRL
from .environments import TrafficEnvironment, PowerGridEnvironment
from .algorithms import GraphPPO, GraphSAC

__all__ = [
    "FederatedGraphRL",
    "TrafficEnvironment", 
    "PowerGridEnvironment",
    "GraphPPO",
    "GraphSAC",
]