"""
pg-neo-graph-rl: Federated Graph-Neural Reinforcement Learning

A toolkit for distributed control of city-scale infrastructure using
dynamic graph neural networks and federated reinforcement learning.
"""

__version__ = "0.1.0"
__author__ = "Daniel Schmidt"

from .algorithms import GraphPPO, GraphSAC
from .core import FederatedGraphRL
from .core.types import GraphState
from .environments import PowerGridEnvironment, SwarmEnvironment, TrafficEnvironment

__all__ = [
    "FederatedGraphRL",
    "GraphState",
    "GraphPPO",
    "GraphSAC",
    "TrafficEnvironment",
    "PowerGridEnvironment",
    "SwarmEnvironment"
]
