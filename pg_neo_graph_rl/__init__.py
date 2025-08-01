"""
pg-neo-graph-rl: Federated Graph-Neural Reinforcement Learning

A toolkit for distributed control of city-scale infrastructure using
dynamic graph neural networks and federated reinforcement learning.
"""

__version__ = "0.1.0"
__author__ = "Daniel Schmidt"

from .core import FederatedGraphRL
from .algorithms import GraphPPO, GraphSAC
from .environments import TrafficEnvironment, PowerGridEnvironment, SwarmEnvironment

__all__ = [
    "FederatedGraphRL",
    "GraphPPO", 
    "GraphSAC",
    "TrafficEnvironment",
    "PowerGridEnvironment", 
    "SwarmEnvironment"
]
