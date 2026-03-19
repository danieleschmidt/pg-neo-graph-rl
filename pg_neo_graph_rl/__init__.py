"""
pg-neo-graph-rl: Policy Gradient for Graph-Structured Environments

A minimal, clean implementation of graph reinforcement learning:
- GraphEnv: dynamic graph environment (network routing / resource allocation)
- GNNPolicy: 2-layer GCN policy network (pure PyTorch, no PyG)
- REINFORCE trainer with moving-average baseline
"""

__version__ = "0.2.0"
__author__ = "Daniel Schmidt"

from .env import GraphEnv, GraphObservation
from .policy import GNNPolicy
from .trainer import REINFORCETrainer

__all__ = [
    "GraphEnv",
    "GraphObservation",
    "GNNPolicy",
    "REINFORCETrainer",
]
