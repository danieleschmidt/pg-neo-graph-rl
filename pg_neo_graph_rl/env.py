"""
Graph environment for network resource allocation.

The task: a network graph with N nodes, each node has a load (0-1).
The agent picks a node pair (source, dest) to route a unit of capacity
from a high-load node to a low-load node, reducing imbalance.

State:  node features = [load, degree_norm]
Action: discrete — pick a directed edge from a fixed edge set
Reward: -std(loads) * 10  →  maximizing reward == balancing load
Episode: 50 steps, random reset each episode
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import List, Tuple

import torch


@dataclass
class GraphObservation:
    """Sparse graph observation returned by the environment."""
    node_features: torch.Tensor   # shape [N, F]
    edge_index: torch.Tensor      # shape [2, E]  (source, dest) columns
    num_nodes: int
    num_edges: int

    def clone(self) -> "GraphObservation":
        return GraphObservation(
            node_features=self.node_features.clone(),
            edge_index=self.edge_index.clone(),
            num_nodes=self.num_nodes,
            num_edges=self.num_edges,
        )


class GraphEnv:
    """
    Dynamic graph environment: network load-balancing.

    Parameters
    ----------
    num_nodes : int
        Number of nodes (entities) in the graph.
    edge_prob : float
        Probability of an edge between any two nodes (Erdős–Rényi).
    max_steps : int
        Maximum steps per episode before truncation.
    seed : int | None
        Optional random seed.
    """

    def __init__(
        self,
        num_nodes: int = 12,
        edge_prob: float = 0.3,
        max_steps: int = 50,
        seed: int | None = None,
    ) -> None:
        self.num_nodes = num_nodes
        self.edge_prob = edge_prob
        self.max_steps = max_steps
        self._rng = random.Random(seed)
        self._torch_rng = torch.Generator()
        if seed is not None:
            self._torch_rng.manual_seed(seed)

        # Build fixed graph topology once
        self.edges: List[Tuple[int, int]] = self._build_graph()
        self.num_edges = len(self.edges)
        self.edge_index = torch.tensor(self.edges, dtype=torch.long).T  # [2, E]

        # Node loads (mutable per episode)
        self.loads: torch.Tensor = torch.zeros(num_nodes)
        self._step_count = 0

    # ------------------------------------------------------------------
    # Topology
    # ------------------------------------------------------------------

    def _build_graph(self) -> List[Tuple[int, int]]:
        """Build a random connected-ish undirected graph, return as directed edges."""
        edges = []
        # Ensure connectivity via a spanning chain first
        for i in range(self.num_nodes - 1):
            edges.append((i, i + 1))
            edges.append((i + 1, i))
        # Random additional edges
        for i in range(self.num_nodes):
            for j in range(i + 2, self.num_nodes):
                if self._rng.random() < self.edge_prob:
                    edges.append((i, j))
                    edges.append((j, i))
        return edges

    # ------------------------------------------------------------------
    # Gym-like interface
    # ------------------------------------------------------------------

    def reset(self) -> GraphObservation:
        """Reset episode: randomize node loads."""
        self.loads = torch.rand(self.num_nodes)
        self._step_count = 0
        return self._observe()

    def step(self, action: int) -> Tuple[GraphObservation, float, bool]:
        """
        Take an action (edge index) — route capacity from source → dest.

        Returns
        -------
        obs : GraphObservation
        reward : float
        done : bool
        """
        src, dst = self.edges[action]

        # Transfer a fraction of excess load from src to dst
        transfer = 0.1 * (self.loads[src] - self.loads[dst])
        self.loads[src] -= transfer
        self.loads[dst] += transfer
        # Clip to [0, 1]
        self.loads = self.loads.clamp(0.0, 1.0)

        reward = float(-self.loads.std() * 10.0)

        self._step_count += 1
        done = self._step_count >= self.max_steps

        return self._observe(), reward, done

    # ------------------------------------------------------------------
    # Observation
    # ------------------------------------------------------------------

    def _observe(self) -> GraphObservation:
        degrees = torch.zeros(self.num_nodes)
        for src, _ in self.edges:
            degrees[src] += 1
        degree_norm = degrees / (degrees.max() + 1e-8)

        node_features = torch.stack([self.loads, degree_norm], dim=1)  # [N, 2]

        return GraphObservation(
            node_features=node_features,
            edge_index=self.edge_index,
            num_nodes=self.num_nodes,
            num_edges=self.num_edges,
        )

    @property
    def action_space_size(self) -> int:
        return self.num_edges

    @property
    def observation_feature_dim(self) -> int:
        return 2  # [load, degree_norm]
