"""
GNNPolicy: 2-layer GCN policy network.

Pure PyTorch — no PyG, no DGL, no external graph libraries.

Architecture:
  Input: node features [N, in_features]
  GCN layer 1: [N, in_features] → [N, hidden_dim]  (with ReLU)
  GCN layer 2: [N, hidden_dim]  → [N, hidden_dim]  (with ReLU)
  Graph pool:  mean across nodes → [hidden_dim]
  Edge scorer: for each edge, concat pooled_graph with src+dst embeddings
               → linear → scalar logit
  Output: softmax distribution over edges
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical

from .env import GraphObservation


class GCNLayer(nn.Module):
    """Single Graph Convolutional Network layer (symmetric-normalized)."""

    def __init__(self, in_features: int, out_features: int) -> None:
        super().__init__()
        self.linear = nn.Linear(in_features, out_features, bias=False)

    def forward(
        self,
        x: torch.Tensor,       # [N, in_features]
        edge_index: torch.Tensor,  # [2, E]
        num_nodes: int,
    ) -> torch.Tensor:            # [N, out_features]
        """
        A_hat x W  where A_hat = symmetric-normalized A + I.
        Implemented via scatter_add for efficiency.
        """
        src, dst = edge_index[0], edge_index[1]

        # Degree for normalization (add 1 for self-loop)
        deg = torch.zeros(num_nodes, device=x.device)
        deg.scatter_add_(0, dst, torch.ones(dst.shape[0], device=x.device))
        deg = deg + 1.0  # self-loop

        # Normalize: d_i^{-0.5}
        deg_inv_sqrt = deg.pow(-0.5)

        # Aggregate: sum neighbor features
        agg = torch.zeros_like(x)
        # Weighted source contribution
        src_feat = x[src] * deg_inv_sqrt[src].unsqueeze(1)
        agg.scatter_add_(0, dst.unsqueeze(1).expand_as(src_feat), src_feat)
        # Self-loop
        agg = agg + x * deg_inv_sqrt.unsqueeze(1)
        # Normalize by destination degree
        agg = agg * deg_inv_sqrt.unsqueeze(1)

        return self.linear(agg)


class GNNPolicy(nn.Module):
    """
    2-layer GCN policy that maps a GraphObservation to a distribution
    over edges (actions).

    Parameters
    ----------
    in_features : int
        Node feature dimension.
    hidden_dim : int
        GCN hidden dimension.
    num_edges : int
        Number of edges in the fixed graph topology (action space size).
    """

    def __init__(
        self,
        in_features: int,
        hidden_dim: int,
        num_edges: int,
    ) -> None:
        super().__init__()
        self.gcn1 = GCNLayer(in_features, hidden_dim)
        self.gcn2 = GCNLayer(hidden_dim, hidden_dim)
        # Edge scorer: graph_pool + src_emb + dst_emb → logit
        self.edge_scorer = nn.Linear(hidden_dim * 3, 1)

    def forward(self, obs: GraphObservation) -> Categorical:
        """
        Forward pass.

        Parameters
        ----------
        obs : GraphObservation

        Returns
        -------
        dist : Categorical distribution over edges
        """
        x = obs.node_features          # [N, F]
        ei = obs.edge_index            # [2, E]
        N = obs.num_nodes

        # GCN layers
        h = F.relu(self.gcn1(x, ei, N))
        h = F.relu(self.gcn2(h, ei, N))   # [N, hidden_dim]

        # Global graph representation (mean pooling)
        g = h.mean(dim=0, keepdim=True)   # [1, hidden_dim]

        # Score each edge: [E, 3*hidden_dim] → [E, 1]
        src, dst = ei[0], ei[1]
        h_src = h[src]   # [E, hidden_dim]
        h_dst = h[dst]   # [E, hidden_dim]
        g_exp = g.expand(obs.num_edges, -1)  # [E, hidden_dim]

        edge_features = torch.cat([g_exp, h_src, h_dst], dim=1)  # [E, 3H]
        logits = self.edge_scorer(edge_features).squeeze(1)       # [E]

        return Categorical(logits=logits)

    def act(self, obs: GraphObservation) -> tuple[int, torch.Tensor]:
        """Sample an action and return (action, log_prob)."""
        dist = self.forward(obs)
        action = dist.sample()
        return int(action.item()), dist.log_prob(action)
