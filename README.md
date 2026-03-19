# pg-neo-graph-rl

**Policy gradient reinforcement learning on dynamic graph-structured environments.**

A clean, minimal implementation — no Terragon, no bloat. Just the core ideas:

- **GraphEnv** — a network load-balancing environment where the agent routes capacity across graph edges to reduce imbalance
- **GNNPolicy** — a 2-layer Graph Convolutional Network (GCN) that reads node/edge structure and produces a distribution over edges
- **REINFORCE** — policy gradient trainer with moving-average baseline, gradient clipping, and discounted returns

Pure PyTorch. No PyG, no DGL, no external graph libraries.

---

## Problem Framing

The graph is a network with `N` nodes (entities) and `E` directed edges (connections). Each node carries a **load** in `[0, 1]`.

At each step the agent picks an edge `(src → dst)` and routes a unit of capacity — reducing `src` load and increasing `dst` load. The reward is the **negative load standard deviation** across all nodes:

```
reward = -std(loads) * 10
```

A balanced network (equal loads) maximizes reward. The agent must learn which edges to activate to minimize imbalance.

---

## Architecture

```
GraphObservation
  node_features: [N, 2]     ← [load, degree_norm]
  edge_index:    [2, E]     ← sparse adjacency (COO format)

GNNPolicy
  GCNLayer(in=2, out=H)     ← symmetric-normalized message passing
  GCNLayer(in=H, out=H)
  mean_pool → [H]           ← global graph representation
  edge_scorer(3H → 1)       ← [graph_pool | src_emb | dst_emb] per edge
  softmax → Categorical(E)  ← distribution over actions
```

The GCN uses symmetric normalization (`D^{-0.5} A D^{-0.5}`), implemented with `scatter_add` — no adjacency matrix materializiation.

---

## Usage

```python
from pg_neo_graph_rl import GraphEnv, GNNPolicy, REINFORCETrainer

env = GraphEnv(num_nodes=14, edge_prob=0.35, max_steps=50, seed=7)

policy = GNNPolicy(
    in_features=env.observation_feature_dim,
    hidden_dim=64,
    num_edges=env.num_edges,
)

trainer = REINFORCETrainer(env, policy, lr=3e-4)
history = trainer.train(num_episodes=500, log_every=50)
```

Run the demo:

```bash
python examples/demo.py
```

---

## Tests

```bash
pytest tests/ -v
```

17 tests covering `GraphEnv`, `GNNPolicy`, and `REINFORCETrainer`.

---

## Requirements

- Python ≥ 3.10
- PyTorch ≥ 2.0

```bash
pip install torch
```

---

## Notes on REINFORCE

REINFORCE is high-variance. For this environment:
- The load-balancing signal is clear but cumulative reward is noisy over 50-step episodes
- A greedy evaluation (policy.eval()) shows the agent does reduce load imbalance — typically from std ~0.30 → ~0.14 in 50 steps
- For faster convergence, try: more episodes (1000+), larger hidden dim (128), or switch to PPO

---

## Project

`pg-neo-graph-rl` = **P**olicy **G**radient **Neo** (new, revisited) **Graph RL** — a revival and clean reimplementation focused on the core algorithm, not scaffolding.
