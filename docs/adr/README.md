# Architecture Decision Records (ADRs)

This directory contains Architecture Decision Records (ADRs) for PG-Neo-Graph-RL. ADRs are lightweight documents that capture important architectural decisions along with their context and consequences.

## Format

We use the format described in [this article by Michael Nygard](http://thinkrelevance.com/blog/2011/11/15/documenting-architecture-decisions).

Each ADR should have:
1. **Title**: A short phrase describing the decision
2. **Status**: Proposed, Accepted, Deprecated, or Superseded
3. **Context**: The forces at play, including technological, political, social, and project local
4. **Decision**: The change we're proposing or have agreed to implement
5. **Consequences**: The context after applying the decision, including positive and negative impacts

## Naming Convention

ADRs should be named using the following pattern:
```
NNNN-title-of-decision.md
```

Where `NNNN` is a 4-digit number (padded with zeros) representing the sequence number.

## Current ADRs

| Number | Title | Status |
|--------|-------|--------|
| [0001](0001-use-jax-for-ml-backend.md) | Use JAX/Flax for ML Backend | Accepted |
| [0002](0002-gossip-protocol-for-federated-learning.md) | Gossip Protocol for Federated Learning | Accepted |
| [0003](0003-graph-neural-network-architecture.md) | Graph Neural Network Architecture | Accepted |

## Creating New ADRs

1. Copy the [template](template.md) to a new file
2. Assign the next sequential number
3. Fill in the template with your decision
4. Create a pull request for review
5. Update this README with the new ADR once merged

## ADR Lifecycle

- **Proposed**: The ADR is under discussion
- **Accepted**: The decision has been made and is being implemented
- **Deprecated**: The decision is no longer relevant but kept for historical context
- **Superseded**: A newer ADR has replaced this decision

## Tools

We recommend using [adr-tools](https://github.com/npryce/adr-tools) for managing ADRs, though manual creation is also acceptable.