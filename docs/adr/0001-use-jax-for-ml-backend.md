# 0001. Use JAX/Flax for ML Backend

Date: 2025-08-01

## Status

Accepted

## Context

PG-Neo-Graph-RL requires a high-performance machine learning backend that can handle:

1. **Large-scale graph operations** with dynamic topologies
2. **Distributed training** across multiple agents
3. **Real-time inference** for control applications
4. **Gradient computation** for federated learning aggregation
5. **Hardware acceleration** on both CPUs and GPUs

The main alternatives considered were:
- **PyTorch**: Mature ecosystem, dynamic graphs, but slower compilation
- **TensorFlow**: Good distributed support, but complex API for research
- **JAX**: Functional programming, fast compilation, excellent for numerical computing

## Decision

We will use JAX as the primary ML backend with Flax for neural network definitions.

**Key components:**
- **JAX**: Core numerical computing, automatic differentiation, JIT compilation
- **Flax**: Neural network library with clean functional API
- **Optax**: Gradient-based optimization algorithms
- **Haiku**: Alternative NN library for specific use cases if needed

## Consequences

### Positive Consequences

- **Performance**: JIT compilation provides significant speedups for graph operations
- **Functional Programming**: Pure functions enable better testing and debugging
- **Automatic Differentiation**: Excellent support for custom gradient computations
- **Vectorization**: vmap enables efficient batch processing of agent updates
- **Research Friendly**: Active development in ML research community
- **Hardware Agnostic**: Same code runs on CPU, GPU, TPU without modification

### Negative Consequences

- **Learning Curve**: JAX requires functional programming mindset
- **Ecosystem Maturity**: Smaller ecosystem compared to PyTorch/TensorFlow
- **Debugging**: Functional style can make debugging more challenging
- **Memory Management**: Requires explicit attention to memory patterns
- **Dynamic Shapes**: Less flexible than PyTorch for dynamic architectures

### Risks

- **Community Support**: Smaller community than PyTorch, though growing rapidly
  - *Mitigation*: Active Google backing and strong research adoption
- **API Stability**: Relatively new framework with occasional breaking changes
  - *Mitigation*: Pin versions, maintain compatibility layers
- **Integration**: Some third-party tools may not support JAX
  - *Mitigation*: Use JAX-native alternatives or write wrappers

### Implementation Guidelines

1. **Use pure functions** for all neural network computations
2. **JIT compile** performance-critical functions
3. **Vectorize** operations using vmap when possible
4. **Profile memory usage** to avoid OOM errors
5. **Document JAX patterns** for new contributors