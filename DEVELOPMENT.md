# Development Guide

This guide provides detailed information for developers working on pg-neo-graph-rl.

## Project Structure

```
pg-neo-graph-rl/
├── pg_neo_graph_rl/          # Main package source
│   ├── __init__.py
│   ├── core/                 # Core federated learning components
│   ├── algorithms/           # RL algorithms (PPO, SAC, etc.)
│   ├── environments/         # Simulation environments
│   ├── networks/             # Graph neural networks
│   ├── utils/                # Utilities and helpers
│   └── monitoring/           # Metrics and monitoring
├── tests/                    # Test suite
│   ├── unit/                # Unit tests
│   ├── integration/         # Integration tests
│   └── benchmarks/          # Performance benchmarks
├── docs/                    # Documentation
├── examples/                # Usage examples
├── scripts/                 # Development scripts
└── benchmarks/              # Benchmark scenarios
```

## Development Workflow

### 1. Environment Setup

```bash
# Clone repository
git clone https://github.com/yourusername/pg-neo-graph-rl.git
cd pg-neo-graph-rl

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install development dependencies
pip install -e ".[dev]"
```

### 2. GPU Development

For CUDA development:

```bash
# Install JAX with CUDA support
pip install -e ".[dev,gpu]"

# Verify CUDA installation
python -c "import jax; print(jax.devices())"
```

### 3. Pre-commit Setup

```bash
# Install pre-commit hooks
pre-commit install

# Run on all files (optional)
pre-commit run --all-files
```

## Code Organization

### Core Components

- **`core/`**: Federated learning infrastructure
  - `federated_learner.py`: Base federated learning class
  - `communication.py`: Agent communication protocols
  - `aggregation.py`: Gradient aggregation methods

- **`algorithms/`**: Reinforcement learning algorithms
  - `graph_ppo.py`: Graph-based PPO implementation
  - `graph_sac.py`: Graph-based SAC implementation
  - `base_algorithm.py`: Base RL algorithm interface

- **`networks/`**: Neural network architectures
  - `graph_networks.py`: Graph neural network layers
  - `attention.py`: Graph attention mechanisms
  - `encoders.py`: Graph encoders and decoders

### Environment Guidelines

- **`environments/`**: Simulation environments
  - Follow Gymnasium interface
  - Provide both single-agent and multi-agent versions
  - Include realistic dynamics and constraints

## Testing Strategy

### Test Categories

1. **Unit Tests** (`tests/unit/`)
   - Test individual functions and classes
   - Fast execution (< 1 second each)
   - No external dependencies

2. **Integration Tests** (`tests/integration/`)
   - Test component interactions
   - End-to-end workflows
   - May involve file I/O or networking

3. **Benchmark Tests** (`tests/benchmarks/`)
   - Performance regression tests
   - Scalability validation
   - Resource usage monitoring

### Running Tests

```bash
# All tests
pytest

# Specific categories
pytest -m unit
pytest -m integration
pytest -m "not slow"

# With coverage
pytest --cov=pg_neo_graph_rl --cov-report=html

# Parallel execution
pytest -n auto
```

### Writing Tests

Follow this structure for new tests:

```python
import pytest
import jax.numpy as jnp
from pg_neo_graph_rl.algorithms import GraphPPO


class TestGraphPPO:
    """Test suite for GraphPPO algorithm."""
    
    @pytest.fixture
    def ppo_config(self):
        """Configuration for PPO tests."""
        return {
            "learning_rate": 3e-4,
            "gnn_layers": 2,
            "hidden_dim": 64
        }
    
    def test_initialization(self, ppo_config):
        """Test PPO initialization."""
        agent = GraphPPO(**ppo_config)
        assert agent.learning_rate == 3e-4
    
    @pytest.mark.slow
    def test_training_convergence(self, ppo_config, sample_graph):
        """Test training convergence (slow test)."""
        # Implementation
        pass
```

## Code Quality Standards

### Style Guidelines

```bash
# Format code
black .

# Sort imports
ruff check --fix .

# Type checking
mypy pg_neo_graph_rl/

# Documentation
pydocstyle pg_neo_graph_rl/
```

### Performance Considerations

1. **JAX Best Practices**
   - Use `@jax.jit` for hot paths
   - Avoid Python loops in compute-heavy functions
   - Use `jax.vmap` for vectorization

2. **Memory Management**
   - Profile memory usage with `jax.profiler`
   - Use gradient checkpointing for large models
   - Clear cached compilations when needed

3. **Graph Operations**
   - Batch graph operations when possible
   - Use sparse representations for large graphs
   - Optimize message passing implementations

## Documentation

### API Documentation

Use Google-style docstrings:

```python
def federated_aggregate(gradients: List[jnp.ndarray],
                       weights: Optional[List[float]] = None) -> jnp.ndarray:
    """Aggregate gradients from federated agents.
    
    This function implements FedAvg algorithm for gradient aggregation
    across multiple federated learning agents.
    
    Args:
        gradients: List of gradient arrays from each agent.
        weights: Optional aggregation weights. If None, uses uniform weighting.
        
    Returns:
        Aggregated gradient array ready for parameter updates.
        
    Raises:
        ValueError: If gradients list is empty or shapes don't match.
        
    Example:
        >>> agent_grads = [grad1, grad2, grad3]
        >>> weights = [0.5, 0.3, 0.2]
        >>> aggregated = federated_aggregate(agent_grads, weights)
    """
```

### Building Documentation

```bash
cd docs/
make html  # Build HTML docs
make clean # Clean build files
```

## Debugging and Profiling

### JAX Debugging

```python
# Enable debugging
jax.config.update("jax_debug_nans", True)
jax.config.update("jax_debug_infs", True)

# Disable JIT for debugging
jax.config.update("jax_disable_jit", True)
```

### Performance Profiling

```python
# Profile memory usage
with jax.profiler.trace("/tmp/jaxprof"):
    result = expensive_function(data)

# Profile computation time
import time
start = time.time()
result = jax.block_until_ready(computation(data))
duration = time.time() - start
```

## Release Process

### Version Management

Versions are managed automatically via `hatch-vcs` based on git tags:

```bash
# Create release tag
git tag v0.1.0
git push origin v0.1.0

# Build package
python -m build

# Upload to PyPI (maintainers only)
twine upload dist/*
```

### Changelog

Update `CHANGELOG.md` with:
- New features
- Bug fixes
- Breaking changes
- Performance improvements

## Common Issues

### JAX Installation

If JAX installation fails:
```bash
# For CPU-only
pip install --upgrade "jax[cpu]"

# For CUDA (specify version)
pip install --upgrade "jax[cuda12_pip]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
```

### Memory Issues

For out-of-memory errors:
```python
# Reduce batch size
config.batch_size = 16

# Use gradient accumulation
config.gradient_accumulation_steps = 4

# Enable gradient checkpointing
config.use_gradient_checkpointing = True
```

### Graph Size Limitations

For large graphs:
```python
# Use graph sampling
from pg_neo_graph_rl.utils import GraphSampler
sampler = GraphSampler(max_nodes=1000)

# Use hierarchical graph processing
from pg_neo_graph_rl.utils import HierarchicalGraphProcessor
processor = HierarchicalGraphProcessor(levels=3)
```

## Getting Help

- Check existing issues on GitHub
- Read the documentation
- Ask questions in GitHub Discussions
- Contact maintainers for urgent issues

For more specific guidance, see:
- [Algorithm Development](docs/dev/algorithms.md)
- [Environment Creation](docs/dev/environments.md)
- [Network Architecture](docs/dev/networks.md)