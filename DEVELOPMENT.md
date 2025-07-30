# Development Guide

This guide covers setting up a development environment for pg-neo-graph-rl.

## Quick Setup

### Prerequisites

- Python 3.9 or higher
- Git
- (Optional) CUDA for GPU acceleration

### Development Environment

1. **Clone the repository**
   ```bash
   git clone https://github.com/terragon-labs/pg-neo-graph-rl.git
   cd pg-neo-graph-rl
   ```

2. **Create virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # Linux/Mac
   # OR
   venv\Scripts\activate     # Windows
   ```

3. **Install in development mode**
   ```bash
   make install-dev
   # OR manually:
   pip install -e ".[dev,test,docs]"
   pre-commit install
   ```

4. **Verify setup**
   ```bash
   make test
   ```

## Development Workflow

### Code Quality

We maintain high code quality standards:

```bash
# Format code
make format

# Run linting
make lint

# Run all checks
pre-commit run --all-files
```

### Testing

```bash
# Run all tests
make test

# Run with coverage
make test-cov

# Run specific test
pytest tests/unit/test_example.py -v

# Run tests by marker
pytest -m "not slow"
pytest -m "gpu" --gpu-available
```

### Documentation

```bash
# Build docs locally
make docs

# Serve docs with auto-reload
make docs-serve
```

## Project Structure

```
pg-neo-graph-rl/
├── src/pg_neo_graph_rl/         # Main package
│   ├── __init__.py
│   ├── core/                    # Core functionality
│   ├── algorithms/              # RL algorithms
│   ├── environments/            # Environment implementations
│   ├── federated/               # Federated learning
│   ├── monitoring/              # Metrics and monitoring
│   └── utils/                   # Utility functions
├── tests/                       # Test suite
│   ├── unit/                    # Unit tests
│   ├── integration/             # Integration tests
│   └── e2e/                     # End-to-end tests
├── docs/                        # Documentation
├── monitoring/                  # Monitoring configs
├── examples/                    # Example scripts
└── benchmarks/                  # Performance benchmarks
```

## Development Guidelines

### Code Style

- **Python Style**: Follow PEP 8 with Black formatting
- **Line Length**: 88 characters maximum
- **Type Hints**: Required for all public functions
- **Docstrings**: Google-style for all public APIs

Example:
```python
def train_agent(
    agent: GraphPPO,
    environment: TrafficEnvironment,
    num_episodes: int = 1000,
) -> Dict[str, float]:
    """Train a graph PPO agent on traffic environment.
    
    Args:
        agent: The PPO agent to train
        environment: Traffic environment instance
        num_episodes: Number of training episodes
        
    Returns:
        Dictionary containing training metrics
        
    Raises:
        ValueError: If num_episodes is negative
    """
```

### Testing Philosophy

- **Unit Tests**: Fast, isolated tests for individual functions
- **Integration Tests**: Test component interactions
- **End-to-End Tests**: Full system validation
- **Property-Based Testing**: Use Hypothesis for complex scenarios

### Git Workflow

1. **Create feature branch**
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. **Make changes with clear commits**
   ```bash
   git add .
   git commit -m "feat: add federated averaging algorithm"
   ```

3. **Push and create PR**
   ```bash
   git push origin feature/your-feature-name
   ```

## GPU Development

### JAX GPU Setup

```bash
# Install JAX with CUDA support
pip install -e ".[gpu]"

# Verify GPU availability
python -c "import jax; print(jax.devices())"
```

### GPU Testing

```bash
# Run GPU-specific tests
pytest -m gpu --gpu-available

# Skip GPU tests
pytest -m "not gpu"
```

## Performance Development

### Profiling

```python
import jax.profiler

# Profile JAX operations
with jax.profiler.trace("/tmp/jax-trace", create_perfetto_link=True):
    result = expensive_jax_function()
```

### Benchmarking

```bash
# Run benchmarks
python -m pg_neo_graph_rl.benchmarks.run_all

# Specific benchmark
python -m pg_neo_graph_rl.benchmarks.federated_training
```

## Container Development

### Development Container

```bash
# Build and run development container
docker-compose --profile dev up -d dev

# Execute commands in container
docker-compose exec dev bash
```

### Jupyter Development

```bash
# Start Jupyter environment
docker-compose --profile jupyter up -d jupyter

# Access at http://localhost:8888
```

## Monitoring Development

### Local Monitoring Stack

```bash
# Start monitoring services
docker-compose up -d grafana prometheus

# Access Grafana: http://localhost:3000 (admin/admin)
# Access Prometheus: http://localhost:9090
```

### Custom Metrics

```python
from pg_neo_graph_rl.monitoring import MetricsCollector

metrics = MetricsCollector()
metrics.log_scalar("training/loss", loss_value, step=episode)
metrics.log_histogram("rewards", reward_distribution)
```

## Debugging

### Common Issues

1. **JAX Installation**: Ensure correct CUDA version match
2. **Memory Issues**: Use `jax.config.update("jax_platform_name", "cpu")` for debugging
3. **Import Errors**: Check PYTHONPATH includes `src/`

### Debug Mode

```python
import jax
jax.config.update("jax_debug_nans", True)
jax.config.update("jax_debug_infs", True)
```

### Logging

```python
import logging
logging.basicConfig(level=logging.DEBUG)

logger = logging.getLogger(__name__)
logger.debug("Debug message")
```

## Contributing Process

1. **Check issues**: Look for existing issues or create new one
2. **Discuss approach**: Comment on issue with proposed solution
3. **Implement**: Follow development workflow above
4. **Test thoroughly**: Ensure all tests pass
5. **Document**: Update docs and add examples
6. **Submit PR**: Use PR template and request review

## Getting Help

- **GitHub Discussions**: For questions and ideas
- **Issues**: For bugs and feature requests
- **Discord**: Join our community chat (link in README)
- **Email**: development@terragon.ai for private inquiries

## Advanced Topics

### Custom Environments

```python
from pg_neo_graph_rl.environments.base import GraphEnvironment

class MyEnvironment(GraphEnvironment):
    def __init__(self, config):
        super().__init__()
        self.config = config
    
    def reset(self):
        # Return initial graph state
        pass
    
    def step(self, actions):
        # Execute actions and return next state
        pass
```

### Custom Algorithms

```python
from pg_neo_graph_rl.algorithms.base import GraphRLAlgorithm

class MyAlgorithm(GraphRLAlgorithm):
    def __init__(self, config):
        super().__init__(config)
    
    def train_step(self, batch):
        # Implement training logic
        pass
```

Happy coding! 🚀