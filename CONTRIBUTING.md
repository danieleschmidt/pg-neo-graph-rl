# Contributing to pg-neo-graph-rl

We welcome contributions to the Federated Graph-Neural Reinforcement Learning toolkit! This document provides guidelines for contributing to the project.

## Quick Start

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/your-feature`
3. Make your changes
4. Run tests: `pytest`
5. Submit a pull request

## Development Setup

### Prerequisites
- Python 3.9+
- Git

### Installation

```bash
# Clone your fork
git clone https://github.com/yourusername/pg-neo-graph-rl.git
cd pg-neo-graph-rl

# Install in development mode
pip install -e ".[dev]"

# Install pre-commit hooks
pre-commit install
```

### Development Dependencies

```bash
# For GPU development
pip install -e ".[dev,gpu]"

# For monitoring development
pip install -e ".[dev,monitoring]"

# For benchmark development
pip install -e ".[dev,benchmarks]"
```

## Code Standards

### Style Guidelines
- Follow PEP 8 style guide
- Use Black for code formatting: `black .`
- Use Ruff for linting: `ruff check .`
- Use mypy for type checking: `mypy pg_neo_graph_rl/`

### Code Quality
- Maintain test coverage above 80%
- Write docstrings for all public functions and classes
- Use type hints for all function signatures
- Follow existing patterns in the codebase

## Testing

### Running Tests
```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=pg_neo_graph_rl

# Run specific test categories
pytest -m unit          # Unit tests only
pytest -m integration   # Integration tests only
pytest -m "not slow"    # Exclude slow tests
```

### Writing Tests
- Place unit tests in `tests/unit/`
- Place integration tests in `tests/integration/`
- Use descriptive test names
- Follow AAA pattern (Arrange, Act, Assert)

## Documentation

### Docstring Format
Use Google-style docstrings:

```python
def federated_aggregate(gradients: List[jnp.ndarray], 
                       weights: Optional[List[float]] = None) -> jnp.ndarray:
    """Aggregate gradients from multiple federated agents.
    
    Args:
        gradients: List of gradient arrays from each agent.
        weights: Optional weights for weighted averaging.
        
    Returns:
        Aggregated gradient array.
        
    Raises:
        ValueError: If gradients list is empty.
    """
```

### Building Documentation
```bash
cd docs/
make html
```

## Pull Request Process

### Before Submitting
1. Ensure all tests pass
2. Run code quality checks
3. Update documentation if needed
4. Add entry to CHANGELOG.md

### PR Guidelines
- Use descriptive titles and descriptions
- Reference related issues
- Keep changes focused and atomic
- Provide benchmarks for performance changes

### Review Process
- All PRs require at least one review
- Address reviewer feedback promptly
- Maintain a clean commit history

## Issue Guidelines

### Bug Reports
Include:
- Python version and OS
- JAX/CUDA versions if relevant
- Minimal reproduction code
- Expected vs actual behavior
- Full error traceback

### Feature Requests
Include:
- Clear use case description
- Proposed API design
- Performance considerations
- Breaking change impact

## Priority Areas

We especially welcome contributions in:

### Algorithms
- New federated learning algorithms
- Graph neural network architectures
- Privacy-preserving methods
- Communication-efficient protocols

### Environments
- Real-world simulation environments
- Benchmark scenarios
- Environment wrappers

### Infrastructure
- Performance optimizations
- Monitoring and visualization tools
- Documentation improvements
- Testing infrastructure

### Integration
- Framework integrations (ROS, OpenAI Gym)
- Cloud deployment tools
- CI/CD improvements

## Community Guidelines

### Code of Conduct
- Be respectful and inclusive
- Focus on constructive feedback
- Help newcomers learn
- Maintain professional communication

### Getting Help
- Check existing issues and documentation
- Ask questions in GitHub Discussions
- Join our community channels (links in README)

## Recognition

Contributors are recognized in:
- CONTRIBUTORS.md file
- Release notes
- Academic paper acknowledgments

## License

By contributing, you agree that your contributions will be licensed under the MIT License.

For questions about contributing, please open an issue or contact the maintainers.