# Contributing to pg-neo-graph-rl

We welcome contributions to the pg-neo-graph-rl project! This document provides guidelines for contributing.

## Code of Conduct

Please note that this project is released with a [Contributor Code of Conduct](CODE_OF_CONDUCT.md). By participating in this project you agree to abide by its terms.

## Getting Started

### Development Setup

1. **Fork and clone the repository**
   ```bash
   git clone https://github.com/your-username/pg-neo-graph-rl.git
   cd pg-neo-graph-rl
   ```

2. **Create a virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install development dependencies**
   ```bash
   pip install -e ".[dev,test]"
   ```

4. **Install pre-commit hooks**
   ```bash
   pre-commit install
   ```

5. **Run tests to verify setup**
   ```bash
   pytest
   ```

## Development Workflow

### Making Changes

1. **Create a feature branch**
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. **Make your changes**
   - Write clean, readable code
   - Add tests for new functionality
   - Update documentation as needed

3. **Run quality checks**
   ```bash
   # Format code
   black src/ tests/
   isort src/ tests/
   
   # Run linting
   flake8 src/ tests/
   mypy src/
   
   # Run tests
   pytest
   ```

4. **Commit your changes**
   ```bash
   git add .
   git commit -m "feat: add your feature description"
   ```

5. **Push and create a pull request**
   ```bash
   git push origin feature/your-feature-name
   ```

### Commit Message Convention

We follow the [Conventional Commits](https://www.conventionalcommits.org/) specification:

- `feat:` new features
- `fix:` bug fixes
- `docs:` documentation changes
- `style:` formatting changes
- `refactor:` code refactoring
- `test:` adding or updating tests
- `chore:` maintenance tasks

## Code Style

- **Python Style**: We use [Black](https://black.readthedocs.io/) for code formatting
- **Import Sorting**: We use [isort](https://pycqa.github.io/isort/) for import organization
- **Line Length**: 88 characters maximum
- **Type Hints**: Required for all public functions and methods
- **Docstrings**: Use Google-style docstrings for all public APIs

## Testing

### Running Tests

```bash
# Run all tests
pytest

# Run specific test file
pytest tests/test_example.py

# Run with coverage
pytest --cov=pg_neo_graph_rl

# Run tests in parallel
pytest -n auto
```

### Test Categories

- **Unit Tests**: Fast, isolated tests (`tests/unit/`)
- **Integration Tests**: Tests that verify component interaction (`tests/integration/`)
- **End-to-End Tests**: Full system tests (`tests/e2e/`)

### Writing Tests

- Use descriptive test names
- Follow the Arrange-Act-Assert pattern
- Mock external dependencies
- Test both success and failure cases

## Documentation

### Building Documentation

```bash
cd docs/
make html  # Generate HTML documentation
make livehtml  # Auto-reload during development
```

### Documentation Standards

- Use clear, concise language
- Include code examples
- Document all public APIs
- Update relevant documentation when making changes

## Pull Request Process

1. **Before submitting:**
   - Ensure all tests pass
   - Update documentation if needed
   - Add entry to CHANGELOG.md
   - Rebase your branch on latest main

2. **Pull Request Guidelines:**
   - Use a descriptive title
   - Include a detailed description
   - Reference related issues
   - Add appropriate labels

3. **Review Process:**
   - At least one maintainer review required
   - Address all feedback before merging
   - Maintain a clean commit history

## Issue Reporting

### Bug Reports

When reporting bugs, please include:
- Python version and platform
- JAX/CUDA version (if applicable)
- Minimal reproducible example
- Expected vs actual behavior
- Full error traceback

### Feature Requests

For feature requests, please:
- Describe the use case
- Explain why the feature is needed
- Suggest implementation approach
- Consider backwards compatibility

## Development Areas

We particularly welcome contributions in:

### Algorithms
- New graph RL algorithms
- Federated learning optimizations
- Communication-efficient methods

### Environments
- Real-world environment implementations
- Benchmark scenarios
- Simulator integrations

### Infrastructure
- Performance optimizations
- Distributed computing support
- Monitoring and visualization tools

### Documentation
- Tutorials and examples
- API documentation
- Deployment guides

## Getting Help

- **Discussions**: Use GitHub Discussions for questions
- **Issues**: Report bugs and request features via GitHub Issues
- **Chat**: Join our community Discord (link in README)

## Recognition

Contributors will be recognized in:
- CHANGELOG.md for their contributions
- README.md contributors section
- Release notes for significant contributions

Thank you for contributing to pg-neo-graph-rl! 🚀