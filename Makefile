.PHONY: help install install-dev test test-cov lint format clean docs docs-serve build

# Default target
help:
	@echo "Available commands:"
	@echo "  install      Install package in production mode"
	@echo "  install-dev  Install package in development mode with dev dependencies"
	@echo "  test         Run tests"
	@echo "  test-cov     Run tests with coverage report"
	@echo "  lint         Run linting checks"
	@echo "  format       Format code"
	@echo "  clean        Clean build artifacts"
	@echo "  docs         Build documentation"
	@echo "  docs-serve   Serve documentation locally"
	@echo "  build        Build package"

# Installation
install:
	pip install -e .

install-dev:
	pip install -e ".[dev,test,docs]"
	pre-commit install

# Testing
test:
	pytest

test-cov:
	pytest --cov=pg_neo_graph_rl --cov-report=term-missing --cov-report=html

# Code quality
lint:
	flake8 src/ tests/
	mypy src/
	black --check src/ tests/
	isort --check-only src/ tests/

format:
	black src/ tests/
	isort src/ tests/

# Documentation  
docs:
	cd docs && make html

docs-serve:
	cd docs && make livehtml

# Build and clean
build:
	python -m build

clean:
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info/
	rm -rf .pytest_cache/
	rm -rf .mypy_cache/
	rm -rf .coverage
	rm -rf htmlcov/
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete