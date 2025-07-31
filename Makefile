.PHONY: help install install-dev test test-fast lint format type-check clean build docs serve-docs
.DEFAULT_GOAL := help

PYTHON := python
PIP := pip
PYTEST := pytest

help: ## Show this help message
	@echo "Available commands:"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-20s\033[0m %s\n", $$1, $$2}'

install: ## Install package for production
	$(PIP) install -e .

install-dev: ## Install package with development dependencies  
	$(PIP) install -e ".[dev,monitoring,benchmarks]"
	pre-commit install

install-gpu: ## Install package with GPU support
	$(PIP) install -e ".[dev,gpu,monitoring,benchmarks]"
	pre-commit install

test: ## Run all tests with coverage
	$(PYTEST) --cov=pg_neo_graph_rl --cov-report=html --cov-report=term

test-fast: ## Run tests excluding slow ones
	$(PYTEST) -m "not slow" --cov=pg_neo_graph_rl --cov-report=term

test-unit: ## Run unit tests only
	$(PYTEST) tests/unit/ --cov=pg_neo_graph_rl --cov-report=term

test-integration: ## Run integration tests only
	$(PYTEST) tests/integration/ --cov=pg_neo_graph_rl --cov-report=term

lint: ## Run linting checks
	pre-commit run --all-files

format: ## Format code
	black .
	ruff check --fix .

type-check: ## Run type checking
	mypy pg_neo_graph_rl/

security-check: ## Run security checks
	bandit -r pg_neo_graph_rl/ -f json -o bandit-report.json || true
	safety check --json --output safety-report.json || true

clean: ## Clean build artifacts
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info/
	rm -rf .pytest_cache/
	rm -rf .coverage
	rm -rf htmlcov/
	rm -rf .mypy_cache/
	rm -rf .ruff_cache/
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete

build: clean ## Build package
	$(PYTHON) -m build

docs: ## Build documentation
	cd docs && make html

serve-docs: docs ## Serve documentation locally
	cd docs/_build/html && $(PYTHON) -m http.server 8000

benchmark: ## Run performance benchmarks
	$(PYTEST) tests/benchmarks/ -v

profile: ## Run profiling
	$(PYTHON) -m cProfile -o profile_output.prof scripts/profile_run.py

docker-build: ## Build Docker image
	docker build -t pg-neo-graph-rl .

docker-run: ## Run container interactively
	docker run -it --rm pg-neo-graph-rl

docker-test: ## Run tests in container
	docker run --rm pg-neo-graph-rl make test-fast

monitoring-stack: ## Start monitoring stack with Docker Compose
	docker-compose -f docker/docker-compose.monitoring.yml up -d

monitoring-down: ## Stop monitoring stack
	docker-compose -f docker/docker-compose.monitoring.yml down

release-check: ## Check release readiness
	$(PYTHON) -m build
	twine check dist/*
	@echo "Release check completed successfully"

all: install-dev lint test build docs ## Run complete development workflow