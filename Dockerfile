# Multi-stage production Dockerfile for pg-neo-graph-rl
# Optimized for federated graph reinforcement learning workloads

# Build stage
FROM python:3.11-slim as builder

# Install system dependencies for building
RUN apt-get update && apt-get install -y \
    build-essential \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements first for better Docker layer caching
COPY requirements.txt .
COPY pyproject.toml .

# Create virtual environment and install dependencies
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip setuptools wheel
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install --no-cache-dir jax[cpu] flax optax

# Copy source code
COPY pg_neo_graph_rl/ ./pg_neo_graph_rl/
COPY examples/ ./examples/
COPY scripts/ ./scripts/

# Install the package
RUN pip install --no-cache-dir -e .

# Production stage
FROM python:3.11-slim as production

# Install runtime dependencies
RUN apt-get update && apt-get install -y \
    curl \
    procps \
    htop \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user for security
RUN groupadd -r pgneo && useradd -r -g pgneo pgneo

# Copy virtual environment from builder
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Copy application code
COPY --from=builder /app /app
WORKDIR /app

# Create necessary directories
RUN mkdir -p /app/data /app/logs /app/checkpoints /app/config && \
    chown -R pgneo:pgneo /app

# Copy configuration files
COPY deployment/config/ ./config/

# Health check script
COPY deployment/healthcheck.py ./healthcheck.py
RUN chmod +x ./healthcheck.py

# Expose ports
EXPOSE 8080 8081

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python healthcheck.py

# Switch to non-root user
USER pgneo

# Environment variables
ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1
ENV LOG_LEVEL=INFO
ENV COMPONENT_TYPE=agent

# Default command
CMD ["python", "-m", "pg_neo_graph_rl.training.cli", "--config", "/app/config/training_config.yaml"]

# Development stage (for local development)
FROM production as development

USER root

# Install development dependencies
RUN pip install --no-cache-dir \
    pytest \
    pytest-cov \
    black \
    ruff \
    mypy \
    jupyter \
    ipython

# Install debugging tools
RUN apt-get update && apt-get install -y \
    vim \
    less \
    tree \
    && rm -rf /var/lib/apt/lists/*

USER pgneo

# Override default command for development
CMD ["python", "-m", "pg_neo_graph_rl.cli", "demo", "--environment", "traffic"]

# GPU-enabled stage
FROM nvcr.io/nvidia/jax:23.08-py3 as gpu

# Install additional dependencies
RUN pip install --no-cache-dir \
    prometheus-client \
    grafana-api \
    psutil

# Copy application
COPY --from=builder /app /app
WORKDIR /app

# Install the package with GPU support
RUN pip install --no-cache-dir -e ".[gpu]"

# Create user and directories
RUN groupadd -r pgneo && useradd -r -g pgneo pgneo && \
    mkdir -p /app/data /app/logs /app/checkpoints /app/config && \
    chown -R pgneo:pgneo /app

# Copy configuration
COPY deployment/config/ ./config/
COPY deployment/healthcheck.py ./healthcheck.py

EXPOSE 8080 8081

USER pgneo

ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1
ENV JAX_PLATFORM_NAME=gpu
ENV XLA_PYTHON_CLIENT_PREALLOCATE=false

CMD ["python", "-m", "pg_neo_graph_rl.training.cli", "--config", "/app/config/training_config.yaml", "--gpu"]