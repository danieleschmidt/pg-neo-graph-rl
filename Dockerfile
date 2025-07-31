# Multi-stage build for pg-neo-graph-rl
FROM python:3.9-slim as base

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONHASHSEED=random \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user
RUN useradd --create-home --shell /bin/bash app

# Development stage
FROM base as development

# Install development dependencies
RUN apt-get update && apt-get install -y \
    graphviz \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
USER app

# Copy requirements first for better caching
COPY --chown=app:app pyproject.toml ./
RUN pip install -e ".[dev,monitoring,benchmarks]"

# Copy source code
COPY --chown=app:app . .

# Install package in development mode
RUN pip install -e .

CMD ["python", "-m", "pytest"]

# Production stage
FROM base as production

WORKDIR /app
USER app

# Copy only necessary files
COPY --chown=app:app pyproject.toml README.md LICENSE ./
COPY --chown=app:app pg_neo_graph_rl/ ./pg_neo_graph_rl/

# Install production dependencies only
RUN pip install -e .

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD python -c "import pg_neo_graph_rl; print('OK')" || exit 1

# Default command
CMD ["python", "-c", "import pg_neo_graph_rl; print('pg-neo-graph-rl ready')"]

# GPU-enabled stage
FROM nvidia/cuda:11.8-devel-ubuntu20.04 as gpu

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    DEBIAN_FRONTEND=noninteractive

# Install Python and system dependencies
RUN apt-get update && apt-get install -y \
    python3.9 \
    python3.9-dev \
    python3.9-distutils \
    python3-pip \
    build-essential \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Create symlinks
RUN ln -s /usr/bin/python3.9 /usr/bin/python

# Create non-root user
RUN useradd --create-home --shell /bin/bash app

WORKDIR /app
USER app

# Copy and install package
COPY --chown=app:app pyproject.toml README.md LICENSE ./
COPY --chown=app:app pg_neo_graph_rl/ ./pg_neo_graph_rl/

# Install with GPU support
RUN pip install -e ".[gpu]"

# Health check for GPU
HEALTHCHECK --interval=30s --timeout=30s --start-period=10s --retries=3 \
    CMD python -c "import jax; print(f'JAX devices: {jax.devices()}'); assert len(jax.devices()) > 0" || exit 1

CMD ["python", "-c", "import jax; print(f'JAX devices: {jax.devices()}')"]