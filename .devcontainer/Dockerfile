# Development container for PG-Neo-Graph-RL
FROM mcr.microsoft.com/devcontainers/python:3.11-bullseye

# Install system dependencies
RUN apt-get update && export DEBIAN_FRONTEND=noninteractive \
    && apt-get -y install --no-install-recommends \
        build-essential \
        git \
        curl \
        wget \
        vim \
        htop \
        tree \
        graphviz \
        graphviz-dev \
        pkg-config \
    && apt-get autoremove -y && apt-get clean -y && rm -rf /var/lib/apt/lists/*

# Install CUDA development tools (for GPU support)
RUN curl -fsSL https://developer.download.nvidia.com/compute/cuda/repos/debian11/x86_64/3bf863cc.pub | apt-key add - \
    && echo "deb https://developer.download.nvidia.com/compute/cuda/repos/debian11/x86_64 /" > /etc/apt/sources.list.d/cuda.list \
    && apt-get update \
    && apt-get install -y --no-install-recommends cuda-toolkit-12-2 \
    && apt-get autoremove -y && apt-get clean -y && rm -rf /var/lib/apt/lists/* || true

# Set up environment variables
ENV PYTHONPATH=/workspaces/pg-neo-graph-rl:$PYTHONPATH
ENV CUDA_ROOT=/usr/local/cuda
ENV PATH=$CUDA_ROOT/bin:$PATH
ENV LD_LIBRARY_PATH=$CUDA_ROOT/lib64:$LD_LIBRARY_PATH

# Install Python development tools
RUN pip install --upgrade pip setuptools wheel

# Create workspace directory
WORKDIR /workspaces/pg-neo-graph-rl

# Copy requirements for caching
COPY requirements.txt pyproject.toml ./
RUN pip install -e ".[dev,monitoring,benchmarks]"

# Install additional development tools
RUN pip install \
    jupyterlab \
    tensorboard \
    wandb \
    pre-commit \
    adr-tools

# Set up git configuration
RUN git config --global --add safe.directory /workspaces/pg-neo-graph-rl

# Expose ports for development services
EXPOSE 3000 8888 6006 9090