FROM python:3.11-slim

LABEL maintainer="Daniel Schmidt <daniel@terragon.ai>"
LABEL description="pg-neo-graph-rl: Federated Graph RL toolkit"

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user
RUN useradd --create-home --shell /bin/bash app
USER app
WORKDIR /home/app

# Copy requirements first for better caching
COPY --chown=app:app pyproject.toml ./

# Install Python dependencies
RUN pip install --user --upgrade pip setuptools wheel && \
    pip install --user -e .

# Copy application code
COPY --chown=app:app . .

# Install the package
RUN pip install --user -e .

# Expose port for Jupyter/web interfaces
EXPOSE 8888

# Default command
CMD ["python", "-m", "pg_neo_graph_rl.cli"]