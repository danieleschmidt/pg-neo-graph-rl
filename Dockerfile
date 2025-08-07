# Sentiment Analyzer Pro - Production Dockerfile

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
COPY --chown=app:app sentiment_analyzer_pro/ ./sentiment_analyzer_pro/
COPY --chown=app:app requirements.txt ./

# Install production dependencies
RUN pip install -r requirements.txt && pip install -e .

# Create necessary directories
RUN mkdir -p /app/models /app/logs /app/cache

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Expose port
EXPOSE 8000

# Default command - API server
CMD ["uvicorn", "sentiment_analyzer_pro.api:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]

# GPU-enabled stage
FROM nvidia/cuda:11.8-runtime-ubuntu20.04 as gpu

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
COPY --chown=app:app sentiment_analyzer_pro/ ./sentiment_analyzer_pro/
COPY --chown=app:app requirements.txt ./

# Install with GPU support
RUN pip install -r requirements.txt && pip install -e ".[gpu]"

# Create directories
RUN mkdir -p /app/models /app/logs /app/cache

# Health check for GPU
HEALTHCHECK --interval=30s --timeout=10s --start-period=10s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

EXPOSE 8000

CMD ["uvicorn", "sentiment_analyzer_pro.api:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "2"]