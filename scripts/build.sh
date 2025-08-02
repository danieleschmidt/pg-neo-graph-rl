#!/bin/bash
# Build script for PG-Neo-Graph-RL

set -e  # Exit on any error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Default values
TARGET="development"
TAG="latest"
REGISTRY=""
PUSH=false
BUILD_ARGS=""
PLATFORM=""
CACHE=true
VERBOSE=false

# Help function
show_help() {
    cat << EOF
Usage: $0 [OPTIONS]

Build Docker images for PG-Neo-Graph-RL

Options:
    -t, --target TARGET      Build target (development|production|gpu) [default: development]
    -g, --tag TAG           Docker image tag [default: latest]
    -r, --registry REGISTRY Docker registry URL
    -p, --push              Push image to registry after build
    -b, --build-arg ARG     Pass build argument (can be used multiple times)
    --platform PLATFORM     Target platform (e.g., linux/amd64,linux/arm64)
    --no-cache              Disable Docker build cache
    -v, --verbose           Verbose output
    -h, --help              Show this help message

Examples:
    $0 --target production --tag v1.0.0
    $0 --target gpu --push --registry myregistry.com
    $0 --build-arg PYTHON_VERSION=3.10 --platform linux/amd64
EOF
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -t|--target)
            TARGET="$2"
            shift 2
            ;;
        -g|--tag)
            TAG="$2"
            shift 2
            ;;
        -r|--registry)
            REGISTRY="$2"
            shift 2
            ;;
        -p|--push)
            PUSH=true
            shift
            ;;
        -b|--build-arg)
            BUILD_ARGS="$BUILD_ARGS --build-arg $2"
            shift 2
            ;;
        --platform)
            PLATFORM="--platform $2"
            shift 2
            ;;
        --no-cache)
            CACHE=false
            shift
            ;;
        -v|--verbose)
            VERBOSE=true
            shift
            ;;
        -h|--help)
            show_help
            exit 0
            ;;
        *)
            echo -e "${RED}Unknown option: $1${NC}"
            show_help
            exit 1
            ;;
    esac
done

# Validate target
if [[ ! "$TARGET" =~ ^(development|production|gpu)$ ]]; then
    echo -e "${RED}Invalid target: $TARGET. Must be one of: development, production, gpu${NC}"
    exit 1
fi

# Set image name
IMAGE_NAME="pg-neo-graph-rl"
if [[ -n "$REGISTRY" ]]; then
    FULL_IMAGE_NAME="$REGISTRY/$IMAGE_NAME:$TAG"
else
    FULL_IMAGE_NAME="$IMAGE_NAME:$TAG"
fi

# Build command construction
BUILD_CMD="docker build"

if [[ "$CACHE" == "false" ]]; then
    BUILD_CMD="$BUILD_CMD --no-cache"
fi

if [[ -n "$PLATFORM" ]]; then
    BUILD_CMD="$BUILD_CMD $PLATFORM"
fi

if [[ -n "$BUILD_ARGS" ]]; then
    BUILD_CMD="$BUILD_CMD $BUILD_ARGS"
fi

BUILD_CMD="$BUILD_CMD --target $TARGET --tag $FULL_IMAGE_NAME ."

# Logging function
log() {
    echo -e "${BLUE}[$(date +'%Y-%m-%d %H:%M:%S')] $1${NC}"
}

log_success() {
    echo -e "${GREEN}[$(date +'%Y-%m-%d %H:%M:%S')] $1${NC}"
}

log_warning() {
    echo -e "${YELLOW}[$(date +'%Y-%m-%d %H:%M:%S')] $1${NC}"
}

log_error() {
    echo -e "${RED}[$(date +'%Y-%m-%d %H:%M:%S')] $1${NC}"
}

# Pre-build checks
log "Starting build process..."
log "Target: $TARGET"
log "Image: $FULL_IMAGE_NAME"
log "Build command: $BUILD_CMD"

# Check if Docker is running
if ! docker info > /dev/null 2>&1; then
    log_error "Docker is not running. Please start Docker and try again."
    exit 1
fi

# Check if Dockerfile exists
if [[ ! -f "Dockerfile" ]]; then
    log_error "Dockerfile not found in current directory."
    exit 1
fi

# GPU-specific checks
if [[ "$TARGET" == "gpu" ]]; then
    log_warning "Building GPU image. Make sure NVIDIA Docker is installed."
    if ! docker run --rm --gpus all nvidia/cuda:11.8-base-ubuntu20.04 nvidia-smi > /dev/null 2>&1; then
        log_warning "GPU not available or NVIDIA Docker not properly configured."
        log_warning "The GPU image will still build but may not work at runtime."
    fi
fi

# Build the image
log "Building Docker image..."
if [[ "$VERBOSE" == "true" ]]; then
    eval $BUILD_CMD
else
    eval $BUILD_CMD > /dev/null 2>&1
fi

if [[ $? -eq 0 ]]; then
    log_success "Build completed successfully!"
else
    log_error "Build failed!"
    exit 1
fi

# Run basic tests on the image
log "Running basic image tests..."
if docker run --rm $FULL_IMAGE_NAME python -c "import pg_neo_graph_rl; print('Import successful')" > /dev/null 2>&1; then
    log_success "Basic import test passed"
else
    log_error "Basic import test failed"
    exit 1
fi

# Push if requested
if [[ "$PUSH" == "true" ]]; then
    if [[ -z "$REGISTRY" ]]; then
        log_error "Cannot push without registry. Use --registry option."
        exit 1
    fi
    
    log "Pushing image to registry..."
    if docker push $FULL_IMAGE_NAME; then
        log_success "Image pushed successfully to $REGISTRY"
    else
        log_error "Failed to push image"
        exit 1
    fi
fi

# Show image information
log "Image build summary:"
echo "  Image Name: $FULL_IMAGE_NAME"
echo "  Target: $TARGET"
echo "  Size: $(docker images $FULL_IMAGE_NAME --format 'table {{.Size}}' | tail -n 1)"
echo "  Created: $(docker images $FULL_IMAGE_NAME --format 'table {{.CreatedAt}}' | tail -n 1)"

# Show next steps
log_success "Build completed! Next steps:"
echo "  • Run development: docker run -it --rm $FULL_IMAGE_NAME bash"
echo "  • Run tests: docker run --rm $FULL_IMAGE_NAME pytest"
if [[ "$TARGET" == "gpu" ]]; then
    echo "  • Run with GPU: docker run --rm --gpus all $FULL_IMAGE_NAME"
fi
echo "  • Start services: docker-compose up"

log_success "Done!"
