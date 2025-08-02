#!/bin/bash
# Docker utilities for PG-Neo-Graph-RL

set -e

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

# Configuration
IMAGE_NAME="pg-neo-graph-rl"
COMPOSE_FILE="docker-compose.yml"

# Logging functions
log() { echo -e "${BLUE}[$(date +'%H:%M:%S')] $1${NC}"; }
log_success() { echo -e "${GREEN}[$(date +'%H:%M:%S')] $1${NC}"; }
log_warning() { echo -e "${YELLOW}[$(date +'%H:%M:%S')] $1${NC}"; }
log_error() { echo -e "${RED}[$(date +'%H:%M:%S')] $1${NC}"; }

# Help function
show_help() {
    cat << EOF
Docker utilities for PG-Neo-Graph-RL

Usage: $0 COMMAND [OPTIONS]

Commands:
    setup           Initial setup of Docker environment
    start           Start all services
    stop            Stop all services
    restart         Restart all services
    status          Show status of all services
    logs            Show logs from services
    clean           Clean up containers, images, and volumes
    test            Run tests in container
    shell           Open shell in development container
    jupyter         Start Jupyter lab server
    monitoring      Start monitoring stack (Prometheus + Grafana)
    build           Build all images
    health          Check health of all services
    backup          Backup volumes and data
    restore         Restore from backup

Options:
    -f, --follow    Follow logs (for logs command)
    -v, --verbose   Verbose output
    -h, --help      Show this help

Examples:
    $0 setup
    $0 start
    $0 logs -f
    $0 shell
    $0 clean
EOF
}

# Setup function
setup() {
    log "Setting up Docker environment..."
    
    # Check if Docker is running
    if ! docker info > /dev/null 2>&1; then
        log_error "Docker is not running. Please start Docker first."
        exit 1
    fi
    
    # Check if docker-compose is available
    if ! command -v docker-compose > /dev/null 2>&1; then
        log_error "docker-compose not found. Please install docker-compose."
        exit 1
    fi
    
    # Create necessary directories
    mkdir -p docker/grafana/provisioning/{dashboards,datasources}
    mkdir -p logs
    mkdir -p data
    
    # Build images
    log "Building Docker images..."
    docker-compose build --parallel
    
    # Pull external images
    log "Pulling external images..."
    docker-compose pull
    
    log_success "Docker environment setup complete!"
}

# Start services
start() {
    log "Starting services..."
    docker-compose up -d
    
    # Wait for services to be healthy
    log "Waiting for services to start..."
    sleep 10
    
    # Show status
    status
    
    log_success "Services started!"
    log "Access points:"
    echo "  • Jupyter Lab: http://localhost:8888"
    echo "  • Grafana: http://localhost:3000 (admin/admin)"
    echo "  • Prometheus: http://localhost:9090"
}

# Stop services
stop() {
    log "Stopping services..."
    docker-compose down
    log_success "Services stopped!"
}

# Restart services
restart() {
    log "Restarting services..."
    docker-compose restart
    log_success "Services restarted!"
}

# Show status
status() {
    log "Service status:"
    docker-compose ps
}

# Show logs
show_logs() {
    local follow_flag=""
    if [[ "$1" == "-f" ]] || [[ "$1" == "--follow" ]]; then
        follow_flag="-f"
    fi
    
    docker-compose logs $follow_flag
}

# Clean up
clean() {
    log_warning "This will remove all containers, images, and volumes. Are you sure? (y/N)"
    read -r response
    if [[ "$response" =~ ^[Yy]$ ]]; then
        log "Cleaning up..."
        
        # Stop and remove containers
        docker-compose down -v
        
        # Remove images
        docker rmi $(docker images "$IMAGE_NAME*" -q) 2>/dev/null || true
        
        # Clean up Docker system
        docker system prune -f
        
        log_success "Cleanup complete!"
    else
        log "Cleanup cancelled."
    fi
}

# Run tests
run_tests() {
    log "Running tests in container..."
    docker-compose run --rm test
}

# Open shell
open_shell() {
    log "Opening shell in development container..."
    docker-compose run --rm dev bash
}

# Start Jupyter
start_jupyter() {
    log "Starting Jupyter Lab..."
    docker-compose up -d jupyter
    
    sleep 5
    log_success "Jupyter Lab started!"
    log "Access at: http://localhost:8888"
}

# Start monitoring
start_monitoring() {
    log "Starting monitoring stack..."
    docker-compose up -d prometheus grafana
    
    sleep 10
    log_success "Monitoring stack started!"
    log "Grafana: http://localhost:3000 (admin/admin)"
    log "Prometheus: http://localhost:9090"
}

# Build all images
build_all() {
    log "Building all images..."
    docker-compose build --parallel
    log_success "All images built!"
}

# Health check
health_check() {
    log "Checking service health..."
    
    services=("dev" "test" "prod" "jupyter")
    
    for service in "${services[@]}"; do
        if docker-compose ps | grep -q "$service.*Up"; then
            log_success "$service: Healthy"
        else
            log_warning "$service: Not running"
        fi
    done
    
    # Check external services
    if curl -s http://localhost:3000 > /dev/null; then
        log_success "Grafana: Accessible"
    else
        log_warning "Grafana: Not accessible"
    fi
    
    if curl -s http://localhost:9090 > /dev/null; then
        log_success "Prometheus: Accessible"
    else
        log_warning "Prometheus: Not accessible"
    fi
}

# Backup
backup() {
    local backup_dir="backup/$(date +%Y%m%d_%H%M%S)"
    mkdir -p "$backup_dir"
    
    log "Creating backup in $backup_dir..."
    
    # Backup volumes
    docker run --rm -v pg-neo-graph-rl_postgres-data:/data -v "$(pwd)/$backup_dir:/backup" alpine tar czf /backup/postgres-data.tar.gz -C /data .
    docker run --rm -v pg-neo-graph-rl_grafana-data:/data -v "$(pwd)/$backup_dir:/backup" alpine tar czf /backup/grafana-data.tar.gz -C /data .
    
    # Backup configuration
    cp -r docker "$backup_dir/"
    cp docker-compose.yml "$backup_dir/"
    
    log_success "Backup created in $backup_dir"
}

# Restore
restore() {
    local backup_dir="$1"
    
    if [[ -z "$backup_dir" ]]; then
        log_error "Please specify backup directory"
        exit 1
    fi
    
    if [[ ! -d "$backup_dir" ]]; then
        log_error "Backup directory not found: $backup_dir"
        exit 1
    fi
    
    log_warning "This will overwrite current data. Are you sure? (y/N)"
    read -r response
    if [[ "$response" =~ ^[Yy]$ ]]; then
        log "Restoring from $backup_dir..."
        
        # Stop services
        docker-compose down
        
        # Restore volumes
        docker run --rm -v pg-neo-graph-rl_postgres-data:/data -v "$(pwd)/$backup_dir:/backup" alpine tar xzf /backup/postgres-data.tar.gz -C /data
        docker run --rm -v pg-neo-graph-rl_grafana-data:/data -v "$(pwd)/$backup_dir:/backup" alpine tar xzf /backup/grafana-data.tar.gz -C /data
        
        log_success "Restore complete!"
        log "You can now start the services with: $0 start"
    else
        log "Restore cancelled."
    fi
}

# Main command processing
case "$1" in
    setup)
        setup
        ;;
    start)
        start
        ;;
    stop)
        stop
        ;;
    restart)
        restart
        ;;
    status)
        status
        ;;
    logs)
        shift
        show_logs "$@"
        ;;
    clean)
        clean
        ;;
    test)
        run_tests
        ;;
    shell)
        open_shell
        ;;
    jupyter)
        start_jupyter
        ;;
    monitoring)
        start_monitoring
        ;;
    build)
        build_all
        ;;
    health)
        health_check
        ;;
    backup)
        backup
        ;;
    restore)
        restore "$2"
        ;;
    -h|--help)
        show_help
        ;;
    "")
        log_error "No command specified"
        show_help
        exit 1
        ;;
    *)
        log_error "Unknown command: $1"
        show_help
        exit 1
        ;;
esac
