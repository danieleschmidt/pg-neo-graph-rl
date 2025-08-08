#!/bin/bash

# Production deployment setup script for pg-neo-graph-rl
# This script sets up the production environment with all necessary components

set -e  # Exit on any error

echo "🚀 Starting pg-neo-graph-rl Production Deployment"
echo "=================================================="

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
DEPLOYMENT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$DEPLOYMENT_DIR")"
DATA_DIR="$PROJECT_ROOT/data"
LOGS_DIR="$PROJECT_ROOT/logs"
BACKUP_DIR="$PROJECT_ROOT/backups"

# Functions
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

check_requirements() {
    log_info "Checking system requirements..."
    
    # Check Docker
    if ! command -v docker &> /dev/null; then
        log_error "Docker is not installed. Please install Docker first."
        exit 1
    fi
    
    # Check Docker Compose
    if ! command -v docker-compose &> /dev/null; then
        log_error "Docker Compose is not installed. Please install Docker Compose first."
        exit 1
    fi
    
    # Check available memory
    AVAILABLE_MEMORY=$(free -m | awk 'NR==2{printf "%.0f", $7}')
    if [ "$AVAILABLE_MEMORY" -lt 4096 ]; then
        log_warning "Available memory is less than 4GB. Recommended: 8GB+"
    fi
    
    # Check available disk space
    AVAILABLE_DISK=$(df -BG "$PROJECT_ROOT" | awk 'NR==2{print $4}' | sed 's/G//')
    if [ "$AVAILABLE_DISK" -lt 20 ]; then
        log_warning "Available disk space is less than 20GB. Recommended: 50GB+"
    fi
    
    log_success "System requirements check completed"
}

create_directories() {
    log_info "Creating necessary directories..."
    
    mkdir -p "$DATA_DIR"/{models,metrics,checkpoints}
    mkdir -p "$LOGS_DIR"/{app,nginx,monitoring}
    mkdir -p "$BACKUP_DIR"
    mkdir -p "$PROJECT_ROOT/deployment/ssl"
    
    # Set proper permissions
    chmod 755 "$DATA_DIR" "$LOGS_DIR" "$BACKUP_DIR"
    
    log_success "Directories created successfully"
}

setup_ssl() {
    log_info "Setting up SSL certificates..."
    
    SSL_DIR="$PROJECT_ROOT/deployment/ssl"
    
    if [ ! -f "$SSL_DIR/server.crt" ] || [ ! -f "$SSL_DIR/server.key" ]; then
        log_info "Generating self-signed SSL certificate..."
        
        openssl req -x509 -nodes -days 365 -newkey rsa:2048 \
            -keyout "$SSL_DIR/server.key" \
            -out "$SSL_DIR/server.crt" \
            -subj "/C=US/ST=State/L=City/O=Organization/CN=pg-neo-graph-rl"
        
        log_success "Self-signed SSL certificate generated"
    else
        log_info "SSL certificates already exist"
    fi
}

setup_environment() {
    log_info "Setting up environment configuration..."
    
    # Create production environment file
    cat > "$PROJECT_ROOT/.env.production" << EOF
# pg-neo-graph-rl Production Configuration
ENVIRONMENT=production
LOG_LEVEL=INFO
DEBUG=false

# Application Settings
MAX_AGENTS=100
CACHE_SIZE=1000
BATCH_SIZE=32
LEARNING_RATE=3e-4

# Monitoring
ENABLE_MONITORING=true
PROMETHEUS_PORT=9090
GRAFANA_PORT=3000

# Security
SESSION_SECRET=$(openssl rand -hex 32)
JWT_SECRET=$(openssl rand -hex 32)

# Database
REDIS_URL=redis://redis:6379

# Resource Limits
MAX_MEMORY_USAGE=0.85
GC_THRESHOLD=0.75
EMERGENCY_THRESHOLD=0.95

# Auto-scaling
MIN_AGENTS=1
MAX_AGENTS=100
SCALING_COOLDOWN=60

# Networking
API_PORT=8080
WORKER_PROCESSES=4
EOF

    log_success "Environment configuration created"
}

build_images() {
    log_info "Building Docker images..."
    
    cd "$PROJECT_ROOT"
    
    # Build main application image
    docker build -t pg-neo-graph-rl:production .
    
    if [ $? -eq 0 ]; then
        log_success "Docker images built successfully"
    else
        log_error "Failed to build Docker images"
        exit 1
    fi
}

deploy_services() {
    log_info "Deploying services..."
    
    cd "$DEPLOYMENT_DIR"
    
    # Stop any existing services
    docker-compose -f docker-compose.production.yml down
    
    # Deploy production services
    docker-compose -f docker-compose.production.yml up -d
    
    if [ $? -eq 0 ]; then
        log_success "Services deployed successfully"
    else
        log_error "Failed to deploy services"
        exit 1
    fi
    
    # Wait for services to be healthy
    log_info "Waiting for services to become healthy..."
    sleep 30
    
    # Check service health
    check_service_health
}

check_service_health() {
    log_info "Checking service health..."
    
    SERVICES=("pg-neo-app" "redis" "prometheus" "grafana" "nginx")
    
    for service in "${SERVICES[@]}"; do
        if docker-compose -f docker-compose.production.yml ps | grep -q "$service.*Up"; then
            log_success "$service is running"
        else
            log_error "$service is not running properly"
        fi
    done
}

setup_monitoring() {
    log_info "Setting up monitoring dashboards..."
    
    # Wait for Grafana to be ready
    timeout=60
    while [ $timeout -gt 0 ]; do
        if curl -s http://localhost:3000/api/health > /dev/null 2>&1; then
            log_success "Grafana is ready"
            break
        fi
        sleep 2
        ((timeout--))
    done
    
    if [ $timeout -eq 0 ]; then
        log_warning "Grafana may not be fully ready"
    fi
    
    log_info "Monitoring setup completed"
}

create_backup_script() {
    log_info "Creating backup script..."
    
    cat > "$PROJECT_ROOT/scripts/backup.sh" << 'EOF'
#!/bin/bash

# Backup script for pg-neo-graph-rl
BACKUP_DIR="/app/backups"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

echo "Starting backup at $TIMESTAMP"

# Create backup directory
mkdir -p "$BACKUP_DIR/$TIMESTAMP"

# Backup data
cp -r /app/data/* "$BACKUP_DIR/$TIMESTAMP/"

# Backup Redis data
docker exec pg-neo-redis redis-cli BGSAVE
sleep 5
docker cp pg-neo-redis:/data/dump.rdb "$BACKUP_DIR/$TIMESTAMP/redis_dump.rdb"

# Backup logs (last 7 days)
find /app/logs -name "*.log" -mtime -7 -exec cp {} "$BACKUP_DIR/$TIMESTAMP/" \;

# Compress backup
tar -czf "$BACKUP_DIR/backup_$TIMESTAMP.tar.gz" -C "$BACKUP_DIR" "$TIMESTAMP"
rm -rf "$BACKUP_DIR/$TIMESTAMP"

echo "Backup completed: backup_$TIMESTAMP.tar.gz"

# Clean old backups (keep last 30 days)
find "$BACKUP_DIR" -name "backup_*.tar.gz" -mtime +30 -delete

echo "Old backups cleaned"
EOF

    chmod +x "$PROJECT_ROOT/scripts/backup.sh"
    
    log_success "Backup script created"
}

setup_monitoring_alerts() {
    log_info "Setting up monitoring alerts..."
    
    # This would configure alerting rules
    # For now, we'll just create a placeholder
    
    cat > "$PROJECT_ROOT/alerts/rules.yml" << 'EOF'
groups:
  - name: pg-neo-graph-rl
    rules:
      - alert: HighMemoryUsage
        expr: memory_usage_percent > 85
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "High memory usage detected"
          description: "Memory usage is above 85% for more than 5 minutes"
      
      - alert: SlowTraining
        expr: avg_training_time > 30
        for: 10m
        labels:
          severity: warning
        annotations:
          summary: "Training performance degraded"
          description: "Average training time is above 30 seconds"
EOF

    log_success "Monitoring alerts configured"
}

print_deployment_info() {
    log_info "Deployment completed successfully!"
    echo ""
    echo "🌍 Access URLs:"
    echo "  • Main Application: http://localhost:8080"
    echo "  • Grafana Dashboard: http://localhost:3000 (admin/pg_neo_admin_2025)"
    echo "  • Prometheus: http://localhost:9090" 
    echo "  • Redis: localhost:6379"
    echo ""
    echo "📁 Important Directories:"
    echo "  • Data: $DATA_DIR"
    echo "  • Logs: $LOGS_DIR"
    echo "  • Backups: $BACKUP_DIR"
    echo ""
    echo "🔧 Management Commands:"
    echo "  • View logs: docker-compose -f deployment/docker-compose.production.yml logs -f"
    echo "  • Restart services: docker-compose -f deployment/docker-compose.production.yml restart"
    echo "  • Stop services: docker-compose -f deployment/docker-compose.production.yml down"
    echo "  • Backup data: ./scripts/backup.sh"
    echo ""
    echo "📊 Monitor system health through Grafana dashboard"
    echo "🔔 Check alerts in AlertManager: http://localhost:9093"
    echo ""
    log_success "pg-neo-graph-rl is ready for production use!"
}

# Main execution
main() {
    log_info "Starting production deployment process..."
    
    check_requirements
    create_directories
    setup_ssl
    setup_environment
    build_images
    deploy_services
    setup_monitoring
    create_backup_script
    setup_monitoring_alerts
    print_deployment_info
    
    log_success "Production deployment completed successfully!"
}

# Run main function
main "$@"