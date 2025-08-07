#!/bin/bash

# Sentiment Analyzer Pro - Production Setup Script
# This script sets up the complete production environment

set -e

echo "🚀 Setting up Sentiment Analyzer Pro production environment..."
echo "==============================================================="

# Configuration
PROJECT_NAME="sentiment-analyzer-pro"
DEPLOY_DIR="/opt/${PROJECT_NAME}"
LOG_DIR="/var/log/${PROJECT_NAME}"
CONFIG_DIR="/etc/${PROJECT_NAME}"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Logging function
log() {
    echo -e "${GREEN}[$(date +'%Y-%m-%d %H:%M:%S')] $1${NC}"
}

warn() {
    echo -e "${YELLOW}[WARNING] $1${NC}"
}

error() {
    echo -e "${RED}[ERROR] $1${NC}"
    exit 1
}

# Check if running as root
if [[ $EUID -eq 0 ]]; then
   error "This script should not be run as root for security reasons"
fi

# Check system requirements
check_requirements() {
    log "Checking system requirements..."
    
    # Check Docker
    if ! command -v docker &> /dev/null; then
        error "Docker is not installed. Please install Docker first."
    fi
    
    # Check Docker Compose
    if ! command -v docker-compose &> /dev/null; then
        error "Docker Compose is not installed. Please install Docker Compose first."
    fi
    
    # Check available memory
    TOTAL_MEM=$(free -g | awk '/^Mem:/{print $2}')
    if [ "$TOTAL_MEM" -lt 4 ]; then
        warn "Less than 4GB RAM available. Performance may be affected."
    fi
    
    # Check available disk space
    AVAILABLE_SPACE=$(df -h . | awk 'NR==2 {print $4}' | sed 's/G//')
    if [ "${AVAILABLE_SPACE%.*}" -lt 10 ]; then
        warn "Less than 10GB disk space available."
    fi
    
    log "✅ System requirements check completed"
}

# Setup directories
setup_directories() {
    log "Setting up directories..."
    
    sudo mkdir -p ${DEPLOY_DIR}
    sudo mkdir -p ${LOG_DIR}
    sudo mkdir -p ${CONFIG_DIR}
    sudo mkdir -p /opt/${PROJECT_NAME}/{models,cache,nginx,monitoring}
    
    # Set permissions
    sudo chown -R $USER:$USER ${DEPLOY_DIR}
    sudo chown -R $USER:$USER ${LOG_DIR}
    
    log "✅ Directories created successfully"
}

# Copy application files
setup_application() {
    log "Setting up application files..."
    
    # Copy source code
    cp -r . ${DEPLOY_DIR}/
    cd ${DEPLOY_DIR}
    
    # Create environment file
    cat > .env << EOF
# Sentiment Analyzer Pro - Production Configuration
LOG_LEVEL=INFO
MODEL_CACHE_DIR=/app/models
REDIS_URL=redis://redis:6379
PROMETHEUS_URL=http://prometheus:9090
MAX_WORKERS=4
BATCH_SIZE=32
ENABLE_MONITORING=true
ENABLE_CACHING=true
ENABLE_SECURITY=true
EOF
    
    log "✅ Application files setup completed"
}

# Setup Nginx configuration
setup_nginx() {
    log "Setting up Nginx configuration..."
    
    mkdir -p nginx/ssl
    
    # Create Nginx config
    cat > nginx/nginx.conf << 'EOF'
worker_processes auto;
error_log /var/log/nginx/error.log warn;
pid /var/run/nginx.pid;

events {
    worker_connections 1024;
    use epoll;
    multi_accept on;
}

http {
    include /etc/nginx/mime.types;
    default_type application/octet-stream;

    # Logging
    log_format main '$remote_addr - $remote_user [$time_local] "$request" '
                   '$status $body_bytes_sent "$http_referer" '
                   '"$http_user_agent" "$http_x_forwarded_for"';
    
    access_log /var/log/nginx/access.log main;

    # Performance
    sendfile on;
    tcp_nopush on;
    tcp_nodelay on;
    keepalive_timeout 65;
    types_hash_max_size 2048;
    client_max_body_size 10M;

    # Gzip compression
    gzip on;
    gzip_vary on;
    gzip_min_length 1000;
    gzip_types text/plain text/css application/json application/javascript text/xml application/xml application/xml+rss text/javascript;

    # Rate limiting
    limit_req_zone $binary_remote_addr zone=api:10m rate=10r/s;
    limit_conn_zone $binary_remote_addr zone=addr:10m;

    # Upstream
    upstream sentiment_api {
        server sentiment-analyzer:8000;
        keepalive 32;
    }

    # HTTP to HTTPS redirect
    server {
        listen 80;
        server_name _;
        return 301 https://$server_name$request_uri;
    }

    # Main server block
    server {
        listen 443 ssl http2;
        server_name _;

        # SSL configuration (replace with your certificates)
        ssl_certificate /etc/nginx/ssl/cert.pem;
        ssl_certificate_key /etc/nginx/ssl/key.pem;
        ssl_protocols TLSv1.2 TLSv1.3;
        ssl_ciphers ECDHE-RSA-AES128-GCM-SHA256:ECDHE-RSA-AES256-GCM-SHA384;
        ssl_prefer_server_ciphers off;

        # Security headers
        add_header X-Frame-Options DENY;
        add_header X-Content-Type-Options nosniff;
        add_header X-XSS-Protection "1; mode=block";
        add_header Strict-Transport-Security "max-age=31536000; includeSubDomains";

        # Rate limiting
        limit_req zone=api burst=20 nodelay;
        limit_conn addr 10;

        # API proxy
        location /api/ {
            proxy_pass http://sentiment_api/;
            proxy_set_header Host $host;
            proxy_set_header X-Real-IP $remote_addr;
            proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
            proxy_set_header X-Forwarded-Proto $scheme;
            proxy_connect_timeout 30s;
            proxy_send_timeout 30s;
            proxy_read_timeout 30s;
        }

        # Health check
        location /health {
            proxy_pass http://sentiment_api/health;
            access_log off;
        }

        # Documentation
        location /docs {
            proxy_pass http://sentiment_api/docs;
        }

        # Static files (if any)
        location /static/ {
            alias /var/www/static/;
            expires 30d;
            add_header Cache-Control "public, no-transform";
        }

        # Default location
        location / {
            return 404;
        }
    }
}
EOF
    
    log "✅ Nginx configuration created"
}

# Setup monitoring configuration
setup_monitoring() {
    log "Setting up monitoring configuration..."
    
    mkdir -p monitoring/grafana/{provisioning/datasources,provisioning/dashboards,dashboards}
    
    # Prometheus configuration
    cat > monitoring/prometheus.yml << 'EOF'
global:
  scrape_interval: 15s
  evaluation_interval: 15s

rule_files: []

scrape_configs:
  - job_name: 'sentiment-analyzer'
    static_configs:
      - targets: ['sentiment-analyzer:8000']
    metrics_path: /metrics
    scrape_interval: 30s

  - job_name: 'node-exporter'
    static_configs:
      - targets: ['node-exporter:9100']

  - job_name: 'redis'
    static_configs:
      - targets: ['redis:6379']

  - job_name: 'nginx'
    static_configs:
      - targets: ['nginx:80']
EOF

    # Grafana datasource
    cat > monitoring/grafana/provisioning/datasources/prometheus.yml << 'EOF'
apiVersion: 1

datasources:
  - name: Prometheus
    type: prometheus
    access: proxy
    url: http://prometheus:9090
    isDefault: true
EOF

    # Grafana dashboard provisioning
    cat > monitoring/grafana/provisioning/dashboards/dashboard.yml << 'EOF'
apiVersion: 1

providers:
  - name: 'default'
    orgId: 1
    folder: ''
    type: file
    disableDeletion: false
    editable: true
    options:
      path: /var/lib/grafana/dashboards
EOF
    
    log "✅ Monitoring configuration created"
}

# Generate SSL certificates (self-signed for demo)
setup_ssl() {
    log "Setting up SSL certificates..."
    
    if [ ! -f nginx/ssl/cert.pem ]; then
        openssl req -x509 -nodes -days 365 -newkey rsa:2048 \
            -keyout nginx/ssl/key.pem \
            -out nginx/ssl/cert.pem \
            -subj "/C=US/ST=State/L=City/O=Organization/CN=localhost"
        
        log "✅ Self-signed SSL certificates generated"
        warn "Replace with proper SSL certificates for production use"
    else
        log "✅ SSL certificates already exist"
    fi
}

# Build and start services
deploy_services() {
    log "Building and starting services..."
    
    # Build images
    docker-compose -f docker-compose.production.yml build --no-cache
    
    # Start services
    docker-compose -f docker-compose.production.yml up -d
    
    # Wait for services to be ready
    log "Waiting for services to be ready..."
    sleep 30
    
    # Check health
    if curl -f http://localhost:8000/health &> /dev/null; then
        log "✅ API service is healthy"
    else
        error "API service health check failed"
    fi
    
    log "✅ Services deployed successfully"
}

# Setup log rotation
setup_log_rotation() {
    log "Setting up log rotation..."
    
    sudo tee /etc/logrotate.d/${PROJECT_NAME} > /dev/null << EOF
${LOG_DIR}/*.log {
    daily
    missingok
    rotate 30
    compress
    delaycompress
    notifempty
    create 644 $USER $USER
    postrotate
        docker-compose -f ${DEPLOY_DIR}/docker-compose.production.yml restart sentiment-analyzer
    endscript
}
EOF
    
    log "✅ Log rotation configured"
}

# Setup systemd service (optional)
setup_systemd() {
    log "Setting up systemd service..."
    
    sudo tee /etc/systemd/system/${PROJECT_NAME}.service > /dev/null << EOF
[Unit]
Description=Sentiment Analyzer Pro
Requires=docker.service
After=docker.service

[Service]
Type=oneshot
RemainAfterExit=yes
WorkingDirectory=${DEPLOY_DIR}
ExecStart=/usr/local/bin/docker-compose -f docker-compose.production.yml up -d
ExecStop=/usr/local/bin/docker-compose -f docker-compose.production.yml down
TimeoutStartSec=0
User=$USER
Group=$USER

[Install]
WantedBy=multi-user.target
EOF
    
    sudo systemctl daemon-reload
    sudo systemctl enable ${PROJECT_NAME}
    
    log "✅ Systemd service configured"
}

# Display deployment information
show_deployment_info() {
    log "Deployment completed successfully! 🎉"
    echo
    echo "==============================================================="
    echo "🚀 Sentiment Analyzer Pro is now running in production mode"
    echo "==============================================================="
    echo
    echo "📡 API Endpoints:"
    echo "  • Health Check: https://localhost/health"
    echo "  • API Documentation: https://localhost/docs"
    echo "  • Sentiment Analysis: https://localhost/api/analyze"
    echo
    echo "📊 Monitoring:"
    echo "  • Grafana Dashboard: http://localhost:3000 (admin/admin)"
    echo "  • Prometheus: http://localhost:9090"
    echo
    echo "🔧 Management:"
    echo "  • View logs: docker-compose -f docker-compose.production.yml logs -f"
    echo "  • Stop services: docker-compose -f docker-compose.production.yml down"
    echo "  • Restart services: docker-compose -f docker-compose.production.yml restart"
    echo
    echo "📁 Important Directories:"
    echo "  • Application: ${DEPLOY_DIR}"
    echo "  • Logs: ${LOG_DIR}"
    echo "  • Configuration: ${CONFIG_DIR}"
    echo
    echo "⚠️  Security Notes:"
    echo "  • Replace self-signed SSL certificates with proper ones"
    echo "  • Change default Grafana password"
    echo "  • Review and adjust rate limiting settings"
    echo "  • Configure firewall rules"
    echo
    echo "==============================================================="
}

# Main execution
main() {
    log "Starting production deployment..."
    
    check_requirements
    setup_directories
    setup_application
    setup_nginx
    setup_monitoring
    setup_ssl
    deploy_services
    setup_log_rotation
    setup_systemd
    
    show_deployment_info
}

# Run main function
main "$@"