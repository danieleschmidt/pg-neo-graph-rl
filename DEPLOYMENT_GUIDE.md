# 🚀 pg-neo-graph-rl Production Deployment Guide

## Overview

This guide covers the complete production deployment of **pg-neo-graph-rl**, a federated graph-neural reinforcement learning toolkit for city-scale infrastructure control.

## 🏗️ Architecture

### System Components

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   NGINX Proxy   │    │  Main App       │    │   Monitoring    │
│   Load Balancer │────┤  pg-neo-graph-rl│────┤   Stack         │
│   SSL/TLS       │    │  Port: 8080     │    │   Grafana/Prom  │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                       │                       │
         │                       │                       │
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Redis Cache   │    │  Data Storage   │    │   AlertManager  │
│   Session Store │    │  Models/Metrics │    │   Notifications │
│   Port: 6379    │    │  Persistent     │    │   Port: 9093    │
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

### Deployment Stack

- **Application**: JAX-based federated graph RL system
- **Web Server**: NGINX reverse proxy with SSL termination
- **Caching**: Redis for session storage and caching
- **Monitoring**: Prometheus + Grafana + AlertManager
- **Container Orchestration**: Docker Compose
- **Data Persistence**: Docker volumes for data/logs/metrics

## 🔧 Prerequisites

### System Requirements

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| **CPU** | 4 cores | 8+ cores |
| **RAM** | 8 GB | 16+ GB |
| **Storage** | 50 GB | 200+ GB SSD |
| **Network** | 100 Mbps | 1 Gbps |

### Software Requirements

- Docker 20.10+
- Docker Compose 2.0+
- Linux (Ubuntu 20.04+ recommended)
- SSL certificate (or self-signed)

## 🚀 Quick Start

### 1. Clone and Prepare

```bash
git clone https://github.com/yourusername/pg-neo-graph-rl.git
cd pg-neo-graph-rl
```

### 2. Run Production Setup

```bash
./deployment/production_setup.sh
```

This automated script will:
- ✅ Check system requirements
- ✅ Create necessary directories
- ✅ Generate SSL certificates
- ✅ Build Docker images
- ✅ Deploy all services
- ✅ Configure monitoring
- ✅ Setup backup scripts

### 3. Verify Deployment

Access these URLs to verify successful deployment:

- **Application**: http://localhost:8080
- **Grafana**: http://localhost:3000 (admin/pg_neo_admin_2025)
- **Prometheus**: http://localhost:9090
- **AlertManager**: http://localhost:9093

## 📁 Directory Structure

```
pg-neo-graph-rl/
├── deployment/
│   ├── docker-compose.production.yml
│   ├── production_setup.sh
│   ├── nginx.conf
│   └── ssl/
├── data/
│   ├── models/
│   ├── metrics/
│   └── checkpoints/
├── logs/
│   ├── app/
│   ├── nginx/
│   └── monitoring/
├── backups/
└── scripts/
    └── backup.sh
```

## ⚙️ Configuration

### Environment Variables

Key configuration in `.env.production`:

```bash
# Core Settings
ENVIRONMENT=production
MAX_AGENTS=100
CACHE_SIZE=1000
BATCH_SIZE=32

# Resource Management
MAX_MEMORY_USAGE=0.85
GC_THRESHOLD=0.75
EMERGENCY_THRESHOLD=0.95

# Auto-scaling
MIN_AGENTS=1
MAX_AGENTS=100
SCALING_COOLDOWN=60
```

### Performance Tuning

#### For High Performance Workloads:
```bash
# Increase batch size and agent count
BATCH_SIZE=64
MAX_AGENTS=200
WORKER_PROCESSES=8

# Optimize memory usage
MAX_MEMORY_USAGE=0.90
GC_THRESHOLD=0.80
```

#### For Resource-Constrained Environments:
```bash
# Reduce resource usage
BATCH_SIZE=16
MAX_AGENTS=50
WORKER_PROCESSES=2

# More aggressive memory management
MAX_MEMORY_USAGE=0.75
GC_THRESHOLD=0.60
```

## 🔒 Security

### SSL/TLS Configuration

The setup script generates self-signed certificates by default. For production, replace with proper certificates:

```bash
# Copy your certificates
cp your-certificate.crt deployment/ssl/server.crt
cp your-private-key.key deployment/ssl/server.key

# Restart NGINX
docker-compose -f deployment/docker-compose.production.yml restart nginx
```

### Firewall Configuration

```bash
# Allow necessary ports
sudo ufw allow 22/tcp    # SSH
sudo ufw allow 80/tcp    # HTTP
sudo ufw allow 443/tcp   # HTTPS
sudo ufw allow 3000/tcp  # Grafana (optional)
sudo ufw enable
```

## 📊 Monitoring & Observability

### Grafana Dashboards

Pre-configured dashboards include:

1. **System Overview**: CPU, memory, disk usage
2. **Federated Learning**: Agent performance, convergence metrics
3. **Application Performance**: Training time, communication latency
4. **Infrastructure**: Docker containers, network metrics

### Key Metrics to Monitor

| Metric | Alert Threshold | Description |
|--------|----------------|-------------|
| Memory Usage | >85% | System memory pressure |
| Training Time | >30s | Performance degradation |
| Agent Failures | >10% | Training instability |
| Convergence Rate | <0.01 | Learning stagnation |

### Alert Configuration

Alerts are automatically configured for:
- High memory usage (>85%)
- Slow training performance (>30s)
- Service failures
- Disk space warnings (>90%)

## 🔄 Operations

### Daily Operations

```bash
# Check service status
docker-compose -f deployment/docker-compose.production.yml ps

# View logs
docker-compose -f deployment/docker-compose.production.yml logs -f pg-neo-app

# Restart services
docker-compose -f deployment/docker-compose.production.yml restart
```

### Backup & Recovery

```bash
# Manual backup
./scripts/backup.sh

# Restore from backup
tar -xzf backups/backup_YYYYMMDD_HHMMSS.tar.gz
# Copy restored data to appropriate directories
```

### Scaling Operations

```bash
# Scale application instances
docker-compose -f deployment/docker-compose.production.yml up -d --scale pg-neo-app=3

# Update resource limits
# Edit .env.production and restart services
```

## 🐛 Troubleshooting

### Common Issues

#### High Memory Usage
```bash
# Check memory usage
docker stats

# Force garbage collection
docker exec pg-neo-app python3 -c "import gc; gc.collect()"

# Reduce batch size in configuration
```

#### Slow Training Performance
```bash
# Check GPU availability
docker exec pg-neo-app python3 -c "import jax; print(jax.devices())"

# Monitor system resources
htop

# Check network latency between agents
```

#### Service Connection Issues
```bash
# Check network connectivity
docker network ls
docker network inspect pg-neo_pg-neo-network

# Restart networking
docker-compose -f deployment/docker-compose.production.yml down
docker-compose -f deployment/docker-compose.production.yml up -d
```

### Log Analysis

```bash
# Application logs
tail -f logs/app/pg-neo-graph-rl.log

# NGINX access logs
tail -f logs/nginx/access.log

# System performance logs
tail -f logs/monitoring/performance.log
```

## 📈 Performance Optimization

### Auto-scaling Configuration

The system automatically scales based on:
- Training performance metrics
- Memory usage patterns
- Agent utilization rates

### Caching Optimization

- **Graph Cache**: 1000 entries (configurable)
- **Parameter Cache**: 512MB (configurable)
- **Redis TTL**: 300 seconds (configurable)

### Resource Management

- **Memory Management**: Automatic garbage collection
- **Gradient Accumulation**: Adaptive batch sizing
- **Connection Pooling**: Redis connection reuse

## 🚦 Health Checks

### Automated Health Monitoring

The system includes comprehensive health checks:

```bash
# Application health
curl http://localhost:8080/health

# Service health
docker-compose -f deployment/docker-compose.production.yml ps

# System health
curl http://localhost:3000/api/health
```

### Health Check Endpoints

- `/health` - Application health status
- `/metrics` - Prometheus metrics
- `/ready` - Readiness probe
- `/alive` - Liveness probe

## 🔄 Updates & Maintenance

### Application Updates

```bash
# Pull latest changes
git pull origin main

# Rebuild and deploy
./deployment/production_setup.sh

# Or manual update
docker-compose -f deployment/docker-compose.production.yml down
docker build -t pg-neo-graph-rl:production .
docker-compose -f deployment/docker-compose.production.yml up -d
```

### Maintenance Windows

Recommended maintenance schedule:
- **Daily**: Log rotation, backup verification
- **Weekly**: Security updates, performance review
- **Monthly**: Full system backup, capacity planning

## 📞 Support

For production support:

1. **Documentation**: Check this guide and README.md
2. **Logs**: Review application and system logs
3. **Monitoring**: Use Grafana dashboards for insights
4. **Community**: GitHub Issues and Discussions

## 🎯 Success Metrics

Track these KPIs for production success:

- **Uptime**: >99.9%
- **Response Time**: <100ms for API calls
- **Training Performance**: Stable convergence rates
- **Resource Utilization**: 60-80% optimal range
- **Error Rate**: <0.1%

---

**Ready for Production!** 🚀

Your pg-neo-graph-rl system is now deployed and ready to handle city-scale federated graph reinforcement learning workloads.