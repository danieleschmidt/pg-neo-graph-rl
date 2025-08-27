# Production Deployment Guide

## 🚀 Quick Production Deployment

### Prerequisites
- Docker & Docker Compose
- Kubernetes cluster (optional)
- 8GB+ RAM, 4+ CPU cores
- GPU support (optional but recommended)

### Environment Setup

1. **Clone and Configure**:
```bash
git clone https://github.com/yourusername/pg-neo-graph-rl.git
cd pg-neo-graph-rl/deployment
cp config/.env.example .env
# Edit .env with your configuration
```

2. **Docker Deployment**:
```bash
# Production deployment
docker-compose -f docker-compose.production.yml up -d

# Monitor logs
docker-compose logs -f
```

3. **Kubernetes Deployment**:
```bash
kubectl apply -f kubernetes/
kubectl get pods -n pg-neo-rl
```

### Configuration

#### Environment Variables
```bash
# Core Configuration
PG_NEO_ENV=production
PG_NEO_LOG_LEVEL=INFO
PG_NEO_WORKERS=4

# Performance Settings  
PG_NEO_CACHE_SIZE=1000
PG_NEO_BATCH_SIZE=64
PG_NEO_MAX_AGENTS=1000

# Security
PG_NEO_SECRET_KEY=your-secret-key-here
PG_NEO_JWT_EXPIRY=3600

# Monitoring
PROMETHEUS_ENDPOINT=http://prometheus:9090
GRAFANA_ENDPOINT=http://grafana:3000
```

#### Resource Limits
```yaml
# kubernetes/application.yaml
resources:
  limits:
    memory: "8Gi"
    cpu: "4"
  requests:
    memory: "4Gi" 
    cpu: "2"
```

### Monitoring & Observability

#### Grafana Dashboards
- Access: http://localhost:3000
- Default credentials: admin/admin
- Pre-configured dashboards for federated learning metrics

#### Prometheus Metrics
- Access: http://localhost:9090
- Monitors training progress, system resources, agent performance

#### Health Checks
```bash
# Application health
curl http://localhost:8000/health

# Detailed status
curl http://localhost:8000/status
```

### Scaling & Performance

#### Horizontal Pod Autoscaling
```yaml
# kubernetes/hpa.yaml
minReplicas: 2
maxReplicas: 10
targetCPUUtilizationPercentage: 70
```

#### Performance Tuning
```python
# config/production_config.yaml
performance:
  jax_backend: "gpu"  # or "cpu"
  batch_size: 128
  parallel_agents: 4
  cache_strategy: "adaptive"
  memory_limit: "6GB"
```

### Security

#### Container Security
- Non-root user execution
- Read-only root filesystem
- Security context constraints
- Network policies

#### Data Protection
- TLS/SSL encryption
- JWT authentication
- Input validation
- Rate limiting

### Backup & Recovery

#### Automated Backups
```bash
# Run backup script
./scripts/backup.sh

# Restore from backup
./scripts/restore.sh backup-2025-01-15.tar.gz
```

#### Model Checkpoints
- Automatic model checkpointing every 100 episodes
- S3/GCS backup integration
- Point-in-time recovery

### Troubleshooting

#### Common Issues

1. **High Memory Usage**:
```bash
# Check memory consumption
kubectl top pods -n pg-neo-rl

# Adjust batch size
export PG_NEO_BATCH_SIZE=32
```

2. **Training Not Converging**:
```bash
# Check learning rate
export PG_NEO_LEARNING_RATE=1e-4

# Monitor gradients
curl http://localhost:8000/debug/gradients
```

3. **Communication Failures**:
```bash
# Check network connectivity
kubectl get networkpolicies

# Test gossip protocol
curl http://localhost:8000/debug/gossip
```

#### Log Analysis
```bash
# Application logs
kubectl logs -f deployment/pg-neo-rl

# System metrics
kubectl logs -f deployment/prometheus
```

### Production Checklist

- [ ] SSL/TLS certificates configured
- [ ] Database connections secured
- [ ] Monitoring dashboards operational
- [ ] Backup strategy implemented
- [ ] Health checks passing
- [ ] Resource limits configured
- [ ] Security policies applied
- [ ] Load testing completed
- [ ] Documentation updated
- [ ] Team training completed

### Support & Maintenance

#### Regular Tasks
- Weekly security updates
- Monthly performance reviews
- Quarterly disaster recovery tests
- Annual architecture reviews

#### Emergency Contacts
- SRE Team: sre@yourcompany.com
- ML Engineering: ml-eng@yourcompany.com
- Security: security@yourcompany.com

For detailed configuration options, see [Configuration Guide](../docs/configuration.md).