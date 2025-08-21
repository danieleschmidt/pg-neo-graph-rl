# Production Deployment Guide
## pg-neo-graph-rl: Federated Graph Neural Network Reinforcement Learning

### 🚀 Quick Start Production Deployment

This guide covers production deployment of the pg-neo-graph-rl system with enterprise-grade features including monitoring, security, auto-scaling, and high availability.

## 📋 Prerequisites

### System Requirements
- **CPU**: 8+ cores recommended
- **Memory**: 16GB+ RAM recommended  
- **Storage**: 100GB+ SSD storage
- **Network**: High-bandwidth connection for distributed training

### Software Dependencies
- Docker 20.10+
- Docker Compose 2.0+
- Kubernetes 1.24+ (for K8s deployment)
- Python 3.10+

## 🐳 Docker Deployment

### 1. Simple Docker Run
```bash
# Quick start with Docker
docker build -t pg-neo-graph-rl .
docker run -p 8080:8080 -p 8090:8090 pg-neo-graph-rl
```

### 2. Production Docker Compose
```bash
# Full production stack with monitoring
cd deploy/
docker-compose up -d

# Check services
docker-compose ps

# View logs
docker-compose logs -f pg-neo-graph-rl
```

**Services included:**
- Main application (port 8080)
- Redis cache (port 6379)
- PostgreSQL database (port 5432)
- Prometheus monitoring (port 9090)
- Grafana dashboards (port 3000)
- Nginx load balancer (port 80/443)

## ☸️ Kubernetes Deployment

### 1. Deploy to Kubernetes
```bash
# Apply Kubernetes configuration
kubectl apply -f deploy/kubernetes.yml

# Check deployment status
kubectl get pods -n pg-neo-graph-rl

# Get service URLs
kubectl get services -n pg-neo-graph-rl
```

### 2. Scale the deployment
```bash
# Manual scaling
kubectl scale deployment pg-neo-graph-rl --replicas=5 -n pg-neo-graph-rl

# Auto-scaling is configured via HPA (CPU: 70%, Memory: 80%)
kubectl get hpa -n pg-neo-graph-rl
```

## 🏗️ Architecture Overview

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Load Balancer │    │   Application   │    │    Database     │
│     (Nginx)     │━━━▶│   (Multi-Pod)   │━━━▶│  (PostgreSQL)   │
│                 │    │                 │    │                 │
└─────────────────┘    └─────────────────┘    └─────────────────┘
                              │                         
                              ▼                         
                       ┌─────────────────┐              
                       │     Cache       │              
                       │    (Redis)      │              
                       └─────────────────┘              
                              │                         
                              ▼                         
                    ┌─────────────────────┐             
                    │     Monitoring      │             
                    │ (Prometheus/Grafana)│             
                    └─────────────────────┘             
```

## 🔧 Configuration

### Environment Variables
```bash
# Core application
ENV=production
WORKERS=4
LOG_LEVEL=info

# Features
MONITORING_ENABLED=true
SECURITY_ENABLED=true
CACHING_ENABLED=true

# Infrastructure
REDIS_URL=redis://redis:6379
POSTGRES_URL=postgresql://user:pass@postgres:5432/db
```

### Resource Limits
```yaml
# Kubernetes resources
resources:
  requests:
    memory: "2Gi"
    cpu: "1000m"
  limits:
    memory: "4Gi" 
    cpu: "2000m"
```

## 📊 Monitoring & Observability

### Metrics Dashboard
- **URL**: http://your-domain:3000
- **Login**: admin / admin_change_in_production
- **Dashboards**: 
  - System metrics (CPU, memory, disk)
  - Application metrics (requests, errors, latency)
  - Federated learning metrics (rounds, convergence)
  - Research metrics (experiments, results)

### Health Checks
```bash
# Application health
curl http://your-domain:8080/health

# Detailed health status
curl http://your-domain:8080/health/detailed

# Metrics endpoint
curl http://your-domain:8080/metrics
```

### Log Management
```bash
# View application logs
docker-compose logs -f pg-neo-graph-rl

# Kubernetes logs
kubectl logs -f deployment/pg-neo-graph-rl -n pg-neo-graph-rl

# Log aggregation (if using ELK stack)
# Logs are automatically shipped to /app/logs/ volume
```

## 🔒 Security Configuration

### Production Security Features
- **Authentication**: JWT-based authentication
- **Authorization**: Role-based access control
- **Encryption**: TLS/SSL for all communications
- **Input Validation**: Comprehensive input sanitization
- **Security Audit**: Real-time security event logging
- **Network Security**: Network policies and firewall rules

### Security Checklist
- [ ] Change default passwords in docker-compose.yml
- [ ] Generate SSL certificates for HTTPS
- [ ] Configure firewall rules
- [ ] Set up backup encryption
- [ ] Enable audit logging
- [ ] Regular security scans

## 🔄 Auto-Scaling Configuration

### Horizontal Pod Autoscaling (HPA)
```yaml
# Automatically scales 2-10 pods based on:
- CPU utilization: 70%
- Memory utilization: 80%
```

### Vertical Pod Autoscaling (VPA)
```bash
# Install VPA (if not available)
kubectl apply -f https://github.com/kubernetes/autoscaler/tree/master/vertical-pod-autoscaler/deploy
```

## 🗄️ Backup & Recovery

### Automated Backups
```bash
# Database backup (daily)
kubectl create cronjob postgres-backup \
  --image=postgres:15-alpine \
  --schedule="0 2 * * *" \
  --restart=OnFailure \
  -- pg_dump postgresql://user:pass@postgres:5432/db

# Application state backup
kubectl create cronjob app-backup \
  --image=pg-neo-graph-rl:latest \
  --schedule="0 3 * * *" \
  --restart=OnFailure \
  -- python -m pg_neo_graph_rl.utils.backup create
```

### Disaster Recovery
```bash
# Restore from backup
python -m pg_neo_graph_rl.utils.backup restore --backup-file=/path/to/backup.gz

# Database restore
psql postgresql://user:pass@postgres:5432/db < backup.sql
```

## 🚀 Performance Optimization

### Production Optimizations
- **JAX Compilation**: Ahead-of-time compilation for faster execution
- **Memory Management**: Optimized memory allocation and garbage collection
- **Caching**: Multi-level caching (Redis + application cache)
- **Connection Pooling**: Database connection pooling
- **Load Balancing**: Nginx with round-robin load balancing

### Performance Monitoring
```bash
# Check performance metrics
curl http://your-domain:8090/metrics/performance

# Resource utilization
kubectl top pods -n pg-neo-graph-rl
kubectl top nodes
```

## 🔍 Troubleshooting

### Common Issues

#### Service Discovery Issues
```bash
# Check service connectivity
kubectl exec -it pod-name -n pg-neo-graph-rl -- nslookup redis-service
kubectl exec -it pod-name -n pg-neo-graph-rl -- telnet postgres-service 5432
```

#### Memory Issues
```bash
# Check memory usage
kubectl describe pod pod-name -n pg-neo-graph-rl
kubectl get events -n pg-neo-graph-rl --sort-by='.firstTimestamp'
```

#### Networking Issues
```bash
# Check network policies
kubectl get networkpolicies -n pg-neo-graph-rl
kubectl describe networkpolicy pg-neo-graph-rl-netpol -n pg-neo-graph-rl
```

### Debug Mode
```bash
# Enable debug logging
kubectl set env deployment/pg-neo-graph-rl LOG_LEVEL=debug -n pg-neo-graph-rl

# Access debug endpoints
curl http://your-domain:8080/debug/health
curl http://your-domain:8080/debug/metrics
```

## 📈 Scaling Considerations

### Horizontal Scaling
- **Stateless Design**: Application pods are stateless and can be scaled horizontally
- **Load Distribution**: Nginx distributes load across multiple pods
- **Auto-scaling**: HPA automatically scales based on resource utilization

### Vertical Scaling
- **Resource Limits**: Adjust CPU/memory limits based on workload
- **JAX Memory**: Configure JAX memory preallocation for GPU workloads
- **Database Scaling**: Consider read replicas for database scaling

### Multi-Region Deployment
```bash
# Deploy across multiple regions for high availability
kubectl apply -f deploy/kubernetes-multi-region.yml
```

## 🛡️ Production Checklist

### Pre-Deployment
- [ ] Load testing completed
- [ ] Security scan passed
- [ ] Backup strategy implemented
- [ ] Monitoring configured
- [ ] SSL certificates installed
- [ ] Resource limits set
- [ ] Network policies configured

### Post-Deployment
- [ ] Health checks passing
- [ ] Metrics being collected
- [ ] Logs being aggregated
- [ ] Auto-scaling working
- [ ] Backup jobs running
- [ ] Performance within SLA
- [ ] Security audit clean

## 🔗 Additional Resources

- **API Documentation**: http://your-domain:8080/docs
- **Monitoring Dashboard**: http://your-domain:3000
- **Health Status**: http://your-domain:8080/health
- **Metrics**: http://your-domain:8080/metrics

## 📞 Support

For production support:
- **Documentation**: Check the /docs directory
- **Health Checks**: Use built-in health endpoints
- **Monitoring**: Use Grafana dashboards for system monitoring
- **Logs**: Check application logs for detailed error information

---

**Note**: This is a production-ready deployment with enterprise features. Ensure all security configurations are properly set before deploying to production environments.