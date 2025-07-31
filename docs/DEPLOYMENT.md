# Deployment Guide

This guide covers deployment strategies for pg-neo-graph-rl in various environments.

## Production Deployment

### Container Deployment

#### Docker
```bash
# Build production image
docker build -t pg-neo-graph-rl:latest --target production .

# Run container
docker run -d \
  --name pg-neo-rl \
  --restart unless-stopped \
  -p 8000:8000 \
  -v /path/to/data:/app/data \
  -v /path/to/logs:/app/logs \
  pg-neo-graph-rl:latest
```

#### Docker Compose
```bash
# Production deployment
docker-compose -f docker-compose.prod.yml up -d

# With monitoring stack
docker-compose -f docker-compose.prod.yml -f docker/docker-compose.monitoring.yml up -d
```

### Kubernetes Deployment

#### Basic Deployment
```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: pg-neo-graph-rl
spec:
  replicas: 3
  selector:
    matchLabels:
      app: pg-neo-graph-rl
  template:
    metadata:
      labels:
        app: pg-neo-graph-rl
    spec:
      containers:
      - name: pg-neo-graph-rl
        image: pg-neo-graph-rl:latest
        ports:
        - containerPort: 8000
        resources:
          requests:
            memory: "1Gi"
            cpu: "500m"
          limits:
            memory: "4Gi"
            cpu: "2000m"
        env:
        - name: PYTHONPATH
          value: "/app"
        volumeMounts:
        - name: data-volume
          mountPath: /app/data
      volumes:
      - name: data-volume
        persistentVolumeClaim:
          claimName: pg-neo-data-pvc
```

#### Service Configuration
```yaml
apiVersion: v1
kind: Service
metadata:
  name: pg-neo-graph-rl-service
spec:
  selector:
    app: pg-neo-graph-rl
  ports:
  - protocol: TCP
    port: 80
    targetPort: 8000
  type: LoadBalancer
```

### Cloud Deployments

#### AWS ECS
```json
{
  "family": "pg-neo-graph-rl",
  "networkMode": "awsvpc",
  "requiresCompatibilities": ["FARGATE"],
  "cpu": "1024",
  "memory": "4096",
  "executionRoleArn": "arn:aws:iam::account:role/ecsTaskExecutionRole",
  "containerDefinitions": [
    {
      "name": "pg-neo-graph-rl",
      "image": "your-account.dkr.ecr.region.amazonaws.com/pg-neo-graph-rl:latest",
      "portMappings": [
        {
          "containerPort": 8000,
          "protocol": "tcp"
        }
      ],
      "logConfiguration": {
        "logDriver": "awslogs",
        "options": {
          "awslogs-group": "/ecs/pg-neo-graph-rl",
          "awslogs-region": "us-west-2",
          "awslogs-stream-prefix": "ecs"
        }
      }
    }
  ]
}
```

#### Google Cloud Run
```yaml
apiVersion: serving.knative.dev/v1
kind: Service
metadata:
  name: pg-neo-graph-rl
  annotations:
    run.googleapis.com/ingress: all
spec:
  template:
    metadata:
      annotations:
        autoscaling.knative.dev/maxScale: "10"
        run.googleapis.com/memory: "4Gi"
        run.googleapis.com/cpu: "2"
    spec:
      containers:
      - image: gcr.io/project-id/pg-neo-graph-rl:latest
        ports:
        - containerPort: 8000
        resources:
          limits:
            memory: "4Gi"
            cpu: "2000m"
```

## Development Deployment

### Local Development
```bash
# Using Docker Compose
docker-compose up -d dev

# Using Make
make install-dev
make test
```

### Staging Environment
```bash
# Deploy to staging
docker-compose -f docker-compose.staging.yml up -d

# Run integration tests
docker-compose exec app pytest tests/integration/
```

## Monitoring and Observability

### Metrics Collection
```bash
# Start monitoring stack
make monitoring-stack

# Access dashboards
# Grafana: http://localhost:3000
# Prometheus: http://localhost:9090
```

### Logging Configuration
```python
# logging_config.py
import logging
from rich.logging import RichHandler

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        RichHandler(rich_tracebacks=True),
        logging.FileHandler("pg_neo_graph_rl.log")
    ]
)
```

## Security Considerations

### Production Security Checklist
- [ ] Use non-root user in containers
- [ ] Scan images for vulnerabilities
- [ ] Enable security contexts in Kubernetes
- [ ] Use secrets management for sensitive data
- [ ] Enable network policies
- [ ] Set up monitoring and alerting
- [ ] Regular security updates

### Environment Variables
```bash
# Required environment variables
export PYTHONPATH=/app
export LOG_LEVEL=INFO
export MONITORING_ENABLED=true

# Optional security settings
export SECURE_MODE=true
export API_KEY_FILE=/secrets/api_key
```

## Scaling and Performance

### Horizontal Scaling
```yaml
# Kubernetes HPA
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: pg-neo-graph-rl-hpa
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: pg-neo-graph-rl
  minReplicas: 2
  maxReplicas: 10
  metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: 70
```

### GPU Deployment
```yaml
# Kubernetes GPU deployment
spec:
  containers:
  - name: pg-neo-graph-rl-gpu
    image: pg-neo-graph-rl:gpu
    resources:
      limits:
        nvidia.com/gpu: 1
        memory: "8Gi"
        cpu: "4000m"
```

## Troubleshooting

### Common Issues

#### Container Won't Start
```bash
# Check logs
docker logs pg-neo-rl

# Check resource usage
docker stats pg-neo-rl

# Debug interactively
docker run -it --rm pg-neo-graph-rl:latest bash
```

#### Performance Issues
```bash
# Monitor resource usage
kubectl top pods
kubectl describe pod <pod-name>

# Check application metrics
curl http://localhost:8000/metrics
```

#### Network Issues
```bash
# Test connectivity
kubectl exec -it <pod-name> -- wget -qO- http://service-name:port/health

# Check DNS resolution
kubectl exec -it <pod-name> -- nslookup service-name
```

## Backup and Recovery

### Data Backup
```bash
# Backup training data and models
kubectl create job backup-$(date +%Y%m%d) --from=cronjob/backup-cron

# Manual backup
kubectl cp <pod-name>:/app/data ./backup-$(date +%Y%m%d)
```

### Disaster Recovery
```bash
# Restore from backup
kubectl apply -f backup-restore-job.yaml

# Verify restoration
kubectl exec -it <pod-name> -- ls -la /app/data
```

For specific deployment scenarios, refer to the platform-specific guides in the `deployment/` directory.