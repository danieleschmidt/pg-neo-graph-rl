# Health Check Configuration

Health checks ensure system reliability and enable automated recovery procedures.

## Health Check Endpoints

### Application Health
```python
# pg_neo_graph_rl/monitoring/health.py
from fastapi import FastAPI, status
from fastapi.responses import JSONResponse
import time
import psutil

app = FastAPI()

@app.get("/health")
async def health_check():
    """Basic health check endpoint"""
    return {"status": "healthy", "timestamp": time.time()}

@app.get("/health/detailed")
async def detailed_health():
    """Detailed health information"""
    return {
        "status": "healthy",
        "timestamp": time.time(),
        "system": {
            "cpu_percent": psutil.cpu_percent(),
            "memory_percent": psutil.virtual_memory().percent,
            "disk_percent": psutil.disk_usage('/').percent
        },
        "services": {
            "prometheus": check_prometheus_connectivity(),
            "agents": check_federated_agents(),
            "graph_env": check_environment_health()
        }
    }

@app.get("/health/ready")
async def readiness_check():
    """Kubernetes readiness probe"""
    if not all_services_ready():
        return JSONResponse(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            content={"status": "not_ready"}
        )
    return {"status": "ready"}

@app.get("/health/live")
async def liveness_check():
    """Kubernetes liveness probe"""
    return {"status": "alive", "timestamp": time.time()}
```

### Database Health
```python
async def check_database_health():
    """Check database connectivity and performance"""
    try:
        # Test connection
        start_time = time.time()
        result = await db.execute("SELECT 1")
        response_time = time.time() - start_time
        
        return {
            "status": "healthy",
            "response_time_ms": response_time * 1000,
            "connection_pool": {
                "active": db.pool.active_connections,
                "idle": db.pool.idle_connections
            }
        }
    except Exception as e:
        return {
            "status": "unhealthy",
            "error": str(e)
        }
```

## Docker Health Checks

### Dockerfile Configuration
```dockerfile
# Add health check to Dockerfile
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1
```

### Docker Compose Health Checks
```yaml
# docker-compose.yml
services:
  pg-neo-graph-rl:
    build: .
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/health"]
      interval: 30s
      timeout: 10s
      retries: 3
      start_period: 40s
    depends_on:
      prometheus:
        condition: service_healthy
      
  prometheus:
    image: prom/prometheus:latest
    healthcheck:
      test: ["CMD", "wget", "--no-verbose", "--tries=1", "--spider", "http://localhost:9090/-/healthy"]
      interval: 30s
      timeout: 10s
      retries: 3
```

## Kubernetes Health Checks

### Deployment Configuration
```yaml
# k8s/deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: pg-neo-graph-rl
spec:
  template:
    spec:
      containers:
      - name: app
        image: pg-neo-graph-rl:latest
        ports:
        - containerPort: 8000
        livenessProbe:
          httpGet:
            path: /health/live
            port: 8000
          initialDelaySeconds: 30
          periodSeconds: 10
          timeoutSeconds: 5
          failureThreshold: 3
        readinessProbe:
          httpGet:
            path: /health/ready
            port: 8000
          initialDelaySeconds: 10
          periodSeconds: 5
          timeoutSeconds: 3
          failureThreshold: 3
```

## Monitoring Integration

### Prometheus Health Metrics
```python
from prometheus_client import Gauge, Counter

# Health check metrics
health_status = Gauge('service_health_status', 'Service health status (1=healthy, 0=unhealthy)', ['service'])
health_check_duration = Histogram('health_check_duration_seconds', 'Health check duration')
health_check_failures = Counter('health_check_failures_total', 'Health check failures', ['service'])

@health_check_duration.time()
async def perform_health_check(service_name):
    try:
        # Perform health check
        result = await check_service_health(service_name)
        health_status.labels(service=service_name).set(1 if result['healthy'] else 0)
        return result
    except Exception as e:
        health_check_failures.labels(service=service_name).inc()
        health_status.labels(service=service_name).set(0)
        raise
```

## Alert Rules

### Health Check Alerts
```yaml
# prometheus/health_alerts.yml
groups:
- name: health_checks
  rules:
  - alert: ServiceUnhealthy
    expr: service_health_status == 0
    for: 5m
    labels:
      severity: critical
    annotations:
      summary: "Service {{ $labels.service }} is unhealthy"
      description: "Service {{ $labels.service }} has been unhealthy for more than 5 minutes"

  - alert: HighHealthCheckFailureRate
    expr: rate(health_check_failures_total[5m]) > 0.1
    for: 2m
    labels:
      severity: warning
    annotations:
      summary: "High health check failure rate for {{ $labels.service }}"
      description: "Health check failure rate is {{ $value }} failures/sec"
```

## Health Check Best Practices

### Implementation Guidelines
1. **Response Time**: Keep health checks fast (<1s)
2. **Dependencies**: Check critical dependencies only
3. **Graceful Degradation**: Distinguish between critical and non-critical failures
4. **Resource Usage**: Minimize resource consumption during checks

### Monitoring Strategy
1. **Layered Checks**: Basic -> Detailed -> Deep dependency checks
2. **Appropriate Timeouts**: Balance between false positives and detection speed
3. **Retry Logic**: Implement exponential backoff for transient failures
4. **Circuit Breakers**: Prevent cascade failures

### Example Implementation
```python
import asyncio
import logging
from typing import Dict, Any
from enum import Enum

class HealthStatus(Enum):
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"

class HealthChecker:
    def __init__(self):
        self.checks = {}
        self.circuit_breakers = {}
    
    def register_check(self, name: str, check_func, critical: bool = True):
        self.checks[name] = {
            'func': check_func,
            'critical': critical,
            'circuit_breaker': CircuitBreaker(failure_threshold=3)
        }
    
    async def run_all_checks(self) -> Dict[str, Any]:
        results = {}
        overall_status = HealthStatus.HEALTHY
        
        for name, check_config in self.checks.items():
            try:
                if check_config['circuit_breaker'].is_open():
                    results[name] = {
                        'status': HealthStatus.UNHEALTHY.value,
                        'message': 'Circuit breaker open'
                    }
                else:
                    result = await asyncio.wait_for(
                        check_config['func'](), 
                        timeout=5.0
                    )
                    results[name] = result
                    check_config['circuit_breaker'].record_success()
                    
            except Exception as e:
                check_config['circuit_breaker'].record_failure()
                results[name] = {
                    'status': HealthStatus.UNHEALTHY.value,
                    'error': str(e)
                }
                
                if check_config['critical']:
                    overall_status = HealthStatus.UNHEALTHY
                elif overall_status == HealthStatus.HEALTHY:
                    overall_status = HealthStatus.DEGRADED
        
        return {
            'overall_status': overall_status.value,
            'checks': results,
            'timestamp': time.time()
        }
```

This health check system provides comprehensive monitoring capabilities while maintaining performance and reliability.