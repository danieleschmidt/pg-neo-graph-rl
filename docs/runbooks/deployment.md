# Deployment Procedures

## Safe Deployment Process

### Pre-Deployment Checklist

#### Code Review
- [ ] All code changes peer-reviewed
- [ ] Unit tests passing (>90% coverage)
- [ ] Integration tests passing
- [ ] Security scan completed
- [ ] Performance impact assessed

#### Environment Preparation
- [ ] Backup current deployment
- [ ] Verify monitoring systems operational
- [ ] Check resource availability
- [ ] Confirm rollback plan
- [ ] Schedule maintenance window if needed

#### Dependencies
- [ ] Database migrations tested
- [ ] Third-party service availability
- [ ] Feature flags configured
- [ ] Configuration updates validated

### Deployment Steps

#### Step 1: Pre-Deployment Backup
```bash
# Backup database
pg_dump production_db > backups/pre_deploy_$(date +%Y%m%d_%H%M%S).sql

# Backup current container images
docker save pg-neo-graph-rl:current > backups/image_backup_$(date +%Y%m%d_%H%M%S).tar

# Backup configuration
cp docker-compose.yml backups/
cp -r config/ backups/config_$(date +%Y%m%d_%H%M%S)/
```

#### Step 2: Build and Test
```bash
# Build new image
docker build -t pg-neo-graph-rl:$(git rev-parse --short HEAD) .

# Run smoke tests
docker run --rm pg-neo-graph-rl:$(git rev-parse --short HEAD) python -m pytest tests/smoke/

# Security scan
docker run --rm -v /var/run/docker.sock:/var/run/docker.sock \
  clair-scanner pg-neo-graph-rl:$(git rev-parse --short HEAD)
```

#### Step 3: Staged Deployment

##### Blue-Green Deployment (Recommended)
```bash
# Deploy to green environment
export DEPLOY_ENV=green
docker-compose -f docker-compose.yml -f docker-compose.green.yml up -d

# Health check green environment
curl -f http://green.localhost:8000/health/detailed

# Run integration tests against green
python -m pytest tests/integration/ --env=green

# Switch traffic to green (update load balancer)
kubectl patch service pg-neo-graph-rl -p '{"spec":{"selector":{"version":"green"}}}'

# Monitor metrics for 10 minutes
# If successful, decomission blue environment
```

##### Rolling Deployment (Alternative)
```bash
# Kubernetes rolling update
kubectl set image deployment/pg-neo-graph-rl \
  app=pg-neo-graph-rl:$(git rev-parse --short HEAD) -n production

# Monitor rollout
kubectl rollout status deployment/pg-neo-graph-rl -n production --timeout=300s

# Verify deployment
kubectl get pods -n production -l app=pg-neo-graph-rl
```

#### Step 4: Post-Deployment Validation
```bash
# Health checks
curl -f http://localhost:8000/health/detailed

# Functional tests
python -m pytest tests/integration/ --env=production

# Performance baseline
python scripts/performance_benchmark.py --compare-baseline

# Monitor key metrics
curl -s "http://localhost:9090/api/v1/query?query=up{job='pg-neo-graph-rl'}"
```

### Database Migrations

#### Safe Migration Process
```bash
# 1. Backup database
pg_dump production_db > migration_backup_$(date +%Y%m%d_%H%M%S).sql

# 2. Test migration on copy
createdb migration_test
pg_restore -d migration_test migration_backup_*.sql
python manage.py migrate --database=migration_test

# 3. Apply to production (during maintenance window)
python manage.py migrate --database=production_db

# 4. Verify migration
python manage.py check --database=production_db
```

#### Migration Rollback
```sql
-- Rollback migration if needed
BEGIN;
-- Execute reverse migration SQL
-- Verify data integrity
ROLLBACK; -- or COMMIT if verified
```

### Rollback Procedures

#### Immediate Rollback (Emergency)
```bash
# Kubernetes
kubectl rollout undo deployment/pg-neo-graph-rl -n production

# Docker Compose  
docker-compose down
docker tag pg-neo-graph-rl:previous pg-neo-graph-rl:current
docker-compose up -d

# Verify rollback
curl -f http://localhost:8000/health
```

#### Planned Rollback
```bash
# 1. Switch traffic back to previous version
kubectl patch service pg-neo-graph-rl -p '{"spec":{"selector":{"version":"blue"}}}'

# 2. Monitor for 5 minutes
# 3. Scale down new version
kubectl scale deployment pg-neo-graph-rl-green --replicas=0 -n production

# 4. Clean up resources
kubectl delete deployment pg-neo-graph-rl-green -n production
```

### Environment-Specific Procedures

#### Development Environment
```bash
# Simple deployment for dev
git pull origin main
docker-compose build
docker-compose up -d
python -m pytest tests/smoke/
```

#### Staging Environment  
```bash
# Deploy release candidate
git checkout release/v1.2.0
docker build -t pg-neo-graph-rl:v1.2.0-rc .
docker-compose up -d

# Full test suite
python -m pytest tests/ --env=staging
python scripts/load_test.py --duration=600
```

#### Production Environment
```bash
# Follow complete deployment process
./scripts/deploy_production.sh --version=v1.2.0 --strategy=blue-green
```

### Monitoring During Deployment

#### Key Metrics to Watch
```prometheus
# Error rate
rate(http_requests_total{status=~"5.."}[5m])

# Response time
histogram_quantile(0.95, rate(http_request_duration_seconds_bucket[5m]))

# Federated learning performance  
rate(agent_reward_total[5m])

# System resources
container_memory_usage_bytes / container_spec_memory_limit_bytes > 0.8
```

#### Alert Thresholds
- Error rate > 5%: Consider rollback
- P95 latency > 2x baseline: Investigate
- Memory usage > 85%: Scale or rollback
- Agent performance drop > 20%: Review changes

### Deployment Automation

#### CI/CD Pipeline
```yaml
# .github/workflows/deploy.yml
name: Deploy
on:
  push:
    tags: ['v*']

jobs:
  deploy:
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v2
    
    - name: Run tests
      run: python -m pytest tests/
      
    - name: Build image
      run: docker build -t pg-neo-graph-rl:${{ github.ref_name }} .
      
    - name: Deploy to staging
      run: ./scripts/deploy_staging.sh ${{ github.ref_name }}
      
    - name: Integration tests
      run: python -m pytest tests/integration/ --env=staging
      
    - name: Deploy to production  
      run: ./scripts/deploy_production.sh ${{ github.ref_name }}
```

#### Deployment Script Template
```bash
#!/bin/bash
# deploy_production.sh

set -euo pipefail

VERSION=${1:-latest}
STRATEGY=${2:-rolling}

echo "Deploying pg-neo-graph-rl:$VERSION using $STRATEGY strategy"

# Pre-deployment checks
./scripts/pre_deploy_checks.sh

# Backup
./scripts/backup.sh

# Deploy based on strategy
case $STRATEGY in
  "blue-green")
    ./scripts/blue_green_deploy.sh "$VERSION"
    ;;
  "rolling")
    ./scripts/rolling_deploy.sh "$VERSION"
    ;;
  *)
    echo "Unknown strategy: $STRATEGY"
    exit 1
    ;;
esac

# Post-deployment validation
./scripts/post_deploy_validation.sh

echo "Deployment completed successfully"
```

### Troubleshooting Deployments

#### Common Issues

1. **Health Check Failures**
```bash
# Check application logs
kubectl logs -n production deployment/pg-neo-graph-rl --tail=50

# Verify configuration
kubectl get configmap -n production
kubectl describe deployment pg-neo-graph-rl -n production
```

2. **Database Connection Issues**  
```bash
# Test database connectivity
kubectl run -it --rm debug --image=postgres:13 --restart=Never -- \
  psql -h postgres-service -U username -d production_db -c "SELECT 1;"
```

3. **Resource Constraints**
```bash
# Check resource usage
kubectl top pods -n production
kubectl describe node

# Check resource requests/limits
kubectl describe deployment pg-neo-graph-rl -n production
```

### Post-Deployment Actions

1. **Monitoring**: Watch dashboards for 30 minutes
2. **Documentation**: Update deployment log
3. **Communication**: Notify team of successful deployment
4. **Cleanup**: Remove old images and backups after 24 hours
5. **Retrospective**: Document any issues or improvements

### Emergency Procedures

#### Immediate Issues (0-5 minutes)
- Stop deployment if critical errors detected
- Implement immediate rollback
- Activate incident response procedures

#### Degraded Performance (5-15 minutes)  
- Analyze metrics and logs
- Consider gradual traffic shifting
- Prepare rollback if no improvement

#### Extended Issues (15+ minutes)
- Execute full rollback
- Convene incident response team
- Schedule post-mortem review

Remember: Better to rollback quickly than debug in production.