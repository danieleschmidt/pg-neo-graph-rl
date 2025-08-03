# Incident Response Runbook

## System Outage Response

### Situation
Complete or partial system unavailability affecting users.

**Symptoms:**
- Health check endpoints returning 5xx errors
- Prometheus targets showing as down
- User reports of service unavailability
- High error rates in application logs

### Investigation

#### Step 1: Confirm Outage Scope
```bash
# Check overall system health
curl -f http://localhost:8000/health || echo "Primary service down"

# Check individual components
docker-compose ps
kubectl get pods -n production

# Verify monitoring stack
curl -f http://localhost:9090/-/healthy || echo "Prometheus down"
curl -f http://localhost:3000/api/health || echo "Grafana down"
```

#### Step 2: Check Recent Changes
```bash
# Review recent deployments
git log --oneline -10
kubectl rollout history deployment/pg-neo-graph-rl

# Check configuration changes
docker-compose config --quiet || echo "Configuration error"

# Review recent alerts
curl -s "http://localhost:9090/api/v1/alerts" | jq '.data.alerts[] | select(.state=="firing")'
```

#### Step 3: Examine System Resources
```bash
# System resources
df -h
free -m
top -n 1

# Container resources
docker stats --no-stream
kubectl top pods -n production

# Network connectivity
ping prometheus
ping grafana
netstat -tlnp | grep :8000
```

#### Step 4: Review Logs
```bash
# Application logs
docker-compose logs --tail=100 pg-neo-graph-rl
kubectl logs -n production deployment/pg-neo-graph-rl --tail=100

# System logs
journalctl -u docker --since "1 hour ago"
dmesg | tail -50

# Look for errors
grep -i error /var/log/pg-neo-graph-rl/app.log
```

### Resolution

#### Option A: Service Restart
```bash
# For Docker Compose
docker-compose restart pg-neo-graph-rl

# For Kubernetes
kubectl rollout restart deployment/pg-neo-graph-rl -n production

# Wait for health check
sleep 30
curl -f http://localhost:8000/health
```

#### Option B: Rollback Deployment
```bash
# Docker Compose rollback
git checkout HEAD~1
docker-compose up -d --build

# Kubernetes rollback
kubectl rollout undo deployment/pg-neo-graph-rl -n production
kubectl rollout status deployment/pg-neo-graph-rl -n production
```

#### Option C: Scale Down/Up
```bash
# Kubernetes
kubectl scale deployment pg-neo-graph-rl --replicas=0 -n production
sleep 10
kubectl scale deployment pg-neo-graph-rl --replicas=3 -n production

# Docker Compose
docker-compose stop pg-neo-graph-rl
docker-compose up -d pg-neo-graph-rl
```

#### Option D: Emergency Maintenance Mode
```bash
# Enable maintenance page
kubectl apply -f k8s/maintenance-mode.yaml

# Or redirect traffic
# Update load balancer to show maintenance page
```

### Verification

#### Step 1: Health Checks
```bash
# Basic health
curl -f http://localhost:8000/health

# Detailed health  
curl -s http://localhost:8000/health/detailed | jq .

# Kubernetes readiness
kubectl get pods -n production -o wide
```

#### Step 2: Functional Testing
```bash
# Test key endpoints
curl -f http://localhost:8000/api/agents
curl -f http://localhost:8000/api/environments

# Run integration tests
python -m pytest tests/integration/ -v
```

#### Step 3: Monitoring Verification
```bash
# Check Prometheus targets
curl -s http://localhost:9090/api/v1/targets | jq '.data.activeTargets[] | select(.health != "up")'

# Verify metrics collection
curl -s http://localhost:9090/api/v1/query?query=up | jq .
```

### Communication

#### During Incident
1. **Immediate**: Post in #production-alerts Slack channel
2. **Status Page**: Update status.company.com if user-facing
3. **Stakeholders**: Notify via incident management system
4. **Updates**: Provide updates every 15 minutes

#### Message Templates
```
INITIAL: 🔴 INCIDENT: pg-neo-graph-rl system outage detected at [TIME]. Investigating root cause.

UPDATE: 🔄 UPDATE: Root cause identified as [CAUSE]. Implementing fix via [SOLUTION]. ETA: [TIME]

RESOLVED: ✅ RESOLVED: System restored at [TIME]. Total downtime: [DURATION]. Post-mortem scheduled.
```

### Prevention

#### Immediate Actions
- [ ] Review monitoring alerts that were missed
- [ ] Check if circuit breakers should have activated
- [ ] Verify health check configurations
- [ ] Review resource limits and requests

#### Long-term Improvements  
- [ ] Enhance monitoring coverage
- [ ] Implement additional health checks
- [ ] Review deployment procedures
- [ ] Conduct chaos engineering tests
- [ ] Update documentation

### Post-Incident Actions

1. **Document Timeline**: Record exact sequence of events
2. **Root Cause Analysis**: Identify primary and contributing factors  
3. **Action Items**: Create tickets for prevention measures
4. **Runbook Updates**: Improve procedures based on learnings
5. **Team Review**: Schedule post-mortem meeting

### Escalation

**Escalate to L3 if:**
- Outage persists beyond 15 minutes
- Multiple system components affected
- Data integrity concerns
- Unknown root cause

**Contact:**
- On-call engineer: +1-555-ONCALL
- Senior engineer: +1-555-SENIOR  
- Engineering manager: +1-555-MANAGER

---

## Critical System Recovery

### Database Recovery
```sql
-- Check database connectivity
SELECT version();

-- Verify recent backups
SELECT * FROM pg_stat_archiver;

-- Restore from backup if needed
pg_restore -d production_db latest_backup.dump
```

### Federated Learning State Recovery
```python
# Check agent states
from pg_neo_graph_rl.monitoring import check_agent_health

for agent_id in range(num_agents):
    status = check_agent_health(agent_id)
    if not status['healthy']:
        # Reset agent state
        reset_agent_state(agent_id)
```

### Configuration Recovery
```bash
# Restore known good configuration
cp /backup/config/docker-compose.yml.backup docker-compose.yml
cp /backup/config/prometheus.yml.backup docker/prometheus.yml

# Validate configuration
docker-compose config --quiet
promtool check config docker/prometheus.yml
```

Remember: Document everything and communicate early and often during incidents.