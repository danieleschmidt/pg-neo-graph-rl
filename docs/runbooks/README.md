# Operational Runbooks

This directory contains operational procedures for managing pg-neo-graph-rl in production environments.

## Available Runbooks

### Incident Response
- [System Outage Response](incident-response.md) - Critical system failure procedures
- [Performance Degradation](performance-issues.md) - Handling performance problems
- [Security Incident Response](security-incidents.md) - Security breach procedures

### Maintenance Operations  
- [Deployment Procedures](deployment.md) - Safe deployment and rollback
- [Backup and Recovery](backup-recovery.md) - Data backup and disaster recovery
- [Scaling Operations](scaling.md) - Horizontal and vertical scaling

### Monitoring and Alerting
- [Alert Investigation](alert-investigation.md) - Alert triage and resolution
- [Monitoring Troubleshooting](monitoring-troubleshooting.md) - Monitoring stack issues

### Application-Specific
- [Federated Learning Issues](federated-learning.md) - FL-specific troubleshooting
- [Graph Environment Problems](graph-environments.md) - Environment-specific issues

## General Procedures

### Emergency Contacts
```
On-Call Engineer: +1-555-ONCALL
DevOps Team: devops@company.com  
Security Team: security@company.com
Management: engineering-mgmt@company.com
```

### Escalation Matrix
1. **L1 - Automated Response**: Automated remediation attempts
2. **L2 - On-Call Engineer**: Initial human investigation  
3. **L3 - Senior Engineer**: Complex technical issues
4. **L4 - Architecture Team**: System design issues
5. **L5 - Management**: Business impact decisions

### Communication Channels
- **Slack**: #production-alerts, #incident-response
- **Email**: production-alerts@company.com
- **Status Page**: https://status.company.com
- **Conference Bridge**: +1-555-BRIDGE

## Runbook Usage Guidelines

### When to Use Runbooks
- System alerts are firing
- Performance degradation observed  
- User-reported issues
- Scheduled maintenance activities
- Post-incident remediation

### Runbook Format
Each runbook follows this structure:
1. **Situation**: Problem description and symptoms
2. **Investigation**: Diagnostic steps and tools
3. **Resolution**: Step-by-step remediation
4. **Verification**: Confirm issue resolution
5. **Prevention**: Long-term improvement actions

### Best Practices
- Follow runbooks step-by-step
- Document any deviations or new findings
- Update runbooks after incidents
- Test runbooks during non-critical periods
- Keep contact information current

### Tools and Access
- **Monitoring**: Grafana dashboards, Prometheus queries
- **Logs**: Centralized logging system
- **Infrastructure**: Cloud provider consoles
- **Communication**: Incident management system
- **Documentation**: Wiki and knowledge base

## Quick Reference

### Common Commands
```bash
# Check system health
curl http://localhost:8000/health/detailed

# View recent logs  
docker-compose logs --tail=100 -f pg-neo-graph-rl

# Restart services
docker-compose restart pg-neo-graph-rl

# Scale federated agents
docker-compose up -d --scale federated-agent=5

# Database backup
pg_dump production_db > backup_$(date +%Y%m%d_%H%M%S).sql
```

### Critical Metrics
- **System Health**: service_health_status
- **Agent Performance**: agent_reward_total  
- **Communication**: communication_latency_seconds
- **Resource Usage**: container_memory_usage_bytes
- **Error Rate**: http_requests_total{status=~"5.."}

### Emergency Procedures
1. **Immediate Response**: Stop traffic, isolate affected systems
2. **Assessment**: Determine scope and impact
3. **Communication**: Notify stakeholders
4. **Remediation**: Execute appropriate runbook
5. **Recovery**: Restore normal operations
6. **Post-Mortem**: Document lessons learned

Remember: When in doubt, escalate early and communicate clearly.