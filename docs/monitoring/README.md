# Monitoring & Observability

This document outlines the monitoring and observability strategy for pg-neo-graph-rl, providing comprehensive visibility into federated graph reinforcement learning systems.

## Overview

The monitoring stack includes:
- **Prometheus**: Metrics collection and storage
- **Grafana**: Visualization and dashboards
- **Alertmanager**: Alert routing and notification
- **Node Exporter**: System-level metrics
- **cAdvisor**: Container metrics

## Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Application   │    │   Prometheus    │    │     Grafana     │
│   (pg-neo-*)    │───▶│   (Scraping)    │───▶│  (Dashboards)   │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                       │                       │
         │              ┌─────────────────┐              │
         │              │  Alertmanager   │              │
         └──────────────│   (Alerting)    │──────────────┘
                        └─────────────────┘
```

## Key Metrics Categories

### 1. Federated Learning Metrics
- **Agent Performance**: Individual agent rewards, convergence rates
- **Communication**: Message passing frequency, gossip rounds
- **Aggregation**: Parameter synchronization latency
- **Privacy**: Differential privacy noise levels

### 2. Graph Environment Metrics  
- **Graph Topology**: Node/edge counts, clustering coefficients
- **Dynamic Changes**: Graph evolution rates, temporal patterns
- **Environment State**: Traffic flow, power grid stability, swarm formation

### 3. System Performance Metrics
- **Computational**: GPU utilization, memory usage, JAX compilation time
- **Network**: Bandwidth utilization, communication latency
- **Storage**: Model checkpoint sizes, data persistence

### 4. Application Health
- **Service Availability**: Uptime, response times
- **Error Rates**: Exception frequencies, failure modes
- **Resource Utilization**: CPU, memory, disk usage

## Dashboard Organization

### Executive Dashboard
- High-level KPIs and system health
- Federated learning progress overview
- Critical alerts and incidents

### Technical Dashboard
- Detailed performance metrics
- Resource utilization trends  
- Communication patterns

### Environment-Specific Dashboards
- Traffic Control: Flow rates, congestion metrics
- Power Grid: Voltage stability, frequency deviation
- Swarm Control: Formation quality, collision avoidance

## Alert Definitions

### Critical Alerts
- System downtime or service unavailability
- Federated learning divergence
- Resource exhaustion (memory, disk)

### Warning Alerts  
- Performance degradation trends
- High error rates
- Unusual communication patterns

### Info Alerts
- Deployment notifications
- Scheduled maintenance windows
- Configuration changes

## Setup Instructions

1. **Start Monitoring Stack**:
   ```bash
   docker-compose up -d grafana prometheus alertmanager
   ```

2. **Configure Application Metrics**:
   ```python
   from pg_neo_graph_rl.monitoring import MetricsCollector
   
   metrics = MetricsCollector(
       prometheus_gateway="localhost:9091",
       job_name="federated_graph_rl"
   )
   ```

3. **Access Dashboards**:
   - Grafana: http://localhost:3000 (admin/admin)
   - Prometheus: http://localhost:9090
   - Alertmanager: http://localhost:9093

## Custom Metrics Integration

### Application Instrumentation
```python
from prometheus_client import Counter, Histogram, Gauge

# Agent performance metrics
agent_reward = Gauge('agent_reward_total', 'Total reward per agent', ['agent_id'])
communication_latency = Histogram('communication_latency_seconds', 'Agent communication latency')
convergence_rate = Gauge('convergence_rate', 'Learning convergence rate')

# Graph environment metrics  
graph_size = Gauge('graph_nodes_total', 'Number of nodes in graph')
edge_count = Gauge('graph_edges_total', 'Number of edges in graph')
clustering_coeff = Gauge('graph_clustering_coefficient', 'Graph clustering coefficient')
```

### Custom Collectors
```python
from prometheus_client.core import CollectorRegistry, Metric

class FederatedLearningCollector:
    def collect(self):
        # Collect federated learning specific metrics
        metric = Metric('federated_agents_active', 'Active federated agents', 'gauge')
        metric.add_sample('federated_agents_active', value=len(self.active_agents))
        yield metric
```

## Troubleshooting

### Common Issues

1. **Missing Metrics**: Verify application instrumentation and Prometheus scraping configuration
2. **High Latency**: Check network connectivity between services
3. **Memory Issues**: Monitor Prometheus retention settings and storage usage

### Debug Commands
```bash
# Check Prometheus targets
curl http://localhost:9090/api/v1/targets

# Verify metrics endpoint
curl http://localhost:8000/metrics

# Check Grafana data sources
curl -u admin:admin http://localhost:3000/api/datasources
```

See [runbooks](../runbooks/) for detailed operational procedures.