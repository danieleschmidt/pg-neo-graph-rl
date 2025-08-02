# Automation Infrastructure

This document describes the comprehensive automation infrastructure implemented for PG-Neo-Graph-RL.

## Overview

The automation system provides:
- **Metrics Collection**: Automated system, application, and project metrics
- **Reporting**: Scheduled and on-demand report generation and distribution  
- **Task Orchestration**: Centralized management of all automation tasks
- **Real-time Monitoring**: Web-based dashboard for live system visibility
- **Integration Management**: Unified control of all automation components

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    Integration Manager                          │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐  │
│  │   Service       │  │   Health        │  │   Configuration │  │
│  │   Manager       │  │   Monitor       │  │   Manager       │  │
│  └─────────────────┘  └─────────────────┘  └─────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
           │                    │                    │
           ▼                    ▼                    ▼
┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐
│   Automation    │  │   Monitoring    │  │   Metrics       │
│   Orchestrator  │  │   Dashboard     │  │   Collector     │
└─────────────────┘  └─────────────────┘  └─────────────────┘
           │                    │                    │
           ▼                    ▼                    ▼
┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐
│   Scheduled     │  │   Real-time     │  │   System &      │
│   Tasks         │  │   WebSocket     │  │   App Metrics   │
└─────────────────┘  └─────────────────┘  └─────────────────┘
           │                    │                    │
           ▼                    ▼                    ▼
┌─────────────────────────────────────────────────────────────────┐
│                    Automated Reporting                         │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐  │
│  │   Email         │  │   Slack         │  │   File Output   │  │
│  │   Reports       │  │   Notifications │  │   & Archives    │  │
│  └─────────────────┘  └─────────────────┘  └─────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
```

## Components

### 1. Metrics Collector (`scripts/metrics_collector.py`)

Collects comprehensive metrics across three categories:

**System Metrics:**
- CPU, memory, and disk usage
- Network I/O statistics
- Process counts and load averages

**Application Metrics:**
- JAX device availability and platform
- Training and federated learning metrics
- Performance timing data

**Project Metrics:**
- Code coverage and test results
- Code quality scores
- Security vulnerabilities
- Lines of code and technical debt

**Usage:**
```bash
# Collect current metrics
python scripts/metrics_collector.py collect

# Generate comprehensive report
python scripts/metrics_collector.py report --format markdown --output report.md

# Start continuous monitoring
python scripts/metrics_collector.py monitor

# Check for alerts only
python scripts/metrics_collector.py alert
```

### 2. Automated Reporting (`scripts/automated_reporting.py`)

Generates and distributes scheduled reports via multiple channels:

**Report Types:**
- Daily health checks
- Weekly executive summaries
- Monthly detailed reports
- Custom threshold-based alerts

**Delivery Methods:**
- Email with attachments
- Slack notifications
- Webhooks for integration
- File system storage

**Usage:**
```bash
# Run all configured reports
python scripts/automated_reporting.py run

# Run specific report
python scripts/automated_reporting.py run --report daily_health_check

# Test report delivery
python scripts/automated_reporting.py test
```

### 3. Automation Orchestrator (`scripts/automation_orchestrator.py`)

Centralized task scheduling and execution engine:

**Features:**
- Cron-like scheduling with natural language
- Dependency management between tasks
- Automatic retry with backoff
- Health monitoring and auto-restart
- Priority-based execution

**Default Tasks:**
- Daily metrics collection
- Code quality checks every 4 hours
- Security scans daily at 2 AM
- Full test suite nightly
- Weekly dependency updates
- Log rotation and cleanup

**Usage:**
```bash
# Start the orchestrator daemon
python scripts/automation_orchestrator.py start

# Run specific task
python scripts/automation_orchestrator.py run --task daily_metrics_collection

# Check status of all tasks
python scripts/automation_orchestrator.py status
```

### 4. Monitoring Dashboard (`scripts/monitoring_dashboard.py`)

Real-time web-based monitoring interface:

**Features:**
- Live metrics visualization
- Real-time alerts and notifications
- Task execution controls
- WebSocket-based updates
- Mobile-responsive design

**Access:**
- URL: `http://localhost:8080`
- API endpoints: `/api/metrics`, `/api/automation`, `/api/health`
- WebSocket: `/ws` for real-time updates

**Usage:**
```bash
# Start dashboard server
python scripts/monitoring_dashboard.py

# Custom host/port
python scripts/monitoring_dashboard.py --host 0.0.0.0 --port 9090
```

### 5. Integration Manager (`scripts/integration_manager.py`)

Unified control system for all automation components:

**Features:**
- Service lifecycle management
- Health monitoring with auto-restart
- Centralized configuration
- Comprehensive status reporting
- Graceful shutdown handling

**Usage:**
```bash
# Start complete automation stack
python scripts/integration_manager.py start

# Check overall status
python scripts/integration_manager.py status

# Run health check
python scripts/integration_manager.py health

# Stop all services
python scripts/integration_manager.py stop
```

## Configuration

### Directory Structure
```
config/
├── README.md                 # Configuration documentation
├── integration.json          # Integration manager config (auto-generated)
├── automation.json           # Task definitions (auto-generated)
└── reporting.json            # Report configurations (auto-generated)
```

### Environment Variables
```bash
# Logging
export LOG_LEVEL=INFO
export LOG_FORMAT=text

# Dashboard
export DASHBOARD_HOST=0.0.0.0
export DASHBOARD_PORT=8080

# Automation
export AUTOMATION_ENABLED=true
export METRICS_ENABLED=true

# Email reporting (optional)
export EMAIL_SMTP_SERVER=smtp.gmail.com
export EMAIL_USERNAME=reports@example.com
export EMAIL_PASSWORD=your-app-password

# Slack integration (optional)
export SLACK_WEBHOOK_URL=https://hooks.slack.com/services/YOUR/SLACK/WEBHOOK
```

## Quick Start

1. **Install automation dependencies:**
   ```bash
   pip install -r requirements-automation.txt
   ```

2. **Start the complete automation stack:**
   ```bash
   python scripts/integration_manager.py start
   ```

3. **Access the monitoring dashboard:**
   Open `http://localhost:8080` in your browser

4. **Check system status:**
   ```bash
   python scripts/integration_manager.py status
   ```

## Customization

### Adding Custom Tasks

Edit `config/automation.json` to add new scheduled tasks:

```json
{
  "name": "custom_backup",
  "description": "Backup important data",
  "command": "python scripts/backup.py",
  "schedule": "daily at 03:00",
  "priority": 2,
  "timeout": 600,
  "enabled": true
}
```

### Custom Metrics

Extend the metrics collector by adding new collectors:

```python
class CustomMetricsCollector:
    def collect_metrics(self):
        # Your custom metrics logic
        return custom_metrics
```

### Report Templates

Create custom report formats by extending the `MetricsReporter` class:

```python
def custom_format_report(self, report):
    # Your custom formatting logic
    return formatted_content
```

### Dashboard Customization

The dashboard HTML can be customized by modifying the `_generate_dashboard_html()` method in the `MonitoringDashboard` class.

## Monitoring and Alerting

### Default Alert Thresholds
- CPU usage > 80%
- Memory usage > 85%  
- Disk usage > 90%
- Failed tests > 0
- Security issues > 0
- Code coverage < 70%

### Alert Channels
- Dashboard notifications
- Email reports
- Slack messages
- Log entries
- Webhook notifications

## Troubleshooting

### Common Issues

1. **Services not starting:**
   ```bash
   # Check configuration
   python scripts/integration_manager.py config
   
   # Check health
   python scripts/integration_manager.py health
   ```

2. **Dashboard not accessible:**
   - Check if port 8080 is available
   - Verify firewall settings
   - Check service status

3. **Reports not being delivered:**
   - Verify email/Slack credentials
   - Check network connectivity
   - Review log files in `logs/` directory

4. **High resource usage:**
   - Adjust collection intervals
   - Disable unnecessary tasks
   - Monitor system resources

### Log Files

All components log to the `logs/` directory:
- `integration_manager.log` - Main integration logs
- `automation_orchestrator.log` - Task execution logs
- `monitoring_dashboard.log` - Dashboard access logs
- `metrics_collector.log` - Metrics collection logs

### Performance Tuning

For large-scale deployments:
- Increase collection intervals
- Use external database for metrics storage
- Enable horizontal scaling
- Configure load balancing for dashboard

## Security Considerations

- Sensitive credentials stored in environment variables
- Dashboard access controls (implement authentication)
- Webhook URL security
- Log file permissions
- Network security for external integrations

## Integration with CI/CD

The automation system integrates with GitHub Actions workflows:
- Metrics collection in CI pipelines
- Quality gate enforcement
- Automated security scanning
- Performance regression detection

See `docs/workflows/` for GitHub Actions integration examples.

## Support and Maintenance

### Monitoring System Health
- Use `integration_manager.py health` for comprehensive checks
- Monitor dashboard for real-time status
- Set up external monitoring for critical alerts

### Regular Maintenance
- Review and update alert thresholds
- Archive old metrics and reports
- Update automation dependencies
- Rotate webhook URLs and credentials

### Scaling Considerations
- External metrics storage (InfluxDB, Prometheus)
- Load balancing for dashboard
- Distributed task execution
- Message queuing for reliability