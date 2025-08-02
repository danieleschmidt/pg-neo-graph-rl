# Configuration Directory

This directory contains configuration files for the PG-Neo-Graph-RL automation and monitoring infrastructure.

## Configuration Files

### Core Configuration
- `integration.json` - Main integration manager configuration
- `automation.json` - Automation orchestrator task definitions
- `reporting.json` - Automated reporting configuration

### Service Configuration
- `metrics.json` - Metrics collection configuration
- `monitoring.json` - Monitoring and alerting configuration
- `dashboard.json` - Dashboard customization settings

## Configuration Structure

All configuration files use JSON format and follow a consistent structure:

```json
{
  "metadata": {
    "version": "1.0",
    "created": "timestamp",
    "description": "Configuration description"
  },
  "settings": {
    // Service-specific settings
  },
  "overrides": {
    // Environment-specific overrides
  }
}
```

## Environment Variables

Many configuration values can be overridden using environment variables:

- `LOG_LEVEL` - Logging level (DEBUG, INFO, WARNING, ERROR)
- `LOG_FORMAT` - Log format (text, json)
- `METRICS_ENABLED` - Enable metrics collection (true/false)
- `DASHBOARD_HOST` - Dashboard host (default: 0.0.0.0)
- `DASHBOARD_PORT` - Dashboard port (default: 8080)
- `AUTOMATION_ENABLED` - Enable automation orchestrator (true/false)

## Usage

Configuration files are automatically created with default values on first run. You can customize them as needed for your environment.

### Starting with Custom Configuration

```bash
# Use custom integration config
python scripts/integration_manager.py start --config config/my-integration.json

# Use custom automation config
python scripts/automation_orchestrator.py start --config config/my-automation.json
```

### Configuration Validation

The integration manager includes configuration validation:

```bash
# Check configuration validity
python scripts/integration_manager.py config

# Health check including configuration
python scripts/integration_manager.py health
```

## Security Considerations

- Store sensitive credentials in environment variables, not config files
- Use `.env` files for local development (already gitignored)
- Rotate webhook URLs and API keys regularly
- Limit file permissions on configuration files containing secrets

## Default Configurations

When configuration files are missing, the system will create defaults optimized for:

- Local development environment
- Moderate resource usage
- Essential monitoring and automation
- Secure defaults with minimal external dependencies