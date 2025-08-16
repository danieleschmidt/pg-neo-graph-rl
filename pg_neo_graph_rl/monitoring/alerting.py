"""
Advanced alerting and monitoring system for pg-neo-graph-rl.
Provides real-time alerts, health monitoring, and performance tracking.
"""

import asyncio
import threading
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional

import jax.numpy as jnp

from ..utils.logging import get_logger

logger = get_logger(__name__)


class AlertSeverity(Enum):
    """Alert severity levels."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class AlertType(Enum):
    """Types of alerts."""
    PERFORMANCE = "performance"
    ERROR = "error"
    SECURITY = "security"
    RESOURCE = "resource"
    FEDERATED = "federated"
    TRAINING = "training"


@dataclass
class Alert:
    """Alert data structure."""
    id: str
    type: AlertType
    severity: AlertSeverity
    title: str
    message: str
    timestamp: float = field(default_factory=time.time)
    metadata: Dict[str, Any] = field(default_factory=dict)
    resolved: bool = False
    acknowledged: bool = False


@dataclass
class MetricThreshold:
    """Metric threshold configuration."""
    metric_name: str
    warning_threshold: float
    critical_threshold: float
    comparison: str = "gt"  # gt, lt, eq
    window_size: int = 5  # Number of samples to consider
    min_samples: int = 3  # Minimum samples before alerting


class HealthMonitor:
    """
    Comprehensive health monitoring system.
    Tracks system metrics and generates alerts.
    """

    def __init__(self, check_interval: float = 30.0):
        self.check_interval = check_interval
        self.is_running = False
        self.health_checks: Dict[str, Callable[[], bool]] = {}
        self.metrics: Dict[str, deque] = defaultdict(lambda: deque(maxlen=100))
        self.thresholds: Dict[str, MetricThreshold] = {}
        self.alerts: List[Alert] = []
        self.alert_handlers: List[Callable[[Alert], None]] = []
        self._lock = threading.Lock()

    def register_health_check(self, name: str, check_func: Callable[[], bool]):
        """Register a health check function."""
        self.health_checks[name] = check_func
        logger.info(f"Registered health check: {name}")

    def add_metric_threshold(self, threshold: MetricThreshold):
        """Add metric threshold for monitoring."""
        self.thresholds[threshold.metric_name] = threshold
        logger.info(f"Added threshold for metric: {threshold.metric_name}")

    def add_alert_handler(self, handler: Callable[[Alert], None]):
        """Add alert handler function."""
        self.alert_handlers.append(handler)

    def record_metric(self, metric_name: str, value: float):
        """Record a metric value."""
        with self._lock:
            self.metrics[metric_name].append({
                'value': value,
                'timestamp': time.time()
            })

        # Check thresholds
        self._check_metric_thresholds(metric_name)

    def _check_metric_thresholds(self, metric_name: str):
        """Check if metric exceeds thresholds."""
        if metric_name not in self.thresholds:
            return

        threshold = self.thresholds[metric_name]
        recent_values = list(self.metrics[metric_name])[-threshold.window_size:]

        if len(recent_values) < threshold.min_samples:
            return

        # Calculate average of recent values
        avg_value = sum(item['value'] for item in recent_values) / len(recent_values)

        # Check thresholds
        severity = None
        if threshold.comparison == "gt":
            if avg_value > threshold.critical_threshold:
                severity = AlertSeverity.CRITICAL
            elif avg_value > threshold.warning_threshold:
                severity = AlertSeverity.HIGH
        elif threshold.comparison == "lt":
            if avg_value < threshold.critical_threshold:
                severity = AlertSeverity.CRITICAL
            elif avg_value < threshold.warning_threshold:
                severity = AlertSeverity.HIGH

        if severity:
            self._create_alert(
                AlertType.PERFORMANCE,
                severity,
                f"Metric threshold exceeded: {metric_name}",
                f"Metric {metric_name} = {avg_value:.4f}, threshold = {threshold.warning_threshold}",
                {"metric_name": metric_name, "value": avg_value, "threshold": threshold.warning_threshold}
            )

    def _create_alert(self,
                     alert_type: AlertType,
                     severity: AlertSeverity,
                     title: str,
                     message: str,
                     metadata: Optional[Dict[str, Any]] = None):
        """Create and process an alert."""
        alert = Alert(
            id=f"{alert_type.value}_{int(time.time())}_{len(self.alerts)}",
            type=alert_type,
            severity=severity,
            title=title,
            message=message,
            metadata=metadata or {}
        )

        with self._lock:
            self.alerts.append(alert)

        logger.warning(f"ALERT [{severity.value.upper()}]: {title} - {message}")

        # Notify handlers
        for handler in self.alert_handlers:
            try:
                handler(alert)
            except Exception as e:
                logger.error(f"Alert handler failed: {e}")

    async def start_monitoring(self):
        """Start continuous health monitoring."""
        self.is_running = True
        logger.info("Starting health monitoring")

        while self.is_running:
            try:
                await self._perform_health_checks()
                await asyncio.sleep(self.check_interval)
            except Exception as e:
                logger.error(f"Health check error: {e}")
                await asyncio.sleep(5)  # Brief pause on error

    def stop_monitoring(self):
        """Stop health monitoring."""
        self.is_running = False
        logger.info("Stopping health monitoring")

    async def _perform_health_checks(self):
        """Perform all registered health checks."""
        for name, check_func in self.health_checks.items():
            try:
                is_healthy = check_func()
                self.record_metric(f"health_{name}", 1.0 if is_healthy else 0.0)

                if not is_healthy:
                    self._create_alert(
                        AlertType.ERROR,
                        AlertSeverity.HIGH,
                        f"Health check failed: {name}",
                        f"Health check '{name}' returned unhealthy status",
                        {"component": name}
                    )
            except Exception as e:
                logger.error(f"Health check '{name}' failed with exception: {e}")
                self._create_alert(
                    AlertType.ERROR,
                    AlertSeverity.CRITICAL,
                    f"Health check error: {name}",
                    f"Health check '{name}' threw exception: {str(e)}",
                    {"component": name, "exception": str(e)}
                )

    def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive system status."""
        with self._lock:
            recent_alerts = [a for a in self.alerts if time.time() - a.timestamp < 3600]

            # Health status by component
            health_status = {}
            for name in self.health_checks.keys():
                health_metric = f"health_{name}"
                if health_metric in self.metrics and self.metrics[health_metric]:
                    latest = self.metrics[health_metric][-1]
                    health_status[name] = latest['value'] > 0.5
                else:
                    health_status[name] = None

            # Alert summary
            alert_summary = defaultdict(int)
            for alert in recent_alerts:
                if not alert.resolved:
                    alert_summary[alert.severity.value] += 1

            return {
                "timestamp": time.time(),
                "overall_health": all(status for status in health_status.values() if status is not None),
                "component_health": health_status,
                "recent_alerts": len(recent_alerts),
                "unresolved_alerts": alert_summary,
                "total_alerts": len(self.alerts),
                "metrics_tracked": len(self.metrics)
            }

    def acknowledge_alert(self, alert_id: str) -> bool:
        """Acknowledge an alert."""
        with self._lock:
            for alert in self.alerts:
                if alert.id == alert_id:
                    alert.acknowledged = True
                    logger.info(f"Alert acknowledged: {alert_id}")
                    return True
        return False

    def resolve_alert(self, alert_id: str) -> bool:
        """Resolve an alert."""
        with self._lock:
            for alert in self.alerts:
                if alert.id == alert_id:
                    alert.resolved = True
                    alert.acknowledged = True
                    logger.info(f"Alert resolved: {alert_id}")
                    return True
        return False


class PerformanceMonitor:
    """
    Performance monitoring for federated learning components.
    """

    def __init__(self, health_monitor: HealthMonitor):
        self.health_monitor = health_monitor
        self.training_metrics = defaultdict(list)
        self.communication_metrics = defaultdict(list)

        # Set up performance thresholds
        self._setup_thresholds()

    def _setup_thresholds(self):
        """Setup default performance thresholds."""
        thresholds = [
            MetricThreshold("training_loss", 10.0, 100.0, "gt"),
            MetricThreshold("convergence_rate", 0.001, 0.0001, "lt"),
            MetricThreshold("memory_usage", 0.8, 0.95, "gt"),
            MetricThreshold("communication_latency", 1.0, 5.0, "gt"),
            MetricThreshold("gradient_norm", 10.0, 100.0, "gt"),
        ]

        for threshold in thresholds:
            self.health_monitor.add_metric_threshold(threshold)

    def record_training_metric(self, metric_name: str, value: float, agent_id: Optional[int] = None):
        """Record training performance metric."""
        self.training_metrics[metric_name].append({
            'value': value,
            'timestamp': time.time(),
            'agent_id': agent_id
        })

        self.health_monitor.record_metric(metric_name, value)

    def record_communication_metric(self, metric_name: str, value: float, round_num: Optional[int] = None):
        """Record communication performance metric."""
        self.communication_metrics[metric_name].append({
            'value': value,
            'timestamp': time.time(),
            'round': round_num
        })

        self.health_monitor.record_metric(f"comm_{metric_name}", value)

    def check_training_performance(self,
                                 loss_values: List[float],
                                 episode: int,
                                 agent_id: Optional[int] = None):
        """Check training performance and generate alerts if needed."""
        if not loss_values:
            return

        recent_loss = loss_values[-1]
        self.record_training_metric("training_loss", recent_loss, agent_id)

        # Check for exploding loss
        if recent_loss > 1000:
            self.health_monitor._create_alert(
                AlertType.TRAINING,
                AlertSeverity.CRITICAL,
                "Exploding loss detected",
                f"Training loss = {recent_loss:.4f} at episode {episode}",
                {"loss": recent_loss, "episode": episode, "agent_id": agent_id}
            )

        # Check convergence
        if len(loss_values) >= 10:
            recent_losses = loss_values[-10:]
            loss_variance = jnp.var(jnp.array(recent_losses))

            if loss_variance < 1e-6:
                self.health_monitor._create_alert(
                    AlertType.TRAINING,
                    AlertSeverity.MEDIUM,
                    "Training may have converged",
                    f"Loss variance = {loss_variance:.2e} over last 10 episodes",
                    {"loss_variance": float(loss_variance), "episode": episode}
                )

    def check_federated_performance(self,
                                  communication_time: float,
                                  aggregation_time: float,
                                  round_num: int):
        """Check federated learning performance."""
        self.record_communication_metric("communication_time", communication_time, round_num)
        self.record_communication_metric("aggregation_time", aggregation_time, round_num)

        total_time = communication_time + aggregation_time
        self.record_communication_metric("total_round_time", total_time, round_num)

        # Alert on slow rounds
        if total_time > 30.0:  # 30 seconds
            self.health_monitor._create_alert(
                AlertType.FEDERATED,
                AlertSeverity.HIGH,
                "Slow federated learning round",
                f"Round {round_num} took {total_time:.2f}s (comm: {communication_time:.2f}s, agg: {aggregation_time:.2f}s)",
                {"round": round_num, "total_time": total_time, "comm_time": communication_time, "agg_time": aggregation_time}
            )


class ResourceMonitor:
    """
    System resource monitoring.
    """

    def __init__(self, health_monitor: HealthMonitor):
        self.health_monitor = health_monitor

        # Register resource health checks
        self.health_monitor.register_health_check("memory", self._check_memory)
        self.health_monitor.register_health_check("disk", self._check_disk)

        # Set up resource thresholds
        self._setup_resource_thresholds()

    def _setup_resource_thresholds(self):
        """Setup resource monitoring thresholds."""
        thresholds = [
            MetricThreshold("memory_usage_percent", 80.0, 95.0, "gt"),
            MetricThreshold("disk_usage_percent", 85.0, 95.0, "gt"),
            MetricThreshold("cpu_usage_percent", 90.0, 98.0, "gt"),
        ]

        for threshold in thresholds:
            self.health_monitor.add_metric_threshold(threshold)

    def _check_memory(self) -> bool:
        """Check memory usage."""
        try:
            import psutil
            memory = psutil.virtual_memory()
            usage_percent = memory.percent

            self.health_monitor.record_metric("memory_usage_percent", usage_percent)
            return usage_percent < 90.0

        except ImportError:
            logger.warning("psutil not available for memory monitoring")
            return True
        except Exception as e:
            logger.error(f"Memory check failed: {e}")
            return False

    def _check_disk(self) -> bool:
        """Check disk usage."""
        try:
            import psutil
            disk = psutil.disk_usage('/')
            usage_percent = (disk.used / disk.total) * 100

            self.health_monitor.record_metric("disk_usage_percent", usage_percent)
            return usage_percent < 90.0

        except ImportError:
            logger.warning("psutil not available for disk monitoring")
            return True
        except Exception as e:
            logger.error(f"Disk check failed: {e}")
            return False


# Alert handler implementations
def console_alert_handler(alert: Alert):
    """Simple console alert handler."""
    severity_colors = {
        AlertSeverity.LOW: '\033[92m',      # Green
        AlertSeverity.MEDIUM: '\033[93m',   # Yellow
        AlertSeverity.HIGH: '\033[91m',     # Red
        AlertSeverity.CRITICAL: '\033[95m'  # Magenta
    }
    reset_color = '\033[0m'

    color = severity_colors.get(alert.severity, '')
    print(f"{color}[{alert.severity.value.upper()}] {alert.title}: {alert.message}{reset_color}")


def log_alert_handler(alert: Alert):
    """Log-based alert handler."""
    severity_map = {
        AlertSeverity.LOW: logger.info,
        AlertSeverity.MEDIUM: logger.warning,
        AlertSeverity.HIGH: logger.error,
        AlertSeverity.CRITICAL: logger.critical
    }

    log_func = severity_map.get(alert.severity, logger.info)
    log_func(f"ALERT [{alert.type.value}]: {alert.title} - {alert.message}", extra={
        'alert_id': alert.id,
        'alert_type': alert.type.value,
        'alert_severity': alert.severity.value,
        'metadata': alert.metadata
    })


def create_comprehensive_monitor() -> HealthMonitor:
    """
    Create a comprehensive monitoring system with all components.
    
    Returns:
        Configured HealthMonitor instance
    """
    health_monitor = HealthMonitor()

    # Add alert handlers
    health_monitor.add_alert_handler(console_alert_handler)
    health_monitor.add_alert_handler(log_alert_handler)

    # Initialize performance monitoring
    performance_monitor = PerformanceMonitor(health_monitor)

    # Initialize resource monitoring
    resource_monitor = ResourceMonitor(health_monitor)

    # Register basic health checks
    health_monitor.register_health_check("jax_backend", lambda: True)  # Always healthy for now
    health_monitor.register_health_check("import_system", _check_imports)

    logger.info("Comprehensive monitoring system created")
    return health_monitor


def _check_imports() -> bool:
    """Check if critical imports are working."""
    try:
        import flax
        import jax
        import optax
        return True
    except ImportError:
        return False
