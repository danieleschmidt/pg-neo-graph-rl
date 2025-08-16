"""
Health checking and system monitoring utilities.
"""
import threading
import time

try:
    import psutil
except ImportError:
    psutil = None
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, Optional

import jax.numpy as jnp

from .logging import get_logger, log_system_health


class HealthStatus(Enum):
    """Health status levels."""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    CRITICAL = "critical"
    UNKNOWN = "unknown"


@dataclass
class HealthCheck:
    """Individual health check configuration."""
    name: str
    check_function: Callable[[], Dict[str, Any]]
    interval_seconds: float = 60.0
    timeout_seconds: float = 10.0
    critical_threshold: Optional[float] = None
    warning_threshold: Optional[float] = None
    enabled: bool = True
    last_check_time: float = field(default_factory=time.time)
    last_status: HealthStatus = HealthStatus.UNKNOWN
    consecutive_failures: int = 0


class SystemHealthMonitor:
    """Monitors system health and resource usage."""

    def __init__(self, check_interval: float = 30.0):
        self.check_interval = check_interval
        self.logger = get_logger("pg_neo_graph_rl.health")
        self.checks: Dict[str, HealthCheck] = {}
        self.monitoring = False
        self.monitor_thread = None
        self._lock = threading.Lock()

        # Register default health checks
        self._register_default_checks()

    def _register_default_checks(self) -> None:
        """Register default system health checks."""

        # CPU usage check
        self.register_check(
            "cpu_usage",
            self._check_cpu_usage,
            interval_seconds=30.0,
            warning_threshold=70.0,
            critical_threshold=90.0
        )

        # Memory usage check
        self.register_check(
            "memory_usage",
            self._check_memory_usage,
            interval_seconds=30.0,
            warning_threshold=80.0,
            critical_threshold=95.0
        )

        # Disk usage check
        self.register_check(
            "disk_usage",
            self._check_disk_usage,
            interval_seconds=60.0,
            warning_threshold=85.0,
            critical_threshold=95.0
        )

        # JAX device check
        self.register_check(
            "jax_devices",
            self._check_jax_devices,
            interval_seconds=120.0
        )

    def register_check(self,
                      name: str,
                      check_function: Callable[[], Dict[str, Any]],
                      interval_seconds: float = 60.0,
                      timeout_seconds: float = 10.0,
                      warning_threshold: Optional[float] = None,
                      critical_threshold: Optional[float] = None) -> None:
        """Register a new health check."""
        with self._lock:
            self.checks[name] = HealthCheck(
                name=name,
                check_function=check_function,
                interval_seconds=interval_seconds,
                timeout_seconds=timeout_seconds,
                warning_threshold=warning_threshold,
                critical_threshold=critical_threshold
            )

    def start_monitoring(self) -> None:
        """Start the health monitoring thread."""
        if self.monitoring:
            return

        self.monitoring = True
        self.monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self.monitor_thread.start()
        self.logger.info("Health monitoring started")

    def stop_monitoring(self) -> None:
        """Stop the health monitoring thread."""
        self.monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=5.0)
        self.logger.info("Health monitoring stopped")

    def _monitor_loop(self) -> None:
        """Main monitoring loop."""
        while self.monitoring:
            try:
                self.run_all_checks()
                time.sleep(self.check_interval)
            except Exception as e:
                self.logger.error(f"Error in monitoring loop: {e}")
                time.sleep(self.check_interval)

    def run_all_checks(self) -> Dict[str, Dict[str, Any]]:
        """Run all registered health checks."""
        results = {}

        with self._lock:
            checks_to_run = list(self.checks.values())

        for check in checks_to_run:
            if not check.enabled:
                continue

            current_time = time.time()
            if current_time - check.last_check_time < check.interval_seconds:
                continue

            try:
                result = self._run_single_check(check)
                results[check.name] = result
                check.last_check_time = current_time
                check.consecutive_failures = 0

            except Exception as e:
                check.consecutive_failures += 1
                result = {
                    "status": HealthStatus.CRITICAL.value,
                    "error": str(e),
                    "consecutive_failures": check.consecutive_failures
                }
                results[check.name] = result
                self.logger.error(f"Health check '{check.name}' failed: {e}")

        return results

    def _run_single_check(self, check: HealthCheck) -> Dict[str, Any]:
        """Run a single health check with timeout."""
        try:
            # Run the check
            metrics = check.check_function()

            # Determine status based on thresholds
            status = self._determine_status(metrics, check)
            check.last_status = status

            result = {
                "status": status.value,
                "metrics": metrics,
                "last_check": time.time()
            }

            # Log based on status
            log_system_health(self.logger, check.name, status.value, metrics)

            return result

        except Exception as e:
            raise e

    def _determine_status(self, metrics: Dict[str, Any], check: HealthCheck) -> HealthStatus:
        """Determine health status based on metrics and thresholds."""
        # Look for a primary metric to evaluate
        primary_metrics = ["usage", "percentage", "value", "count"]

        for metric_name in primary_metrics:
            if metric_name in metrics:
                value = metrics[metric_name]

                if check.critical_threshold and value >= check.critical_threshold:
                    return HealthStatus.CRITICAL
                elif check.warning_threshold and value >= check.warning_threshold:
                    return HealthStatus.DEGRADED
                else:
                    return HealthStatus.HEALTHY

        # If no thresholds or primary metrics, assume healthy if no errors
        if "error" in metrics:
            return HealthStatus.CRITICAL
        else:
            return HealthStatus.HEALTHY

    def _check_cpu_usage(self) -> Dict[str, Any]:
        """Check CPU usage."""
        if psutil is None:
            return {"error": "psutil not available", "usage": 0.0}

        cpu_percent = psutil.cpu_percent(interval=1.0)
        cpu_count = psutil.cpu_count()

        return {
            "usage": cpu_percent,
            "cpu_count": cpu_count,
            "load_average": psutil.getloadavg() if hasattr(psutil, 'getloadavg') else None
        }

    def _check_memory_usage(self) -> Dict[str, Any]:
        """Check memory usage."""
        if psutil is None:
            return {"error": "psutil not available", "usage": 0.0}

        memory = psutil.virtual_memory()

        return {
            "usage": memory.percent,
            "total_gb": memory.total / (1024**3),
            "available_gb": memory.available / (1024**3),
            "used_gb": memory.used / (1024**3)
        }

    def _check_disk_usage(self) -> Dict[str, Any]:
        """Check disk usage."""
        if psutil is None:
            return {"error": "psutil not available", "usage": 0.0}

        disk = psutil.disk_usage('/')

        return {
            "usage": (disk.used / disk.total) * 100,
            "total_gb": disk.total / (1024**3),
            "used_gb": disk.used / (1024**3),
            "free_gb": disk.free / (1024**3)
        }

    def _check_jax_devices(self) -> Dict[str, Any]:
        """Check JAX device availability."""
        try:
            import jax
            devices = jax.devices()

            device_info = []
            for device in devices:
                device_info.append({
                    "id": device.id,
                    "platform": device.platform,
                    "device_kind": device.device_kind
                })

            return {
                "device_count": len(devices),
                "devices": device_info,
                "default_backend": jax.default_backend()
            }

        except ImportError:
            return {"error": "JAX not available"}
        except Exception as e:
            return {"error": f"JAX device check failed: {str(e)}"}

    def get_overall_status(self) -> HealthStatus:
        """Get overall system health status."""
        with self._lock:
            statuses = [check.last_status for check in self.checks.values() if check.enabled]

        if not statuses:
            return HealthStatus.UNKNOWN

        # If any check is critical, overall is critical
        if HealthStatus.CRITICAL in statuses:
            return HealthStatus.CRITICAL

        # If any check is degraded, overall is degraded
        if HealthStatus.DEGRADED in statuses:
            return HealthStatus.DEGRADED

        # If all are healthy, overall is healthy
        if all(status == HealthStatus.HEALTHY for status in statuses):
            return HealthStatus.HEALTHY

        return HealthStatus.UNKNOWN

    def get_health_summary(self) -> Dict[str, Any]:
        """Get comprehensive health summary."""
        with self._lock:
            check_results = {}
            for name, check in self.checks.items():
                if check.enabled:
                    check_results[name] = {
                        "status": check.last_status.value,
                        "last_check": check.last_check_time,
                        "consecutive_failures": check.consecutive_failures
                    }

        return {
            "overall_status": self.get_overall_status().value,
            "timestamp": time.time(),
            "checks": check_results
        }


class FederatedHealthMonitor:
    """Health monitor specifically for federated learning components."""

    def __init__(self):
        self.logger = get_logger("pg_neo_graph_rl.federated_health")
        self.agent_health = {}
        self.communication_health = {}
        self._lock = threading.Lock()

    def check_agent_health(self, agent_id: int, metrics: Dict[str, Any]) -> HealthStatus:
        """Check health of a specific agent."""
        with self._lock:
            # Check for common issues
            issues = []

            # Check loss convergence
            if "loss" in metrics:
                loss = metrics["loss"]
                if loss > 100.0:  # Very high loss
                    issues.append("high_loss")
                elif jnp.isnan(loss) or jnp.isinf(loss):
                    issues.append("invalid_loss")

            # Check gradient health
            if "gradient_norm" in metrics:
                grad_norm = metrics["gradient_norm"]
                if grad_norm > 10.0:  # Exploding gradients
                    issues.append("exploding_gradients")
                elif grad_norm < 1e-8:  # Vanishing gradients
                    issues.append("vanishing_gradients")

            # Check memory usage
            if "memory_usage" in metrics and metrics["memory_usage"] > 0.9:
                issues.append("high_memory")

            # Determine status
            if "invalid_loss" in issues or "exploding_gradients" in issues:
                status = HealthStatus.CRITICAL
            elif issues:
                status = HealthStatus.DEGRADED
            else:
                status = HealthStatus.HEALTHY

            # Store agent health
            self.agent_health[agent_id] = {
                "status": status,
                "issues": issues,
                "metrics": metrics,
                "timestamp": time.time()
            }

            return status

    def check_communication_health(self,
                                 round_number: int,
                                 num_agents: int,
                                 successful_updates: int,
                                 communication_time: float) -> HealthStatus:
        """Check health of federated communication."""
        with self._lock:
            issues = []

            # Check participation rate
            participation_rate = successful_updates / num_agents if num_agents > 0 else 0
            if participation_rate < 0.5:
                issues.append("low_participation")
            elif participation_rate < 0.8:
                issues.append("reduced_participation")

            # Check communication time
            if communication_time > 60.0:  # More than 1 minute
                issues.append("slow_communication")

            # Determine status
            if "low_participation" in issues:
                status = HealthStatus.CRITICAL
            elif issues:
                status = HealthStatus.DEGRADED
            else:
                status = HealthStatus.HEALTHY

            # Store communication health
            self.communication_health[round_number] = {
                "status": status,
                "issues": issues,
                "participation_rate": participation_rate,
                "communication_time": communication_time,
                "timestamp": time.time()
            }

            return status

    def get_federated_health_summary(self) -> Dict[str, Any]:
        """Get summary of federated learning health."""
        with self._lock:
            # Agent health summary
            agent_statuses = [info["status"] for info in self.agent_health.values()]
            agent_issues = []
            for info in self.agent_health.values():
                agent_issues.extend(info["issues"])

            # Communication health summary (last 10 rounds)
            recent_comm = dict(list(self.communication_health.items())[-10:])
            comm_statuses = [info["status"] for info in recent_comm.values()]

            return {
                "agent_health": {
                    "total_agents": len(self.agent_health),
                    "healthy_agents": sum(1 for s in agent_statuses if s == HealthStatus.HEALTHY),
                    "degraded_agents": sum(1 for s in agent_statuses if s == HealthStatus.DEGRADED),
                    "critical_agents": sum(1 for s in agent_statuses if s == HealthStatus.CRITICAL),
                    "common_issues": list(set(agent_issues))
                },
                "communication_health": {
                    "recent_rounds": len(recent_comm),
                    "healthy_rounds": sum(1 for s in comm_statuses if s == HealthStatus.HEALTHY),
                    "issues_detected": len(comm_statuses) - sum(1 for s in comm_statuses if s == HealthStatus.HEALTHY)
                },
                "timestamp": time.time()
            }


def create_health_monitor(enable_monitoring: bool = True) -> SystemHealthMonitor:
    """Create and optionally start a system health monitor."""
    monitor = SystemHealthMonitor()

    if enable_monitoring:
        monitor.start_monitoring()

    return monitor
