"""
Health checking and system monitoring utilities.
"""
import time
import psutil
import threading
from typing import Dict, Any, List, Optional, Callable
from dataclasses import dataclass, field
from enum import Enum
import jax.numpy as jnp

from .logging import get_logger, log_system_health
from .exceptions import ResourceError, ValidationError


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
        """
        Register a new health check.
        
        Args:
            name: Unique name for the check
            check_function: Function that performs the check
            interval_seconds: How often to run the check
            timeout_seconds: Timeout for the check
            warning_threshold: Threshold for warning status
            critical_threshold: Threshold for critical status
        """
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
                result = {\n                    \"status\": HealthStatus.CRITICAL.value,\n                    \"error\": str(e),\n                    \"consecutive_failures\": check.consecutive_failures\n                }\n                results[check.name] = result\n                self.logger.error(f\"Health check '{check.name}' failed: {e}\")\n        \n        return results\n    \n    def _run_single_check(self, check: HealthCheck) -> Dict[str, Any]:\n        \"\"\"Run a single health check with timeout.\"\"\"\n        import signal\n        \n        def timeout_handler(signum, frame):\n            raise TimeoutError(f\"Health check '{check.name}' timed out\")\n        \n        # Set timeout\n        old_handler = signal.signal(signal.SIGALRM, timeout_handler)\n        signal.alarm(int(check.timeout_seconds))\n        \n        try:\n            # Run the check\n            metrics = check.check_function()\n            \n            # Determine status based on thresholds\n            status = self._determine_status(metrics, check)\n            check.last_status = status\n            \n            result = {\n                \"status\": status.value,\n                \"metrics\": metrics,\n                \"last_check\": time.time()\n            }\n            \n            # Log based on status\n            log_system_health(self.logger, check.name, status.value, metrics)\n            \n            return result\n            \n        finally:\n            signal.alarm(0)\n            signal.signal(signal.SIGALRM, old_handler)\n    \n    def _determine_status(self, metrics: Dict[str, Any], check: HealthCheck) -> HealthStatus:\n        \"\"\"Determine health status based on metrics and thresholds.\"\"\"\n        # Look for a primary metric to evaluate\n        primary_metrics = [\"usage\", \"percentage\", \"value\", \"count\"]\n        \n        for metric_name in primary_metrics:\n            if metric_name in metrics:\n                value = metrics[metric_name]\n                \n                if check.critical_threshold and value >= check.critical_threshold:\n                    return HealthStatus.CRITICAL\n                elif check.warning_threshold and value >= check.warning_threshold:\n                    return HealthStatus.DEGRADED\n                else:\n                    return HealthStatus.HEALTHY\n        \n        # If no thresholds or primary metrics, assume healthy if no errors\n        if \"error\" in metrics:\n            return HealthStatus.CRITICAL\n        else:\n            return HealthStatus.HEALTHY\n    \n    def _check_cpu_usage(self) -> Dict[str, Any]:\n        \"\"\"Check CPU usage.\"\"\"\n        cpu_percent = psutil.cpu_percent(interval=1.0)\n        cpu_count = psutil.cpu_count()\n        \n        return {\n            \"usage\": cpu_percent,\n            \"cpu_count\": cpu_count,\n            \"load_average\": psutil.getloadavg() if hasattr(psutil, 'getloadavg') else None\n        }\n    \n    def _check_memory_usage(self) -> Dict[str, Any]:\n        \"\"\"Check memory usage.\"\"\"\n        memory = psutil.virtual_memory()\n        \n        return {\n            \"usage\": memory.percent,\n            \"total_gb\": memory.total / (1024**3),\n            \"available_gb\": memory.available / (1024**3),\n            \"used_gb\": memory.used / (1024**3)\n        }\n    \n    def _check_disk_usage(self) -> Dict[str, Any]:\n        \"\"\"Check disk usage.\"\"\"\n        disk = psutil.disk_usage('/')\n        \n        return {\n            \"usage\": (disk.used / disk.total) * 100,\n            \"total_gb\": disk.total / (1024**3),\n            \"used_gb\": disk.used / (1024**3),\n            \"free_gb\": disk.free / (1024**3)\n        }\n    \n    def _check_jax_devices(self) -> Dict[str, Any]:\n        \"\"\"Check JAX device availability.\"\"\"\n        try:\n            import jax\n            devices = jax.devices()\n            \n            device_info = []\n            for device in devices:\n                device_info.append({\n                    \"id\": device.id,\n                    \"platform\": device.platform,\n                    \"device_kind\": device.device_kind\n                })\n            \n            return {\n                \"device_count\": len(devices),\n                \"devices\": device_info,\n                \"default_backend\": jax.default_backend()\n            }\n            \n        except ImportError:\n            return {\"error\": \"JAX not available\"}\n        except Exception as e:\n            return {\"error\": f\"JAX device check failed: {str(e)}\"}\n    \n    def get_overall_status(self) -> HealthStatus:\n        \"\"\"Get overall system health status.\"\"\"\n        with self._lock:\n            statuses = [check.last_status for check in self.checks.values() if check.enabled]\n        \n        if not statuses:\n            return HealthStatus.UNKNOWN\n        \n        # If any check is critical, overall is critical\n        if HealthStatus.CRITICAL in statuses:\n            return HealthStatus.CRITICAL\n        \n        # If any check is degraded, overall is degraded\n        if HealthStatus.DEGRADED in statuses:\n            return HealthStatus.DEGRADED\n        \n        # If all are healthy, overall is healthy\n        if all(status == HealthStatus.HEALTHY for status in statuses):\n            return HealthStatus.HEALTHY\n        \n        return HealthStatus.UNKNOWN\n    \n    def get_health_summary(self) -> Dict[str, Any]:\n        \"\"\"Get comprehensive health summary.\"\"\"\n        with self._lock:\n            check_results = {}\n            for name, check in self.checks.items():\n                if check.enabled:\n                    check_results[name] = {\n                        \"status\": check.last_status.value,\n                        \"last_check\": check.last_check_time,\n                        \"consecutive_failures\": check.consecutive_failures\n                    }\n        \n        return {\n            \"overall_status\": self.get_overall_status().value,\n            \"timestamp\": time.time(),\n            \"checks\": check_results\n        }\n\n\nclass FederatedHealthMonitor:\n    \"\"\"Health monitor specifically for federated learning components.\"\"\"\n    \n    def __init__(self):\n        self.logger = get_logger(\"pg_neo_graph_rl.federated_health\")\n        self.agent_health = {}\n        self.communication_health = {}\n        self._lock = threading.Lock()\n    \n    def check_agent_health(self, agent_id: int, metrics: Dict[str, Any]) -> HealthStatus:\n        \"\"\"Check health of a specific agent.\"\"\"\n        with self._lock:\n            # Check for common issues\n            issues = []\n            \n            # Check loss convergence\n            if \"loss\" in metrics:\n                loss = metrics[\"loss\"]\n                if loss > 100.0:  # Very high loss\n                    issues.append(\"high_loss\")\n                elif jnp.isnan(loss) or jnp.isinf(loss):\n                    issues.append(\"invalid_loss\")\n            \n            # Check gradient health\n            if \"gradient_norm\" in metrics:\n                grad_norm = metrics[\"gradient_norm\"]\n                if grad_norm > 10.0:  # Exploding gradients\n                    issues.append(\"exploding_gradients\")\n                elif grad_norm < 1e-8:  # Vanishing gradients\n                    issues.append(\"vanishing_gradients\")\n            \n            # Check memory usage\n            if \"memory_usage\" in metrics and metrics[\"memory_usage\"] > 0.9:\n                issues.append(\"high_memory\")\n            \n            # Determine status\n            if \"invalid_loss\" in issues or \"exploding_gradients\" in issues:\n                status = HealthStatus.CRITICAL\n            elif issues:\n                status = HealthStatus.DEGRADED\n            else:\n                status = HealthStatus.HEALTHY\n            \n            # Store agent health\n            self.agent_health[agent_id] = {\n                \"status\": status,\n                \"issues\": issues,\n                \"metrics\": metrics,\n                \"timestamp\": time.time()\n            }\n            \n            return status\n    \n    def check_communication_health(self, \n                                 round_number: int,\n                                 num_agents: int,\n                                 successful_updates: int,\n                                 communication_time: float) -> HealthStatus:\n        \"\"\"Check health of federated communication.\"\"\"\n        with self._lock:\n            issues = []\n            \n            # Check participation rate\n            participation_rate = successful_updates / num_agents if num_agents > 0 else 0\n            if participation_rate < 0.5:\n                issues.append(\"low_participation\")\n            elif participation_rate < 0.8:\n                issues.append(\"reduced_participation\")\n            \n            # Check communication time\n            if communication_time > 60.0:  # More than 1 minute\n                issues.append(\"slow_communication\")\n            \n            # Determine status\n            if \"low_participation\" in issues:\n                status = HealthStatus.CRITICAL\n            elif issues:\n                status = HealthStatus.DEGRADED\n            else:\n                status = HealthStatus.HEALTHY\n            \n            # Store communication health\n            self.communication_health[round_number] = {\n                \"status\": status,\n                \"issues\": issues,\n                \"participation_rate\": participation_rate,\n                \"communication_time\": communication_time,\n                \"timestamp\": time.time()\n            }\n            \n            return status\n    \n    def get_federated_health_summary(self) -> Dict[str, Any]:\n        \"\"\"Get summary of federated learning health.\"\"\"\n        with self._lock:\n            # Agent health summary\n            agent_statuses = [info[\"status\"] for info in self.agent_health.values()]\n            agent_issues = []\n            for info in self.agent_health.values():\n                agent_issues.extend(info[\"issues\"])\n            \n            # Communication health summary (last 10 rounds)\n            recent_comm = dict(list(self.communication_health.items())[-10:])\n            comm_statuses = [info[\"status\"] for info in recent_comm.values()]\n            \n            return {\n                \"agent_health\": {\n                    \"total_agents\": len(self.agent_health),\n                    \"healthy_agents\": sum(1 for s in agent_statuses if s == HealthStatus.HEALTHY),\n                    \"degraded_agents\": sum(1 for s in agent_statuses if s == HealthStatus.DEGRADED),\n                    \"critical_agents\": sum(1 for s in agent_statuses if s == HealthStatus.CRITICAL),\n                    \"common_issues\": list(set(agent_issues))\n                },\n                \"communication_health\": {\n                    \"recent_rounds\": len(recent_comm),\n                    \"healthy_rounds\": sum(1 for s in comm_statuses if s == HealthStatus.HEALTHY),\n                    \"issues_detected\": len(comm_statuses) - sum(1 for s in comm_statuses if s == HealthStatus.HEALTHY)\n                },\n                \"timestamp\": time.time()\n            }\n\n\ndef create_health_monitor(enable_monitoring: bool = True) -> SystemHealthMonitor:\n    \"\"\"Create and optionally start a system health monitor.\"\"\"\n    monitor = SystemHealthMonitor()\n    \n    if enable_monitoring:\n        monitor.start_monitoring()\n    \n    return monitor"