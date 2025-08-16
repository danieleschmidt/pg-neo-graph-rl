"""
Auto-scaling utilities for federated graph RL.
"""
import threading
import time
from dataclasses import dataclass
from enum import Enum
from typing import Any, Callable, Dict, Optional

from ..utils.logging import get_logger, get_performance_logger


class ScalingDirection(Enum):
    """Scaling direction."""
    UP = "up"
    DOWN = "down"
    STABLE = "stable"


@dataclass
class ScalingMetrics:
    """Metrics for auto-scaling decisions."""
    cpu_usage: float
    memory_usage: float
    training_throughput: float  # episodes/second
    communication_latency: float  # seconds
    queue_length: int
    error_rate: float
    timestamp: float


@dataclass
class ScalingRule:
    """Rule for auto-scaling."""
    metric_name: str
    threshold_up: float
    threshold_down: float
    cooldown_period: float  # seconds
    min_agents: int = 1
    max_agents: int = 100
    last_triggered: float = 0.0


class AutoScaler:
    """Auto-scales federated learning system based on metrics."""

    def __init__(self, initial_agents: int = 3, max_agents: int = 50):
        self.current_agents = initial_agents
        self.max_agents = max_agents
        self.min_agents = 1
        self.logger = get_logger("pg_neo_graph_rl.auto_scaler")
        self.perf_logger = get_performance_logger()

        # Scaling rules
        self.scaling_rules = {
            "cpu_usage": ScalingRule("cpu_usage", 80.0, 30.0, 300.0, max_agents=max_agents),
            "memory_usage": ScalingRule("memory_usage", 85.0, 40.0, 300.0, max_agents=max_agents),
            "training_throughput": ScalingRule("training_throughput", 0.1, 2.0, 300.0, max_agents=max_agents),
            "communication_latency": ScalingRule("communication_latency", 10.0, 2.0, 300.0, max_agents=max_agents),
            "queue_length": ScalingRule("queue_length", 100, 10, 180.0, max_agents=max_agents),
            "error_rate": ScalingRule("error_rate", 0.05, 0.01, 600.0, max_agents=max_agents)
        }

        # Metrics history
        self.metrics_history = []
        self.scaling_history = []

        # Auto-scaling settings
        self.auto_scaling_enabled = True
        self.scaling_factor = 1.5  # Multiplier for scaling up
        self.min_stable_duration = 600.0  # 10 minutes before scaling down

        # Callbacks
        self.scale_up_callback: Optional[Callable[[int], bool]] = None
        self.scale_down_callback: Optional[Callable[[int], bool]] = None

    def set_scale_callbacks(self,
                          scale_up_callback: Callable[[int], bool],
                          scale_down_callback: Callable[[int], bool]) -> None:
        """Set callbacks for scaling operations."""
        self.scale_up_callback = scale_up_callback
        self.scale_down_callback = scale_down_callback

    def record_metrics(self, metrics: ScalingMetrics) -> None:
        """Record metrics for scaling decisions."""
        self.metrics_history.append(metrics)

        # Keep only recent metrics (last 1 hour worth)
        cutoff_time = time.time() - 3600
        self.metrics_history = [
            m for m in self.metrics_history
            if m.timestamp > cutoff_time
        ]

        # Check if scaling is needed
        if self.auto_scaling_enabled and len(self.metrics_history) >= 5:
            self._evaluate_scaling()

    def _evaluate_scaling(self) -> None:
        """Evaluate if scaling is needed."""
        if len(self.metrics_history) < 5:
            return

        current_time = time.time()
        recent_metrics = self.metrics_history[-5:]  # Last 5 data points

        # Check each scaling rule
        scale_up_votes = 0
        scale_down_votes = 0

        for rule_name, rule in self.scaling_rules.items():
            if current_time - rule.last_triggered < rule.cooldown_period:
                continue  # Still in cooldown

            # Get metric values
            metric_values = []
            for metrics in recent_metrics:
                if hasattr(metrics, rule.metric_name):
                    metric_values.append(getattr(metrics, rule.metric_name))

            if not metric_values:
                continue

            avg_value = sum(metric_values) / len(metric_values)

            # Check thresholds
            if rule_name in ["training_throughput"]:
                # For throughput, low values trigger scale up
                if avg_value < rule.threshold_up:
                    scale_up_votes += 1
                elif avg_value > rule.threshold_down:
                    scale_down_votes += 1
            else:
                # For resource usage, high values trigger scale up
                if avg_value > rule.threshold_up:
                    scale_up_votes += 1
                elif avg_value < rule.threshold_down:
                    scale_down_votes += 1

        # Make scaling decision
        scaling_decision = ScalingDirection.STABLE

        if scale_up_votes >= 2 and self.current_agents < self.max_agents:
            scaling_decision = ScalingDirection.UP
        elif scale_down_votes >= 3 and self.current_agents > self.min_agents:
            # More conservative about scaling down
            scaling_decision = ScalingDirection.DOWN

        # Execute scaling if needed
        if scaling_decision != ScalingDirection.STABLE:
            self._execute_scaling(scaling_decision)

    def _execute_scaling(self, direction: ScalingDirection) -> None:
        """Execute scaling operation."""
        current_time = time.time()

        if direction == ScalingDirection.UP:
            # Scale up: increase by scaling factor
            new_agent_count = min(
                int(self.current_agents * self.scaling_factor),
                self.max_agents
            )

            if new_agent_count > self.current_agents:
                self.logger.info(f"Scaling UP: {self.current_agents} -> {new_agent_count} agents")

                if self.scale_up_callback and self.scale_up_callback(new_agent_count):
                    self._record_scaling_action(direction, self.current_agents, new_agent_count)
                    self.current_agents = new_agent_count

                    # Update cooldown for all rules
                    for rule in self.scaling_rules.values():
                        rule.last_triggered = current_time

        elif direction == ScalingDirection.DOWN:
            # Scale down: decrease by inverse scaling factor
            new_agent_count = max(
                int(self.current_agents / self.scaling_factor),
                self.min_agents
            )

            if new_agent_count < self.current_agents:
                # Check if system has been stable long enough
                if not self._is_system_stable(self.min_stable_duration):
                    self.logger.debug("System not stable long enough for scale down")
                    return

                self.logger.info(f"Scaling DOWN: {self.current_agents} -> {new_agent_count} agents")

                if self.scale_down_callback and self.scale_down_callback(new_agent_count):
                    self._record_scaling_action(direction, self.current_agents, new_agent_count)
                    self.current_agents = new_agent_count

                    # Update cooldown for all rules
                    for rule in self.scaling_rules.values():
                        rule.last_triggered = current_time

    def _is_system_stable(self, duration: float) -> bool:
        """Check if system has been stable for given duration."""
        if len(self.scaling_history) == 0:
            return True

        last_scaling = self.scaling_history[-1]
        return time.time() - last_scaling["timestamp"] > duration

    def _record_scaling_action(self,
                             direction: ScalingDirection,
                             old_count: int,
                             new_count: int) -> None:
        """Record scaling action."""
        scaling_event = {
            "timestamp": time.time(),
            "direction": direction.value,
            "old_agent_count": old_count,
            "new_agent_count": new_count,
            "trigger_metrics": self.metrics_history[-1].__dict__ if self.metrics_history else None
        }

        self.scaling_history.append(scaling_event)

        # Keep only recent scaling events
        if len(self.scaling_history) > 100:
            self.scaling_history = self.scaling_history[-50:]

    def get_scaling_recommendations(self) -> Dict[str, Any]:
        """Get current scaling recommendations."""
        if len(self.metrics_history) < 3:
            return {"status": "insufficient_data"}

        recent_metrics = self.metrics_history[-3:]
        current_metrics = recent_metrics[-1]

        recommendations = {
            "current_agents": self.current_agents,
            "max_agents": self.max_agents,
            "auto_scaling_enabled": self.auto_scaling_enabled,
            "current_metrics": current_metrics.__dict__,
            "recommendations": []
        }

        # Analyze each metric
        for rule_name, rule in self.scaling_rules.items():
            if hasattr(current_metrics, rule.metric_name):
                current_value = getattr(current_metrics, rule.metric_name)

                if rule_name in ["training_throughput"]:
                    if current_value < rule.threshold_up:
                        recommendations["recommendations"].append({
                            "metric": rule_name,
                            "current": current_value,
                            "threshold": rule.threshold_up,
                            "action": "scale_up",
                            "reason": f"{rule_name} below threshold"
                        })
                else:
                    if current_value > rule.threshold_up:
                        recommendations["recommendations"].append({
                            "metric": rule_name,
                            "current": current_value,
                            "threshold": rule.threshold_up,
                            "action": "scale_up",
                            "reason": f"{rule_name} above threshold"
                        })
                    elif current_value < rule.threshold_down:
                        recommendations["recommendations"].append({
                            "metric": rule_name,
                            "current": current_value,
                            "threshold": rule.threshold_down,
                            "action": "scale_down",
                            "reason": f"{rule_name} below threshold"
                        })

        return recommendations

    def manual_scale(self, target_agents: int) -> bool:
        """Manually scale to target agent count."""
        if target_agents < self.min_agents or target_agents > self.max_agents:
            self.logger.error(f"Target agents {target_agents} outside valid range")
            return False

        if target_agents == self.current_agents:
            return True

        direction = ScalingDirection.UP if target_agents > self.current_agents else ScalingDirection.DOWN

        self.logger.info(f"Manual scaling: {self.current_agents} -> {target_agents} agents")

        # Execute scaling
        callback = self.scale_up_callback if direction == ScalingDirection.UP else self.scale_down_callback

        if callback and callback(target_agents):
            self._record_scaling_action(direction, self.current_agents, target_agents)
            self.current_agents = target_agents
            return True

        return False

    def enable_auto_scaling(self) -> None:
        """Enable automatic scaling."""
        self.auto_scaling_enabled = True
        self.logger.info("Auto-scaling enabled")

    def disable_auto_scaling(self) -> None:
        """Disable automatic scaling."""
        self.auto_scaling_enabled = False
        self.logger.info("Auto-scaling disabled")

    def update_scaling_rules(self, rule_updates: Dict[str, Dict[str, Any]]) -> None:
        """Update scaling rules."""
        for rule_name, updates in rule_updates.items():
            if rule_name in self.scaling_rules:
                rule = self.scaling_rules[rule_name]

                if "threshold_up" in updates:
                    rule.threshold_up = updates["threshold_up"]
                if "threshold_down" in updates:
                    rule.threshold_down = updates["threshold_down"]
                if "cooldown_period" in updates:
                    rule.cooldown_period = updates["cooldown_period"]

                self.logger.info(f"Updated scaling rule {rule_name}: {updates}")

    def get_scaling_summary(self) -> Dict[str, Any]:
        """Get summary of auto-scaling status."""
        recent_events = self.scaling_history[-10:] if self.scaling_history else []

        return {
            "current_agents": self.current_agents,
            "max_agents": self.max_agents,
            "min_agents": self.min_agents,
            "auto_scaling_enabled": self.auto_scaling_enabled,
            "total_scaling_events": len(self.scaling_history),
            "recent_events": recent_events,
            "scaling_rules": {
                name: {
                    "threshold_up": rule.threshold_up,
                    "threshold_down": rule.threshold_down,
                    "cooldown_period": rule.cooldown_period,
                    "last_triggered": rule.last_triggered
                }
                for name, rule in self.scaling_rules.items()
            }
        }


class ResourceMonitor:
    """Monitor system resources for auto-scaling."""

    def __init__(self, auto_scaler: AutoScaler):
        self.auto_scaler = auto_scaler
        self.logger = get_logger("pg_neo_graph_rl.resource_monitor")
        self.monitoring = False
        self.monitor_thread = None
        self.monitor_interval = 30.0  # seconds

    def start_monitoring(self) -> None:
        """Start resource monitoring."""
        if self.monitoring:
            return

        self.monitoring = True
        self.monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self.monitor_thread.start()
        self.logger.info("Resource monitoring started")

    def stop_monitoring(self) -> None:
        """Stop resource monitoring."""
        self.monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=5.0)
        self.logger.info("Resource monitoring stopped")

    def _monitor_loop(self) -> None:
        """Main monitoring loop."""
        while self.monitoring:
            try:
                metrics = self._collect_metrics()
                self.auto_scaler.record_metrics(metrics)
                time.sleep(self.monitor_interval)
            except Exception as e:
                self.logger.error(f"Error in resource monitoring: {e}")
                time.sleep(self.monitor_interval)

    def _collect_metrics(self) -> ScalingMetrics:
        """Collect current system metrics."""

        # Collect system metrics (with fallback values)
        cpu_usage = self._get_cpu_usage()
        memory_usage = self._get_memory_usage()

        # Collect application metrics (placeholder values for demo)
        training_throughput = 1.0  # episodes/second
        communication_latency = 0.5  # seconds
        queue_length = 0
        error_rate = 0.0

        return ScalingMetrics(
            cpu_usage=cpu_usage,
            memory_usage=memory_usage,
            training_throughput=training_throughput,
            communication_latency=communication_latency,
            queue_length=queue_length,
            error_rate=error_rate,
            timestamp=time.time()
        )

    def _get_cpu_usage(self) -> float:
        """Get CPU usage percentage."""
        try:
            import psutil
            return psutil.cpu_percent(interval=1.0)
        except ImportError:
            # Fallback: estimate based on load
            try:
                import os
                load_avg = os.getloadavg()[0]
                cpu_count = os.cpu_count() or 1
                return min(100.0, (load_avg / cpu_count) * 100.0)
            except:
                return 50.0  # Default fallback

    def _get_memory_usage(self) -> float:
        """Get memory usage percentage."""
        try:
            import psutil
            return psutil.virtual_memory().percent
        except ImportError:
            # Fallback: basic memory check
            try:
                import resource
                usage = resource.getrusage(resource.RUSAGE_SELF)
                # Convert to approximate percentage (very rough estimate)
                return min(100.0, (usage.ru_maxrss / (1024 * 1024)) * 10)  # Very rough
            except:
                return 60.0  # Default fallback
