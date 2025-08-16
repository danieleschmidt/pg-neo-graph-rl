"""
Memory management and optimization utilities.
"""
import gc
import threading
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import jax
import jax.numpy as jnp
import psutil

from ..utils.exceptions import ResourceError
from ..utils.logging import get_logger


@dataclass
class MemoryConfig:
    """Memory management configuration."""
    max_memory_usage: float = 0.85  # 85% of available memory
    gc_threshold: float = 0.75      # Trigger GC at 75%
    emergency_threshold: float = 0.95  # Emergency cleanup at 95%
    gradient_accumulation_limit: int = 32  # Max gradients to accumulate


class MemoryManager:
    """Manages memory usage and garbage collection."""

    def __init__(self, config: MemoryConfig = None):
        self.config = config or MemoryConfig()
        self.logger = get_logger("pg_neo_graph_rl.memory")
        self._lock = threading.RLock()

        # Memory tracking
        self.memory_snapshots = []
        self.last_gc_time = time.time()
        self.gc_count = 0

        # Get system memory info
        self.total_memory = psutil.virtual_memory().total
        self.max_allowed_memory = self.total_memory * self.config.max_memory_usage

        self.logger.info(f"Memory manager initialized. Max allowed: {self.max_allowed_memory / (1024**3):.2f} GB")

    def check_memory_usage(self) -> Dict[str, Any]:
        """Check current memory usage."""
        memory_info = psutil.virtual_memory()
        process = psutil.Process()
        process_memory = process.memory_info()

        usage_info = {
            "system_total_gb": memory_info.total / (1024**3),
            "system_available_gb": memory_info.available / (1024**3),
            "system_used_gb": memory_info.used / (1024**3),
            "system_percent": memory_info.percent,
            "process_rss_gb": process_memory.rss / (1024**3),
            "process_vms_gb": process_memory.vms / (1024**3),
            "threshold_percent": self.config.gc_threshold * 100,
            "emergency_percent": self.config.emergency_threshold * 100
        }

        # Store snapshot
        with self._lock:
            self.memory_snapshots.append({
                "timestamp": time.time(),
                "usage": usage_info
            })

            # Keep only recent snapshots
            if len(self.memory_snapshots) > 1000:
                self.memory_snapshots = self.memory_snapshots[-500:]

        return usage_info

    def should_trigger_gc(self) -> bool:
        """Check if garbage collection should be triggered."""
        usage_info = self.check_memory_usage()

        # Trigger GC if system memory usage is high
        if usage_info["system_percent"] > self.config.gc_threshold * 100:
            return True

        # Trigger GC if process memory is high
        process_usage_ratio = usage_info["process_rss_gb"] * (1024**3) / self.max_allowed_memory
        if process_usage_ratio > self.config.gc_threshold:
            return True

        # Trigger GC if it's been too long since last GC
        time_since_gc = time.time() - self.last_gc_time
        if time_since_gc > 300:  # 5 minutes
            return True

        return False

    def perform_gc(self, force: bool = False) -> Dict[str, Any]:
        """Perform garbage collection."""
        if not force and not self.should_trigger_gc():
            return {"skipped": True, "reason": "not_needed"}

        start_time = time.time()
        memory_before = self.check_memory_usage()

        # Force garbage collection
        collected = gc.collect()

        # JAX-specific cleanup
        try:
            jax.clear_caches()
        except Exception as e:
            self.logger.warning(f"JAX cache clear failed: {e}")

        memory_after = self.check_memory_usage()
        gc_time = time.time() - start_time

        with self._lock:
            self.last_gc_time = time.time()
            self.gc_count += 1

        memory_freed_gb = (memory_before["process_rss_gb"] - memory_after["process_rss_gb"])

        gc_info = {
            "collected_objects": collected,
            "memory_freed_gb": memory_freed_gb,
            "gc_time_seconds": gc_time,
            "memory_before": memory_before,
            "memory_after": memory_after,
            "gc_count": self.gc_count
        }

        self.logger.info(f"GC completed: freed {memory_freed_gb:.3f} GB in {gc_time:.3f}s")

        return gc_info

    def check_emergency_cleanup(self) -> bool:
        """Check if emergency memory cleanup is needed."""
        usage_info = self.check_memory_usage()

        if usage_info["system_percent"] > self.config.emergency_threshold * 100:
            self.logger.warning("Emergency memory threshold reached!")

            # Force aggressive GC
            self.perform_gc(force=True)

            # Check again
            usage_after = self.check_memory_usage()
            if usage_after["system_percent"] > self.config.emergency_threshold * 100:
                raise ResourceError(
                    f"Critical memory usage: {usage_after['system_percent']:.1f}%",
                    resource_type="memory"
                )

            return True

        return False

    def get_memory_summary(self) -> Dict[str, Any]:
        """Get comprehensive memory usage summary."""
        current_usage = self.check_memory_usage()

        with self._lock:
            if len(self.memory_snapshots) > 1:
                # Calculate trends
                recent_snapshots = self.memory_snapshots[-10:]
                memory_trend = recent_snapshots[-1]["usage"]["system_percent"] - recent_snapshots[0]["usage"]["system_percent"]
            else:
                memory_trend = 0.0

        return {
            "current_usage": current_usage,
            "memory_trend_percent": memory_trend,
            "gc_stats": {
                "total_gc_count": self.gc_count,
                "last_gc_time": self.last_gc_time,
                "time_since_gc": time.time() - self.last_gc_time
            },
            "thresholds": {
                "gc_threshold_percent": self.config.gc_threshold * 100,
                "emergency_threshold_percent": self.config.emergency_threshold * 100
            },
            "recommendations": self._get_memory_recommendations(current_usage)
        }

    def _get_memory_recommendations(self, usage_info: Dict[str, Any]) -> List[str]:
        """Get memory optimization recommendations."""
        recommendations = []

        if usage_info["system_percent"] > 80:
            recommendations.append("Consider reducing batch size")
            recommendations.append("Increase garbage collection frequency")

        if usage_info["process_rss_gb"] > 8:
            recommendations.append("Process memory usage is high")
            recommendations.append("Consider gradient accumulation limits")

        process_ratio = usage_info["process_rss_gb"] / usage_info["system_total_gb"]
        if process_ratio > 0.3:
            recommendations.append("Process using >30% of system memory")

        return recommendations


class GradientAccumulator:
    """Accumulates gradients to manage memory usage."""

    def __init__(self,
                 max_accumulation_steps: int = 8,
                 memory_manager: MemoryManager = None):
        self.max_accumulation_steps = max_accumulation_steps
        self.memory_manager = memory_manager or MemoryManager()
        self.accumulated_gradients: Dict[str, jnp.ndarray] = {}
        self.accumulation_count = 0
        self._lock = threading.RLock()
        self.logger = get_logger("pg_neo_graph_rl.gradient_accumulator")

    def accumulate(self, gradients: Dict[str, jnp.ndarray]) -> bool:
        """
        Accumulate gradients.
        
        Returns:
            True if gradients should be applied (accumulation buffer is full)
        """
        with self._lock:
            # Check memory before accumulating
            if self.memory_manager.should_trigger_gc():
                self.memory_manager.perform_gc()

            for key, grad in gradients.items():
                if key not in self.accumulated_gradients:
                    self.accumulated_gradients[key] = jnp.zeros_like(grad)

                self.accumulated_gradients[key] += grad

            self.accumulation_count += 1

            # Check if we should apply gradients
            should_apply = (
                self.accumulation_count >= self.max_accumulation_steps or
                self.memory_manager.check_memory_usage()["system_percent"] > 80
            )

            return should_apply

    def get_accumulated_gradients(self) -> Dict[str, jnp.ndarray]:
        """Get accumulated gradients (averaged)."""
        with self._lock:
            if self.accumulation_count == 0:
                return {}

            # Average the gradients
            averaged_gradients = {
                key: grad / self.accumulation_count
                for key, grad in self.accumulated_gradients.items()
            }

            return averaged_gradients

    def reset(self) -> None:
        """Reset accumulation buffer."""
        with self._lock:
            self.accumulated_gradients.clear()
            self.accumulation_count = 0

    def get_accumulation_stats(self) -> Dict[str, Any]:
        """Get accumulation statistics."""
        with self._lock:
            total_params = sum(grad.size for grad in self.accumulated_gradients.values())
            memory_usage_mb = sum(grad.nbytes for grad in self.accumulated_gradients.values()) / (1024**2)

            return {
                "accumulation_count": self.accumulation_count,
                "max_accumulation_steps": self.max_accumulation_steps,
                "total_parameters": total_params,
                "memory_usage_mb": memory_usage_mb,
                "gradient_keys": list(self.accumulated_gradients.keys()),
                "buffer_full": self.accumulation_count >= self.max_accumulation_steps
            }


class AutoScaler:
    """Automatically scales resources based on demand."""

    def __init__(self, min_agents: int = 1, max_agents: int = 100):
        self.min_agents = min_agents
        self.max_agents = max_agents
        self.current_agents = min_agents
        self.scaling_history = []
        self.logger = get_logger("pg_neo_graph_rl.autoscaler")
        self._lock = threading.RLock()

        # Scaling metrics
        self.metrics_window = []
        self.last_scale_time = time.time()
        self.cooldown_period = 60.0  # seconds

    def record_metrics(self,
                      training_time: float,
                      memory_usage: float,
                      convergence_rate: float,
                      agent_utilization: float) -> None:
        """Record metrics for scaling decisions."""
        metrics = {
            "timestamp": time.time(),
            "training_time": training_time,
            "memory_usage": memory_usage,
            "convergence_rate": convergence_rate,
            "agent_utilization": agent_utilization
        }

        with self._lock:
            self.metrics_window.append(metrics)

            # Keep only recent metrics (last 100 data points)
            if len(self.metrics_window) > 100:
                self.metrics_window = self.metrics_window[-50:]

    def should_scale_up(self) -> bool:
        """Determine if we should scale up."""
        if len(self.metrics_window) < 5:
            return False

        # Check cooldown period
        if time.time() - self.last_scale_time < self.cooldown_period:
            return False

        recent_metrics = self.metrics_window[-5:]

        # Scale up conditions
        avg_training_time = sum(m["training_time"] for m in recent_metrics) / len(recent_metrics)
        avg_utilization = sum(m["agent_utilization"] for m in recent_metrics) / len(recent_metrics)
        avg_convergence = sum(m["convergence_rate"] for m in recent_metrics) / len(recent_metrics)

        scale_up_conditions = [
            avg_training_time > 10.0,  # Training is slow
            avg_utilization > 0.8,     # Agents are highly utilized
            avg_convergence < 0.01,    # Slow convergence
            self.current_agents < self.max_agents  # Can still scale up
        ]

        return all(scale_up_conditions[:3]) and scale_up_conditions[3]

    def should_scale_down(self) -> bool:
        """Determine if we should scale down."""
        if len(self.metrics_window) < 5:
            return False

        # Check cooldown period
        if time.time() - self.last_scale_time < self.cooldown_period:
            return False

        recent_metrics = self.metrics_window[-5:]

        # Scale down conditions
        avg_training_time = sum(m["training_time"] for m in recent_metrics) / len(recent_metrics)
        avg_utilization = sum(m["agent_utilization"] for m in recent_metrics) / len(recent_metrics)
        avg_memory_usage = sum(m["memory_usage"] for m in recent_metrics) / len(recent_metrics)

        scale_down_conditions = [
            avg_training_time < 2.0,    # Training is fast
            avg_utilization < 0.3,      # Agents are underutilized
            avg_memory_usage > 0.8,     # Memory pressure
            self.current_agents > self.min_agents  # Can still scale down
        ]

        return (scale_down_conditions[1] or scale_down_conditions[2]) and scale_down_conditions[3]

    def scale(self) -> Optional[int]:
        """
        Perform scaling decision.
        
        Returns:
            New number of agents, or None if no scaling needed
        """
        with self._lock:
            if self.should_scale_up():
                new_count = min(int(self.current_agents * 1.5), self.max_agents)
                self._record_scaling_decision(self.current_agents, new_count, "scale_up")
                self.current_agents = new_count
                self.last_scale_time = time.time()

                self.logger.info(f"Scaled up to {new_count} agents")
                return new_count

            elif self.should_scale_down():
                new_count = max(int(self.current_agents * 0.7), self.min_agents)
                self._record_scaling_decision(self.current_agents, new_count, "scale_down")
                self.current_agents = new_count
                self.last_scale_time = time.time()

                self.logger.info(f"Scaled down to {new_count} agents")
                return new_count

        return None

    def _record_scaling_decision(self, old_count: int, new_count: int, reason: str) -> None:
        """Record scaling decision."""
        decision = {
            "timestamp": time.time(),
            "old_count": old_count,
            "new_count": new_count,
            "reason": reason
        }

        self.scaling_history.append(decision)

        # Keep only recent history
        if len(self.scaling_history) > 100:
            self.scaling_history = self.scaling_history[-50:]

    def get_scaling_summary(self) -> Dict[str, Any]:
        """Get scaling summary and statistics."""
        with self._lock:
            return {
                "current_agents": self.current_agents,
                "min_agents": self.min_agents,
                "max_agents": self.max_agents,
                "total_scaling_events": len(self.scaling_history),
                "last_scale_time": self.last_scale_time,
                "time_since_last_scale": time.time() - self.last_scale_time,
                "recent_scaling_history": self.scaling_history[-10:],
                "cooldown_remaining": max(0, self.cooldown_period - (time.time() - self.last_scale_time))
            }
