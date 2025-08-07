"""Performance monitoring and optimization utilities."""

import time
import threading
from typing import Dict, List, Any, Optional, Tuple, Callable
from dataclasses import dataclass, asdict
from collections import deque, defaultdict
import statistics
import logging
import asyncio
from concurrent.futures import ThreadPoolExecutor, as_completed
import numpy as np


logger = logging.getLogger(__name__)


@dataclass
class PerformanceMetrics:
    """Performance metrics for sentiment analysis operations."""
    operation: str
    duration: float
    throughput: float
    batch_size: int
    success_count: int
    error_count: int
    memory_usage_mb: float
    timestamp: float
    metadata: Dict[str, Any]


@dataclass
class PerformanceProfile:
    """Performance profile summary."""
    operation: str
    total_calls: int
    avg_duration: float
    min_duration: float
    max_duration: float
    p95_duration: float
    p99_duration: float
    avg_throughput: float
    total_success: int
    total_errors: int
    error_rate: float


class PerformanceMonitor:
    """
    Advanced performance monitoring for sentiment analysis.
    
    Features:
    - Real-time performance tracking
    - Statistical analysis
    - Performance profiling
    - Bottleneck detection
    - Optimization recommendations
    """
    
    def __init__(
        self,
        max_history: int = 10000,
        enable_profiling: bool = True,
        alert_thresholds: Optional[Dict[str, float]] = None
    ):
        """
        Initialize performance monitor.
        
        Args:
            max_history: Maximum number of metrics to keep
            enable_profiling: Enable detailed profiling
            alert_thresholds: Performance alert thresholds
        """
        self.max_history = max_history
        self.enable_profiling = enable_profiling
        self.alert_thresholds = alert_thresholds or {
            "latency_p95": 2.0,      # 2 seconds
            "throughput_min": 5.0,   # 5 texts/second
            "error_rate_max": 0.05   # 5%
        }
        
        self.metrics_history: deque = deque(maxlen=max_history)
        self.operation_profiles: Dict[str, List[PerformanceMetrics]] = defaultdict(list)
        self._lock = threading.RLock()
        
        logger.info(f"PerformanceMonitor initialized with max_history={max_history}")
    
    def record_metric(self, metric: PerformanceMetrics) -> None:
        """Record a performance metric."""
        with self._lock:
            self.metrics_history.append(metric)
            
            if self.enable_profiling:
                self.operation_profiles[metric.operation].append(metric)
                
                # Keep operation profiles manageable
                if len(self.operation_profiles[metric.operation]) > 1000:
                    self.operation_profiles[metric.operation] = \
                        self.operation_profiles[metric.operation][-1000:]
    
    def time_operation(self, operation: str, **metadata) -> Callable:
        """
        Decorator to time operations and record metrics.
        
        Args:
            operation: Operation name
            **metadata: Additional metadata to record
        """
        def decorator(func):
            def wrapper(*args, **kwargs):
                start_time = time.time()
                success_count = 0
                error_count = 0
                batch_size = 1
                
                try:
                    result = func(*args, **kwargs)
                    success_count = 1
                    
                    # Try to extract batch size from result
                    if hasattr(result, '__len__') and not isinstance(result, str):
                        batch_size = len(result)
                    
                    return result
                    
                except Exception as e:
                    error_count = 1
                    raise
                    
                finally:
                    duration = time.time() - start_time
                    throughput = batch_size / duration if duration > 0 else 0
                    
                    metric = PerformanceMetrics(
                        operation=operation,
                        duration=duration,
                        throughput=throughput,
                        batch_size=batch_size,
                        success_count=success_count,
                        error_count=error_count,
                        memory_usage_mb=self._get_current_memory_mb(),
                        timestamp=time.time(),
                        metadata=metadata
                    )
                    
                    self.record_metric(metric)
            
            return wrapper
        return decorator
    
    async def time_async_operation(
        self, 
        operation: str, 
        coro, 
        **metadata
    ) -> Any:
        """
        Time an async operation and record metrics.
        
        Args:
            operation: Operation name
            coro: Coroutine to time
            **metadata: Additional metadata
            
        Returns:
            Result of the coroutine
        """
        start_time = time.time()
        success_count = 0
        error_count = 0
        batch_size = 1
        
        try:
            result = await coro
            success_count = 1
            
            if hasattr(result, '__len__') and not isinstance(result, str):
                batch_size = len(result)
            
            return result
            
        except Exception as e:
            error_count = 1
            raise
            
        finally:
            duration = time.time() - start_time
            throughput = batch_size / duration if duration > 0 else 0
            
            metric = PerformanceMetrics(
                operation=operation,
                duration=duration,
                throughput=throughput,
                batch_size=batch_size,
                success_count=success_count,
                error_count=error_count,
                memory_usage_mb=self._get_current_memory_mb(),
                timestamp=time.time(),
                metadata=metadata
            )
            
            self.record_metric(metric)
    
    def get_performance_profile(self, operation: str) -> Optional[PerformanceProfile]:
        """
        Get performance profile for an operation.
        
        Args:
            operation: Operation name
            
        Returns:
            Performance profile if available
        """
        with self._lock:
            if operation not in self.operation_profiles:
                return None
            
            metrics = self.operation_profiles[operation]
            if not metrics:
                return None
            
            durations = [m.duration for m in metrics]
            throughputs = [m.throughput for m in metrics]
            
            return PerformanceProfile(
                operation=operation,
                total_calls=len(metrics),
                avg_duration=statistics.mean(durations),
                min_duration=min(durations),
                max_duration=max(durations),
                p95_duration=np.percentile(durations, 95),
                p99_duration=np.percentile(durations, 99),
                avg_throughput=statistics.mean(throughputs),
                total_success=sum(m.success_count for m in metrics),
                total_errors=sum(m.error_count for m in metrics),
                error_rate=sum(m.error_count for m in metrics) / len(metrics)
            )
    
    def get_recent_performance(self, minutes: int = 5) -> Dict[str, Any]:
        """
        Get performance summary for recent operations.
        
        Args:
            minutes: Number of recent minutes to analyze
            
        Returns:
            Recent performance summary
        """
        cutoff_time = time.time() - (minutes * 60)
        
        with self._lock:
            recent_metrics = [
                m for m in self.metrics_history 
                if m.timestamp >= cutoff_time
            ]
        
        if not recent_metrics:
            return {"status": "no_recent_data", "analyzed_minutes": minutes}
        
        # Group by operation
        operation_metrics = defaultdict(list)
        for metric in recent_metrics:
            operation_metrics[metric.operation].append(metric)
        
        summary = {
            "analyzed_minutes": minutes,
            "total_operations": len(recent_metrics),
            "unique_operations": len(operation_metrics),
            "operations": {}
        }
        
        for operation, metrics in operation_metrics.items():
            durations = [m.duration for m in metrics]
            throughputs = [m.throughput for m in metrics]
            
            summary["operations"][operation] = {
                "call_count": len(metrics),
                "avg_duration": statistics.mean(durations),
                "avg_throughput": statistics.mean(throughputs),
                "success_rate": sum(m.success_count for m in metrics) / len(metrics),
                "total_processed": sum(m.batch_size for m in metrics)
            }
        
        return summary
    
    def detect_performance_issues(self) -> List[Dict[str, Any]]:
        """
        Detect performance issues based on thresholds.
        
        Returns:
            List of detected issues
        """
        issues = []
        
        for operation in self.operation_profiles:
            profile = self.get_performance_profile(operation)
            if not profile:
                continue
            
            # Check latency
            if profile.p95_duration > self.alert_thresholds["latency_p95"]:
                issues.append({
                    "type": "high_latency",
                    "operation": operation,
                    "current_p95": profile.p95_duration,
                    "threshold": self.alert_thresholds["latency_p95"],
                    "severity": "high" if profile.p95_duration > self.alert_thresholds["latency_p95"] * 2 else "medium"
                })
            
            # Check throughput
            if profile.avg_throughput < self.alert_thresholds["throughput_min"]:
                issues.append({
                    "type": "low_throughput",
                    "operation": operation,
                    "current_throughput": profile.avg_throughput,
                    "threshold": self.alert_thresholds["throughput_min"],
                    "severity": "medium"
                })
            
            # Check error rate
            if profile.error_rate > self.alert_thresholds["error_rate_max"]:
                issues.append({
                    "type": "high_error_rate",
                    "operation": operation,
                    "current_error_rate": profile.error_rate,
                    "threshold": self.alert_thresholds["error_rate_max"],
                    "severity": "high"
                })
        
        return issues
    
    def generate_recommendations(self) -> List[str]:
        """Generate performance optimization recommendations."""
        recommendations = []
        issues = self.detect_performance_issues()
        
        for issue in issues:
            if issue["type"] == "high_latency":
                recommendations.append(
                    f"Consider optimizing {issue['operation']} - current P95 latency is "
                    f"{issue['current_p95']:.2f}s (threshold: {issue['threshold']:.2f}s)"
                )
                recommendations.append("Try reducing batch size or enabling GPU acceleration")
            
            elif issue["type"] == "low_throughput":
                recommendations.append(
                    f"Increase throughput for {issue['operation']} - current: "
                    f"{issue['current_throughput']:.1f} ops/s"
                )
                recommendations.append("Consider increasing batch size or parallel processing")
            
            elif issue["type"] == "high_error_rate":
                recommendations.append(
                    f"Investigate errors in {issue['operation']} - error rate: "
                    f"{issue['current_error_rate']:.1%}"
                )
        
        if not recommendations:
            recommendations.append("Performance is within acceptable thresholds")
        
        return recommendations
    
    def _get_current_memory_mb(self) -> float:
        """Get current memory usage in MB."""
        try:
            import psutil
            process = psutil.Process()
            return process.memory_info().rss / (1024**2)
        except Exception:
            return 0.0
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get comprehensive performance summary."""
        summary = {
            "total_metrics": len(self.metrics_history),
            "tracked_operations": list(self.operation_profiles.keys()),
            "recent_performance": self.get_recent_performance(5),
            "performance_issues": self.detect_performance_issues(),
            "recommendations": self.generate_recommendations()
        }
        
        # Add operation profiles
        summary["operation_profiles"] = {}
        for operation in self.operation_profiles:
            profile = self.get_performance_profile(operation)
            if profile:
                summary["operation_profiles"][operation] = asdict(profile)
        
        return summary


class BatchOptimizer:
    """
    Intelligent batch size optimization for sentiment analysis.
    
    Features:
    - Dynamic batch sizing
    - Memory-aware optimization
    - Performance-based tuning
    - Load balancing
    """
    
    def __init__(
        self,
        initial_batch_size: int = 16,
        min_batch_size: int = 1,
        max_batch_size: int = 128,
        target_memory_usage: float = 0.7,
        performance_monitor: Optional[PerformanceMonitor] = None
    ):
        """
        Initialize batch optimizer.
        
        Args:
            initial_batch_size: Starting batch size
            min_batch_size: Minimum allowed batch size
            max_batch_size: Maximum allowed batch size
            target_memory_usage: Target memory usage ratio
            performance_monitor: Performance monitor instance
        """
        self.initial_batch_size = initial_batch_size
        self.min_batch_size = min_batch_size
        self.max_batch_size = max_batch_size
        self.target_memory_usage = target_memory_usage
        self.performance_monitor = performance_monitor
        
        self.current_batch_size = initial_batch_size
        self.optimization_history: List[Dict[str, Any]] = []
        
        logger.info(f"BatchOptimizer initialized with batch size {initial_batch_size}")
    
    def optimize_batch_size(
        self,
        current_memory_usage: float,
        recent_performance: Dict[str, Any],
        target_throughput: Optional[float] = None
    ) -> int:
        """
        Optimize batch size based on current conditions.
        
        Args:
            current_memory_usage: Current memory usage ratio (0-1)
            recent_performance: Recent performance metrics
            target_throughput: Target throughput if specified
            
        Returns:
            Optimized batch size
        """
        old_batch_size = self.current_batch_size
        
        # Memory-based adjustment
        if current_memory_usage > self.target_memory_usage + 0.1:
            # High memory usage - reduce batch size
            self.current_batch_size = max(
                self.min_batch_size,
                int(self.current_batch_size * 0.8)
            )
            reason = "high_memory_usage"
            
        elif current_memory_usage < self.target_memory_usage - 0.1:
            # Low memory usage - can increase batch size
            self.current_batch_size = min(
                self.max_batch_size,
                int(self.current_batch_size * 1.2)
            )
            reason = "low_memory_usage"
            
        else:
            reason = "memory_optimal"
        
        # Performance-based adjustment
        if self.performance_monitor and recent_performance:
            avg_duration = recent_performance.get("operations", {}).get("sentiment_analysis", {}).get("avg_duration", 0)
            
            if avg_duration > 2.0:  # Slow performance
                self.current_batch_size = max(
                    self.min_batch_size,
                    int(self.current_batch_size * 0.9)
                )
                reason += "_slow_performance"
            
            elif avg_duration < 0.5:  # Fast performance, can increase
                self.current_batch_size = min(
                    self.max_batch_size,
                    int(self.current_batch_size * 1.1)
                )
                reason += "_fast_performance"
        
        # Target throughput adjustment
        if target_throughput and recent_performance:
            current_throughput = recent_performance.get("operations", {}).get("sentiment_analysis", {}).get("avg_throughput", 0)
            
            if current_throughput < target_throughput * 0.8:
                self.current_batch_size = min(
                    self.max_batch_size,
                    int(self.current_batch_size * 1.3)
                )
                reason += "_low_throughput"
        
        # Ensure within bounds
        self.current_batch_size = max(self.min_batch_size, min(self.max_batch_size, self.current_batch_size))
        
        # Record optimization
        optimization_record = {
            "timestamp": time.time(),
            "old_batch_size": old_batch_size,
            "new_batch_size": self.current_batch_size,
            "memory_usage": current_memory_usage,
            "reason": reason,
            "performance_data": recent_performance
        }
        
        self.optimization_history.append(optimization_record)
        
        # Keep only last 100 optimizations
        if len(self.optimization_history) > 100:
            self.optimization_history = self.optimization_history[-100:]
        
        if old_batch_size != self.current_batch_size:
            logger.info(
                f"Batch size optimized: {old_batch_size} -> {self.current_batch_size} "
                f"(reason: {reason})"
            )
        
        return self.current_batch_size
    
    def get_optimal_batch_sizes(self, text_lengths: List[int]) -> List[int]:
        """
        Get optimal batch sizes for a list of texts based on their lengths.
        
        Args:
            text_lengths: List of text lengths
            
        Returns:
            List of optimal batch sizes for each group
        """
        if not text_lengths:
            return []
        
        # Group texts by length ranges
        length_groups = self._group_by_length(text_lengths)
        batch_sizes = []
        
        for length_range, count in length_groups.items():
            # Adjust batch size based on text length
            if length_range == "short":  # < 100 chars
                base_batch_size = self.current_batch_size * 2
            elif length_range == "medium":  # 100-500 chars
                base_batch_size = self.current_batch_size
            else:  # > 500 chars
                base_batch_size = max(1, self.current_batch_size // 2)
            
            # Ensure within bounds
            optimal_size = max(self.min_batch_size, min(self.max_batch_size, base_batch_size))
            batch_sizes.extend([optimal_size] * count)
        
        return batch_sizes[:len(text_lengths)]
    
    def _group_by_length(self, text_lengths: List[int]) -> Dict[str, int]:
        """Group texts by length ranges."""
        groups = {"short": 0, "medium": 0, "long": 0}
        
        for length in text_lengths:
            if length < 100:
                groups["short"] += 1
            elif length < 500:
                groups["medium"] += 1
            else:
                groups["long"] += 1
        
        return groups
    
    def get_optimization_stats(self) -> Dict[str, Any]:
        """Get batch optimization statistics."""
        if not self.optimization_history:
            return {"status": "no_optimizations"}
        
        # Calculate optimization frequency
        recent_optimizations = [
            opt for opt in self.optimization_history 
            if time.time() - opt["timestamp"] <= 3600  # Last hour
        ]
        
        # Batch size trend
        batch_sizes = [opt["new_batch_size"] for opt in self.optimization_history[-10:]]
        avg_batch_size = sum(batch_sizes) / len(batch_sizes) if batch_sizes else self.current_batch_size
        
        return {
            "current_batch_size": self.current_batch_size,
            "initial_batch_size": self.initial_batch_size,
            "avg_recent_batch_size": avg_batch_size,
            "total_optimizations": len(self.optimization_history),
            "optimizations_last_hour": len(recent_optimizations),
            "optimization_reasons": [opt["reason"] for opt in recent_optimizations],
            "batch_size_range": {
                "min": self.min_batch_size,
                "max": self.max_batch_size,
                "current": self.current_batch_size
            }
        }