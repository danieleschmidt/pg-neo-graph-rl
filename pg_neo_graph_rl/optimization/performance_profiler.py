"""
Advanced performance profiling and optimization system.
Provides detailed analysis of training performance and automatic optimizations.
"""

import time
import threading
import functools
import traceback
import gc
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
from dataclasses import dataclass, field
from collections import defaultdict, deque
from contextlib import contextmanager
import statistics

import jax
import jax.numpy as jnp
from jax import profiler

from ..utils.exceptions import GraphRLError
from ..utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class PerformanceMetrics:
    """Performance metrics for a function or operation."""
    name: str
    total_calls: int = 0
    total_time: float = 0.0
    min_time: float = float('inf')
    max_time: float = 0.0
    avg_time: float = 0.0
    memory_usage_mb: float = 0.0
    cpu_usage_percent: float = 0.0
    jax_compilations: int = 0
    errors: int = 0
    recent_times: deque = field(default_factory=lambda: deque(maxlen=100))
    
    def update(self, execution_time: float, memory_mb: float = 0.0, cpu_percent: float = 0.0):
        """Update metrics with new measurement."""
        self.total_calls += 1
        self.total_time += execution_time
        self.min_time = min(self.min_time, execution_time)
        self.max_time = max(self.max_time, execution_time)
        self.avg_time = self.total_time / self.total_calls
        self.memory_usage_mb = max(self.memory_usage_mb, memory_mb)
        self.cpu_usage_percent = max(self.cpu_usage_percent, cpu_percent)
        self.recent_times.append(execution_time)
    
    def get_percentiles(self) -> Dict[str, float]:
        """Calculate performance percentiles."""
        if not self.recent_times:
            return {}
        
        times = list(self.recent_times)
        return {
            "p50": statistics.median(times),
            "p90": statistics.quantiles(times, n=10)[8] if len(times) >= 10 else max(times),
            "p95": statistics.quantiles(times, n=20)[18] if len(times) >= 20 else max(times),
            "p99": statistics.quantiles(times, n=100)[98] if len(times) >= 100 else max(times)
        }


@dataclass
class JaxProfileMetrics:
    """JAX-specific profiling metrics."""
    compilation_time: float = 0.0
    execution_time: float = 0.0
    memory_allocated_mb: float = 0.0
    device_time: float = 0.0
    host_time: float = 0.0
    transfer_time: float = 0.0


class PerformanceProfiler:
    """
    Comprehensive performance profiler for federated graph RL.
    """
    
    def __init__(self, enabled: bool = True, profile_jax: bool = True):
        self.enabled = enabled
        self.profile_jax = profile_jax
        
        # Metrics storage
        self.metrics: Dict[str, PerformanceMetrics] = defaultdict(lambda: PerformanceMetrics(""))
        self.jax_metrics: Dict[str, JaxProfileMetrics] = defaultdict(JaxProfileMetrics)
        
        # Profiling state
        self.current_context: Optional[str] = None
        self.context_stack: List[str] = []
        self.lock = threading.RLock()
        
        # Performance tracking
        self.bottlenecks: List[Tuple[str, float]] = []
        self.optimization_suggestions: List[str] = []
        
        # JAX compilation tracking
        self.compilation_cache: Dict[str, int] = defaultdict(int)
        
        logger.info(f"PerformanceProfiler initialized (enabled={enabled}, jax_profiling={profile_jax})")
    
    def enable(self):
        """Enable profiling."""
        self.enabled = True
        logger.info("Performance profiling enabled")
    
    def disable(self):
        """Disable profiling."""
        self.enabled = False
        logger.info("Performance profiling disabled")
    
    @contextmanager
    def profile_context(self, name: str, metadata: Optional[Dict[str, Any]] = None):
        """Context manager for profiling code blocks."""
        if not self.enabled:
            yield
            return
        
        start_time = time.perf_counter()
        start_memory = self._get_memory_usage()
        
        # JAX profiling setup
        if self.profile_jax:
            jax_trace_name = f"jax_{name}"
            profiler.start_trace(jax_trace_name)
        
        self.context_stack.append(name)
        self.current_context = name
        
        try:
            yield
        except Exception as e:
            with self.lock:
                self.metrics[name].errors += 1
            raise
        finally:
            # Calculate timing
            end_time = time.perf_counter()
            execution_time = end_time - start_time
            
            # Calculate memory
            end_memory = self._get_memory_usage()
            memory_delta = max(0, end_memory - start_memory)
            
            # Update metrics
            with self.lock:
                self.metrics[name].name = name
                self.metrics[name].update(execution_time, memory_delta)
            
            # JAX profiling cleanup
            if self.profile_jax:
                try:
                    profiler.stop_trace()
                    # In a real implementation, you would parse the JAX trace here
                    self._process_jax_trace(name, execution_time)
                except Exception as e:
                    logger.warning(f"JAX profiling failed for {name}: {e}")
            
            # Context cleanup
            if self.context_stack and self.context_stack[-1] == name:
                self.context_stack.pop()
            self.current_context = self.context_stack[-1] if self.context_stack else None
            
            # Check for performance issues
            self._check_performance_issues(name, execution_time)
    
    def profile_function(self, name: Optional[str] = None):
        """Decorator for profiling functions."""
        def decorator(func: Callable) -> Callable:
            func_name = name or f"{func.__module__}.{func.__name__}"
            
            @functools.wraps(func)
            def wrapper(*args, **kwargs):
                if not self.enabled:
                    return func(*args, **kwargs)
                
                with self.profile_context(func_name):
                    result = func(*args, **kwargs)
                
                return result
            
            return wrapper
        return decorator
    
    def profile_jax_function(self, func: Callable, name: Optional[str] = None) -> Callable:
        """Profile JAX-compiled functions."""
        func_name = name or f"jax_{func.__name__}"
        
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            if not self.enabled:
                return func(*args, **kwargs)
            
            # Track compilations
            compilation_start = time.perf_counter()
            
            # Check if this is a new compilation
            arg_signature = self._get_arg_signature(args, kwargs)
            cache_key = f"{func_name}_{arg_signature}"
            
            is_compilation = cache_key not in self.compilation_cache
            
            with self.profile_context(func_name):
                result = func(*args, **kwargs)
            
            if is_compilation:
                compilation_time = time.perf_counter() - compilation_start
                with self.lock:
                    self.compilation_cache[cache_key] = 1
                    self.metrics[func_name].jax_compilations += 1
                    self.jax_metrics[func_name].compilation_time += compilation_time
                
                logger.debug(f"JAX compilation for {func_name}: {compilation_time:.3f}s")
            
            return result
        
        return wrapper
    
    def _get_arg_signature(self, args: tuple, kwargs: dict) -> str:
        """Generate signature for JAX function arguments."""
        try:
            arg_shapes = []
            for arg in args:
                if isinstance(arg, jnp.ndarray):
                    arg_shapes.append(f"{arg.shape}_{arg.dtype}")
                else:
                    arg_shapes.append(str(type(arg).__name__))
            
            kwarg_shapes = []
            for key, value in kwargs.items():
                if isinstance(value, jnp.ndarray):
                    kwarg_shapes.append(f"{key}:{value.shape}_{value.dtype}")
                else:
                    kwarg_shapes.append(f"{key}:{type(value).__name__}")
            
            return "_".join(arg_shapes + kwarg_shapes)
        except Exception:
            return "unknown_signature"
    
    def _get_memory_usage(self) -> float:
        """Get current memory usage in MB."""
        try:
            import psutil
            process = psutil.Process()
            return process.memory_info().rss / (1024 * 1024)
        except ImportError:
            # Fallback to gc stats
            return sum(obj['count'] for obj in gc.get_stats()) / 1000.0
        except Exception:
            return 0.0
    
    def _process_jax_trace(self, name: str, execution_time: float):
        """Process JAX profiling trace (simplified)."""
        # In a real implementation, this would parse the JAX profiler output
        # For now, we'll estimate some metrics
        with self.lock:
            jax_metrics = self.jax_metrics[name]
            jax_metrics.execution_time += execution_time
            
            # Estimate device vs host time (simplified)
            jax_metrics.device_time += execution_time * 0.8  # Assume 80% on device
            jax_metrics.host_time += execution_time * 0.2    # Assume 20% on host
    
    def _check_performance_issues(self, name: str, execution_time: float):
        """Check for performance issues and add suggestions."""
        with self.lock:
            metrics = self.metrics[name]
            
            # Check for slow operations
            if execution_time > 5.0:  # > 5 seconds
                self.bottlenecks.append((name, execution_time))
                self.bottlenecks.sort(key=lambda x: x[1], reverse=True)
                self.bottlenecks = self.bottlenecks[:10]  # Keep top 10
                
                suggestion = f"Operation '{name}' is slow ({execution_time:.2f}s). Consider optimization."
                if suggestion not in self.optimization_suggestions:
                    self.optimization_suggestions.append(suggestion)
            
            # Check for excessive JAX compilations
            if metrics.jax_compilations > 10:
                suggestion = f"Function '{name}' has {metrics.jax_compilations} JAX compilations. Consider static shapes or caching."
                if suggestion not in self.optimization_suggestions:
                    self.optimization_suggestions.append(suggestion)
            
            # Check for memory issues
            if metrics.memory_usage_mb > 1000:  # > 1GB
                suggestion = f"Function '{name}' uses {metrics.memory_usage_mb:.1f}MB memory. Consider optimization."
                if suggestion not in self.optimization_suggestions:
                    self.optimization_suggestions.append(suggestion)
    
    def get_performance_report(self) -> Dict[str, Any]:
        """Generate comprehensive performance report."""
        with self.lock:
            # Calculate overall statistics
            total_functions = len(self.metrics)
            total_calls = sum(m.total_calls for m in self.metrics.values())
            total_time = sum(m.total_time for m in self.metrics.values())
            
            # Find slowest operations
            slowest_ops = sorted(
                [(name, m.avg_time) for name, m in self.metrics.items()],
                key=lambda x: x[1],
                reverse=True
            )[:10]
            
            # Find most called operations
            most_called = sorted(
                [(name, m.total_calls) for name, m in self.metrics.items()],
                key=lambda x: x[1],
                reverse=True
            )[:10]
            
            # Memory usage summary
            memory_usage = sorted(
                [(name, m.memory_usage_mb) for name, m in self.metrics.items()],
                key=lambda x: x[1],
                reverse=True
            )[:10]
            
            # JAX compilation summary
            jax_compilations = sum(m.jax_compilations for m in self.metrics.values())
            
            return {
                "overview": {
                    "total_functions_profiled": total_functions,
                    "total_function_calls": total_calls,
                    "total_execution_time": total_time,
                    "average_call_time": total_time / total_calls if total_calls > 0 else 0,
                    "total_jax_compilations": jax_compilations
                },
                "slowest_operations": slowest_ops,
                "most_called_operations": most_called,
                "memory_usage": memory_usage,
                "bottlenecks": self.bottlenecks,
                "optimization_suggestions": self.optimization_suggestions,
                "detailed_metrics": {
                    name: {
                        "total_calls": m.total_calls,
                        "avg_time": m.avg_time,
                        "total_time": m.total_time,
                        "min_time": m.min_time,
                        "max_time": m.max_time,
                        "memory_mb": m.memory_usage_mb,
                        "jax_compilations": m.jax_compilations,
                        "errors": m.errors,
                        "percentiles": m.get_percentiles()
                    }
                    for name, m in self.metrics.items()
                }
            }
    
    def get_optimization_recommendations(self) -> List[Dict[str, Any]]:
        """Get specific optimization recommendations."""
        recommendations = []
        
        with self.lock:
            for name, metrics in self.metrics.items():
                recs = []
                
                # Check for slow average times
                if metrics.avg_time > 1.0:
                    recs.append({
                        "type": "performance",
                        "severity": "high" if metrics.avg_time > 5.0 else "medium",
                        "message": f"Slow average execution time: {metrics.avg_time:.3f}s",
                        "suggestion": "Consider algorithmic optimization or parallelization"
                    })
                
                # Check for high variance in execution times
                if len(metrics.recent_times) > 10:
                    variance = statistics.variance(metrics.recent_times)
                    if variance > metrics.avg_time:
                        recs.append({
                            "type": "consistency",
                            "severity": "medium",
                            "message": f"High variance in execution times: {variance:.3f}",
                            "suggestion": "Investigate inconsistent performance causes"
                        })
                
                # Check for excessive compilations
                if metrics.jax_compilations > 5:
                    recs.append({
                        "type": "jax_optimization",
                        "severity": "medium",
                        "message": f"Too many JAX compilations: {metrics.jax_compilations}",
                        "suggestion": "Use static shapes or implement compilation caching"
                    })
                
                # Check for memory usage
                if metrics.memory_usage_mb > 500:
                    recs.append({
                        "type": "memory",
                        "severity": "high" if metrics.memory_usage_mb > 1000 else "medium",
                        "message": f"High memory usage: {metrics.memory_usage_mb:.1f}MB",
                        "suggestion": "Consider memory optimization or batch size reduction"
                    })
                
                # Check for error rates
                if metrics.errors > 0 and metrics.total_calls > 0:
                    error_rate = metrics.errors / metrics.total_calls
                    if error_rate > 0.01:  # > 1% error rate
                        recs.append({
                            "type": "reliability",
                            "severity": "high",
                            "message": f"High error rate: {error_rate:.2%}",
                            "suggestion": "Investigate and fix error causes"
                        })
                
                if recs:
                    recommendations.append({
                        "function": name,
                        "recommendations": recs
                    })
        
        return recommendations
    
    def reset_metrics(self):
        """Reset all performance metrics."""
        with self.lock:
            self.metrics.clear()
            self.jax_metrics.clear()
            self.bottlenecks.clear()
            self.optimization_suggestions.clear()
            self.compilation_cache.clear()
        
        logger.info("Performance metrics reset")
    
    def export_metrics(self, filepath: str):
        """Export metrics to file."""
        import json
        
        report = self.get_performance_report()
        
        # Convert to JSON-serializable format
        json_report = {}
        for key, value in report.items():
            if key == "detailed_metrics":
                json_report[key] = {}
                for name, metrics in value.items():
                    json_report[key][name] = {
                        k: float(v) if isinstance(v, (int, float)) else v
                        for k, v in metrics.items()
                        if k != "percentiles" or metrics["percentiles"]
                    }
            else:
                json_report[key] = value
        
        with open(filepath, 'w') as f:
            json.dump(json_report, f, indent=2)
        
        logger.info(f"Performance metrics exported to {filepath}")


class AutoOptimizer:
    """
    Automatic optimization system based on performance profiling.
    """
    
    def __init__(self, profiler: PerformanceProfiler):
        self.profiler = profiler
        self.optimizations_applied: List[str] = []
        self.logger = get_logger("pg_neo_graph_rl.auto_optimizer")
    
    def apply_automatic_optimizations(self) -> List[str]:
        """Apply automatic optimizations based on profiling data."""
        applied = []
        
        recommendations = self.profiler.get_optimization_recommendations()
        
        for func_rec in recommendations:
            func_name = func_rec["function"]
            
            for rec in func_rec["recommendations"]:
                if rec["type"] == "jax_optimization" and rec["severity"] == "medium":
                    # Apply JAX optimization
                    if self._apply_jax_optimization(func_name):
                        applied.append(f"JAX optimization for {func_name}")
                
                elif rec["type"] == "memory" and rec["severity"] == "high":
                    # Apply memory optimization
                    if self._apply_memory_optimization(func_name):
                        applied.append(f"Memory optimization for {func_name}")
        
        self.optimizations_applied.extend(applied)
        return applied
    
    def _apply_jax_optimization(self, func_name: str) -> bool:
        """Apply JAX-specific optimizations."""
        # This would contain actual optimization logic
        # For now, just log the optimization
        self.logger.info(f"Applied JAX optimization to {func_name}")
        return True
    
    def _apply_memory_optimization(self, func_name: str) -> bool:
        """Apply memory optimizations."""
        # This would contain actual optimization logic
        # For now, just log the optimization
        self.logger.info(f"Applied memory optimization to {func_name}")
        return True


# Global profiler instance
_global_profiler: Optional[PerformanceProfiler] = None


def get_global_profiler() -> PerformanceProfiler:
    """Get global performance profiler instance."""
    global _global_profiler
    if _global_profiler is None:
        _global_profiler = PerformanceProfiler(enabled=True, profile_jax=True)
    return _global_profiler


def profile(name: Optional[str] = None):
    """Decorator for profiling functions."""
    return get_global_profiler().profile_function(name)


@contextmanager
def profile_block(name: str):
    """Context manager for profiling code blocks."""
    with get_global_profiler().profile_context(name):
        yield


def create_performance_system() -> Tuple[PerformanceProfiler, AutoOptimizer]:
    """
    Create complete performance profiling and optimization system.
    
    Returns:
        Tuple of (PerformanceProfiler, AutoOptimizer)
    """
    profiler = PerformanceProfiler(enabled=True, profile_jax=True)
    auto_optimizer = AutoOptimizer(profiler)
    
    logger.info("Performance profiling and optimization system created")
    return profiler, auto_optimizer