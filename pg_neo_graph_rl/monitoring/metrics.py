"""Metrics collection and monitoring for PG-Neo-Graph-RL."""

import time
import psutil
import threading
import logging
from typing import Dict, Any, Optional, List, Callable
from dataclasses import dataclass
from collections import defaultdict, deque
import jax
import jax.numpy as jnp

try:
    from prometheus_client import Counter, Gauge, Histogram, Summary, start_http_server, CollectorRegistry
    PROMETHEUS_AVAILABLE = True
except ImportError:
    PROMETHEUS_AVAILABLE = False

logger = logging.getLogger(__name__)


@dataclass
class MetricValue:
    """Container for metric values with timestamp."""
    value: float
    timestamp: float
    labels: Dict[str, str] = None
    
    def __post_init__(self):
        if self.labels is None:
            self.labels = {}


class MetricsCollector:
    """Central metrics collection system for PG-Neo-Graph-RL."""
    
    def __init__(self, enable_prometheus: bool = True, prometheus_port: int = 8000):
        self.enable_prometheus = enable_prometheus and PROMETHEUS_AVAILABLE
        self.prometheus_port = prometheus_port
        self.registry = CollectorRegistry() if self.enable_prometheus else None
        
        # Internal metrics storage
        self._metrics: Dict[str, deque] = defaultdict(lambda: deque(maxlen=1000))
        self._metric_callbacks: Dict[str, List[Callable]] = defaultdict(list)
        self._lock = threading.Lock()
        
        # Performance tracking
        self._start_time = time.time()
        self._last_collection = time.time()
        
        # Initialize Prometheus metrics if available
        if self.enable_prometheus:
            self._init_prometheus_metrics()
            
        # Start background collection
        self._collection_thread = threading.Thread(target=self._collect_system_metrics, daemon=True)
        self._collection_thread.start()
        
        logger.info("MetricsCollector initialized with Prometheus: %s", self.enable_prometheus)
    
    def _init_prometheus_metrics(self):
        """Initialize Prometheus metrics."""
        if not self.enable_prometheus:
            return
            
        # Training metrics
        self.training_loss = Gauge(
            'pg_neo_training_loss', 
            'Current training loss',
            registry=self.registry
        )
        
        self.training_accuracy = Gauge(
            'pg_neo_training_accuracy',
            'Current training accuracy',
            registry=self.registry
        )
        
        self.training_episodes = Counter(
            'pg_neo_training_episodes_total',
            'Total number of training episodes',
            registry=self.registry
        )
        
        # Federated learning metrics
        self.active_agents = Gauge(
            'pg_neo_active_agents',
            'Number of active federated learning agents',
            registry=self.registry
        )
        
        self.communication_rounds = Counter(
            'pg_neo_communication_rounds_total',
            'Total federated learning communication rounds',
            registry=self.registry
        )
        
        self.communication_latency = Histogram(
            'pg_neo_communication_latency_seconds',
            'Communication latency between agents',
            buckets=[0.01, 0.05, 0.1, 0.5, 1.0, 5.0, 10.0],
            registry=self.registry
        )
        
        # Graph processing metrics
        self.graph_size = Gauge(
            'pg_neo_graph_size',
            'Current graph size (number of nodes)',
            registry=self.registry
        )
        
        self.graph_processing_time = Histogram(
            'pg_neo_graph_processing_seconds',
            'Time spent processing graphs',
            buckets=[0.001, 0.01, 0.1, 1.0, 10.0],
            registry=self.registry
        )
        
        self.graph_processing_errors = Counter(
            'pg_neo_graph_processing_errors_total',
            'Total graph processing errors',
            registry=self.registry
        )
        
        # JAX/GPU metrics
        self.jax_compilation_time = Histogram(
            'pg_neo_jax_compilation_time_seconds',
            'JAX compilation time',
            buckets=[0.1, 1.0, 10.0, 60.0, 300.0],
            registry=self.registry
        )
        
        self.gpu_memory_usage = Gauge(
            'pg_neo_gpu_memory_usage_bytes',
            'GPU memory usage in bytes',
            registry=self.registry
        )
        
        self.gpu_memory_usage_ratio = Gauge(
            'pg_neo_gpu_memory_usage_ratio',
            'GPU memory usage ratio (0-1)',
            registry=self.registry
        )
        
        # System metrics
        self.cpu_usage = Gauge(
            'pg_neo_cpu_usage_percent',
            'CPU usage percentage',
            registry=self.registry
        )
        
        self.memory_usage = Gauge(
            'pg_neo_memory_usage_bytes',
            'Memory usage in bytes',
            registry=self.registry
        )
        
        self.memory_usage_ratio = Gauge(
            'pg_neo_memory_usage_ratio',
            'Memory usage ratio (0-1)',
            registry=self.registry
        )
        
        # Health check metrics
        self.health_check_success = Gauge(
            'pg_neo_health_check_success',
            'Health check success (1=success, 0=failure)',
            registry=self.registry
        )
        
        self.uptime_seconds = Gauge(
            'pg_neo_uptime_seconds',
            'Application uptime in seconds',
            registry=self.registry
        )
    
    def start_prometheus_server(self, port: Optional[int] = None):
        """Start Prometheus metrics server."""
        if not self.enable_prometheus:
            logger.warning("Prometheus not available, cannot start server")
            return
            
        port = port or self.prometheus_port
        try:
            start_http_server(port, registry=self.registry)
            logger.info("Prometheus metrics server started on port %d", port)
        except Exception as e:
            logger.error("Failed to start Prometheus server: %s", e)
    
    def record_metric(self, name: str, value: float, labels: Dict[str, str] = None):
        """Record a metric value."""
        metric_value = MetricValue(value, time.time(), labels or {})
        
        with self._lock:
            self._metrics[name].append(metric_value)
            
        # Call registered callbacks
        for callback in self._metric_callbacks[name]:
            try:
                callback(metric_value)
            except Exception as e:
                logger.error("Error in metric callback for %s: %s", name, e)
    
    def get_metric_history(self, name: str, limit: int = 100) -> List[MetricValue]:
        """Get recent history for a metric."""
        with self._lock:
            return list(self._metrics[name])[-limit:]
    
    def get_metric_summary(self, name: str) -> Dict[str, float]:
        """Get summary statistics for a metric."""
        history = self.get_metric_history(name)
        if not history:
            return {}
            
        values = [m.value for m in history]
        return {
            'count': len(values),
            'mean': sum(values) / len(values),
            'min': min(values),
            'max': max(values),
            'latest': values[-1] if values else 0.0,
            'latest_timestamp': history[-1].timestamp if history else 0.0
        }
    
    def register_callback(self, metric_name: str, callback: Callable[[MetricValue], None]):
        """Register a callback for when a metric is updated."""
        self._metric_callbacks[metric_name].append(callback)
    
    def _collect_system_metrics(self):
        """Background thread to collect system metrics."""
        while True:
            try:
                # CPU and memory
                cpu_percent = psutil.cpu_percent(interval=1)
                memory = psutil.virtual_memory()
                
                self.record_metric('system.cpu_usage_percent', cpu_percent)
                self.record_metric('system.memory_usage_bytes', memory.used)
                self.record_metric('system.memory_usage_ratio', memory.percent / 100.0)
                
                if self.enable_prometheus:
                    self.cpu_usage.set(cpu_percent)
                    self.memory_usage.set(memory.used)
                    self.memory_usage_ratio.set(memory.percent / 100.0)
                    self.uptime_seconds.set(time.time() - self._start_time)
                
                # JAX device information
                try:
                    devices = jax.devices()
                    if devices and hasattr(devices[0], 'platform') and devices[0].platform == 'gpu':
                        # Try to get GPU memory info (this is platform-specific)
                        try:
                            import GPUtil
                            gpus = GPUtil.getGPUs()
                            if gpus:
                                gpu = gpus[0]
                                gpu_memory_used = gpu.memoryUsed * 1024 * 1024  # Convert MB to bytes
                                gpu_memory_total = gpu.memoryTotal * 1024 * 1024
                                gpu_memory_ratio = gpu.memoryUtil
                                
                                self.record_metric('gpu.memory_usage_bytes', gpu_memory_used)
                                self.record_metric('gpu.memory_usage_ratio', gpu_memory_ratio)
                                
                                if self.enable_prometheus:
                                    self.gpu_memory_usage.set(gpu_memory_used)
                                    self.gpu_memory_usage_ratio.set(gpu_memory_ratio)
                        except ImportError:
                            pass  # GPUtil not available
                except Exception as e:
                    logger.debug("Error collecting GPU metrics: %s", e)
                
                time.sleep(10)  # Collect every 10 seconds
                
            except Exception as e:
                logger.error("Error collecting system metrics: %s", e)
                time.sleep(30)  # Wait longer on error
    
    # Training metrics methods
    def record_training_loss(self, loss: float, episode: int = None):
        """Record training loss."""
        labels = {'episode': str(episode)} if episode else {}
        self.record_metric('training.loss', loss, labels)
        
        if self.enable_prometheus:
            self.training_loss.set(loss)
    
    def record_training_accuracy(self, accuracy: float, episode: int = None):
        """Record training accuracy."""
        labels = {'episode': str(episode)} if episode else {}
        self.record_metric('training.accuracy', accuracy, labels)
        
        if self.enable_prometheus:
            self.training_accuracy.set(accuracy)
    
    def increment_training_episodes(self):
        """Increment training episode counter."""
        self.record_metric('training.episodes', 1)
        
        if self.enable_prometheus:
            self.training_episodes.inc()
    
    # Federated learning metrics methods
    def set_active_agents(self, count: int):
        """Set number of active agents."""
        self.record_metric('federated.active_agents', count)
        
        if self.enable_prometheus:
            self.active_agents.set(count)
    
    def increment_communication_rounds(self):
        """Increment communication rounds counter."""
        self.record_metric('federated.communication_rounds', 1)
        
        if self.enable_prometheus:
            self.communication_rounds.inc()
    
    def record_communication_latency(self, latency_seconds: float):
        """Record communication latency."""
        self.record_metric('federated.communication_latency', latency_seconds)
        
        if self.enable_prometheus:
            self.communication_latency.observe(latency_seconds)
    
    # Graph processing metrics methods
    def set_graph_size(self, num_nodes: int):
        """Set current graph size."""
        self.record_metric('graph.size', num_nodes)
        
        if self.enable_prometheus:
            self.graph_size.set(num_nodes)
    
    def record_graph_processing_time(self, processing_time: float):
        """Record graph processing time."""
        self.record_metric('graph.processing_time', processing_time)
        
        if self.enable_prometheus:
            self.graph_processing_time.observe(processing_time)
    
    def increment_graph_processing_errors(self):
        """Increment graph processing error counter."""
        self.record_metric('graph.processing_errors', 1)
        
        if self.enable_prometheus:
            self.graph_processing_errors.inc()
    
    # JAX metrics methods
    def record_jax_compilation_time(self, compilation_time: float):
        """Record JAX compilation time."""
        self.record_metric('jax.compilation_time', compilation_time)
        
        if self.enable_prometheus:
            self.jax_compilation_time.observe(compilation_time)
    
    # Health check methods
    def record_health_check(self, success: bool):
        """Record health check result."""
        self.record_metric('health.check_success', 1.0 if success else 0.0)
        
        if self.enable_prometheus:
            self.health_check_success.set(1 if success else 0)
    
    def get_health_status(self) -> Dict[str, Any]:
        """Get overall health status."""
        return {
            'status': 'healthy',
            'uptime_seconds': time.time() - self._start_time,
            'metrics_collected': len(self._metrics),
            'prometheus_enabled': self.enable_prometheus,
            'jax_devices': len(jax.devices()),
            'jax_platform': jax.devices()[0].platform if jax.devices() else 'unknown'
        }


# Global metrics collector instance
_global_metrics_collector: Optional[MetricsCollector] = None


def get_metrics_collector() -> MetricsCollector:
    """Get the global metrics collector instance."""
    global _global_metrics_collector
    if _global_metrics_collector is None:
        _global_metrics_collector = MetricsCollector()
    return _global_metrics_collector


def initialize_metrics(enable_prometheus: bool = True, prometheus_port: int = 8000) -> MetricsCollector:
    """Initialize the global metrics collector."""
    global _global_metrics_collector
    _global_metrics_collector = MetricsCollector(enable_prometheus, prometheus_port)
    return _global_metrics_collector


# Convenience functions for common metrics
def record_training_loss(loss: float, episode: int = None):
    """Record training loss using global collector."""
    get_metrics_collector().record_training_loss(loss, episode)


def record_training_accuracy(accuracy: float, episode: int = None):
    """Record training accuracy using global collector."""
    get_metrics_collector().record_training_accuracy(accuracy, episode)


def set_active_agents(count: int):
    """Set active agents count using global collector."""
    get_metrics_collector().set_active_agents(count)


def record_communication_latency(latency_seconds: float):
    """Record communication latency using global collector."""
    get_metrics_collector().record_communication_latency(latency_seconds)


def set_graph_size(num_nodes: int):
    """Set graph size using global collector."""
    get_metrics_collector().set_graph_size(num_nodes)


# Context managers for timing
class timer_context:
    """Context manager for timing operations."""
    
    def __init__(self, metric_name: str, collector: Optional[MetricsCollector] = None):
        self.metric_name = metric_name
        self.collector = collector or get_metrics_collector()
        self.start_time = None
    
    def __enter__(self):
        self.start_time = time.time()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.start_time is not None:
            duration = time.time() - self.start_time
            self.collector.record_metric(self.metric_name, duration)


def time_jax_compilation(func):
    """Decorator to time JAX compilation."""
    def wrapper(*args, **kwargs):
        collector = get_metrics_collector()
        start_time = time.time()
        result = func(*args, **kwargs)
        compilation_time = time.time() - start_time
        collector.record_jax_compilation_time(compilation_time)
        return result
    return wrapper


def time_graph_processing(func):
    """Decorator to time graph processing."""
    def wrapper(*args, **kwargs):
        collector = get_metrics_collector()
        start_time = time.time()
        try:
            result = func(*args, **kwargs)
            processing_time = time.time() - start_time
            collector.record_graph_processing_time(processing_time)
            return result
        except Exception as e:
            collector.increment_graph_processing_errors()
            raise
    return wrapper