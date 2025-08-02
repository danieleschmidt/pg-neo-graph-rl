"""Distributed tracing and observability utilities."""

import time
import uuid
import threading
import contextvars
from typing import Dict, Any, Optional, List, Callable
from dataclasses import dataclass, field
from contextlib import contextmanager
import logging

logger = logging.getLogger(__name__)

# Context variables for trace propagation
trace_id_var: contextvars.ContextVar[Optional[str]] = contextvars.ContextVar('trace_id', default=None)
span_id_var: contextvars.ContextVar[Optional[str]] = contextvars.ContextVar('span_id', default=None)
parent_span_id_var: contextvars.ContextVar[Optional[str]] = contextvars.ContextVar('parent_span_id', default=None)


@dataclass
class Span:
    """Represents a traced operation span."""
    
    trace_id: str
    span_id: str
    parent_span_id: Optional[str]
    operation_name: str
    start_time: float = field(default_factory=time.time)
    end_time: Optional[float] = None
    duration: Optional[float] = None
    status: str = "started"  # started, success, error
    tags: Dict[str, Any] = field(default_factory=dict)
    logs: List[Dict[str, Any]] = field(default_factory=list)
    
    def finish(self, status: str = "success"):
        """Finish the span."""
        self.end_time = time.time()
        self.duration = self.end_time - self.start_time
        self.status = status
    
    def add_tag(self, key: str, value: Any):
        """Add a tag to the span."""
        self.tags[key] = value
    
    def log_event(self, event: str, **kwargs):
        """Log an event within the span."""
        log_entry = {
            'timestamp': time.time(),
            'event': event,
            **kwargs
        }
        self.logs.append(log_entry)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert span to dictionary representation."""
        return {
            'trace_id': self.trace_id,
            'span_id': self.span_id,
            'parent_span_id': self.parent_span_id,
            'operation_name': self.operation_name,
            'start_time': self.start_time,
            'end_time': self.end_time,
            'duration': self.duration,
            'status': self.status,
            'tags': self.tags,
            'logs': self.logs
        }


class TracingBackend:
    """Base class for tracing backends."""
    
    def send_span(self, span: Span):
        """Send a completed span to the backend."""
        raise NotImplementedError


class LoggingTracingBackend(TracingBackend):
    """Tracing backend that logs spans."""
    
    def __init__(self, logger: Optional[logging.Logger] = None):
        self.logger = logger or logging.getLogger(__name__)
    
    def send_span(self, span: Span):
        """Log the span."""
        self.logger.info(
            f"Span completed: {span.operation_name}",
            extra={
                'trace_data': span.to_dict(),
                'trace_id': span.trace_id,
                'span_id': span.span_id,
                'operation': span.operation_name,
                'duration': span.duration,
                'status': span.status
            }
        )


class InMemoryTracingBackend(TracingBackend):
    """In-memory tracing backend for testing."""
    
    def __init__(self):
        self.spans: List[Span] = []
        self._lock = threading.Lock()
    
    def send_span(self, span: Span):
        """Store the span in memory."""
        with self._lock:
            self.spans.append(span)
    
    def get_spans(self) -> List[Span]:
        """Get all stored spans."""
        with self._lock:
            return self.spans.copy()
    
    def clear(self):
        """Clear all stored spans."""
        with self._lock:
            self.spans.clear()


class Tracer:
    """Main tracing interface."""
    
    def __init__(self, backend: Optional[TracingBackend] = None):
        self.backend = backend or LoggingTracingBackend()
        self._active_spans: Dict[str, Span] = {}
        self._lock = threading.Lock()
    
    def start_span(
        self,
        operation_name: str,
        parent_span_id: Optional[str] = None,
        trace_id: Optional[str] = None,
        tags: Optional[Dict[str, Any]] = None
    ) -> Span:
        """Start a new span."""
        # Generate IDs
        if trace_id is None:
            current_trace_id = trace_id_var.get()
            trace_id = current_trace_id or str(uuid.uuid4())
        
        span_id = str(uuid.uuid4())
        
        if parent_span_id is None:
            parent_span_id = span_id_var.get()
        
        # Create span
        span = Span(
            trace_id=trace_id,
            span_id=span_id,
            parent_span_id=parent_span_id,
            operation_name=operation_name,
            tags=tags or {}
        )
        
        # Store active span
        with self._lock:
            self._active_spans[span_id] = span
        
        # Set context variables
        trace_id_var.set(trace_id)
        span_id_var.set(span_id)
        parent_span_id_var.set(parent_span_id)
        
        logger.debug(f"Started span: {operation_name} (trace_id={trace_id}, span_id={span_id})")
        
        return span
    
    def finish_span(self, span: Span, status: str = "success"):
        """Finish a span."""
        span.finish(status)
        
        # Remove from active spans
        with self._lock:
            self._active_spans.pop(span.span_id, None)
        
        # Send to backend
        try:
            self.backend.send_span(span)
        except Exception as e:
            logger.error(f"Failed to send span to backend: {e}")
        
        logger.debug(f"Finished span: {span.operation_name} (duration={span.duration:.4f}s)")
    
    def get_current_span(self) -> Optional[Span]:
        """Get the current active span."""
        current_span_id = span_id_var.get()
        if current_span_id:
            with self._lock:
                return self._active_spans.get(current_span_id)
        return None
    
    @contextmanager
    def span(
        self,
        operation_name: str,
        tags: Optional[Dict[str, Any]] = None,
        finish_on_error: bool = True
    ):
        """Context manager for creating spans."""
        span = self.start_span(operation_name, tags=tags)
        try:
            yield span
            self.finish_span(span, "success")
        except Exception as e:
            if finish_on_error:
                span.add_tag("error", True)
                span.add_tag("error.message", str(e))
                span.log_event("error", message=str(e), error_type=type(e).__name__)
                self.finish_span(span, "error")
            raise


# Global tracer instance
_global_tracer: Optional[Tracer] = None


def get_tracer() -> Tracer:
    """Get the global tracer instance."""
    global _global_tracer
    if _global_tracer is None:
        _global_tracer = Tracer()
    return _global_tracer


def initialize_tracing(backend: Optional[TracingBackend] = None) -> Tracer:
    """Initialize the global tracer."""
    global _global_tracer
    _global_tracer = Tracer(backend)
    return _global_tracer


# Convenience functions
def start_span(operation_name: str, **kwargs) -> Span:
    """Start a span using the global tracer."""
    return get_tracer().start_span(operation_name, **kwargs)


def get_current_span() -> Optional[Span]:
    """Get the current span using the global tracer."""
    return get_tracer().get_current_span()


@contextmanager
def span(operation_name: str, **kwargs):
    """Create a span using the global tracer."""
    with get_tracer().span(operation_name, **kwargs) as s:
        yield s


# Decorators for automatic tracing
def trace_function(operation_name: Optional[str] = None, tags: Optional[Dict[str, Any]] = None):
    """Decorator to automatically trace function calls."""
    def decorator(func):
        nonlocal operation_name
        if operation_name is None:
            operation_name = f"{func.__module__}.{func.__name__}"
        
        def wrapper(*args, **kwargs):
            span_tags = tags.copy() if tags else {}
            span_tags.update({
                'function': func.__name__,
                'module': func.__module__,
                'args_count': len(args),
                'kwargs_count': len(kwargs)
            })
            
            with span(operation_name, tags=span_tags) as current_span:
                try:
                    result = func(*args, **kwargs)
                    current_span.add_tag("success", True)
                    return result
                except Exception as e:
                    current_span.add_tag("error", True)
                    current_span.add_tag("error.type", type(e).__name__)
                    current_span.add_tag("error.message", str(e))
                    raise
        
        return wrapper
    return decorator


def trace_class_methods(cls, exclude: Optional[List[str]] = None):
    """Class decorator to automatically trace all methods."""
    exclude = exclude or ['__init__', '__del__', '__str__', '__repr__']
    
    for attr_name in dir(cls):
        attr = getattr(cls, attr_name)
        if (callable(attr) and 
            not attr_name.startswith('_') and 
            attr_name not in exclude):
            
            operation_name = f"{cls.__name__}.{attr_name}"
            traced_method = trace_function(operation_name)(attr)
            setattr(cls, attr_name, traced_method)
    
    return cls


# Correlation ID utilities
def generate_correlation_id() -> str:
    """Generate a new correlation ID."""
    return str(uuid.uuid4())


def get_correlation_id() -> Optional[str]:
    """Get the current correlation ID from trace context."""
    return trace_id_var.get()


def set_correlation_id(correlation_id: str):
    """Set the correlation ID in trace context."""
    trace_id_var.set(correlation_id)


# Baggage utilities for cross-process context propagation
_baggage_var: contextvars.ContextVar[Dict[str, str]] = contextvars.ContextVar('baggage', default={})


def set_baggage(key: str, value: str):
    """Set a baggage item."""
    current_baggage = _baggage_var.get({})
    new_baggage = current_baggage.copy()
    new_baggage[key] = value
    _baggage_var.set(new_baggage)


def get_baggage(key: str) -> Optional[str]:
    """Get a baggage item."""
    return _baggage_var.get({}).get(key)


def get_all_baggage() -> Dict[str, str]:
    """Get all baggage items."""
    return _baggage_var.get({}).copy()


# HTTP header utilities for trace propagation
def inject_trace_headers() -> Dict[str, str]:
    """Inject trace context into HTTP headers."""
    headers = {}
    
    trace_id = trace_id_var.get()
    if trace_id:
        headers['X-Trace-Id'] = trace_id
    
    span_id = span_id_var.get()
    if span_id:
        headers['X-Span-Id'] = span_id
    
    # Add baggage
    baggage = get_all_baggage()
    if baggage:
        baggage_header = ','.join(f"{k}={v}" for k, v in baggage.items())
        headers['X-Baggage'] = baggage_header
    
    return headers


def extract_trace_headers(headers: Dict[str, str]):
    """Extract trace context from HTTP headers."""
    if 'X-Trace-Id' in headers:
        trace_id_var.set(headers['X-Trace-Id'])
    
    if 'X-Span-Id' in headers:
        parent_span_id_var.set(headers['X-Span-Id'])
    
    # Extract baggage
    if 'X-Baggage' in headers:
        baggage_items = headers['X-Baggage'].split(',')
        baggage = {}
        for item in baggage_items:
            if '=' in item:
                key, value = item.split('=', 1)
                baggage[key.strip()] = value.strip()
        if baggage:
            _baggage_var.set(baggage)


# Performance monitoring
class PerformanceTracker:
    """Track performance metrics within spans."""
    
    def __init__(self):
        self._metrics: Dict[str, List[float]] = {}
        self._lock = threading.Lock()
    
    def record_metric(self, name: str, value: float, span: Optional[Span] = None):
        """Record a performance metric."""
        with self._lock:
            if name not in self._metrics:
                self._metrics[name] = []
            self._metrics[name].append(value)
        
        # Add to current span if available
        if span is None:
            span = get_current_span()
        
        if span:
            span.add_tag(f"metric.{name}", value)
    
    def get_metric_summary(self, name: str) -> Dict[str, float]:
        """Get summary statistics for a metric."""
        with self._lock:
            values = self._metrics.get(name, [])
        
        if not values:
            return {}
        
        return {
            'count': len(values),
            'min': min(values),
            'max': max(values),
            'avg': sum(values) / len(values),
            'latest': values[-1]
        }


# Global performance tracker
_performance_tracker = PerformanceTracker()


def record_performance_metric(name: str, value: float):
    """Record a performance metric."""
    _performance_tracker.record_metric(name, value)


def get_performance_summary(name: str) -> Dict[str, float]:
    """Get performance metric summary."""
    return _performance_tracker.get_metric_summary(name)