"""Health check system for PG-Neo-Graph-RL."""

import time
import asyncio
import threading
from typing import Dict, Any, List, Optional, Callable, Awaitable
from dataclasses import dataclass, field
from enum import Enum
import logging
import jax

logger = logging.getLogger(__name__)


class HealthStatus(Enum):
    """Health check status."""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    UNKNOWN = "unknown"


@dataclass
class HealthCheckResult:
    """Result of a health check."""
    name: str
    status: HealthStatus
    message: str = ""
    duration_ms: float = 0.0
    details: Dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            'name': self.name,
            'status': self.status.value,
            'message': self.message,
            'duration_ms': self.duration_ms,
            'details': self.details,
            'timestamp': self.timestamp
        }


class HealthCheck:
    """Base class for health checks."""
    
    def __init__(self, name: str, timeout_seconds: float = 5.0):
        self.name = name
        self.timeout_seconds = timeout_seconds
    
    def check(self) -> HealthCheckResult:
        """Perform the health check."""
        start_time = time.time()
        
        try:
            result = self._perform_check()
            duration_ms = (time.time() - start_time) * 1000
            
            return HealthCheckResult(
                name=self.name,
                status=result.get('status', HealthStatus.UNKNOWN),
                message=result.get('message', ''),
                duration_ms=duration_ms,
                details=result.get('details', {})
            )
        
        except Exception as e:
            duration_ms = (time.time() - start_time) * 1000
            logger.error(f"Health check {self.name} failed: {e}")
            
            return HealthCheckResult(
                name=self.name,
                status=HealthStatus.UNHEALTHY,
                message=f"Check failed: {str(e)}",
                duration_ms=duration_ms,
                details={'error': str(e), 'error_type': type(e).__name__}
            )
    
    def _perform_check(self) -> Dict[str, Any]:
        """Override this method to implement the actual check."""
        raise NotImplementedError


class AsyncHealthCheck(HealthCheck):
    """Base class for async health checks."""
    
    async def check_async(self) -> HealthCheckResult:
        """Perform the async health check."""
        start_time = time.time()
        
        try:
            result = await self._perform_check_async()
            duration_ms = (time.time() - start_time) * 1000
            
            return HealthCheckResult(
                name=self.name,
                status=result.get('status', HealthStatus.UNKNOWN),
                message=result.get('message', ''),
                duration_ms=duration_ms,
                details=result.get('details', {})
            )
        
        except Exception as e:
            duration_ms = (time.time() - start_time) * 1000
            logger.error(f"Async health check {self.name} failed: {e}")
            
            return HealthCheckResult(
                name=self.name,
                status=HealthStatus.UNHEALTHY,
                message=f"Check failed: {str(e)}",
                duration_ms=duration_ms,
                details={'error': str(e), 'error_type': type(e).__name__}
            )
    
    def check(self) -> HealthCheckResult:
        """Run async check in sync context."""
        try:
            loop = asyncio.get_event_loop()
            return loop.run_until_complete(self.check_async())
        except RuntimeError:
            # No event loop running, create a new one
            loop = asyncio.new_event_loop()
            try:
                return loop.run_until_complete(self.check_async())
            finally:
                loop.close()
    
    async def _perform_check_async(self) -> Dict[str, Any]:
        """Override this method to implement the actual async check."""
        raise NotImplementedError


# Built-in health checks
class JAXHealthCheck(HealthCheck):
    """Check JAX device availability and basic operations."""
    
    def __init__(self):
        super().__init__("jax_devices")
    
    def _perform_check(self) -> Dict[str, Any]:
        devices = jax.devices()
        
        if not devices:
            return {
                'status': HealthStatus.UNHEALTHY,
                'message': 'No JAX devices available',
                'details': {'device_count': 0}
            }
        
        # Test basic JAX operation
        try:
            test_array = jax.numpy.array([1, 2, 3])
            result = jax.numpy.sum(test_array)
            jax.block_until_ready(result)
        except Exception as e:
            return {
                'status': HealthStatus.UNHEALTHY,
                'message': f'JAX operation failed: {str(e)}',
                'details': {
                    'device_count': len(devices),
                    'devices': [str(d) for d in devices],
                    'error': str(e)
                }
            }
        
        return {
            'status': HealthStatus.HEALTHY,
            'message': f'{len(devices)} JAX device(s) available and working',
            'details': {
                'device_count': len(devices),
                'devices': [str(d) for d in devices],
                'platform': devices[0].platform if devices else 'unknown'
            }
        }


class MemoryHealthCheck(HealthCheck):
    """Check system memory usage."""
    
    def __init__(self, warning_threshold: float = 0.8, critical_threshold: float = 0.95):
        super().__init__("memory_usage")
        self.warning_threshold = warning_threshold
        self.critical_threshold = critical_threshold
    
    def _perform_check(self) -> Dict[str, Any]:
        try:
            import psutil
            memory = psutil.virtual_memory()
            usage_ratio = memory.percent / 100.0
            
            if usage_ratio >= self.critical_threshold:
                status = HealthStatus.UNHEALTHY
                message = f'Critical memory usage: {memory.percent:.1f}%'
            elif usage_ratio >= self.warning_threshold:
                status = HealthStatus.DEGRADED
                message = f'High memory usage: {memory.percent:.1f}%'
            else:
                status = HealthStatus.HEALTHY
                message = f'Memory usage normal: {memory.percent:.1f}%'
            
            return {
                'status': status,
                'message': message,
                'details': {
                    'total_bytes': memory.total,
                    'used_bytes': memory.used,
                    'available_bytes': memory.available,
                    'usage_percent': memory.percent,
                    'warning_threshold': self.warning_threshold,
                    'critical_threshold': self.critical_threshold
                }
            }
        
        except ImportError:
            return {
                'status': HealthStatus.UNKNOWN,
                'message': 'psutil not available for memory monitoring',
                'details': {}
            }


class DiskSpaceHealthCheck(HealthCheck):
    """Check disk space usage."""
    
    def __init__(self, path: str = "/", warning_threshold: float = 0.8, critical_threshold: float = 0.95):
        super().__init__("disk_space")
        self.path = path
        self.warning_threshold = warning_threshold
        self.critical_threshold = critical_threshold
    
    def _perform_check(self) -> Dict[str, Any]:
        try:
            import shutil
            total, used, free = shutil.disk_usage(self.path)
            usage_ratio = used / total
            
            if usage_ratio >= self.critical_threshold:
                status = HealthStatus.UNHEALTHY
                message = f'Critical disk usage: {usage_ratio*100:.1f}%'
            elif usage_ratio >= self.warning_threshold:
                status = HealthStatus.DEGRADED
                message = f'High disk usage: {usage_ratio*100:.1f}%'
            else:
                status = HealthStatus.HEALTHY
                message = f'Disk usage normal: {usage_ratio*100:.1f}%'
            
            return {
                'status': status,
                'message': message,
                'details': {
                    'path': self.path,
                    'total_bytes': total,
                    'used_bytes': used,
                    'free_bytes': free,
                    'usage_percent': usage_ratio * 100,
                    'warning_threshold': self.warning_threshold,
                    'critical_threshold': self.critical_threshold
                }
            }
        
        except Exception as e:
            return {
                'status': HealthStatus.UNHEALTHY,
                'message': f'Failed to check disk space: {str(e)}',
                'details': {'path': self.path, 'error': str(e)}
            }


class DatabaseHealthCheck(AsyncHealthCheck):
    """Check database connectivity (placeholder for future database integration)."""
    
    def __init__(self, connection_string: Optional[str] = None):
        super().__init__("database")
        self.connection_string = connection_string
    
    async def _perform_check_async(self) -> Dict[str, Any]:
        # Placeholder - implement actual database check when needed
        if not self.connection_string:
            return {
                'status': HealthStatus.UNKNOWN,
                'message': 'No database configured',
                'details': {}
            }
        
        # TODO: Implement actual database connectivity check
        return {
            'status': HealthStatus.HEALTHY,
            'message': 'Database connectivity check not implemented',
            'details': {'connection_string': self.connection_string}
        }


class MetricsHealthCheck(HealthCheck):
    """Check metrics collection system."""
    
    def __init__(self):
        super().__init__("metrics_system")
    
    def _perform_check(self) -> Dict[str, Any]:
        try:
            from .metrics import get_metrics_collector
            collector = get_metrics_collector()
            health_status = collector.get_health_status()
            
            return {
                'status': HealthStatus.HEALTHY,
                'message': 'Metrics system operational',
                'details': health_status
            }
        
        except Exception as e:
            return {
                'status': HealthStatus.UNHEALTHY,
                'message': f'Metrics system error: {str(e)}',
                'details': {'error': str(e)}
            }


class HealthManager:
    """Central health check management."""
    
    def __init__(self):
        self.checks: Dict[str, HealthCheck] = {}
        self.results_cache: Dict[str, HealthCheckResult] = {}
        self.cache_timeout: float = 30.0  # Cache results for 30 seconds
        self._lock = threading.Lock()
        
        # Register default health checks
        self.register_default_checks()
    
    def register_default_checks(self):
        """Register default health checks."""
        self.register_check(JAXHealthCheck())
        self.register_check(MemoryHealthCheck())
        self.register_check(DiskSpaceHealthCheck())
        self.register_check(MetricsHealthCheck())
    
    def register_check(self, check: HealthCheck):
        """Register a health check."""
        with self._lock:
            self.checks[check.name] = check
        logger.info(f"Registered health check: {check.name}")
    
    def unregister_check(self, name: str):
        """Unregister a health check."""
        with self._lock:
            self.checks.pop(name, None)
            self.results_cache.pop(name, None)
        logger.info(f"Unregistered health check: {name}")
    
    def run_check(self, name: str, use_cache: bool = True) -> Optional[HealthCheckResult]:
        """Run a specific health check."""
        with self._lock:
            check = self.checks.get(name)
            if not check:
                return None
            
            # Check cache
            if use_cache and name in self.results_cache:
                cached_result = self.results_cache[name]
                if time.time() - cached_result.timestamp < self.cache_timeout:
                    return cached_result
        
        # Run the check
        result = check.check()
        
        # Cache the result
        with self._lock:
            self.results_cache[name] = result
        
        return result
    
    def run_all_checks(self, use_cache: bool = True) -> Dict[str, HealthCheckResult]:
        """Run all registered health checks."""
        results = {}
        
        with self._lock:
            check_names = list(self.checks.keys())
        
        for name in check_names:
            result = self.run_check(name, use_cache)
            if result:
                results[name] = result
        
        return results
    
    def get_overall_status(self, use_cache: bool = True) -> Dict[str, Any]:
        """Get overall system health status."""
        results = self.run_all_checks(use_cache)
        
        if not results:
            return {
                'status': HealthStatus.UNKNOWN.value,
                'message': 'No health checks configured',
                'checks': {},
                'summary': {
                    'total': 0,
                    'healthy': 0,
                    'degraded': 0,
                    'unhealthy': 0,
                    'unknown': 0
                }
            }
        
        # Count statuses
        summary = {
            'total': len(results),
            'healthy': 0,
            'degraded': 0,
            'unhealthy': 0,
            'unknown': 0
        }
        
        for result in results.values():
            if result.status == HealthStatus.HEALTHY:
                summary['healthy'] += 1
            elif result.status == HealthStatus.DEGRADED:
                summary['degraded'] += 1
            elif result.status == HealthStatus.UNHEALTHY:
                summary['unhealthy'] += 1
            else:
                summary['unknown'] += 1
        
        # Determine overall status
        if summary['unhealthy'] > 0:
            overall_status = HealthStatus.UNHEALTHY
            message = f"{summary['unhealthy']} critical health check(s) failing"
        elif summary['degraded'] > 0:
            overall_status = HealthStatus.DEGRADED
            message = f"{summary['degraded']} health check(s) degraded"
        elif summary['healthy'] == summary['total']:
            overall_status = HealthStatus.HEALTHY
            message = "All health checks passing"
        else:
            overall_status = HealthStatus.UNKNOWN
            message = "Unknown health status"
        
        return {
            'status': overall_status.value,
            'message': message,
            'checks': {name: result.to_dict() for name, result in results.items()},
            'summary': summary,
            'timestamp': time.time()
        }


# Global health manager
_global_health_manager: Optional[HealthManager] = None


def get_health_manager() -> HealthManager:
    """Get the global health manager."""
    global _global_health_manager
    if _global_health_manager is None:
        _global_health_manager = HealthManager()
    return _global_health_manager


def initialize_health_checks() -> HealthManager:
    """Initialize the global health manager."""
    global _global_health_manager
    _global_health_manager = HealthManager()
    return _global_health_manager


# Convenience functions
def register_health_check(check: HealthCheck):
    """Register a health check with the global manager."""
    get_health_manager().register_check(check)


def run_health_check(name: str) -> Optional[HealthCheckResult]:
    """Run a specific health check."""
    return get_health_manager().run_check(name)


def get_health_status() -> Dict[str, Any]:
    """Get overall health status."""
    return get_health_manager().get_overall_status()


# Health check decorators
def health_check_endpoint(func):
    """Decorator to create a health check endpoint."""
    def wrapper(*args, **kwargs):
        try:
            status = get_health_status()
            # Return appropriate HTTP status code based on health
            if status['status'] == HealthStatus.HEALTHY.value:
                return status, 200
            elif status['status'] == HealthStatus.DEGRADED.value:
                return status, 200  # Still operational
            else:
                return status, 503  # Service unavailable
        except Exception as e:
            return {
                'status': HealthStatus.UNHEALTHY.value,
                'message': f'Health check error: {str(e)}',
                'error': str(e)
            }, 500
    
    return wrapper


# Readiness and liveness checks for Kubernetes
def liveness_check() -> Dict[str, Any]:
    """Basic liveness check - just verify the service is running."""
    return {
        'status': HealthStatus.HEALTHY.value,
        'message': 'Service is alive',
        'timestamp': time.time()
    }


def readiness_check() -> Dict[str, Any]:
    """Readiness check - verify service is ready to handle requests."""
    # Run critical checks only for readiness
    manager = get_health_manager()
    critical_checks = ['jax_devices', 'metrics_system']
    
    results = {}
    for check_name in critical_checks:
        result = manager.run_check(check_name)
        if result:
            results[check_name] = result
    
    # Service is ready if all critical checks pass
    unhealthy_checks = [
        name for name, result in results.items()
        if result.status == HealthStatus.UNHEALTHY
    ]
    
    if unhealthy_checks:
        return {
            'status': HealthStatus.UNHEALTHY.value,
            'message': f'Critical checks failing: {", ".join(unhealthy_checks)}',
            'checks': {name: result.to_dict() for name, result in results.items()},
            'timestamp': time.time()
        }
    else:
        return {
            'status': HealthStatus.HEALTHY.value,
            'message': 'Service is ready',
            'checks': {name: result.to_dict() for name, result in results.items()},
            'timestamp': time.time()
        }