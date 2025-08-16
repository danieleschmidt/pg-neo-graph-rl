"""
Fault tolerance mechanisms for pg-neo-graph-rl.
Includes circuit breakers, retry logic, and failover strategies.
"""

import asyncio
import functools
import time
from collections import defaultdict
from dataclasses import dataclass
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Type

from .exceptions import CircuitBreakerError, GraphRLError, TimeoutError
from .logging import get_logger

logger = get_logger(__name__)


class CircuitState(Enum):
    """Circuit breaker states."""
    CLOSED = "closed"      # Normal operation
    OPEN = "open"          # Failing, blocking requests
    HALF_OPEN = "half_open"  # Testing if service recovered


@dataclass
class CircuitBreakerConfig:
    """Configuration for circuit breaker."""
    failure_threshold: int = 5
    recovery_timeout: float = 60.0
    expected_exception: Type[Exception] = Exception
    call_timeout: Optional[float] = None


class CircuitBreaker:
    """
    Circuit breaker implementation for fault tolerance.
    
    Prevents cascade failures by stopping calls to failing services.
    """

    def __init__(self, config: CircuitBreakerConfig):
        self.config = config
        self.state = CircuitState.CLOSED
        self.failure_count = 0
        self.last_failure_time = 0
        self.success_count = 0

    def __call__(self, func: Callable) -> Callable:
        """Decorator to wrap functions with circuit breaker."""
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            return self._call(func, *args, **kwargs)
        return wrapper

    def _call(self, func: Callable, *args, **kwargs) -> Any:
        """Execute function with circuit breaker protection."""
        if self.state == CircuitState.OPEN:
            self._check_recovery_timeout()
            if self.state == CircuitState.OPEN:
                raise CircuitBreakerError(
                    f"Circuit breaker is OPEN for {func.__name__}",
                    service_name=func.__name__,
                    failure_count=self.failure_count
                )

        try:
            if self.config.call_timeout:
                result = self._call_with_timeout(func, *args, **kwargs)
            else:
                result = func(*args, **kwargs)

            self._on_success()
            return result

        except self.config.expected_exception:
            self._on_failure()
            raise

    def _call_with_timeout(self, func: Callable, *args, **kwargs) -> Any:
        """Execute function with timeout protection."""
        start_time = time.time()

        try:
            # Simple timeout implementation
            result = func(*args, **kwargs)
            elapsed = time.time() - start_time

            if elapsed > self.config.call_timeout:
                raise TimeoutError(
                    f"Function {func.__name__} exceeded timeout",
                    timeout_seconds=self.config.call_timeout,
                    operation=func.__name__
                )
            return result

        except Exception as e:
            elapsed = time.time() - start_time
            if elapsed > self.config.call_timeout:
                raise TimeoutError(
                    f"Function {func.__name__} timed out during execution",
                    timeout_seconds=self.config.call_timeout,
                    operation=func.__name__
                ) from e
            raise

    def _on_success(self):
        """Handle successful execution."""
        if self.state == CircuitState.HALF_OPEN:
            self.success_count += 1
            if self.success_count >= 2:  # Require 2 successes to close
                self._close_circuit()
        else:
            self.failure_count = 0

    def _on_failure(self):
        """Handle failed execution."""
        self.failure_count += 1
        self.last_failure_time = time.time()

        if self.failure_count >= self.config.failure_threshold:
            self._open_circuit()

    def _open_circuit(self):
        """Open the circuit breaker."""
        self.state = CircuitState.OPEN
        self.last_failure_time = time.time()
        logger.warning(
            f"Circuit breaker OPENED after {self.failure_count} failures"
        )

    def _close_circuit(self):
        """Close the circuit breaker."""
        self.state = CircuitState.CLOSED
        self.failure_count = 0
        self.success_count = 0
        logger.info("Circuit breaker CLOSED - service recovered")

    def _check_recovery_timeout(self):
        """Check if recovery timeout has elapsed."""
        if (time.time() - self.last_failure_time) >= self.config.recovery_timeout:
            self.state = CircuitState.HALF_OPEN
            self.success_count = 0
            logger.info("Circuit breaker HALF_OPEN - testing recovery")


class RetryConfig:
    """Configuration for retry mechanisms."""

    def __init__(
        self,
        max_attempts: int = 3,
        base_delay: float = 1.0,
        max_delay: float = 60.0,
        backoff_factor: float = 2.0,
        jitter: bool = True,
        retryable_exceptions: Optional[List[Type[Exception]]] = None
    ):
        self.max_attempts = max_attempts
        self.base_delay = base_delay
        self.max_delay = max_delay
        self.backoff_factor = backoff_factor
        self.jitter = jitter
        self.retryable_exceptions = retryable_exceptions or [Exception]


def retry_with_backoff(config: RetryConfig = None):
    """
    Decorator for retrying functions with exponential backoff.
    
    Args:
        config: Retry configuration
    """
    if config is None:
        config = RetryConfig()

    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            last_exception = None

            for attempt in range(config.max_attempts):
                try:
                    return func(*args, **kwargs)

                except Exception as e:
                    last_exception = e

                    # Check if exception is retryable
                    if not any(isinstance(e, exc_type) for exc_type in config.retryable_exceptions):
                        logger.error(f"Non-retryable exception in {func.__name__}: {e}")
                        raise

                    if attempt == config.max_attempts - 1:
                        logger.error(f"All retry attempts failed for {func.__name__}")
                        break

                    # Calculate delay with exponential backoff
                    delay = min(
                        config.base_delay * (config.backoff_factor ** attempt),
                        config.max_delay
                    )

                    # Add jitter to avoid thundering herd
                    if config.jitter:
                        import random
                        delay *= (0.5 + 0.5 * random.random())

                    logger.warning(
                        f"Attempt {attempt + 1}/{config.max_attempts} failed for {func.__name__}: {e}. "
                        f"Retrying in {delay:.2f}s"
                    )

                    time.sleep(delay)

            raise last_exception

        return wrapper
    return decorator


class FailoverManager:
    """
    Manages failover between multiple service instances.
    """

    def __init__(self, services: List[Callable], max_failures: int = 3):
        self.services = services
        self.max_failures = max_failures
        self.failure_counts = defaultdict(int)
        self.current_service_index = 0

    def execute(self, *args, **kwargs) -> Any:
        """Execute function with failover to backup services."""
        attempts = 0
        total_services = len(self.services)

        while attempts < total_services:
            service = self.services[self.current_service_index]

            try:
                result = service(*args, **kwargs)
                # Success - reset failure count
                self.failure_counts[self.current_service_index] = 0
                return result

            except Exception as e:
                self.failure_counts[self.current_service_index] += 1

                logger.warning(
                    f"Service {self.current_service_index} failed: {e}. "
                    f"Failure count: {self.failure_counts[self.current_service_index]}"
                )

                # Move to next service
                self._switch_service()
                attempts += 1

        raise GraphRLError("All failover services have failed")

    def _switch_service(self):
        """Switch to the next available service."""
        self.current_service_index = (self.current_service_index + 1) % len(self.services)

        # Skip services that have exceeded failure threshold
        attempts = 0
        while (self.failure_counts[self.current_service_index] >= self.max_failures and
               attempts < len(self.services)):
            self.current_service_index = (self.current_service_index + 1) % len(self.services)
            attempts += 1


class HealthChecker:
    """
    Periodic health checking for system components.
    """

    def __init__(self, check_interval: float = 30.0):
        self.check_interval = check_interval
        self.health_status: Dict[str, bool] = {}
        self.health_checks: Dict[str, Callable] = {}
        self.running = False

    def register_health_check(self, name: str, check_func: Callable[[], bool]):
        """Register a health check function."""
        self.health_checks[name] = check_func
        self.health_status[name] = True

    async def start_monitoring(self):
        """Start continuous health monitoring."""
        self.running = True

        while self.running:
            await self._perform_health_checks()
            await asyncio.sleep(self.check_interval)

    def stop_monitoring(self):
        """Stop health monitoring."""
        self.running = False

    async def _perform_health_checks(self):
        """Perform all registered health checks."""
        for name, check_func in self.health_checks.items():
            try:
                self.health_status[name] = check_func()
                if not self.health_status[name]:
                    logger.warning(f"Health check failed: {name}")
            except Exception as e:
                logger.error(f"Health check error for {name}: {e}")
                self.health_status[name] = False

    def is_healthy(self, component: str = None) -> bool:
        """Check if component (or all components) are healthy."""
        if component:
            return self.health_status.get(component, False)

        return all(self.health_status.values()) if self.health_status else True

    def get_health_report(self) -> Dict[str, bool]:
        """Get full health status report."""
        return self.health_status.copy()


# Convenience decorators with default configurations
default_circuit_breaker = CircuitBreaker(CircuitBreakerConfig())
robust_circuit_breaker = CircuitBreaker(CircuitBreakerConfig(
    failure_threshold=3,
    recovery_timeout=30.0,
    call_timeout=10.0
))

default_retry = retry_with_backoff()
robust_retry = retry_with_backoff(RetryConfig(
    max_attempts=5,
    base_delay=0.5,
    max_delay=30.0,
    backoff_factor=1.5
))
