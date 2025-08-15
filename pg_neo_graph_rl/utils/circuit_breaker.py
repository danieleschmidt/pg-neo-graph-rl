"""Circuit breaker pattern for fault tolerance in federated systems."""
import time
import threading
from enum import Enum
from typing import Callable, Any, Optional, Dict
from dataclasses import dataclass
from ..utils.logging import get_logger
from ..utils.exceptions import CircuitBreakerError

logger = get_logger(__name__)

class CircuitBreakerState(Enum):
    """Circuit breaker states."""
    CLOSED = "closed"      # Normal operation
    OPEN = "open"          # Preventing calls
    HALF_OPEN = "half_open" # Testing if service recovered

@dataclass
class CircuitBreakerStats:
    """Circuit breaker statistics."""
    total_requests: int = 0
    failed_requests: int = 0
    successful_requests: int = 0
    last_failure_time: Optional[float] = None
    last_success_time: Optional[float] = None
    consecutive_failures: int = 0
    consecutive_successes: int = 0

class CircuitBreaker:
    """Circuit breaker for protecting against cascading failures."""
    
    def __init__(self, 
                 failure_threshold: int = 5,
                 recovery_timeout: float = 60.0,
                 expected_exception: type = Exception,
                 name: str = "circuit_breaker"):
        """
        Initialize circuit breaker.
        
        Args:
            failure_threshold: Number of failures to trigger OPEN state
            recovery_timeout: Seconds before trying HALF_OPEN state
            expected_exception: Exception type that counts as failure
            name: Name for logging
        """
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.expected_exception = expected_exception
        self.name = name
        
        self._state = CircuitBreakerState.CLOSED
        self._stats = CircuitBreakerStats()
        self._lock = threading.RLock()
        self._last_failure_time = None
        
        logger.info(f"Initialized circuit breaker '{name}' with threshold={failure_threshold}")
    
    @property
    def state(self) -> CircuitBreakerState:
        """Get current circuit breaker state."""
        with self._lock:
            return self._state
    
    @property 
    def stats(self) -> CircuitBreakerStats:
        """Get circuit breaker statistics."""
        with self._lock:
            return self._stats
    
    def call(self, func: Callable, *args, **kwargs) -> Any:
        """
        Execute function through circuit breaker.
        
        Args:
            func: Function to execute
            *args, **kwargs: Function arguments
            
        Returns:
            Function result
            
        Raises:
            CircuitBreakerError: If circuit is open
        """
        with self._lock:
            current_state = self._state
            
        if current_state == CircuitBreakerState.OPEN:
            if self._should_attempt_reset():
                self._move_to_half_open()
            else:
                self._stats.total_requests += 1
                raise CircuitBreakerError(f"Circuit breaker '{self.name}' is OPEN")
        
        try:
            result = func(*args, **kwargs)
            self._on_success()
            return result
            
        except self.expected_exception as e:
            self._on_failure(e)
            raise
        except Exception as e:
            # Unexpected exceptions don't count as failures
            logger.warning(f"Unexpected exception in circuit breaker '{self.name}': {e}")
            self._stats.total_requests += 1
            raise
    
    def _should_attempt_reset(self) -> bool:
        """Check if enough time has passed to attempt reset."""
        if self._last_failure_time is None:
            return True
        return time.time() - self._last_failure_time >= self.recovery_timeout
    
    def _move_to_half_open(self) -> None:
        """Move circuit breaker to HALF_OPEN state."""
        with self._lock:
            self._state = CircuitBreakerState.HALF_OPEN
        logger.info(f"Circuit breaker '{self.name}' moved to HALF_OPEN")
    
    def _on_success(self) -> None:
        """Handle successful function execution."""
        with self._lock:
            self._stats.total_requests += 1
            self._stats.successful_requests += 1
            self._stats.consecutive_successes += 1
            self._stats.consecutive_failures = 0
            self._stats.last_success_time = time.time()
            
            if self._state == CircuitBreakerState.HALF_OPEN:
                self._state = CircuitBreakerState.CLOSED
                logger.info(f"Circuit breaker '{self.name}' moved to CLOSED after success")
    
    def _on_failure(self, exception: Exception) -> None:
        """Handle failed function execution."""
        with self._lock:
            self._stats.total_requests += 1
            self._stats.failed_requests += 1
            self._stats.consecutive_failures += 1
            self._stats.consecutive_successes = 0
            self._stats.last_failure_time = time.time()
            self._last_failure_time = self._stats.last_failure_time
            
            if (self._stats.consecutive_failures >= self.failure_threshold and 
                self._state != CircuitBreakerState.OPEN):
                self._state = CircuitBreakerState.OPEN
                logger.warning(
                    f"Circuit breaker '{self.name}' moved to OPEN after {self._stats.consecutive_failures} failures"
                )
    
    def force_open(self) -> None:
        """Manually force circuit breaker to OPEN state."""
        with self._lock:
            self._state = CircuitBreakerState.OPEN
            self._last_failure_time = time.time()
        logger.warning(f"Circuit breaker '{self.name}' forced to OPEN state")
    
    def force_close(self) -> None:
        """Manually force circuit breaker to CLOSED state."""
        with self._lock:
            self._state = CircuitBreakerState.CLOSED
            self._stats.consecutive_failures = 0
        logger.info(f"Circuit breaker '{self.name}' forced to CLOSED state")
    
    def reset_stats(self) -> None:
        """Reset circuit breaker statistics."""
        with self._lock:
            self._stats = CircuitBreakerStats()
        logger.info(f"Reset statistics for circuit breaker '{self.name}'")

class FederatedCircuitBreakerManager:
    """Manage circuit breakers for federated learning components."""
    
    def __init__(self):
        self.breakers: Dict[str, CircuitBreaker] = {}
        self._lock = threading.RLock()
    
    def create_breaker(self, name: str, **kwargs) -> CircuitBreaker:
        """Create and register a new circuit breaker."""
        with self._lock:
            if name in self.breakers:
                logger.warning(f"Circuit breaker '{name}' already exists")
                return self.breakers[name]
                
            breaker = CircuitBreaker(name=name, **kwargs)
            self.breakers[name] = breaker
            logger.info(f"Created circuit breaker '{name}'")
            return breaker
    
    def get_breaker(self, name: str) -> Optional[CircuitBreaker]:
        """Get circuit breaker by name."""
        with self._lock:
            return self.breakers.get(name)
    
    def get_all_stats(self) -> Dict[str, CircuitBreakerStats]:
        """Get statistics for all circuit breakers."""
        with self._lock:
            return {name: breaker.stats for name, breaker in self.breakers.items()}
    
    def get_health_summary(self) -> Dict[str, Any]:
        """Get health summary of all circuit breakers."""
        with self._lock:
            summary = {
                'total_breakers': len(self.breakers),
                'open_breakers': 0,
                'half_open_breakers': 0,
                'closed_breakers': 0,
                'unhealthy_breakers': []
            }
            
            for name, breaker in self.breakers.items():
                state = breaker.state
                stats = breaker.stats
                
                if state == CircuitBreakerState.OPEN:
                    summary['open_breakers'] += 1
                    summary['unhealthy_breakers'].append(name)
                elif state == CircuitBreakerState.HALF_OPEN:
                    summary['half_open_breakers'] += 1
                else:
                    summary['closed_breakers'] += 1
                
                # Consider unhealthy if failure rate > 50% and has significant traffic
                if (stats.total_requests > 10 and 
                    stats.failed_requests / stats.total_requests > 0.5):
                    if name not in summary['unhealthy_breakers']:
                        summary['unhealthy_breakers'].append(name)
            
            return summary
    
    def emergency_open_all(self) -> None:
        """Emergency: open all circuit breakers."""
        with self._lock:
            for breaker in self.breakers.values():
                breaker.force_open()
        logger.error("EMERGENCY: All circuit breakers forced OPEN")
    
    def reset_all(self) -> None:
        """Reset all circuit breakers to CLOSED state."""
        with self._lock:
            for breaker in self.breakers.values():
                breaker.force_close()
                breaker.reset_stats()
        logger.info("Reset all circuit breakers")

def circuit_breaker(failure_threshold: int = 5,
                   recovery_timeout: float = 60.0,
                   expected_exception: type = Exception,
                   name: str = None):
    """
    Decorator to add circuit breaker protection to functions.
    
    Args:
        failure_threshold: Number of failures to trigger OPEN state
        recovery_timeout: Seconds before trying HALF_OPEN state  
        expected_exception: Exception type that counts as failure
        name: Circuit breaker name (defaults to function name)
    """
    def decorator(func):
        breaker_name = name or func.__name__
        breaker = CircuitBreaker(
            failure_threshold=failure_threshold,
            recovery_timeout=recovery_timeout,
            expected_exception=expected_exception,
            name=breaker_name
        )
        
        def wrapper(*args, **kwargs):
            return breaker.call(func, *args, **kwargs)
        
        wrapper._circuit_breaker = breaker
        return wrapper
    return decorator

# Global circuit breaker manager instance
_global_manager = FederatedCircuitBreakerManager()

def get_circuit_breaker_manager() -> FederatedCircuitBreakerManager:
    """Get the global circuit breaker manager."""
    return _global_manager