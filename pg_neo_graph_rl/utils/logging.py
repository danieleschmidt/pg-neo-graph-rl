"""
Structured logging utilities for pg-neo-graph-rl.
"""
import json
import logging
import sys
import threading
import time
from datetime import datetime
from typing import Any, Dict, Optional


class StructuredFormatter(logging.Formatter):
    """Custom formatter for structured logging."""

    def format(self, record: logging.LogRecord) -> str:
        """Format log record as structured JSON."""
        log_entry = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno
        }

        # Add extra fields if present
        if hasattr(record, 'agent_id'):
            log_entry["agent_id"] = record.agent_id
        if hasattr(record, 'episode'):
            log_entry["episode"] = record.episode
        if hasattr(record, 'environment'):
            log_entry["environment"] = record.environment
        if hasattr(record, 'metrics'):
            log_entry["metrics"] = record.metrics
        if hasattr(record, 'error_type'):
            log_entry["error_type"] = record.error_type
        if hasattr(record, 'stack_trace'):
            log_entry["stack_trace"] = record.stack_trace

        return json.dumps(log_entry)


class AgentLoggerAdapter(logging.LoggerAdapter):
    """Logger adapter that adds agent context to log messages."""

    def __init__(self, logger: logging.Logger, agent_id: int):
        super().__init__(logger, {"agent_id": agent_id})

    def process(self, msg: str, kwargs: Dict[str, Any]) -> tuple:
        """Add agent ID to log record."""
        if "extra" not in kwargs:
            kwargs["extra"] = {}
        kwargs["extra"]["agent_id"] = self.extra["agent_id"]
        return msg, kwargs


class PerformanceLogger:
    """Logger for performance monitoring."""

    def __init__(self, logger: logging.Logger):
        self.logger = logger
        self._timers = {}
        self._lock = threading.Lock()

    def start_timer(self, name: str) -> None:
        """Start a performance timer."""
        with self._lock:
            self._timers[name] = time.time()

    def end_timer(self, name: str, log_level: int = logging.INFO) -> float:
        """End a performance timer and log the duration."""
        with self._lock:
            if name not in self._timers:
                self.logger.warning(f"Timer '{name}' was not started")
                return 0.0

            duration = time.time() - self._timers[name]
            del self._timers[name]

        self.logger.log(
            log_level,
            f"Performance: {name} completed",
            extra={"metrics": {"duration_seconds": duration, "operation": name}}
        )

        return duration

    def timer_context(self, name: str, log_level: int = logging.INFO):
        """Context manager for timing operations."""
        return TimerContext(self, name, log_level)


class TimerContext:
    """Context manager for performance timing."""

    def __init__(self, perf_logger: PerformanceLogger, name: str, log_level: int):
        self.perf_logger = perf_logger
        self.name = name
        self.log_level = log_level

    def __enter__(self):
        self.perf_logger.start_timer(self.name)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.perf_logger.end_timer(self.name, self.log_level)


def setup_logger(name: str,
                level: int = logging.INFO,
                structured: bool = True,
                log_file: Optional[str] = None) -> logging.Logger:
    """
    Set up a logger with optional structured formatting.
    
    Args:
        name: Logger name
        level: Logging level
        structured: Whether to use structured JSON formatting
        log_file: Optional file to log to
        
    Returns:
        Configured logger
    """
    logger = logging.getLogger(name)

    # Clear existing handlers
    logger.handlers.clear()
    logger.setLevel(level)

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)

    if structured:
        formatter = StructuredFormatter()
    else:
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )

    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # File handler if specified
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    # Prevent propagation to root logger
    logger.propagate = False

    return logger


def get_logger(name: str) -> logging.Logger:
    """Get existing logger or create new one with default settings."""
    logger = logging.getLogger(name)

    # If logger has no handlers, set it up with defaults
    if not logger.handlers:
        return setup_logger(name)

    return logger


def get_agent_logger(agent_id: int, base_name: str = "pg_neo_graph_rl") -> AgentLoggerAdapter:
    """Get logger adapter for specific agent."""
    base_logger = get_logger(f"{base_name}.agent_{agent_id}")
    return AgentLoggerAdapter(base_logger, agent_id)


def get_performance_logger(name: str = "pg_neo_graph_rl.performance") -> PerformanceLogger:
    """Get performance logger."""
    logger = get_logger(name)
    return PerformanceLogger(logger)


def log_exception(logger: logging.Logger,
                 exception: Exception,
                 context: Optional[Dict[str, Any]] = None) -> None:
    """
    Log exception with full context.
    
    Args:
        logger: Logger to use
        exception: Exception to log
        context: Additional context information
    """
    import traceback

    extra = {
        "error_type": type(exception).__name__,
        "stack_trace": traceback.format_exc()
    }

    if context:
        extra.update(context)

    logger.error(f"Exception occurred: {str(exception)}", extra=extra)


def log_federated_round(logger: logging.Logger,
                       round_number: int,
                       num_agents: int,
                       aggregation_method: str,
                       metrics: Dict[str, Any]) -> None:
    """
    Log federated learning round information.
    
    Args:
        logger: Logger to use
        round_number: Current round number
        num_agents: Number of participating agents
        aggregation_method: Aggregation method used
        metrics: Round metrics
    """
    logger.info(
        f"Federated round {round_number} completed",
        extra={
            "round_number": round_number,
            "num_agents": num_agents,
            "aggregation_method": aggregation_method,
            "metrics": metrics
        }
    )


def log_training_episode(logger: logging.Logger,
                        episode: int,
                        environment: str,
                        reward: float,
                        steps: int,
                        metrics: Dict[str, Any]) -> None:
    """
    Log training episode information.
    
    Args:
        logger: Logger to use
        episode: Episode number
        environment: Environment name
        reward: Episode reward
        steps: Number of steps
        metrics: Episode metrics
    """
    logger.info(
        f"Episode {episode} completed",
        extra={
            "episode": episode,
            "environment": environment,
            "reward": reward,
            "steps": steps,
            "metrics": metrics
        }
    )


def log_system_health(logger: logging.Logger,
                     component: str,
                     status: str,
                     metrics: Dict[str, Any]) -> None:
    """
    Log system health status.
    
    Args:
        logger: Logger to use
        component: Component name
        status: Health status (healthy, degraded, critical)
        metrics: Health metrics
    """
    level = logging.INFO
    if status == "degraded":
        level = logging.WARNING
    elif status == "critical":
        level = logging.ERROR

    logger.log(
        level,
        f"Health check: {component} is {status}",
        extra={
            "component": component,
            "status": status,
            "metrics": metrics
        }
    )
