"""Logging configuration for PG-Neo-Graph-RL."""

import os
import sys
import json
import logging
import logging.handlers
from typing import Dict, Any, Optional
from pathlib import Path
from datetime import datetime


class JSONFormatter(logging.Formatter):
    """JSON formatter for structured logging."""
    
    def __init__(self, include_extra: bool = True):
        super().__init__()
        self.include_extra = include_extra
    
    def format(self, record: logging.LogRecord) -> str:
        # Basic log record
        log_data = {
            'timestamp': datetime.utcfromtimestamp(record.created).isoformat() + 'Z',
            'level': record.levelname,
            'logger': record.name,
            'message': record.getMessage(),
            'module': record.module,
            'function': record.funcName,
            'line': record.lineno,
        }
        
        # Add thread info if available
        if hasattr(record, 'thread') and record.thread:
            log_data['thread_id'] = record.thread
            log_data['thread_name'] = getattr(record, 'threadName', None)
        
        # Add process info if available  
        if hasattr(record, 'process') and record.process:
            log_data['process_id'] = record.process
            log_data['process_name'] = getattr(record, 'processName', None)
        
        # Add exception info if present
        if record.exc_info:
            log_data['exception'] = {
                'type': record.exc_info[0].__name__,
                'message': str(record.exc_info[1]),
                'traceback': self.formatException(record.exc_info)
            }
        
        # Add extra fields
        if self.include_extra:
            for key, value in record.__dict__.items():
                if key not in ['name', 'msg', 'args', 'levelname', 'levelno', 'pathname', 
                              'filename', 'module', 'lineno', 'funcName', 'created', 
                              'msecs', 'relativeCreated', 'thread', 'threadName', 
                              'processName', 'process', 'exc_info', 'exc_text', 'stack_info']:
                    try:
                        # Only include JSON-serializable values
                        json.dumps(value)
                        log_data[key] = value
                    except (TypeError, ValueError):
                        log_data[key] = str(value)
        
        return json.dumps(log_data, ensure_ascii=False)


class ColoredFormatter(logging.Formatter):
    """Colored formatter for console output."""
    
    COLORS = {
        'DEBUG': '\033[36m',     # Cyan
        'INFO': '\033[32m',      # Green  
        'WARNING': '\033[33m',   # Yellow
        'ERROR': '\033[31m',     # Red
        'CRITICAL': '\033[35m',  # Magenta
    }
    RESET = '\033[0m'
    
    def __init__(self, fmt: str = None, datefmt: str = None):
        super().__init__(fmt, datefmt)
        
    def format(self, record: logging.LogRecord) -> str:
        # Add color to level name
        levelname = record.levelname
        if levelname in self.COLORS:
            colored_levelname = f"{self.COLORS[levelname]}{levelname}{self.RESET}"
            record.levelname = colored_levelname
        
        # Format the message
        formatted = super().format(record)
        
        # Reset levelname for subsequent formatters
        record.levelname = levelname
        
        return formatted


def setup_logging(
    level: str = "INFO",
    format_type: str = "text",
    log_file: Optional[str] = None,
    log_dir: Optional[str] = None,
    max_file_size: int = 10 * 1024 * 1024,  # 10MB
    backup_count: int = 5,
    enable_console: bool = True,
    logger_configs: Optional[Dict[str, Dict[str, Any]]] = None
) -> logging.Logger:
    """Setup logging configuration.
    
    Args:
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        format_type: Format type ('text', 'json')
        log_file: Specific log file path
        log_dir: Directory for log files
        max_file_size: Maximum size of log files before rotation
        backup_count: Number of backup files to keep
        enable_console: Whether to enable console logging
        logger_configs: Custom configurations for specific loggers
        
    Returns:
        Configured root logger
    """
    # Convert level string to logging level
    numeric_level = getattr(logging, level.upper(), logging.INFO)
    
    # Create log directory if specified
    if log_dir:
        log_dir_path = Path(log_dir)
        log_dir_path.mkdir(parents=True, exist_ok=True)
        if not log_file:
            log_file = log_dir_path / "pg_neo_graph_rl.log"
    
    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(numeric_level)
    
    # Clear existing handlers
    root_logger.handlers.clear()
    
    # Setup formatters
    if format_type.lower() == 'json':
        file_formatter = JSONFormatter()
        console_formatter = JSONFormatter()
    else:
        file_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        console_formatter = ColoredFormatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
    
    # Setup console handler
    if enable_console:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(numeric_level)
        console_handler.setFormatter(console_formatter)
        root_logger.addHandler(console_handler)
    
    # Setup file handler
    if log_file:
        file_handler = logging.handlers.RotatingFileHandler(
            log_file,
            maxBytes=max_file_size,
            backupCount=backup_count
        )
        file_handler.setLevel(numeric_level)
        file_handler.setFormatter(file_formatter)
        root_logger.addHandler(file_handler)
    
    # Configure specific loggers
    if logger_configs:
        for logger_name, config in logger_configs.items():
            logger = logging.getLogger(logger_name)
            if 'level' in config:
                logger.setLevel(getattr(logging, config['level'].upper()))
            if 'handlers' in config:
                logger.handlers = config['handlers']
    
    # Suppress noisy third-party loggers
    logging.getLogger('urllib3').setLevel(logging.WARNING)
    logging.getLogger('requests').setLevel(logging.WARNING)
    logging.getLogger('matplotlib').setLevel(logging.WARNING)
    
    return root_logger


def setup_logging_from_env() -> logging.Logger:
    """Setup logging from environment variables."""
    config = {
        'level': os.getenv('LOG_LEVEL', 'INFO'),
        'format_type': os.getenv('LOG_FORMAT', 'text'),
        'log_file': os.getenv('LOG_FILE'),
        'log_dir': os.getenv('LOG_DIR', 'logs'),
        'enable_console': os.getenv('LOG_CONSOLE', 'true').lower() == 'true',
    }
    
    # Remove None values
    config = {k: v for k, v in config.items() if v is not None}
    
    return setup_logging(**config)


def get_logger(name: str) -> logging.Logger:
    """Get a logger with the specified name."""
    return logging.getLogger(name)


# Context manager for logging additional context
class LogContext:
    """Context manager for adding context to log messages."""
    
    def __init__(self, logger: logging. Logger, **context):
        self.logger = logger
        self.context = context
        self.old_factory = None
    
    def __enter__(self):
        self.old_factory = logging.getLogRecordFactory()
        
        def record_factory(*args, **kwargs):
            record = self.old_factory(*args, **kwargs)
            for key, value in self.context.items():
                setattr(record, key, value)
            return record
        
        logging.setLogRecordFactory(record_factory)
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        logging.setLogRecordFactory(self.old_factory)


# Logging decorators
def log_function_call(logger: Optional[logging.Logger] = None, level: str = "DEBUG"):
    """Decorator to log function calls."""
    def decorator(func):
        nonlocal logger
        if logger is None:
            logger = logging.getLogger(func.__module__)
        
        def wrapper(*args, **kwargs):
            func_name = f"{func.__module__}.{func.__name__}"
            getattr(logger, level.lower())(
                f"Calling {func_name} with args={args}, kwargs={kwargs}"
            )
            
            try:
                result = func(*args, **kwargs)
                getattr(logger, level.lower())(
                    f"Completed {func_name} successfully"
                )
                return result
            except Exception as e:
                logger.error(
                    f"Error in {func_name}: {e}",
                    exc_info=True
                )
                raise
        
        return wrapper
    return decorator


def log_execution_time(logger: Optional[logging.Logger] = None, level: str = "INFO"):
    """Decorator to log function execution time."""
    def decorator(func):
        nonlocal logger
        if logger is None:
            logger = logging.getLogger(func.__module__)
        
        def wrapper(*args, **kwargs):
            import time
            start_time = time.time()
            
            try:
                result = func(*args, **kwargs)
                execution_time = time.time() - start_time
                
                getattr(logger, level.lower())(
                    f"Function {func.__name__} executed in {execution_time:.4f}s",
                    extra={
                        'function': func.__name__,
                        'execution_time': execution_time,
                        'module': func.__module__
                    }
                )
                return result
            except Exception as e:
                execution_time = time.time() - start_time
                logger.error(
                    f"Function {func.__name__} failed after {execution_time:.4f}s: {e}",
                    extra={
                        'function': func.__name__,
                        'execution_time': execution_time,
                        'module': func.__module__,
                        'error': str(e)
                    },
                    exc_info=True
                )
                raise
        
        return wrapper
    return decorator


# Pre-configured logger instances
def get_training_logger() -> logging.Logger:
    """Get logger for training operations."""
    return logging.getLogger('pg_neo_graph_rl.training')


def get_federated_logger() -> logging.Logger:
    """Get logger for federated learning operations."""
    return logging.getLogger('pg_neo_graph_rl.federated')


def get_graph_logger() -> logging.Logger:
    """Get logger for graph processing operations."""
    return logging.getLogger('pg_neo_graph_rl.graph')


def get_monitoring_logger() -> logging.Logger:
    """Get logger for monitoring operations."""
    return logging.getLogger('pg_neo_graph_rl.monitoring')


# Health check logging
def log_health_status(logger: logging.Logger, status: Dict[str, Any]):
    """Log health status information."""
    logger.info(
        "Health check completed",
        extra={
            'health_status': status['status'],
            'uptime_seconds': status.get('uptime_seconds'),
            'metrics_collected': status.get('metrics_collected'),
            'jax_devices': status.get('jax_devices'),
            'jax_platform': status.get('jax_platform')
        }
    )


# Initialize logging on module import if environment variables are set
if os.getenv('INIT_LOGGING_ON_IMPORT', 'false').lower() == 'true':
    setup_logging_from_env()