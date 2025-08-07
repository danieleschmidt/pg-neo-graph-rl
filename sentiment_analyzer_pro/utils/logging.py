"""Advanced logging configuration for sentiment analysis."""

import logging
import logging.handlers
import json
import time
import sys
from typing import Dict, Any, Optional
from datetime import datetime
import os


class JSONFormatter(logging.Formatter):
    """Custom JSON formatter for structured logging."""
    
    def format(self, record: logging.LogRecord) -> str:
        """Format log record as JSON."""
        log_entry = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno,
            "process_id": os.getpid(),
            "thread_id": record.thread
        }
        
        # Add exception information if present
        if record.exc_info:
            log_entry["exception"] = self.formatException(record.exc_info)
        
        # Add custom fields if present
        if hasattr(record, 'custom_fields'):
            log_entry.update(record.custom_fields)
            
        return json.dumps(log_entry)


class PerformanceFilter(logging.Filter):
    """Filter to add performance metrics to log records."""
    
    def __init__(self):
        super().__init__()
        self.start_time = time.time()
    
    def filter(self, record: logging.LogRecord) -> bool:
        """Add performance metrics to record."""
        record.uptime = time.time() - self.start_time
        return True


class SentimentAnalysisLogger:
    """
    Advanced logger for sentiment analysis operations.
    
    Features:
    - Structured JSON logging
    - Performance tracking
    - Security event logging
    - Error aggregation
    - Multiple output handlers
    """
    
    def __init__(
        self,
        name: str = "sentiment_analyzer_pro",
        level: str = "INFO",
        log_file: Optional[str] = None,
        max_file_size: int = 10 * 1024 * 1024,  # 10MB
        backup_count: int = 5,
        enable_console: bool = True,
        enable_json: bool = True
    ):
        """
        Initialize logger.
        
        Args:
            name: Logger name
            level: Logging level
            log_file: Log file path (if None, no file logging)
            max_file_size: Maximum log file size in bytes
            backup_count: Number of backup files to keep
            enable_console: Enable console logging
            enable_json: Use JSON formatting
        """
        self.logger = logging.getLogger(name)
        self.logger.setLevel(getattr(logging, level.upper()))
        
        # Clear existing handlers
        self.logger.handlers.clear()
        
        # Add performance filter
        perf_filter = PerformanceFilter()
        
        # Console handler
        if enable_console:
            console_handler = logging.StreamHandler(sys.stdout)
            if enable_json:
                console_handler.setFormatter(JSONFormatter())
            else:
                console_handler.setFormatter(
                    logging.Formatter(
                        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
                    )
                )
            console_handler.addFilter(perf_filter)
            self.logger.addHandler(console_handler)
        
        # File handler
        if log_file:
            file_handler = logging.handlers.RotatingFileHandler(
                log_file,
                maxBytes=max_file_size,
                backupCount=backup_count
            )
            if enable_json:
                file_handler.setFormatter(JSONFormatter())
            else:
                file_handler.setFormatter(
                    logging.Formatter(
                        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
                    )
                )
            file_handler.addFilter(perf_filter)
            self.logger.addHandler(file_handler)
        
        # Error aggregation
        self.error_counts = {}
        self.security_events = []
        
    def log_analysis(
        self,
        text_length: int,
        confidence: float,
        label: str,
        processing_time: float,
        model_name: str,
        success: bool = True,
        error: Optional[str] = None
    ) -> None:
        """Log sentiment analysis operation."""
        custom_fields = {
            "operation": "sentiment_analysis",
            "text_length": text_length,
            "confidence": confidence,
            "predicted_label": label,
            "processing_time": processing_time,
            "model_name": model_name,
            "success": success
        }
        
        if error:
            custom_fields["error"] = error
        
        record = logging.LogRecord(
            name=self.logger.name,
            level=logging.INFO if success else logging.ERROR,
            pathname="",
            lineno=0,
            msg=f"Analysis {'completed' if success else 'failed'} - {label} ({confidence:.3f})",
            args=(),
            exc_info=None
        )
        record.custom_fields = custom_fields
        
        self.logger.handle(record)
    
    def log_batch_analysis(
        self,
        batch_size: int,
        total_time: float,
        success_count: int,
        error_count: int,
        model_name: str
    ) -> None:
        """Log batch analysis operation."""
        custom_fields = {
            "operation": "batch_analysis",
            "batch_size": batch_size,
            "total_processing_time": total_time,
            "success_count": success_count,
            "error_count": error_count,
            "model_name": model_name,
            "avg_time_per_text": total_time / batch_size if batch_size > 0 else 0
        }
        
        record = logging.LogRecord(
            name=self.logger.name,
            level=logging.INFO,
            pathname="",
            lineno=0,
            msg=f"Batch analysis completed - {success_count}/{batch_size} successful",
            args=(),
            exc_info=None
        )
        record.custom_fields = custom_fields
        
        self.logger.handle(record)
    
    def log_security_event(
        self,
        event_type: str,
        description: str,
        severity: str = "medium",
        details: Optional[Dict[str, Any]] = None
    ) -> None:
        """Log security-related events."""
        security_event = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "event_type": event_type,
            "description": description,
            "severity": severity,
            "details": details or {}
        }
        
        self.security_events.append(security_event)
        
        # Keep only last 1000 events
        if len(self.security_events) > 1000:
            self.security_events = self.security_events[-1000:]
        
        custom_fields = {
            "operation": "security_event",
            "event_type": event_type,
            "severity": severity,
            "details": details or {}
        }
        
        record = logging.LogRecord(
            name=self.logger.name,
            level=logging.WARNING if severity in ["medium", "high"] else logging.INFO,
            pathname="",
            lineno=0,
            msg=f"Security event: {event_type} - {description}",
            args=(),
            exc_info=None
        )
        record.custom_fields = custom_fields
        
        self.logger.handle(record)
    
    def log_performance_metrics(
        self,
        operation: str,
        duration: float,
        memory_usage: Optional[Dict[str, float]] = None,
        gpu_usage: Optional[Dict[str, float]] = None
    ) -> None:
        """Log performance metrics."""
        custom_fields = {
            "operation": "performance_metrics",
            "metric_type": operation,
            "duration": duration,
            "memory_usage": memory_usage or {},
            "gpu_usage": gpu_usage or {}
        }
        
        record = logging.LogRecord(
            name=self.logger.name,
            level=logging.INFO,
            pathname="",
            lineno=0,
            msg=f"Performance metric - {operation}: {duration:.3f}s",
            args=(),
            exc_info=None
        )
        record.custom_fields = custom_fields
        
        self.logger.handle(record)
    
    def log_error(
        self,
        error_type: str,
        message: str,
        details: Optional[Dict[str, Any]] = None,
        exc_info: bool = False
    ) -> None:
        """Log error with aggregation."""
        # Count errors
        if error_type not in self.error_counts:
            self.error_counts[error_type] = 0
        self.error_counts[error_type] += 1
        
        custom_fields = {
            "operation": "error",
            "error_type": error_type,
            "error_count": self.error_counts[error_type],
            "details": details or {}
        }
        
        record = logging.LogRecord(
            name=self.logger.name,
            level=logging.ERROR,
            pathname="",
            lineno=0,
            msg=f"Error: {error_type} - {message}",
            args=(),
            exc_info=exc_info
        )
        record.custom_fields = custom_fields
        
        self.logger.handle(record)
    
    def get_error_summary(self) -> Dict[str, int]:
        """Get summary of error counts."""
        return self.error_counts.copy()
    
    def get_security_events(self, limit: int = 100) -> list:
        """Get recent security events."""
        return self.security_events[-limit:]
    
    def info(self, message: str, **kwargs) -> None:
        """Log info message."""
        self.logger.info(message, extra={'custom_fields': kwargs} if kwargs else None)
    
    def warning(self, message: str, **kwargs) -> None:
        """Log warning message."""
        self.logger.warning(message, extra={'custom_fields': kwargs} if kwargs else None)
    
    def error(self, message: str, **kwargs) -> None:
        """Log error message."""
        self.logger.error(message, extra={'custom_fields': kwargs} if kwargs else None)
    
    def debug(self, message: str, **kwargs) -> None:
        """Log debug message."""
        self.logger.debug(message, extra={'custom_fields': kwargs} if kwargs else None)


# Global logger instance
logger = SentimentAnalysisLogger()


def setup_logging(
    level: str = "INFO",
    log_file: Optional[str] = None,
    enable_json: bool = True
) -> SentimentAnalysisLogger:
    """Setup global logging configuration."""
    global logger
    logger = SentimentAnalysisLogger(
        level=level,
        log_file=log_file,
        enable_json=enable_json
    )
    return logger