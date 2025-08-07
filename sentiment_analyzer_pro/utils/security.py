"""Security utilities for sentiment analysis."""

import re
import hashlib
import hmac
import secrets
import time
from typing import Dict, List, Optional, Any, Set
from dataclasses import dataclass
from collections import defaultdict, deque
import logging

from .exceptions import ValidationError, SecurityError


logger = logging.getLogger(__name__)


@dataclass
class SecurityEvent:
    """Security event data structure."""
    timestamp: float
    event_type: str
    severity: str
    source: str
    description: str
    details: Dict[str, Any]


class InputValidator:
    """
    Advanced input validation and sanitization for sentiment analysis.
    
    Features:
    - SQL injection detection
    - XSS prevention
    - Command injection detection
    - Content filtering
    - Rate limiting
    - Suspicious pattern detection
    """
    
    def __init__(
        self,
        max_length: int = 10000,
        min_length: int = 1,
        allowed_languages: Optional[Set[str]] = None,
        enable_content_filtering: bool = True,
        enable_injection_detection: bool = True
    ):
        """
        Initialize input validator.
        
        Args:
            max_length: Maximum allowed text length
            min_length: Minimum allowed text length
            allowed_languages: Set of allowed language codes (if None, all allowed)
            enable_content_filtering: Enable content filtering
            enable_injection_detection: Enable injection attack detection
        """
        self.max_length = max_length
        self.min_length = min_length
        self.allowed_languages = allowed_languages
        self.enable_content_filtering = enable_content_filtering
        self.enable_injection_detection = enable_injection_detection
        
        # Compile security patterns
        self._compile_security_patterns()
        
        # Content filtering patterns
        self.sensitive_patterns = [
            # Personal information
            r'\b\d{3}-\d{2}-\d{4}\b',  # SSN
            r'\b\d{4}[- ]?\d{4}[- ]?\d{4}[- ]?\d{4}\b',  # Credit card
            r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',  # Email
            r'\b\d{3}[.-]?\d{3}[.-]?\d{4}\b',  # Phone number
            
            # API keys and tokens (common patterns)
            r'sk-[a-zA-Z0-9]{48}',  # OpenAI API key
            r'xox[baprs]-[a-zA-Z0-9-]{10,72}',  # Slack tokens
            r'ghp_[a-zA-Z0-9]{36}',  # GitHub personal access token
            r'AKIA[0-9A-Z]{16}',  # AWS access key
        ]
        
        # Compile patterns
        self.compiled_sensitive_patterns = [
            re.compile(pattern, re.IGNORECASE) for pattern in self.sensitive_patterns
        ]
        
        logger.info("InputValidator initialized with security patterns")
    
    def _compile_security_patterns(self) -> None:
        """Compile regex patterns for security detection."""
        # SQL injection patterns
        self.sql_patterns = [
            r'(\bUNION\b.*\bSELECT\b)',
            r'(\bSELECT\b.*\bFROM\b)',
            r'(\bINSERT\b.*\bINTO\b)',
            r'(\bUPDATE\b.*\bSET\b)',
            r'(\bDELETE\b.*\bFROM\b)',
            r'(\bDROP\b.*\bTABLE\b)',
            r'(\bALTER\b.*\bTABLE\b)',
            r'(\bCREATE\b.*\bTABLE\b)',
            r'(--|\#|\/\*|\*\/)',
            r'(\bOR\b.*=.*)',
            r'(\bAND\b.*=.*)',
            r'(\'.*=.*\')',
            r'(\";.*--)',
            r'(\bEXEC\b|\bEXECUTE\b)',
        ]
        
        # XSS patterns
        self.xss_patterns = [
            r'<script.*?>.*?</script.*?>',
            r'<iframe.*?>.*?</iframe.*?>',
            r'<object.*?>.*?</object.*?>',
            r'<embed.*?>.*?</embed.*?>',
            r'<link.*?>',
            r'<meta.*?>',
            r'javascript:',
            r'vbscript:',
            r'onload.*=',
            r'onerror.*=',
            r'onclick.*=',
            r'onmouseover.*=',
        ]
        
        # Command injection patterns
        self.command_patterns = [
            r'[;&|`$]',
            r'\.\./',
            r'/etc/passwd',
            r'/etc/shadow',
            r'cmd\.exe',
            r'powershell',
            r'bash',
            r'/bin/',
            r'wget\s',
            r'curl\s',
            r'nc\s',
            r'netcat\s',
        ]
        
        # Compile all patterns
        self.compiled_sql_patterns = [
            re.compile(pattern, re.IGNORECASE | re.DOTALL) for pattern in self.sql_patterns
        ]
        self.compiled_xss_patterns = [
            re.compile(pattern, re.IGNORECASE | re.DOTALL) for pattern in self.xss_patterns
        ]
        self.compiled_command_patterns = [
            re.compile(pattern, re.IGNORECASE) for pattern in self.command_patterns
        ]
    
    def validate_input(self, text: str, source: str = "unknown") -> str:
        """
        Validate and sanitize input text.
        
        Args:
            text: Input text to validate
            source: Source of the input for logging
            
        Returns:
            Sanitized text
            
        Raises:
            ValidationError: If input is invalid or potentially malicious
        """
        if not isinstance(text, str):
            raise ValidationError(f"Input must be a string, got {type(text)}")
        
        # Length validation
        if len(text) < self.min_length:
            raise ValidationError(f"Text too short (minimum {self.min_length} characters)")
        
        if len(text) > self.max_length:
            logger.warning(f"Text truncated from {len(text)} to {self.max_length} characters")
            text = text[:self.max_length]
        
        # Security checks
        if self.enable_injection_detection:
            self._detect_injection_attacks(text, source)
        
        # Content filtering
        if self.enable_content_filtering:
            text = self._filter_sensitive_content(text, source)
        
        # Basic sanitization
        text = self._sanitize_text(text)
        
        return text
    
    def _detect_injection_attacks(self, text: str, source: str) -> None:
        """Detect potential injection attacks."""
        
        # SQL injection detection
        for pattern in self.compiled_sql_patterns:
            if pattern.search(text):
                logger.warning(f"Potential SQL injection detected from {source}: {pattern.pattern}")
                raise ValidationError("Input contains potentially malicious SQL patterns")
        
        # XSS detection
        for pattern in self.compiled_xss_patterns:
            if pattern.search(text):
                logger.warning(f"Potential XSS attack detected from {source}: {pattern.pattern}")
                raise ValidationError("Input contains potentially malicious script content")
        
        # Command injection detection
        for pattern in self.compiled_command_patterns:
            if pattern.search(text):
                logger.warning(f"Potential command injection detected from {source}: {pattern.pattern}")
                raise ValidationError("Input contains potentially malicious system commands")
    
    def _filter_sensitive_content(self, text: str, source: str) -> str:
        """Filter sensitive content from text."""
        original_text = text
        
        # Remove sensitive patterns
        for pattern in self.compiled_sensitive_patterns:
            matches = pattern.findall(text)
            if matches:
                logger.warning(f"Sensitive content filtered from {source}: {len(matches)} matches")
                text = pattern.sub('[FILTERED]', text)
        
        # Log if content was modified
        if text != original_text:
            logger.info(f"Content filtering applied to input from {source}")
        
        return text
    
    def _sanitize_text(self, text: str) -> str:
        """Basic text sanitization."""
        # Remove null bytes
        text = text.replace('\x00', '')
        
        # Normalize line endings
        text = text.replace('\r\n', '\n').replace('\r', '\n')
        
        # Limit consecutive whitespace
        text = re.sub(r'\s{10,}', ' ' * 10, text)
        
        # Remove control characters (except common ones like newline, tab)
        text = re.sub(r'[\x00-\x08\x0B-\x0C\x0E-\x1F\x7F]', '', text)
        
        return text.strip()


class RateLimiter:
    """
    Rate limiter to prevent abuse.
    
    Features:
    - Token bucket algorithm
    - Per-source rate limiting
    - Sliding window tracking
    - Automatic cleanup
    """
    
    def __init__(
        self,
        max_requests: int = 100,
        time_window: int = 60,
        cleanup_interval: int = 300
    ):
        """
        Initialize rate limiter.
        
        Args:
            max_requests: Maximum requests per time window
            time_window: Time window in seconds
            cleanup_interval: Cleanup interval for old entries
        """
        self.max_requests = max_requests
        self.time_window = time_window
        self.cleanup_interval = cleanup_interval
        
        self.requests = defaultdict(deque)
        self.last_cleanup = time.time()
        
        logger.info(f"RateLimiter initialized: {max_requests} requests per {time_window}s")
    
    def is_allowed(self, source: str) -> bool:
        """
        Check if request from source is allowed.
        
        Args:
            source: Source identifier (IP, user ID, etc.)
            
        Returns:
            True if request is allowed, False otherwise
        """
        now = time.time()
        
        # Cleanup old entries periodically
        if now - self.last_cleanup > self.cleanup_interval:
            self._cleanup_old_entries(now)
            self.last_cleanup = now
        
        # Get request history for this source
        source_requests = self.requests[source]
        
        # Remove requests outside the time window
        while source_requests and source_requests[0] <= now - self.time_window:
            source_requests.popleft()
        
        # Check if limit exceeded
        if len(source_requests) >= self.max_requests:
            logger.warning(f"Rate limit exceeded for source: {source}")
            return False
        
        # Add current request
        source_requests.append(now)
        return True
    
    def _cleanup_old_entries(self, now: float) -> None:
        """Clean up old request entries."""
        cutoff = now - self.time_window
        sources_to_remove = []
        
        for source, requests in self.requests.items():
            # Remove old requests
            while requests and requests[0] <= cutoff:
                requests.popleft()
            
            # Remove empty entries
            if not requests:
                sources_to_remove.append(source)
        
        for source in sources_to_remove:
            del self.requests[source]
        
        logger.debug(f"Cleaned up {len(sources_to_remove)} empty rate limit entries")
    
    def get_remaining_requests(self, source: str) -> int:
        """Get remaining requests for source."""
        now = time.time()
        source_requests = self.requests[source]
        
        # Remove old requests
        while source_requests and source_requests[0] <= now - self.time_window:
            source_requests.popleft()
        
        return max(0, self.max_requests - len(source_requests))


class SecurityManager:
    """
    Centralized security manager for sentiment analysis operations.
    
    Features:
    - Input validation
    - Rate limiting
    - Security event logging
    - Threat detection
    - Access control
    """
    
    def __init__(
        self,
        validator: Optional[InputValidator] = None,
        rate_limiter: Optional[RateLimiter] = None,
        enable_monitoring: bool = True
    ):
        """
        Initialize security manager.
        
        Args:
            validator: Input validator instance
            rate_limiter: Rate limiter instance
            enable_monitoring: Enable security monitoring
        """
        self.validator = validator or InputValidator()
        self.rate_limiter = rate_limiter or RateLimiter()
        self.enable_monitoring = enable_monitoring
        
        # Security event tracking
        self.security_events = deque(maxlen=1000)
        self.threat_scores = defaultdict(int)
        
        logger.info("SecurityManager initialized")
    
    def validate_request(
        self,
        text: str,
        source: str = "unknown",
        user_id: Optional[str] = None
    ) -> str:
        """
        Validate a sentiment analysis request.
        
        Args:
            text: Input text to validate
            source: Source of the request
            user_id: User identifier if available
            
        Returns:
            Validated and sanitized text
            
        Raises:
            ValidationError: If request is invalid
            SecurityError: If request is blocked for security reasons
        """
        # Rate limiting check
        if not self.rate_limiter.is_allowed(source):
            self._log_security_event(
                event_type="rate_limit_exceeded",
                severity="medium",
                source=source,
                description=f"Rate limit exceeded for source: {source}",
                details={"user_id": user_id, "text_length": len(text)}
            )
            raise SecurityError("Rate limit exceeded. Please try again later.")
        
        # Input validation
        try:
            validated_text = self.validator.validate_input(text, source)
        except ValidationError as e:
            self._log_security_event(
                event_type="validation_failed",
                severity="high",
                source=source,
                description=f"Input validation failed: {str(e)}",
                details={
                    "user_id": user_id,
                    "text_length": len(text),
                    "error": str(e)
                }
            )
            # Increase threat score
            self.threat_scores[source] += 5
            raise
        
        return validated_text
    
    def check_threat_level(self, source: str) -> str:
        """
        Check threat level for a source.
        
        Args:
            source: Source to check
            
        Returns:
            Threat level ("low", "medium", "high")
        """
        score = self.threat_scores[source]
        
        if score >= 20:
            return "high"
        elif score >= 10:
            return "medium"
        else:
            return "low"
    
    def _log_security_event(
        self,
        event_type: str,
        severity: str,
        source: str,
        description: str,
        details: Dict[str, Any]
    ) -> None:
        """Log a security event."""
        if not self.enable_monitoring:
            return
        
        event = SecurityEvent(
            timestamp=time.time(),
            event_type=event_type,
            severity=severity,
            source=source,
            description=description,
            details=details
        )
        
        self.security_events.append(event)
        
        # Log to standard logger
        logger.warning(f"Security event: {event_type} from {source} - {description}")
    
    def get_security_summary(self) -> Dict[str, Any]:
        """Get security summary."""
        now = time.time()
        recent_events = [
            event for event in self.security_events 
            if now - event.timestamp <= 3600  # Last hour
        ]
        
        event_counts = defaultdict(int)
        severity_counts = defaultdict(int)
        
        for event in recent_events:
            event_counts[event.event_type] += 1
            severity_counts[event.severity] += 1
        
        return {
            "total_events_last_hour": len(recent_events),
            "event_types": dict(event_counts),
            "severity_breakdown": dict(severity_counts),
            "high_threat_sources": {
                source: score for source, score in self.threat_scores.items()
                if score >= 10
            },
            "rate_limiter_stats": {
                "tracked_sources": len(self.rate_limiter.requests),
                "max_requests_per_window": self.rate_limiter.max_requests,
                "time_window": self.rate_limiter.time_window
            }
        }