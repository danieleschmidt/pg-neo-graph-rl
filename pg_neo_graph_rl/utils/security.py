"""
Security utilities for pg-neo-graph-rl.
"""
import hashlib
import hmac
import secrets
import time
from typing import Any, Dict, Optional

import jax
import jax.numpy as jnp

from .exceptions import SecurityError, ValidationError


class SecurityValidator:
    """Validates inputs for security issues."""

    @staticmethod
    def validate_file_path(path: str) -> str:
        """Validate file path for security issues."""
        if not isinstance(path, str):
            raise ValidationError("Path must be string")

        # Check for path traversal
        if ".." in path:
            raise SecurityError("Path traversal detected")

        # Check for suspicious patterns
        suspicious = ["eval", "exec", "import", "subprocess"]
        if any(s in path.lower() for s in suspicious):
            raise SecurityError("Suspicious pattern in path")

        return path

    @staticmethod
    def validate_string_input(text: str, field_name: str = "input") -> None:
        """Validate string input for security risks."""
        if not isinstance(text, str):
            raise SecurityError(f"Input {field_name} must be string")

        if len(text) > 10000:
            raise SecurityError(f"Input {field_name} too long")

    @staticmethod
    def sanitize_config(config: Dict[str, Any]) -> Dict[str, Any]:
        """Sanitize configuration dictionary."""
        sanitized = {}

        for key, value in config.items():
            # Remove private keys
            if "password" in key.lower() or "secret" in key.lower():
                continue

            # Validate numeric ranges
            if isinstance(value, (int, float)):
                if abs(value) > 1e10:
                    raise SecurityError(f"Value too large: {key}")

            # Validate string inputs
            if isinstance(value, str):
                SecurityValidator.validate_string_input(value, key)

            sanitized[key] = value

        return sanitized


def add_differential_privacy_noise(params: Dict[str, jnp.ndarray],
                                 epsilon: float = 1.0,
                                 delta: float = 1e-5) -> Dict[str, jnp.ndarray]:
    """Add differential privacy noise to parameters."""
    if epsilon <= 0:
        raise ValidationError("Epsilon must be positive")

    noisy_params = {}
    sensitivity = 1.0 / epsilon  # Simplified sensitivity

    for key, param in params.items():
        noise = jax.random.laplace(
            jax.random.PRNGKey(int(time.time())),
            param.shape
        ) * sensitivity
        noisy_params[key] = param + noise

    return noisy_params


def validate_input_data(data: Any, max_size: int = 1024 * 1024) -> None:
    """Validate input data for security issues."""
    import sys

    # Check data size
    data_size = sys.getsizeof(data)
    if data_size > max_size:
        raise SecurityError(f"Input data too large: {data_size} bytes")

    # Check for JAX arrays
    if isinstance(data, jnp.ndarray):
        # Check for NaN/Inf
        if jnp.any(jnp.isnan(data)) or jnp.any(jnp.isinf(data)):
            raise SecurityError("Invalid values (NaN/Inf) detected in input")

        # Check for extremely large values
        if jnp.any(jnp.abs(data) > 1e10):
            raise SecurityError("Extremely large values detected in input")


def check_gradient_norm(gradients: Dict[str, jnp.ndarray],
                       max_norm: float = 10.0) -> float:
    """Check gradient norm for exploding gradients."""
    total_norm = 0.0

    for key, grad in gradients.items():
        if grad is not None:
            grad_norm = jnp.linalg.norm(grad.flatten())
            total_norm += grad_norm ** 2

    total_norm = jnp.sqrt(total_norm)

    if total_norm > max_norm:
        raise SecurityError(f"Gradient norm too large: {total_norm:.4f} > {max_norm}")

    return float(total_norm)


def secure_random_key(seed: Optional[int] = None) -> jnp.ndarray:
    """Generate cryptographically secure random key for JAX."""
    if seed is None:
        seed = secrets.randbits(32)

    # Use system entropy to enhance randomness
    entropy = secrets.token_bytes(4)
    enhanced_seed = seed ^ int.from_bytes(entropy, 'big')

    return jax.random.PRNGKey(enhanced_seed)


def compute_data_hash(data: jnp.ndarray) -> str:
    """Compute SHA-256 hash of data for integrity checking."""
    data_bytes = data.tobytes()
    return hashlib.sha256(data_bytes).hexdigest()


def verify_data_integrity(data: jnp.ndarray, expected_hash: str) -> bool:
    """Verify data integrity using hash."""
    actual_hash = compute_data_hash(data)
    return hmac.compare_digest(actual_hash, expected_hash)


class SecurityAudit:
    """Security audit logger for tracking security events."""

    def __init__(self):
        from .logging import get_logger
        self.logger = get_logger("pg_neo_graph_rl.security")
        self._events = []

    def log_security_event(self,
                          event_type: str,
                          description: str,
                          severity: str = "INFO",
                          metadata: Optional[Dict[str, Any]] = None) -> None:
        """Log a security event."""
        event = {
            "timestamp": time.time(),
            "event_type": event_type,
            "description": description,
            "severity": severity,
            "metadata": metadata or {}
        }

        self._events.append(event)

        # Log based on severity
        if severity == "CRITICAL":
            self.logger.critical(f"Security event: {description}", extra=event)
        elif severity == "HIGH":
            self.logger.error(f"Security event: {description}", extra=event)
        elif severity == "MEDIUM":
            self.logger.warning(f"Security event: {description}", extra=event)
        else:
            self.logger.info(f"Security event: {description}", extra=event)

    def get_security_summary(self) -> Dict[str, Any]:
        """Get summary of security events."""
        event_counts = {}
        for event in self._events:
            event_type = event["event_type"]
            severity = event["severity"]
            key = f"{event_type}_{severity}"
            event_counts[key] = event_counts.get(key, 0) + 1

        return {
            "total_events": len(self._events),
            "event_counts": event_counts,
            "recent_events": self._events[-10:] if self._events else []
        }
