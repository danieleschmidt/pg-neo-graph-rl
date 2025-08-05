"""
Security utilities for pg-neo-graph-rl.
"""
import hashlib
import hmac
import secrets
import time
from typing import Any, Dict, Optional, Union
import jax.numpy as jnp

from .exceptions import SecurityError, ValidationError


class SecureValidator:
    """Validates inputs for security issues."""
    
    @staticmethod
    def validate_file_path(path: str) -> str:
        """Validate file path for security issues."""
        if not isinstance(path, str):
            raise ValidationError("Path must be string")
        
        # Check for path traversal
        if ".." in path or path.startswith("/"):
            raise SecurityError("Path traversal detected")
        
        # Check for suspicious patterns
        suspicious = ["eval", "exec", "import", "subprocess"]
        if any(s in path.lower() for s in suspicious):
            raise SecurityError("Suspicious pattern in path")
        
        return path
    
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