"""
Caching mechanisms for performance optimization.
"""
import time
import threading
from typing import Any, Dict, Optional, Tuple, Callable
from dataclasses import dataclass
import jax.numpy as jnp
from functools import lru_cache, wraps

from ..utils.logging import get_logger


@dataclass
class CacheEntry:
    """Cache entry with metadata."""
    value: Any
    timestamp: float
    access_count: int = 0
    size_bytes: int = 0


class GraphCache:
    """Cache for graph computations and embeddings."""
    
    def __init__(self, max_size: int = 1000, ttl_seconds: float = 300.0):
        self.max_size = max_size
        self.ttl_seconds = ttl_seconds
        self.cache: Dict[str, CacheEntry] = {}
        self.access_order = []
        self._lock = threading.RLock()
        self.logger = get_logger("pg_neo_graph_rl.cache")
        
        # Performance tracking
        self.hits = 0
        self.misses = 0
    
    def _make_key(self, *args, **kwargs) -> str:
        """Create cache key from arguments."""
        key_parts = []
        
        for arg in args:
            if isinstance(arg, jnp.ndarray):
                # Use shape and hash of first few elements for arrays
                key_parts.append(f"array_{arg.shape}_{hash(tuple(arg.flatten()[:10]))}")
            else:
                key_parts.append(str(arg))
        
        for k, v in sorted(kwargs.items()):
            if isinstance(v, jnp.ndarray):
                key_parts.append(f"{k}_array_{v.shape}_{hash(tuple(v.flatten()[:10]))}")
            else:
                key_parts.append(f"{k}_{v}")
        
        return "|".join(key_parts)
    
    def get(self, key: str) -> Optional[Any]:
        """Get value from cache."""
        with self._lock:
            if key not in self.cache:
                self.misses += 1
                return None
            
            entry = self.cache[key]
            
            # Check TTL
            if time.time() - entry.timestamp > self.ttl_seconds:
                del self.cache[key]
                self.access_order.remove(key)
                self.misses += 1
                return None
            
            # Update access info
            entry.access_count += 1
            self.access_order.remove(key)
            self.access_order.append(key)
            
            self.hits += 1
            return entry.value
    
    def put(self, key: str, value: Any) -> None:
        """Put value in cache."""
        with self._lock:
            # Estimate size
            size_bytes = self._estimate_size(value)
            
            # Remove old entry if exists
            if key in self.cache:
                self.access_order.remove(key)
            
            # Evict if necessary
            while len(self.cache) >= self.max_size:
                self._evict_lru()
            
            # Add new entry
            entry = CacheEntry(
                value=value,
                timestamp=time.time(),
                size_bytes=size_bytes
            )
            
            self.cache[key] = entry
            self.access_order.append(key)
    
    def _evict_lru(self) -> None:
        """Evict least recently used entry."""
        if not self.access_order:
            return
        
        lru_key = self.access_order.pop(0)
        del self.cache[lru_key]
    
    def _estimate_size(self, value: Any) -> int:
        """Estimate memory size of value."""
        if isinstance(value, jnp.ndarray):
            return value.nbytes
        elif isinstance(value, (list, tuple)):
            return sum(self._estimate_size(item) for item in value)
        elif isinstance(value, dict):
            return sum(self._estimate_size(k) + self._estimate_size(v) for k, v in value.items())
        else:
            return 64  # Rough estimate for other types
    
    def cached_computation(self, func: Callable) -> Callable:
        """Decorator for caching function results."""
        @wraps(func)
        def wrapper(*args, **kwargs):
            key = f"{func.__name__}_{self._make_key(*args, **kwargs)}"
            
            # Try cache first
            result = self.get(key)
            if result is not None:
                return result
            
            # Compute and cache
            result = func(*args, **kwargs)
            self.put(key, result)
            return result
        
        return wrapper
    
    def clear(self) -> None:
        """Clear all cache entries."""
        with self._lock:
            self.cache.clear()
            self.access_order.clear()
            self.hits = 0
            self.misses = 0
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        with self._lock:
            total_requests = self.hits + self.misses
            hit_rate = self.hits / total_requests if total_requests > 0 else 0
            
            total_size = sum(entry.size_bytes for entry in self.cache.values())
            
            return {
                "entries": len(self.cache),
                "max_size": self.max_size,
                "hits": self.hits,
                "misses": self.misses,
                "hit_rate": hit_rate,
                "total_size_bytes": total_size,
                "avg_entry_size": total_size / len(self.cache) if self.cache else 0
            }


class ParameterCache:
    """Cache for model parameters and gradients."""
    
    def __init__(self, max_memory_mb: int = 512):
        self.max_memory_bytes = max_memory_mb * 1024 * 1024
        self.parameter_cache: Dict[str, Dict[str, jnp.ndarray]] = {}
        self.gradient_cache: Dict[str, Dict[str, jnp.ndarray]] = {}
        self._lock = threading.RLock()
        self.logger = get_logger("pg_neo_graph_rl.param_cache")
    
    def cache_parameters(self, agent_id: int, parameters: Dict[str, jnp.ndarray]) -> None:
        """Cache agent parameters."""
        with self._lock:
            key = f"agent_{agent_id}"
            self.parameter_cache[key] = {k: v.copy() for k, v in parameters.items()}
            self._manage_memory()
    
    def get_parameters(self, agent_id: int) -> Optional[Dict[str, jnp.ndarray]]:
        """Get cached parameters."""
        with self._lock:
            key = f"agent_{agent_id}"
            return self.parameter_cache.get(key)
    
    def cache_gradients(self, agent_id: int, gradients: Dict[str, jnp.ndarray]) -> None:
        """Cache agent gradients."""
        with self._lock:
            key = f"agent_{agent_id}"
            self.gradient_cache[key] = {k: v.copy() for k, v in gradients.items()}
            self._manage_memory()
    
    def get_gradients(self, agent_id: int) -> Optional[Dict[str, jnp.ndarray]]:
        """Get cached gradients."""
        with self._lock:
            key = f"agent_{agent_id}"
            return self.gradient_cache.get(key)
    
    def _manage_memory(self) -> None:
        """Manage cache memory usage."""
        current_memory = self._calculate_memory_usage()
        
        while current_memory > self.max_memory_bytes:
            # Remove oldest entries
            if self.parameter_cache:
                oldest_key = next(iter(self.parameter_cache))
                del self.parameter_cache[oldest_key]
            elif self.gradient_cache:
                oldest_key = next(iter(self.gradient_cache))
                del self.gradient_cache[oldest_key]
            else:
                break
            
            current_memory = self._calculate_memory_usage()
    
    def _calculate_memory_usage(self) -> int:
        """Calculate total memory usage."""
        total = 0
        
        for params in self.parameter_cache.values():
            for param in params.values():
                total += param.nbytes
        
        for grads in self.gradient_cache.values():
            for grad in grads.values():
                total += grad.nbytes
        
        return total


class AdaptiveCache:
    """Cache that adapts based on access patterns."""
    
    def __init__(self, initial_size: int = 100):
        self.cache = GraphCache(max_size=initial_size)
        self.access_pattern = {}
        self.adaptation_interval = 100  # Adapt every N requests
        self.request_count = 0
        self._lock = threading.RLock()
        self.logger = get_logger("pg_neo_graph_rl.adaptive_cache")
    
    def get(self, key: str) -> Optional[Any]:
        """Get with adaptive behavior."""
        with self._lock:
            self.request_count += 1
            
            # Track access pattern
            if key not in self.access_pattern:
                self.access_pattern[key] = {"frequency": 0, "last_access": time.time()}
            
            self.access_pattern[key]["frequency"] += 1
            self.access_pattern[key]["last_access"] = time.time()
            
            # Adapt cache if needed
            if self.request_count % self.adaptation_interval == 0:
                self._adapt_cache_size()
            
            return self.cache.get(key)
    
    def put(self, key: str, value: Any) -> None:
        """Put with adaptive behavior."""
        with self._lock:
            self.cache.put(key, value)
    
    def _adapt_cache_size(self) -> None:
        """Adapt cache size based on access patterns."""
        stats = self.cache.get_stats()
        hit_rate = stats["hit_rate"]
        
        current_size = self.cache.max_size
        
        # Increase size if hit rate is high and we're using most of the cache
        if hit_rate > 0.8 and stats["entries"] > current_size * 0.9:
            new_size = min(current_size * 2, 10000)  # Cap at 10k entries
            self.cache.max_size = new_size
            self.logger.info(f"Increased cache size to {new_size} (hit rate: {hit_rate:.2f})")
        
        # Decrease size if hit rate is low
        elif hit_rate < 0.3 and current_size > 50:
            new_size = max(current_size // 2, 50)  # Minimum 50 entries
            self.cache.max_size = new_size
            self.logger.info(f"Decreased cache size to {new_size} (hit rate: {hit_rate:.2f})")
    
    def get_adaptation_stats(self) -> Dict[str, Any]:
        """Get adaptation statistics."""
        with self._lock:
            cache_stats = self.cache.get_stats()
            
            # Analyze access patterns
            frequent_keys = sorted(
                self.access_pattern.items(),
                key=lambda x: x[1]["frequency"],
                reverse=True
            )[:10]
            
            return {
                "cache_stats": cache_stats,
                "total_requests": self.request_count,
                "unique_keys": len(self.access_pattern),
                "most_frequent_keys": [(k, v["frequency"]) for k, v in frequent_keys],
                "adaptation_interval": self.adaptation_interval
            }