"""
Advanced caching system for federated graph RL.
"""
import hashlib
import pickle
import threading
import time
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

import jax.numpy as jnp

from ..core.types import GraphState
from ..utils.logging import get_logger


@dataclass
class CacheEntry:
    """Cache entry with metadata."""
    key: str
    value: Any
    timestamp: float
    access_count: int = 0
    last_accessed: float = field(default_factory=time.time)
    size_bytes: int = 0
    ttl: Optional[float] = None  # Time to live in seconds

    def is_expired(self) -> bool:
        """Check if cache entry is expired."""
        if self.ttl is None:
            return False
        return time.time() - self.timestamp > self.ttl

    def update_access(self) -> None:
        """Update access statistics."""
        self.access_count += 1
        self.last_accessed = time.time()


class LRUCache:
    """LRU Cache with size limits and TTL support."""

    def __init__(self, max_size: int = 1000, max_memory_mb: int = 100):
        self.max_size = max_size
        self.max_memory_bytes = max_memory_mb * 1024 * 1024
        self.cache: OrderedDict[str, CacheEntry] = OrderedDict()
        self.current_memory = 0
        self.lock = threading.RLock()
        self.logger = get_logger("pg_neo_graph_rl.lru_cache")

        # Statistics
        self.hits = 0
        self.misses = 0
        self.evictions = 0

    def get(self, key: str) -> Optional[Any]:
        """Get item from cache."""
        with self.lock:
            if key not in self.cache:
                self.misses += 1
                return None

            entry = self.cache[key]

            # Check if expired
            if entry.is_expired():
                del self.cache[key]
                self.current_memory -= entry.size_bytes
                self.misses += 1
                return None

            # Move to end (most recently used)
            self.cache.move_to_end(key)
            entry.update_access()
            self.hits += 1

            return entry.value

    def put(self, key: str, value: Any, ttl: Optional[float] = None) -> None:
        """Put item in cache."""
        with self.lock:
            # Calculate size
            try:
                size_bytes = len(pickle.dumps(value))
            except:
                size_bytes = 1024  # Default estimate

            # Remove existing entry if present
            if key in self.cache:
                old_entry = self.cache[key]
                self.current_memory -= old_entry.size_bytes
                del self.cache[key]

            # Create new entry
            entry = CacheEntry(
                key=key,
                value=value,
                timestamp=time.time(),
                size_bytes=size_bytes,
                ttl=ttl
            )

            # Check memory constraints
            while (self.current_memory + size_bytes > self.max_memory_bytes or
                   len(self.cache) >= self.max_size) and self.cache:
                self._evict_least_recent()

            # Add new entry
            self.cache[key] = entry
            self.current_memory += size_bytes

    def _evict_least_recent(self) -> None:
        """Evict least recently used item."""
        if not self.cache:
            return

        key, entry = self.cache.popitem(last=False)
        self.current_memory -= entry.size_bytes
        self.evictions += 1
        self.logger.debug(f"Evicted cache entry: {key}")

    def clear(self) -> None:
        """Clear all cache entries."""
        with self.lock:
            self.cache.clear()
            self.current_memory = 0

    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        with self.lock:
            total_requests = self.hits + self.misses
            hit_rate = self.hits / total_requests if total_requests > 0 else 0.0

            return {
                "hits": self.hits,
                "misses": self.misses,
                "hit_rate": hit_rate,
                "evictions": self.evictions,
                "current_size": len(self.cache),
                "max_size": self.max_size,
                "current_memory_mb": self.current_memory / (1024 * 1024),
                "max_memory_mb": self.max_memory_bytes / (1024 * 1024)
            }


class SmartCache:
    """Intelligent cache with adaptive policies."""

    def __init__(self,
                 max_size: int = 1000,
                 max_memory_mb: int = 100,
                 enable_prefetch: bool = True):
        self.lru_cache = LRUCache(max_size, max_memory_mb)
        self.enable_prefetch = enable_prefetch
        self.lock = threading.RLock()
        self.logger = get_logger("pg_neo_graph_rl.smart_cache")

        # Access patterns
        self.access_patterns: Dict[str, List[float]] = {}
        self.prefetch_candidates: Dict[str, float] = {}

        # Async prefetching
        self.prefetch_executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="cache_prefetch")
        self.prefetch_functions: Dict[str, Callable] = {}

        # Cache warming
        self.warm_functions: List[Callable] = []

    def get(self, key: str, compute_func: Optional[Callable] = None) -> Any:
        """Get item from cache with optional compute function."""
        # Try cache first
        value = self.lru_cache.get(key)

        if value is not None:
            self._record_access(key)
            return value

        # Compute if function provided
        if compute_func is not None:
            start_time = time.time()
            value = compute_func()
            compute_time = time.time() - start_time

            # Cache if computation was expensive
            if compute_time > 0.01:  # Cache if > 10ms
                self.put(key, value)

            self._record_access(key)
            return value

        return None

    def put(self, key: str, value: Any, ttl: Optional[float] = None) -> None:
        """Put item in cache."""
        self.lru_cache.put(key, value, ttl)
        self._record_access(key)

    def _record_access(self, key: str) -> None:
        """Record access pattern for intelligent prefetching."""
        if not self.enable_prefetch:
            return

        with self.lock:
            current_time = time.time()

            if key not in self.access_patterns:
                self.access_patterns[key] = []

            self.access_patterns[key].append(current_time)

            # Keep only recent accesses
            cutoff_time = current_time - 3600  # Last hour
            self.access_patterns[key] = [
                t for t in self.access_patterns[key] if t > cutoff_time
            ]

            # Update prefetch candidates
            self._update_prefetch_candidates(key)

    def _update_prefetch_candidates(self, key: str) -> None:
        """Update prefetch candidates based on access patterns."""
        accesses = self.access_patterns.get(key, [])

        if len(accesses) < 3:
            return

        # Calculate access frequency
        recent_accesses = [t for t in accesses if t > time.time() - 1800]  # Last 30 minutes
        frequency = len(recent_accesses) / 1800  # Accesses per second

        # Calculate access regularity (low variance in intervals = more regular)
        if len(accesses) >= 3:
            intervals = [accesses[i] - accesses[i-1] for i in range(1, len(accesses))]
            if intervals:
                avg_interval = sum(intervals) / len(intervals)
                variance = sum((x - avg_interval) ** 2 for x in intervals) / len(intervals)
                regularity_score = 1.0 / (1.0 + variance)  # Higher score = more regular
            else:
                regularity_score = 0.0
        else:
            regularity_score = 0.0

        # Combine frequency and regularity
        prefetch_score = frequency * regularity_score

        if prefetch_score > 0.01:  # Threshold for prefetch candidates
            self.prefetch_candidates[key] = prefetch_score

    def register_prefetch_function(self, key_pattern: str, func: Callable) -> None:
        """Register function for prefetching specific key patterns."""
        self.prefetch_functions[key_pattern] = func
        self.logger.info(f"Registered prefetch function for pattern: {key_pattern}")

    def trigger_prefetch(self) -> None:
        """Trigger prefetching for high-score candidates."""
        if not self.enable_prefetch:
            return

        with self.lock:
            candidates = list(self.prefetch_candidates.items())

        # Sort by score and prefetch top candidates
        candidates.sort(key=lambda x: x[1], reverse=True)

        for key, score in candidates[:5]:  # Top 5 candidates
            if key not in [entry.key for entry in self.lru_cache.cache.values()]:
                self._async_prefetch(key)

    def _async_prefetch(self, key: str) -> None:
        """Asynchronously prefetch a key."""
        # Find matching prefetch function
        prefetch_func = None
        for pattern, func in self.prefetch_functions.items():
            if pattern in key or key.startswith(pattern):
                prefetch_func = func
                break

        if prefetch_func is None:
            return

        def prefetch_task():
            try:
                value = prefetch_func(key)
                if value is not None:
                    self.put(key, value, ttl=300.0)  # 5 minute TTL for prefetched items
                    self.logger.debug(f"Prefetched key: {key}")
            except Exception as e:
                self.logger.error(f"Prefetch failed for key {key}: {e}")

        self.prefetch_executor.submit(prefetch_task)

    def add_warm_function(self, func: Callable) -> None:
        """Add function for cache warming."""
        self.warm_functions.append(func)

    def warm_cache(self) -> None:
        """Warm cache with commonly used data."""
        self.logger.info("Starting cache warming")

        for warm_func in self.warm_functions:
            try:
                warm_items = warm_func()
                for key, value in warm_items.items():
                    self.put(key, value, ttl=1800.0)  # 30 minute TTL for warmed items
            except Exception as e:
                self.logger.error(f"Cache warming failed: {e}")

        self.logger.info(f"Cache warming completed. Cache size: {len(self.lru_cache.cache)}")

    def get_analytics(self) -> Dict[str, Any]:
        """Get cache analytics."""
        with self.lock:
            base_stats = self.lru_cache.get_stats()

            # Access pattern analysis
            pattern_analysis = {}
            for key, accesses in self.access_patterns.items():
                if len(accesses) >= 2:
                    frequency = len(accesses) / 3600  # Per hour
                    intervals = [accesses[i] - accesses[i-1] for i in range(1, len(accesses))]
                    avg_interval = sum(intervals) / len(intervals) if intervals else 0

                    pattern_analysis[key] = {
                        "access_count": len(accesses),
                        "frequency_per_hour": frequency,
                        "avg_interval_seconds": avg_interval
                    }

            return {
                **base_stats,
                "prefetch_enabled": self.enable_prefetch,
                "prefetch_candidates_count": len(self.prefetch_candidates),
                "top_prefetch_candidates": dict(sorted(
                    self.prefetch_candidates.items(),
                    key=lambda x: x[1],
                    reverse=True
                )[:10]),
                "access_patterns_tracked": len(self.access_patterns),
                "pattern_analysis": pattern_analysis
            }

    def clear(self) -> None:
        """Clear cache and patterns."""
        self.lru_cache.clear()
        with self.lock:
            self.access_patterns.clear()
            self.prefetch_candidates.clear()


class GraphStateCache(SmartCache):
    """Specialized cache for graph states."""

    def __init__(self, max_size: int = 500, max_memory_mb: int = 50):
        super().__init__(max_size, max_memory_mb)
        self.logger = get_logger("pg_neo_graph_rl.graph_state_cache")

    def get_graph_state_key(self, graph_state: GraphState, agent_id: int) -> str:
        """Generate cache key for graph state."""
        # Create hash of graph structure and features
        nodes_hash = hashlib.md5(graph_state.nodes.tobytes()).hexdigest()[:8]
        edges_hash = hashlib.md5(graph_state.edges.tobytes()).hexdigest()[:8]

        return f"graph_state_{agent_id}_{nodes_hash}_{edges_hash}"

    def cache_subgraph(self,
                      agent_id: int,
                      full_graph: GraphState,
                      subgraph: GraphState) -> None:
        """Cache computed subgraph."""
        key = f"subgraph_{self.get_graph_state_key(full_graph, agent_id)}"
        self.put(key, subgraph, ttl=600.0)  # 10 minute TTL

    def get_cached_subgraph(self,
                           agent_id: int,
                           full_graph: GraphState) -> Optional[GraphState]:
        """Get cached subgraph."""
        key = f"subgraph_{self.get_graph_state_key(full_graph, agent_id)}"
        return self.get(key)

    def cache_embeddings(self,
                        graph_state: GraphState,
                        embeddings: jnp.ndarray,
                        model_name: str) -> None:
        """Cache computed graph embeddings."""
        key = f"embeddings_{model_name}_{self.get_graph_state_key(graph_state, 0)}"
        self.put(key, embeddings, ttl=1800.0)  # 30 minute TTL

    def get_cached_embeddings(self,
                             graph_state: GraphState,
                             model_name: str) -> Optional[jnp.ndarray]:
        """Get cached embeddings."""
        key = f"embeddings_{model_name}_{self.get_graph_state_key(graph_state, 0)}"
        return self.get(key)


# Global cache instances
_global_smart_cache: Optional[SmartCache] = None
_global_graph_cache: Optional[GraphStateCache] = None


def get_global_cache() -> SmartCache:
    """Get global smart cache instance."""
    global _global_smart_cache
    if _global_smart_cache is None:
        _global_smart_cache = SmartCache(max_size=1000, max_memory_mb=100)
    return _global_smart_cache


def get_graph_cache() -> GraphStateCache:
    """Get global graph state cache instance."""
    global _global_graph_cache
    if _global_graph_cache is None:
        _global_graph_cache = GraphStateCache(max_size=500, max_memory_mb=50)
    return _global_graph_cache


def cached(ttl: Optional[float] = None,
          key_func: Optional[Callable] = None):
    """Decorator for caching function results."""
    def decorator(func: Callable):
        def wrapper(*args, **kwargs):
            cache = get_global_cache()

            # Generate cache key
            if key_func:
                cache_key = key_func(*args, **kwargs)
            else:
                # Default key generation
                key_parts = [func.__name__]
                key_parts.extend(str(arg) for arg in args)
                key_parts.extend(f"{k}={v}" for k, v in sorted(kwargs.items()))
                cache_key = "_".join(key_parts)

            # Try cache first
            cached_result = cache.get(cache_key)
            if cached_result is not None:
                return cached_result

            # Compute and cache result
            result = func(*args, **kwargs)
            cache.put(cache_key, result, ttl=ttl)

            return result

        return wrapper
    return decorator
