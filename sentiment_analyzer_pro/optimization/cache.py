"""Advanced caching system for sentiment analysis."""

import hashlib
import time
import json
import pickle
import logging
from typing import Any, Dict, List, Optional, Tuple, Union
from dataclasses import dataclass, asdict
from collections import OrderedDict
from threading import RLock
import numpy as np

from ..core.base import SentimentResult


logger = logging.getLogger(__name__)


@dataclass
class CacheStats:
    """Cache statistics."""
    hits: int = 0
    misses: int = 0
    evictions: int = 0
    total_size: int = 0
    max_size: int = 0
    hit_rate: float = 0.0
    
    def update_hit_rate(self):
        """Update hit rate calculation."""
        total_requests = self.hits + self.misses
        self.hit_rate = self.hits / total_requests if total_requests > 0 else 0.0


@dataclass
class CacheEntry:
    """Cache entry with metadata."""
    key: str
    value: Any
    timestamp: float
    access_count: int
    last_access: float
    size: int
    
    def is_expired(self, ttl: float) -> bool:
        """Check if entry is expired."""
        return time.time() - self.timestamp > ttl
    
    def update_access(self):
        """Update access statistics."""
        self.access_count += 1
        self.last_access = time.time()


class ResultCache:
    """
    High-performance LRU cache for sentiment analysis results.
    
    Features:
    - TTL-based expiration
    - Size-based eviction
    - Access frequency tracking
    - Thread-safe operations
    - Memory-efficient storage
    """
    
    def __init__(
        self,
        max_size: int = 10000,
        ttl_seconds: float = 3600,
        enable_stats: bool = True,
        cleanup_interval: int = 300
    ):
        """
        Initialize result cache.
        
        Args:
            max_size: Maximum number of entries
            ttl_seconds: Time-to-live for entries
            enable_stats: Enable statistics tracking
            cleanup_interval: Cleanup interval in seconds
        """
        self.max_size = max_size
        self.ttl_seconds = ttl_seconds
        self.enable_stats = enable_stats
        self.cleanup_interval = cleanup_interval
        
        self._cache: OrderedDict[str, CacheEntry] = OrderedDict()
        self._lock = RLock()
        self._stats = CacheStats(max_size=max_size)
        self._last_cleanup = time.time()
        
        logger.info(f"ResultCache initialized: max_size={max_size}, ttl={ttl_seconds}s")
    
    def get(self, key: str) -> Optional[SentimentResult]:
        """
        Get cached result.
        
        Args:
            key: Cache key
            
        Returns:
            Cached result if available, None otherwise
        """
        with self._lock:
            # Periodic cleanup
            if time.time() - self._last_cleanup > self.cleanup_interval:
                self._cleanup_expired()
            
            if key in self._cache:
                entry = self._cache[key]
                
                # Check if expired
                if entry.is_expired(self.ttl_seconds):
                    del self._cache[key]
                    if self.enable_stats:
                        self._stats.misses += 1
                    return None
                
                # Update access and move to end (LRU)
                entry.update_access()
                self._cache.move_to_end(key)
                
                if self.enable_stats:
                    self._stats.hits += 1
                    self._stats.update_hit_rate()
                
                return entry.value
            else:
                if self.enable_stats:
                    self._stats.misses += 1
                    self._stats.update_hit_rate()
                return None
    
    def put(self, key: str, value: SentimentResult) -> None:
        """
        Cache a result.
        
        Args:
            key: Cache key
            value: Result to cache
        """
        with self._lock:
            now = time.time()
            
            # Calculate entry size (approximate)
            entry_size = self._estimate_size(value)
            
            entry = CacheEntry(
                key=key,
                value=value,
                timestamp=now,
                access_count=1,
                last_access=now,
                size=entry_size
            )
            
            # Remove existing entry if present
            if key in self._cache:
                old_entry = self._cache[key]
                self._stats.total_size -= old_entry.size
                del self._cache[key]
            
            # Evict if necessary
            while len(self._cache) >= self.max_size:
                self._evict_lru()
            
            # Add new entry
            self._cache[key] = entry
            self._stats.total_size += entry_size
    
    def _estimate_size(self, value: SentimentResult) -> int:
        """Estimate memory size of a result."""
        try:
            # Rough estimation based on content
            size = len(value.text) * 2  # Unicode characters
            size += 8 * len(value.probabilities)  # Float array
            size += 100  # Other fields
            return size
        except Exception:
            return 200  # Default estimate
    
    def _evict_lru(self) -> None:
        """Evict least recently used entry."""
        if self._cache:
            key, entry = self._cache.popitem(last=False)  # Remove oldest
            self._stats.total_size -= entry.size
            if self.enable_stats:
                self._stats.evictions += 1
            logger.debug(f"Evicted cache entry: {key}")
    
    def _cleanup_expired(self) -> None:
        """Remove expired entries."""
        now = time.time()
        expired_keys = []
        
        for key, entry in self._cache.items():
            if entry.is_expired(self.ttl_seconds):
                expired_keys.append(key)
        
        for key in expired_keys:
            entry = self._cache[key]
            self._stats.total_size -= entry.size
            del self._cache[key]
        
        if expired_keys:
            logger.debug(f"Cleaned up {len(expired_keys)} expired cache entries")
        
        self._last_cleanup = now
    
    def clear(self) -> None:
        """Clear all cached entries."""
        with self._lock:
            self._cache.clear()
            self._stats = CacheStats(max_size=self.max_size)
            logger.info("Cache cleared")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        with self._lock:
            stats_dict = asdict(self._stats)
            stats_dict.update({
                "current_size": len(self._cache),
                "memory_usage_bytes": self._stats.total_size,
                "memory_usage_mb": round(self._stats.total_size / (1024**2), 2)
            })
            return stats_dict


class CacheManager:
    """
    Centralized cache manager for different types of cached data.
    
    Features:
    - Multiple cache instances
    - Adaptive cache sizing
    - Performance monitoring
    - Memory pressure handling
    """
    
    def __init__(
        self,
        result_cache_size: int = 10000,
        model_cache_size: int = 5,
        adaptive_sizing: bool = True,
        memory_threshold: float = 0.8
    ):
        """
        Initialize cache manager.
        
        Args:
            result_cache_size: Size for result cache
            model_cache_size: Size for model cache
            adaptive_sizing: Enable adaptive cache sizing
            memory_threshold: Memory usage threshold for cache reduction
        """
        self.adaptive_sizing = adaptive_sizing
        self.memory_threshold = memory_threshold
        
        # Initialize caches
        self.result_cache = ResultCache(
            max_size=result_cache_size,
            ttl_seconds=3600
        )
        
        self.model_cache = ResultCache(
            max_size=model_cache_size,
            ttl_seconds=86400  # 24 hours for models
        )
        
        # Preprocessing cache for expensive operations
        self.preprocessing_cache = ResultCache(
            max_size=5000,
            ttl_seconds=7200  # 2 hours
        )
        
        logger.info("CacheManager initialized with multiple cache instances")
    
    def get_result(self, text: str, model_name: str) -> Optional[SentimentResult]:
        """Get cached sentiment analysis result."""
        cache_key = self._generate_result_key(text, model_name)
        return self.result_cache.get(cache_key)
    
    def cache_result(self, text: str, model_name: str, result: SentimentResult) -> None:
        """Cache sentiment analysis result."""
        cache_key = self._generate_result_key(text, model_name)
        self.result_cache.put(cache_key, result)
    
    def get_preprocessed_text(self, text: str, preprocessing_config: Dict[str, Any]) -> Optional[str]:
        """Get cached preprocessed text."""
        cache_key = self._generate_preprocessing_key(text, preprocessing_config)
        return self.preprocessing_cache.get(cache_key)
    
    def cache_preprocessed_text(
        self, 
        text: str, 
        preprocessing_config: Dict[str, Any], 
        processed_text: str
    ) -> None:
        """Cache preprocessed text."""
        cache_key = self._generate_preprocessing_key(text, preprocessing_config)
        # Create a dummy result object for caching
        dummy_result = SentimentResult(
            text=processed_text,
            label="CACHED",
            confidence=1.0,
            probabilities=np.array([1.0]),
            processing_time=0.0
        )
        self.preprocessing_cache.put(cache_key, dummy_result)
    
    def _generate_result_key(self, text: str, model_name: str) -> str:
        """Generate cache key for results."""
        content = f"{text}:{model_name}"
        return hashlib.sha256(content.encode()).hexdigest()[:16]
    
    def _generate_preprocessing_key(self, text: str, config: Dict[str, Any]) -> str:
        """Generate cache key for preprocessing."""
        config_str = json.dumps(config, sort_keys=True)
        content = f"{text}:{config_str}"
        return hashlib.sha256(content.encode()).hexdigest()[:16]
    
    def optimize_cache_sizes(self, memory_usage: float) -> None:
        """Optimize cache sizes based on memory usage."""
        if not self.adaptive_sizing:
            return
        
        if memory_usage > self.memory_threshold:
            # Reduce cache sizes
            reduction_factor = 0.8
            
            new_result_size = int(self.result_cache.max_size * reduction_factor)
            new_preprocessing_size = int(self.preprocessing_cache.max_size * reduction_factor)
            
            self._resize_cache(self.result_cache, new_result_size)
            self._resize_cache(self.preprocessing_cache, new_preprocessing_size)
            
            logger.warning(f"Reduced cache sizes due to memory pressure: {memory_usage:.1%}")
        
        elif memory_usage < self.memory_threshold * 0.7:
            # Increase cache sizes if memory is abundant
            expansion_factor = 1.1
            
            new_result_size = int(self.result_cache.max_size * expansion_factor)
            new_preprocessing_size = int(self.preprocessing_cache.max_size * expansion_factor)
            
            # Cap at reasonable limits
            new_result_size = min(new_result_size, 50000)
            new_preprocessing_size = min(new_preprocessing_size, 25000)
            
            self._resize_cache(self.result_cache, new_result_size)
            self._resize_cache(self.preprocessing_cache, new_preprocessing_size)
            
            logger.info(f"Expanded cache sizes due to low memory usage: {memory_usage:.1%}")
    
    def _resize_cache(self, cache: ResultCache, new_size: int) -> None:
        """Resize a cache instance."""
        if new_size == cache.max_size:
            return
        
        with cache._lock:
            cache.max_size = new_size
            cache._stats.max_size = new_size
            
            # Evict entries if new size is smaller
            while len(cache._cache) > new_size:
                cache._evict_lru()
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get comprehensive cache statistics."""
        return {
            "result_cache": self.result_cache.get_stats(),
            "model_cache": self.model_cache.get_stats(),
            "preprocessing_cache": self.preprocessing_cache.get_stats(),
            "total_memory_mb": round(
                (self.result_cache._stats.total_size + 
                 self.model_cache._stats.total_size + 
                 self.preprocessing_cache._stats.total_size) / (1024**2), 
                2
            ),
            "adaptive_sizing_enabled": self.adaptive_sizing
        }
    
    def clear_all_caches(self) -> None:
        """Clear all cache instances."""
        self.result_cache.clear()
        self.model_cache.clear()
        self.preprocessing_cache.clear()
        logger.info("All caches cleared")
    
    def warmup_cache(self, common_texts: List[str], model_name: str) -> None:
        """Warm up cache with common texts."""
        logger.info(f"Starting cache warmup with {len(common_texts)} texts")
        
        # This would typically involve running predictions on common texts
        # For now, we'll just log the operation
        for i, text in enumerate(common_texts):
            cache_key = self._generate_result_key(text, model_name)
            logger.debug(f"Warmup cache key {i+1}: {cache_key}")
        
        logger.info("Cache warmup completed")