"""Production-grade performance optimization for federated learning at scale."""
import gc
import threading
import time
from collections import defaultdict, deque
from dataclasses import dataclass
from typing import Any, Callable, Dict, List

import jax
import jax.numpy as jnp
import numpy as np

from ..core.types import GraphState
from ..monitoring.advanced_metrics import AdvancedMetricsCollector
from ..utils.logging import get_logger, get_performance_logger

logger = get_logger(__name__)
perf_logger = get_performance_logger(__name__)

@dataclass
class OptimizationProfile:
    """Performance optimization profile."""
    target_latency_ms: float = 100.0
    target_throughput_qps: float = 1000.0
    memory_limit_gb: float = 8.0
    cpu_target_percent: float = 80.0
    enable_jit_compilation: bool = True
    enable_memory_pooling: bool = True
    enable_computation_caching: bool = True
    batch_optimization: bool = True

class ProductionOptimizer:
    """Production-grade optimizer for federated learning systems."""

    def __init__(self, profile: OptimizationProfile = None):
        self.profile = profile or OptimizationProfile()

        # Performance tracking
        self.metrics_collector = AdvancedMetricsCollector()
        self.optimization_history = deque(maxlen=1000)

        # JIT compilation cache
        self.jit_cache = {}
        self.compilation_stats = defaultdict(int)

        # Memory management
        self.memory_pool = MemoryPool()
        self.object_cache = {}

        # Batch processing
        self.batch_processor = BatchProcessor()

        # Computation graph optimization
        self.graph_optimizer = ComputationGraphOptimizer()

        # Runtime optimization
        self.runtime_optimizer = RuntimeOptimizer()

        # Background optimization thread
        self.optimization_thread = None
        self.optimization_active = False
        self.stop_event = threading.Event()

        logger.info("Production optimizer initialized")

    def start_optimization(self) -> None:
        """Start background optimization processes."""
        if self.optimization_active:
            logger.warning("Optimization already active")
            return

        self.optimization_active = True
        self.stop_event.clear()

        # Start background optimization
        self.optimization_thread = threading.Thread(
            target=self._optimization_loop,
            daemon=True
        )
        self.optimization_thread.start()

        # Initialize memory pool
        if self.profile.enable_memory_pooling:
            self.memory_pool.initialize()

        # Start batch processor
        if self.profile.batch_optimization:
            self.batch_processor.start()

        logger.info("Production optimization started")

    def stop_optimization(self) -> None:
        """Stop optimization processes."""
        if not self.optimization_active:
            return

        self.optimization_active = False
        self.stop_event.set()

        # Stop batch processor
        self.batch_processor.stop()

        # Join optimization thread
        if self.optimization_thread and self.optimization_thread.is_alive():
            self.optimization_thread.join(timeout=10.0)

        logger.info("Production optimization stopped")

    def _optimization_loop(self) -> None:
        """Background optimization loop."""
        while self.optimization_active and not self.stop_event.wait(30.0):
            try:
                self._optimize_runtime_performance()
                self._optimize_memory_usage()
                self._optimize_compilation_cache()
                self._collect_optimization_metrics()
            except Exception as e:
                logger.error(f"Optimization loop error: {e}")

    def optimize_computation(self, func: Callable, *args, **kwargs) -> Any:
        """Optimize a computation with production-grade techniques."""
        start_time = time.time()

        try:
            # Check if function is JIT compiled
            if self.profile.enable_jit_compilation:
                optimized_func = self._get_or_compile_jit(func, *args)
            else:
                optimized_func = func

            # Use memory pool for arrays
            if self.profile.enable_memory_pooling:
                args = self._optimize_memory_allocation(args)

            # Execute optimized computation
            result = optimized_func(*args, **kwargs)

            # Record performance metrics
            duration = time.time() - start_time
            self.metrics_collector.record('computation_duration_ms', duration * 1000)

            return result

        except Exception as e:
            logger.error(f"Computation optimization failed: {e}")
            # Fallback to original function
            return func(*args, **kwargs)

    def _get_or_compile_jit(self, func: Callable, *args) -> Callable:
        """Get JIT compiled version of function or compile if needed."""
        # Create cache key based on function and argument shapes
        cache_key = self._create_jit_cache_key(func, *args)

        if cache_key in self.jit_cache:
            return self.jit_cache[cache_key]

        # Compile function with JAX JIT
        try:
            jit_func = jax.jit(func)

            # Warm up compilation
            if args:
                _ = jit_func(*args)

            self.jit_cache[cache_key] = jit_func
            self.compilation_stats[func.__name__] += 1

            logger.debug(f"JIT compiled function: {func.__name__}")
            return jit_func

        except Exception as e:
            logger.warning(f"JIT compilation failed for {func.__name__}: {e}")
            return func

    def _create_jit_cache_key(self, func: Callable, *args) -> str:
        """Create cache key for JIT compiled functions."""
        func_name = getattr(func, '__name__', str(func))

        # Include argument shapes in key
        arg_shapes = []
        for arg in args:
            if hasattr(arg, 'shape'):
                arg_shapes.append(str(arg.shape))
            elif hasattr(arg, '__len__'):
                arg_shapes.append(str(len(arg)))
            else:
                arg_shapes.append(str(type(arg)))

        return f"{func_name}_{'_'.join(arg_shapes)}"

    def _optimize_memory_allocation(self, args: tuple) -> tuple:
        """Optimize memory allocation using memory pool."""
        optimized_args = []

        for arg in args:
            if isinstance(arg, (np.ndarray, jnp.ndarray)):
                # Use memory pool for large arrays
                if arg.size > 1000:
                    pooled_array = self.memory_pool.get_array(arg.shape, arg.dtype)
                    pooled_array = pooled_array.at[:].set(arg)
                    optimized_args.append(pooled_array)
                else:
                    optimized_args.append(arg)
            else:
                optimized_args.append(arg)

        return tuple(optimized_args)

    def optimize_graph_state(self, state: GraphState) -> GraphState:
        """Optimize graph state representation."""
        start_time = time.time()

        try:
            # Optimize node features
            optimized_nodes = self._optimize_array(state.nodes)

            # Optimize adjacency representation
            optimized_adj = self._optimize_sparse_matrix(state.adjacency)

            # Optimize edge features
            optimized_edge_attr = None
            if state.edge_attr is not None:
                optimized_edge_attr = self._optimize_array(state.edge_attr)

            # Create optimized state
            optimized_state = GraphState(
                nodes=optimized_nodes,
                edges=state.edges,
                adjacency=optimized_adj,
                edge_attr=optimized_edge_attr,
                timestamps=state.timestamps
            )

            duration = time.time() - start_time
            self.metrics_collector.record('graph_optimization_ms', duration * 1000)

            return optimized_state

        except Exception as e:
            logger.error(f"Graph state optimization failed: {e}")
            return state

    def _optimize_array(self, array: jnp.ndarray) -> jnp.ndarray:
        """Optimize array representation and storage."""
        if array is None:
            return array

        # Check if array can be compressed
        if self._should_compress_array(array):
            # Use lower precision if possible
            if array.dtype == jnp.float64:
                return array.astype(jnp.float32)
            elif array.dtype == jnp.int64:
                if jnp.all(array < 2**31):
                    return array.astype(jnp.int32)

        return array

    def _should_compress_array(self, array: jnp.ndarray) -> bool:
        """Check if array should be compressed."""
        # Compress large arrays to save memory
        return array.size > 10000

    def _optimize_sparse_matrix(self, matrix: jnp.ndarray) -> jnp.ndarray:
        """Optimize sparse matrix representation."""
        if matrix is None:
            return matrix

        # Check sparsity
        sparsity = 1.0 - (jnp.count_nonzero(matrix) / matrix.size)

        if sparsity > 0.9:  # Very sparse
            logger.debug(f"Matrix sparsity: {sparsity:.2f}")
            # Could implement sparse matrix format here
            # For now, just ensure efficient dtype
            return self._optimize_array(matrix)

        return matrix

    def _optimize_runtime_performance(self) -> None:
        """Optimize runtime performance characteristics."""
        try:
            # Check current performance metrics
            recent_latencies = self.metrics_collector.get_recent_values('computation_duration_ms', 100)

            if recent_latencies:
                avg_latency = np.mean(recent_latencies)

                if avg_latency > self.profile.target_latency_ms:
                    logger.info(f"High latency detected: {avg_latency:.1f}ms")
                    self._apply_latency_optimizations()

            # Check throughput
            recent_throughput = self.metrics_collector.get_recent_values('throughput', 10)
            if recent_throughput:
                avg_throughput = np.mean(recent_throughput)

                if avg_throughput < self.profile.target_throughput_qps:
                    logger.info(f"Low throughput detected: {avg_throughput:.1f} qps")
                    self._apply_throughput_optimizations()

        except Exception as e:
            logger.error(f"Runtime optimization error: {e}")

    def _apply_latency_optimizations(self) -> None:
        """Apply optimizations to reduce latency."""
        # Clear old JIT cache entries
        if len(self.jit_cache) > 100:
            # Remove least recently used entries
            self.jit_cache = dict(list(self.jit_cache.items())[-50:])

        # Trigger garbage collection
        gc.collect()

        logger.debug("Applied latency optimizations")

    def _apply_throughput_optimizations(self) -> None:
        """Apply optimizations to increase throughput."""
        # Enable more aggressive batch processing
        self.batch_processor.increase_batch_size()

        # Precompile common functions
        self._precompile_hot_functions()

        logger.debug("Applied throughput optimizations")

    def _precompile_hot_functions(self) -> None:
        """Precompile frequently used functions."""
        # Identify hot functions from compilation stats
        hot_functions = sorted(
            self.compilation_stats.items(),
            key=lambda x: x[1],
            reverse=True
        )[:10]

        logger.debug(f"Hot functions: {[name for name, count in hot_functions]}")

    def _optimize_memory_usage(self) -> None:
        """Optimize memory usage."""
        try:
            # Get current memory usage
            import psutil
            memory_usage = psutil.virtual_memory().percent

            if memory_usage > 85.0:  # High memory usage
                logger.warning(f"High memory usage: {memory_usage:.1f}%")

                # Clear caches
                self._clear_old_caches()

                # Force garbage collection
                collected = gc.collect()
                logger.info(f"Garbage collected {collected} objects")

                # Clean memory pool
                self.memory_pool.cleanup()

        except ImportError:
            logger.debug("psutil not available for memory monitoring")
        except Exception as e:
            logger.error(f"Memory optimization error: {e}")

    def _clear_old_caches(self) -> None:
        """Clear old cache entries."""
        # Clear object cache
        old_size = len(self.object_cache)
        self.object_cache.clear()

        # Clear JIT cache partially
        if len(self.jit_cache) > 50:
            self.jit_cache = dict(list(self.jit_cache.items())[-25:])

        logger.debug(f"Cleared {old_size} cache entries")

    def _optimize_compilation_cache(self) -> None:
        """Optimize JIT compilation cache."""
        # Remove unused compiled functions
        current_time = time.time()

        # This is simplified - in practice, would track usage timestamps
        if len(self.jit_cache) > 200:
            # Keep only most recent entries
            self.jit_cache = dict(list(self.jit_cache.items())[-100:])
            logger.debug("Optimized compilation cache")

    def _collect_optimization_metrics(self) -> None:
        """Collect optimization-specific metrics."""
        metrics = {
            'jit_cache_size': len(self.jit_cache),
            'object_cache_size': len(self.object_cache),
            'memory_pool_size': self.memory_pool.get_size(),
            'batch_queue_size': self.batch_processor.get_queue_size()
        }

        for name, value in metrics.items():
            self.metrics_collector.record(name, value)

    def get_optimization_stats(self) -> Dict[str, Any]:
        """Get current optimization statistics."""
        return {
            'profile': {
                'target_latency_ms': self.profile.target_latency_ms,
                'target_throughput_qps': self.profile.target_throughput_qps,
                'memory_limit_gb': self.profile.memory_limit_gb
            },
            'cache_stats': {
                'jit_cache_size': len(self.jit_cache),
                'object_cache_size': len(self.object_cache),
                'compilation_stats': dict(self.compilation_stats)
            },
            'memory_stats': self.memory_pool.get_stats(),
            'batch_stats': self.batch_processor.get_stats(),
            'is_active': self.optimization_active
        }

class MemoryPool:
    """Memory pool for efficient array allocation."""

    def __init__(self):
        self.pools = defaultdict(list)  # dtype -> list of arrays
        self.pool_sizes = defaultdict(int)
        self.max_pool_size = 100
        self.allocated_count = 0
        self.reused_count = 0

    def initialize(self) -> None:
        """Initialize memory pool."""
        # Pre-allocate common array sizes
        common_shapes = [
            (100, 64), (1000, 128), (50, 32), (500, 256)
        ]

        for shape in common_shapes:
            for dtype in [jnp.float32, jnp.int32]:
                for _ in range(5):  # Pre-allocate 5 of each
                    array = jnp.zeros(shape, dtype=dtype)
                    key = (shape, dtype)
                    self.pools[key].append(array)
                    self.pool_sizes[key] += 1

        logger.debug("Memory pool initialized")

    def get_array(self, shape: tuple, dtype: type) -> jnp.ndarray:
        """Get array from pool or allocate new one."""
        key = (shape, dtype)

        if self.pools[key]:
            array = self.pools[key].pop()
            self.pool_sizes[key] -= 1
            self.reused_count += 1
            return array.at[:].set(0)  # Reset to zeros
        else:
            self.allocated_count += 1
            return jnp.zeros(shape, dtype=dtype)

    def return_array(self, array: jnp.ndarray) -> None:
        """Return array to pool for reuse."""
        key = (array.shape, array.dtype)

        if self.pool_sizes[key] < self.max_pool_size:
            self.pools[key].append(array)
            self.pool_sizes[key] += 1

    def get_size(self) -> int:
        """Get total number of pooled arrays."""
        return sum(self.pool_sizes.values())

    def get_stats(self) -> Dict[str, Any]:
        """Get memory pool statistics."""
        return {
            'total_pooled': self.get_size(),
            'allocated_count': self.allocated_count,
            'reused_count': self.reused_count,
            'reuse_rate': self.reused_count / max(self.allocated_count, 1),
            'pool_sizes': dict(self.pool_sizes)
        }

    def cleanup(self) -> None:
        """Clean up memory pool."""
        cleared = 0
        for key in list(self.pools.keys()):
            cleared += len(self.pools[key])
            self.pools[key].clear()
            self.pool_sizes[key] = 0

        logger.debug(f"Cleaned up {cleared} pooled arrays")

class BatchProcessor:
    """Batch processor for improved throughput."""

    def __init__(self, initial_batch_size: int = 32):
        self.batch_size = initial_batch_size
        self.max_batch_size = 256
        self.batch_queue = deque()
        self.processing_active = False
        self.process_thread = None
        self.processed_count = 0

    def start(self) -> None:
        """Start batch processing."""
        self.processing_active = True
        self.process_thread = threading.Thread(target=self._process_batches, daemon=True)
        self.process_thread.start()
        logger.debug("Batch processor started")

    def stop(self) -> None:
        """Stop batch processing."""
        self.processing_active = False
        if self.process_thread and self.process_thread.is_alive():
            self.process_thread.join(timeout=5.0)
        logger.debug("Batch processor stopped")

    def _process_batches(self) -> None:
        """Process batched computations."""
        while self.processing_active:
            if len(self.batch_queue) >= self.batch_size:
                batch = []
                for _ in range(min(self.batch_size, len(self.batch_queue))):
                    if self.batch_queue:
                        batch.append(self.batch_queue.popleft())

                if batch:
                    self._execute_batch(batch)
            else:
                time.sleep(0.01)  # Small delay

    def _execute_batch(self, batch: List[Any]) -> None:
        """Execute a batch of computations."""
        try:
            # This is a placeholder for batch execution logic
            self.processed_count += len(batch)
        except Exception as e:
            logger.error(f"Batch execution error: {e}")

    def increase_batch_size(self) -> None:
        """Increase batch size for higher throughput."""
        if self.batch_size < self.max_batch_size:
            self.batch_size = min(self.batch_size * 2, self.max_batch_size)
            logger.debug(f"Increased batch size to {self.batch_size}")

    def get_queue_size(self) -> int:
        """Get current queue size."""
        return len(self.batch_queue)

    def get_stats(self) -> Dict[str, Any]:
        """Get batch processor statistics."""
        return {
            'batch_size': self.batch_size,
            'queue_size': self.get_queue_size(),
            'processed_count': self.processed_count,
            'is_active': self.processing_active
        }

class ComputationGraphOptimizer:
    """Optimizes computation graphs for better performance."""

    def __init__(self):
        self.optimized_graphs = {}

    def optimize_graph(self, computation_fn: Callable) -> Callable:
        """Optimize computation graph."""
        # This is a placeholder for graph optimization
        return computation_fn

class RuntimeOptimizer:
    """Optimizes runtime characteristics."""

    def __init__(self):
        self.runtime_stats = defaultdict(list)

    def optimize_runtime(self, func: Callable) -> Callable:
        """Apply runtime optimizations."""
        # This is a placeholder for runtime optimization
        return func

# Global production optimizer instance
_global_optimizer = None

def get_production_optimizer(**kwargs) -> ProductionOptimizer:
    """Get or create global production optimizer."""
    global _global_optimizer
    if _global_optimizer is None:
        _global_optimizer = ProductionOptimizer(**kwargs)
    return _global_optimizer
