"""
Distributed computing utilities for federated graph RL.
"""
import concurrent.futures
import multiprocessing as mp
import threading
import time
from dataclasses import dataclass
from queue import Empty, Queue
from typing import Any, Callable, Dict, List, Optional, Tuple

try:
    import jax
    import jax.numpy as jnp
    JAX_AVAILABLE = True
except ImportError:
    JAX_AVAILABLE = False
    # Mock JAX for testing
    class MockJax:
        @staticmethod
        def devices(): return []
        @staticmethod  
        def local_devices(): return []
        @staticmethod
        def device_count(): return 1
        @staticmethod
        def default_backend(): return "cpu"
    jax = MockJax()
    jnp = None

from ..utils.logging import get_logger


@dataclass
class ComputeNode:
    """Represents a compute node in the distributed system."""
    node_id: str
    host: str
    port: int
    capabilities: Dict[str, Any]
    status: str = "active"
    load: float = 0.0
    last_heartbeat: float = 0.0


class DistributedComputeManager:
    """Manages distributed computation across multiple nodes."""
    
    def __init__(self, max_local_workers: int = None):
        self.max_workers = max_local_workers or min(32, (mp.cpu_count() or 1) + 4)
        self.compute_nodes: Dict[str, ComputeNode] = {}
        self.task_queue = Queue()
        self.result_cache: Dict[str, Any] = {}
        
        # Thread pools for different types of work
        self.cpu_executor = concurrent.futures.ThreadPoolExecutor(
            max_workers=self.max_workers // 2,
            thread_name_prefix="cpu_worker"
        )
        self.io_executor = concurrent.futures.ThreadPoolExecutor(
            max_workers=self.max_workers // 2,
            thread_name_prefix="io_worker"
        )
        
        self._lock = threading.RLock()
        self.logger = get_logger("pg_neo_graph_rl.distributed_compute")
        
        # Performance tracking
        self.task_metrics: Dict[str, Dict[str, float]] = {}
        
    def register_compute_node(self, node: ComputeNode):
        """Register a new compute node."""
        with self._lock:
            self.compute_nodes[node.node_id] = node
            
        self.logger.info(f"Registered compute node {node.node_id} at {node.host}:{node.port}")
        
    def submit_computation(self,
                          task_id: str,
                          computation_func: Callable,
                          *args,
                          priority: str = "normal",
                          **kwargs) -> concurrent.futures.Future:
        """Submit a computation task."""
        
        # Check cache first
        cache_key = f"{task_id}_{hash(str(args))}{hash(str(kwargs))}"
        if cache_key in self.result_cache:
            future = concurrent.futures.Future()
            future.set_result(self.result_cache[cache_key])
            return future
            
        # Select executor based on task type
        executor = self._select_executor(computation_func, priority)
        
        # Wrap computation with monitoring
        def monitored_computation():
            start_time = time.time()
            try:
                result = computation_func(*args, **kwargs)
                
                # Cache result
                self.result_cache[cache_key] = result
                
                # Record metrics
                execution_time = time.time() - start_time
                self._record_task_metrics(task_id, execution_time, "success")
                
                return result
                
            except Exception as e:
                execution_time = time.time() - start_time
                self._record_task_metrics(task_id, execution_time, "error")
                raise e
                
        future = executor.submit(monitored_computation)
        return future
        
    def _select_executor(self, func: Callable, priority: str) -> concurrent.futures.Executor:
        """Select appropriate executor based on function characteristics."""
        # Simple heuristic: if function name suggests I/O, use I/O executor
        func_name = getattr(func, '__name__', 'unknown')
        
        io_keywords = ['load', 'save', 'read', 'write', 'fetch', 'send', 'receive']
        if any(keyword in func_name.lower() for keyword in io_keywords):
            return self.io_executor
        else:
            return self.cpu_executor
            
    def _record_task_metrics(self, task_id: str, execution_time: float, status: str):
        """Record task execution metrics."""
        with self._lock:
            if task_id not in self.task_metrics:
                self.task_metrics[task_id] = {
                    "total_executions": 0,
                    "total_time": 0.0,
                    "success_count": 0,
                    "error_count": 0
                }
                
            metrics = self.task_metrics[task_id]
            metrics["total_executions"] += 1
            metrics["total_time"] += execution_time
            
            if status == "success":
                metrics["success_count"] += 1
            else:
                metrics["error_count"] += 1
    
    def get_compute_stats(self) -> Dict[str, Any]:
        """Get compute system statistics."""
        with self._lock:
            total_tasks = sum(m["total_executions"] for m in self.task_metrics.values())
            total_time = sum(m["total_time"] for m in self.task_metrics.values())
            success_count = sum(m["success_count"] for m in self.task_metrics.values())
            
            return {
                "registered_nodes": len(self.compute_nodes),
                "max_workers": self.max_workers,
                "cache_size": len(self.result_cache),
                "task_metrics": {
                    "total_tasks": total_tasks,
                    "total_execution_time": total_time,
                    "success_rate": success_count / total_tasks if total_tasks > 0 else 1.0,
                    "avg_execution_time": total_time / total_tasks if total_tasks > 0 else 0.0
                }
            }


class JAXDistributedCompute:
    """JAX-specific distributed computation utilities."""
    
    def __init__(self):
        self.logger = get_logger("pg_neo_graph_rl.jax_distributed")
        
        if JAX_AVAILABLE:
            try:
                self.devices = jax.devices()
                self.local_devices = jax.local_devices() 
                self.device_count = jax.device_count()
            except Exception as e:
                self.logger.error(f"JAX initialization failed: {e}")
                self.devices = []
                self.device_count = 1
        else:
            self.devices = []
            self.device_count = 1
            
    def get_device_info(self) -> Dict[str, Any]:
        """Get information about available JAX devices."""
        return {
            "total_devices": self.device_count,
            "jax_available": JAX_AVAILABLE,
            "backend": jax.default_backend() if JAX_AVAILABLE else "none"
        }