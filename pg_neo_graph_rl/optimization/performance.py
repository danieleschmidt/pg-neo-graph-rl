"""
Performance optimization utilities.
"""
import time
import threading
import concurrent.futures
from typing import Any, Dict, List, Optional, Callable, Tuple
from queue import Queue, Empty
from dataclasses import dataclass
import jax
import jax.numpy as jnp

from ..utils.logging import get_logger, get_performance_logger
from ..core.federated import GraphState


@dataclass 
class BatchConfig:
    """Configuration for batch processing."""
    min_batch_size: int = 1
    max_batch_size: int = 32
    max_wait_time: float = 0.1  # seconds
    adaptive: bool = True


class BatchProcessor:
    """Processes batches of requests efficiently."""
    
    def __init__(self, 
                 process_function: Callable,
                 batch_config: BatchConfig = None):
        self.process_function = process_function
        self.config = batch_config or BatchConfig()
        self.request_queue = Queue()
        self.response_map = {}
        self.processing = False
        self.worker_thread = None
        self._lock = threading.RLock()
        self.logger = get_logger("pg_neo_graph_rl.batch_processor")
        
        # Adaptive batching metrics
        self.batch_times = []
        self.optimal_batch_size = self.config.min_batch_size
    
    def start(self) -> None:
        """Start batch processing."""
        if self.processing:
            return
        
        self.processing = True
        self.worker_thread = threading.Thread(target=self._process_loop, daemon=True)
        self.worker_thread.start()
        self.logger.info("Batch processor started")
    
    def stop(self) -> None:
        """Stop batch processing."""
        self.processing = False
        if self.worker_thread:
            self.worker_thread.join(timeout=5.0)
        self.logger.info("Batch processor stopped")
    
    def submit(self, request_id: str, *args, **kwargs) -> None:
        """Submit request for batch processing."""
        future = concurrent.futures.Future()
        
        with self._lock:
            self.response_map[request_id] = future
        
        self.request_queue.put((request_id, args, kwargs))
        return future
    
    def _process_loop(self) -> None:
        """Main batch processing loop."""
        while self.processing:
            try:
                batch = self._collect_batch()
                if batch:
                    self._process_batch(batch)
                else:
                    time.sleep(0.001)  # Short sleep if no requests
            except Exception as e:
                self.logger.error(f"Error in batch processing: {e}")
    
    def _collect_batch(self) -> List[Tuple[str, tuple, dict]]:
        """Collect requests into a batch."""
        batch = []
        start_time = time.time()
        
        # Get first request (blocking with timeout)
        try:
            first_request = self.request_queue.get(timeout=0.01)
            batch.append(first_request)
        except Empty:
            return batch
        
        # Collect additional requests
        while (len(batch) < self.optimal_batch_size and 
               time.time() - start_time < self.config.max_wait_time):
            try:
                request = self.request_queue.get_nowait()
                batch.append(request)
            except Empty:
                break
        
        return batch
    
    def _process_batch(self, batch: List[Tuple[str, tuple, dict]]) -> None:
        """Process a batch of requests."""
        if not batch:
            return
        
        start_time = time.time()
        
        try:
            # Extract requests
            request_ids = [item[0] for item in batch]
            args_list = [item[1] for item in batch]
            kwargs_list = [item[2] for item in batch]
            
            # Process batch
            results = self.process_function(args_list, kwargs_list)
            
            # Distribute results
            with self._lock:
                for request_id, result in zip(request_ids, results):
                    if request_id in self.response_map:
                        self.response_map[request_id].set_result(result)
                        del self.response_map[request_id]
            
            # Update metrics
            batch_time = time.time() - start_time
            self.batch_times.append(batch_time)
            
            # Adapt batch size if enabled
            if self.config.adaptive and len(self.batch_times) % 100 == 0:
                self._adapt_batch_size()
                
        except Exception as e:
            # Set exception for all requests in batch
            with self._lock:
                for request_id, _, _ in batch:
                    if request_id in self.response_map:
                        self.response_map[request_id].set_exception(e)
                        del self.response_map[request_id]
    
    def _adapt_batch_size(self) -> None:
        """Adapt batch size based on performance."""
        if len(self.batch_times) < 50:
            return
        
        recent_times = self.batch_times[-50:]
        avg_time = sum(recent_times) / len(recent_times)
        
        # Simple adaptation: increase batch size if processing is fast
        if avg_time < 0.01:  # Less than 10ms per batch
            self.optimal_batch_size = min(
                self.optimal_batch_size + 2,
                self.config.max_batch_size
            )
        elif avg_time > 0.1:  # More than 100ms per batch
            self.optimal_batch_size = max(
                self.optimal_batch_size - 2,
                self.config.min_batch_size
            )
        
        self.logger.debug(f"Adapted batch size to {self.optimal_batch_size}")


class ConcurrentTrainer:
    """Manages concurrent training across multiple agents."""
    
    def __init__(self, 
                 max_workers: int = None,
                 queue_size: int = 1000):
        self.max_workers = max_workers or min(32, (threading.active_count() or 1) + 4)
        self.executor = concurrent.futures.ThreadPoolExecutor(
            max_workers=self.max_workers,
            thread_name_prefix="trainer"
        )
        self.task_queue = Queue(maxsize=queue_size)
        self.active_tasks = {}
        self._lock = threading.RLock()
        self.logger = get_logger("pg_neo_graph_rl.concurrent_trainer")
        self.perf_logger = get_performance_logger()
    
    def submit_training_task(self, 
                           agent_id: int,
                           training_function: Callable,
                           *args, **kwargs) -> concurrent.futures.Future:
        """Submit training task for concurrent execution."""
        
        def wrapped_training():
            with self.perf_logger.timer_context(f"agent_{agent_id}_training"):
                try:
                    return training_function(*args, **kwargs)
                except Exception as e:
                    self.logger.error(f"Training failed for agent {agent_id}: {e}")
                    raise
        
        future = self.executor.submit(wrapped_training)
        
        with self._lock:
            self.active_tasks[agent_id] = future
        
        # Clean up completed tasks
        future.add_done_callback(lambda f: self._cleanup_task(agent_id))
        
        return future
    
    def wait_for_all_agents(self, 
                          timeout: Optional[float] = None) -> Dict[int, Any]:
        """Wait for all active training tasks to complete."""
        with self._lock:
            active_futures = dict(self.active_tasks)
        
        results = {}
        
        # Wait for all futures
        for agent_id, future in active_futures.items():
            try:
                result = future.result(timeout=timeout)
                results[agent_id] = result
            except Exception as e:
                self.logger.error(f"Agent {agent_id} training failed: {e}")
                results[agent_id] = None
        
        return results
    
    def wait_for_agents(self, 
                       agent_ids: List[int],
                       timeout: Optional[float] = None) -> Dict[int, Any]:
        """Wait for specific agents to complete training."""
        futures_to_wait = {}
        
        with self._lock:
            for agent_id in agent_ids:
                if agent_id in self.active_tasks:
                    futures_to_wait[agent_id] = self.active_tasks[agent_id]
        
        results = {}
        
        for agent_id, future in futures_to_wait.items():
            try:
                result = future.result(timeout=timeout)
                results[agent_id] = result
            except Exception as e:
                self.logger.error(f"Agent {agent_id} training failed: {e}")
                results[agent_id] = None
        
        return results
    
    def _cleanup_task(self, agent_id: int) -> None:
        """Clean up completed task."""
        with self._lock:
            if agent_id in self.active_tasks:
                del self.active_tasks[agent_id]
    
    def get_active_agents(self) -> List[int]:
        """Get list of agents currently training."""
        with self._lock:
            return list(self.active_tasks.keys())
    
    def cancel_agent_training(self, agent_id: int) -> bool:
        """Cancel training for specific agent."""
        with self._lock:
            if agent_id in self.active_tasks:
                future = self.active_tasks[agent_id]
                return future.cancel()
        return False
    
    def shutdown(self, wait: bool = True) -> None:
        """Shutdown the concurrent trainer."""
        self.executor.shutdown(wait=wait)
        self.logger.info("Concurrent trainer shutdown")


class PerformanceOptimizer:
    """Optimizes performance based on runtime metrics."""
    
    def __init__(self):
        self.metrics_history = []
        self.optimization_history = []
        self.logger = get_logger("pg_neo_graph_rl.optimizer")
        
        # Current optimization settings
        self.current_settings = {
            "batch_size": 16,
            "learning_rate": 3e-4,
            "communication_frequency": 10,
            "gradient_accumulation_steps": 1
        }
    
    def record_metrics(self, 
                      episode: int,
                      training_time: float,
                      communication_time: float,
                      memory_usage: float,
                      convergence_rate: float) -> None:
        """Record performance metrics."""
        metrics = {
            "episode": episode,
            "training_time": training_time,
            "communication_time": communication_time,
            "memory_usage": memory_usage,
            "convergence_rate": convergence_rate,
            "timestamp": time.time()
        }
        
        self.metrics_history.append(metrics)
        
        # Keep only recent metrics
        if len(self.metrics_history) > 1000:
            self.metrics_history = self.metrics_history[-500:]
    
    def optimize(self) -> Dict[str, Any]:
        """Analyze metrics and suggest optimizations."""
        if len(self.metrics_history) < 10:
            return self.current_settings
        
        recent_metrics = self.metrics_history[-10:]
        avg_training_time = sum(m["training_time"] for m in recent_metrics) / len(recent_metrics)
        avg_memory_usage = sum(m["memory_usage"] for m in recent_metrics) / len(recent_metrics)
        avg_convergence = sum(m["convergence_rate"] for m in recent_metrics) / len(recent_metrics)
        
        new_settings = self.current_settings.copy()
        
        # Optimize batch size based on training time and memory
        if avg_training_time > 5.0 and avg_memory_usage < 0.7:
            # Training is slow but memory is available - increase batch size
            new_settings["batch_size"] = min(self.current_settings["batch_size"] * 2, 128)
            self.logger.info(f"Increased batch size to {new_settings['batch_size']}")
            
        elif avg_memory_usage > 0.9:
            # Memory pressure - reduce batch size
            new_settings["batch_size"] = max(self.current_settings["batch_size"] // 2, 4)
            self.logger.info(f"Reduced batch size to {new_settings['batch_size']}")
        
        # Optimize communication frequency based on convergence
        if avg_convergence < 0.01:  # Slow convergence
            # More frequent communication might help
            new_settings["communication_frequency"] = max(
                self.current_settings["communication_frequency"] - 2, 1
            )
        elif avg_convergence > 0.1:  # Fast convergence
            # Can reduce communication frequency
            new_settings["communication_frequency"] = min(
                self.current_settings["communication_frequency"] + 2, 50
            )
        
        # Update settings
        if new_settings != self.current_settings:
            self.optimization_history.append({
                "timestamp": time.time(),
                "old_settings": self.current_settings.copy(),
                "new_settings": new_settings.copy(),
                "reason": "performance_optimization"
            })
            
            self.current_settings = new_settings
        
        return self.current_settings
    
    def get_optimization_summary(self) -> Dict[str, Any]:
        """Get summary of optimizations performed."""
        if not self.metrics_history:
            return {"status": "no_data"}
        
        recent_metrics = self.metrics_history[-50:] if len(self.metrics_history) >= 50 else self.metrics_history
        
        return {
            "current_settings": self.current_settings,
            "total_optimizations": len(self.optimization_history),
            "avg_training_time": sum(m["training_time"] for m in recent_metrics) / len(recent_metrics),
            "avg_memory_usage": sum(m["memory_usage"] for m in recent_metrics) / len(recent_metrics),
            "recent_convergence_rate": sum(m["convergence_rate"] for m in recent_metrics) / len(recent_metrics),
            "optimization_history": self.optimization_history[-10:]  # Last 10 optimizations
        }