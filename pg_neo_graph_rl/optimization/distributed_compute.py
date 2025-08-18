"""Distributed computing infrastructure for federated learning at scale."""
import time
import asyncio
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
from typing import Dict, List, Any, Optional, Callable, Union, Tuple
from dataclasses import dataclass
import threading
import multiprocessing as mp
from queue import Queue
import numpy as np
import jax
import jax.numpy as jnp
from jax import pmap, vmap

from ..utils.logging import get_logger, get_performance_logger
from ..core.types import GraphState
from ..utils.exceptions import DistributedComputeError
from ..monitoring.advanced_metrics import AdvancedMetricsCollector

logger = get_logger(__name__)
perf_logger = get_performance_logger(__name__)

@dataclass
class ComputeNode:
    """Represents a compute node in the distributed system."""
    node_id: str
    cpu_count: int
    memory_gb: float
    gpu_count: int = 0
    load_factor: float = 0.0
    last_heartbeat: float = 0.0
    is_active: bool = True
    specialization: Optional[str] = None  # 'cpu', 'gpu', 'memory_intensive'

@dataclass
class ComputeTask:
    """Represents a compute task to be executed."""
    task_id: str
    task_type: str
    data: Any
    priority: int = 0
    estimated_duration: float = 0.0
    required_resources: Dict[str, Any] = None
    callback: Optional[Callable] = None

class DistributedComputeManager:
    """Manages distributed computation across multiple nodes."""
    
    def __init__(self, max_workers: int = None, enable_gpu: bool = True):
        self.max_workers = max_workers or mp.cpu_count()
        self.enable_gpu = enable_gpu and len(jax.devices('gpu')) > 0
        
        # Compute resources
        self.thread_pool = ThreadPoolExecutor(max_workers=self.max_workers)
        self.process_pool = ProcessPoolExecutor(max_workers=max(1, self.max_workers // 2))
        
        # Task management
        self.task_queue = Queue()
        self.active_tasks = {}
        self.completed_tasks = {}
        self.task_counter = 0
        
        # Node management
        self.compute_nodes = {}
        self.local_node = self._initialize_local_node()
        
        # Performance tracking
        self.metrics_collector = AdvancedMetricsCollector()
        
        # Load balancing
        self.load_balancer = LoadBalancer()
        
        # JAX setup for GPU acceleration
        if self.enable_gpu:
            self._setup_jax_distributed()
            
        logger.info(f"Distributed compute manager initialized with {self.max_workers} workers")
        logger.info(f"GPU acceleration: {'enabled' if self.enable_gpu else 'disabled'}")
    
    def _initialize_local_node(self) -> ComputeNode:
        """Initialize local compute node information."""
        try:
            import psutil
            
            memory_gb = psutil.virtual_memory().total / (1024**3)
            cpu_count = mp.cpu_count()
            gpu_count = len(jax.devices('gpu')) if self.enable_gpu else 0
            
            node = ComputeNode(
                node_id="local",
                cpu_count=cpu_count,
                memory_gb=memory_gb,
                gpu_count=gpu_count,
                last_heartbeat=time.time()
            )
            
            self.compute_nodes["local"] = node
            return node
            
        except ImportError:
            logger.warning("psutil not available, using default node config")
            return ComputeNode(
                node_id="local",
                cpu_count=self.max_workers,
                memory_gb=8.0,
                gpu_count=1 if self.enable_gpu else 0,
                last_heartbeat=time.time()
            )
    
    def _setup_jax_distributed(self) -> None:
        """Setup JAX for distributed/multi-GPU computation."""
        try:
            devices = jax.devices()
            logger.info(f"JAX devices available: {devices}")
            
            # Setup for multi-device computation
            self.device_count = len(devices)
            if self.device_count > 1:
                logger.info(f"Multi-device computation enabled with {self.device_count} devices")
                
        except Exception as e:
            logger.warning(f"JAX distributed setup error: {e}")
    
    def submit_task(self, task_type: str, data: Any, priority: int = 0, 
                   node_preference: str = None) -> str:
        """Submit a compute task for execution."""
        task_id = f"task_{self.task_counter}"
        self.task_counter += 1
        
        task = ComputeTask(
            task_id=task_id,
            task_type=task_type,
            data=data,
            priority=priority,
            estimated_duration=self._estimate_task_duration(task_type, data)
        )
        
        # Select optimal compute node
        target_node = self._select_compute_node(task, node_preference)
        
        # Execute task
        if task_type in ['graph_forward', 'gradient_computation', 'parameter_update']:
            future = self._execute_ml_task(task, target_node)
        elif task_type in ['data_processing', 'metric_aggregation']:
            future = self._execute_data_task(task, target_node)
        else:
            future = self._execute_generic_task(task, target_node)
        
        self.active_tasks[task_id] = {
            'task': task,
            'future': future,
            'node': target_node,
            'start_time': time.time()
        }
        
        logger.debug(f"Submitted task {task_id} to node {target_node}")
        return task_id
    
    def _select_compute_node(self, task: ComputeTask, preference: str = None) -> str:
        """Select optimal compute node for task execution."""
        if preference and preference in self.compute_nodes:
            if self.compute_nodes[preference].is_active:
                return preference
        
        # Use load balancer to select node
        return self.load_balancer.select_node(
            task, 
            list(self.compute_nodes.values())
        )
    
    def _estimate_task_duration(self, task_type: str, data: Any) -> float:
        """Estimate task execution duration."""
        base_times = {
            'graph_forward': 0.01,
            'gradient_computation': 0.05,
            'parameter_update': 0.02,
            'data_processing': 0.001,
            'metric_aggregation': 0.005
        }
        
        base_time = base_times.get(task_type, 0.01)
        
        # Adjust based on data size
        if isinstance(data, (list, tuple)):
            size_factor = len(data) / 1000
        elif hasattr(data, 'shape'):
            size_factor = np.prod(data.shape) / 10000
        else:
            size_factor = 1.0
            
        return base_time * (1 + size_factor)
    
    def _execute_ml_task(self, task: ComputeTask, node: str) -> Any:
        """Execute machine learning task with GPU acceleration if available."""
        if self.enable_gpu and task.task_type in ['graph_forward', 'gradient_computation']:
            return self.thread_pool.submit(self._gpu_ml_task, task)
        else:
            return self.process_pool.submit(self._cpu_ml_task, task)
    
    def _execute_data_task(self, task: ComputeTask, node: str) -> Any:
        """Execute data processing task."""
        return self.thread_pool.submit(self._data_processing_task, task)
    
    def _execute_generic_task(self, task: ComputeTask, node: str) -> Any:
        """Execute generic compute task."""
        return self.thread_pool.submit(self._generic_task, task)
    
    def _gpu_ml_task(self, task: ComputeTask) -> Any:
        """Execute ML task on GPU."""
        start_time = time.time()
        
        try:
            if task.task_type == 'graph_forward':
                result = self._gpu_graph_forward(task.data)
            elif task.task_type == 'gradient_computation':
                result = self._gpu_gradient_computation(task.data)
            else:
                result = self._cpu_ml_task(task)  # Fallback
                
            duration = time.time() - start_time
            self.metrics_collector.record('gpu_task_duration', duration)
            
            return result
            
        except Exception as e:
            logger.error(f"GPU task {task.task_id} failed: {e}")
            # Fallback to CPU
            return self._cpu_ml_task(task)
    
    def _gpu_graph_forward(self, data: Dict[str, Any]) -> jnp.ndarray:
        """GPU-accelerated graph forward pass."""
        @jax.jit
        def forward_pass(nodes, edges, adjacency):
            # Simple graph convolution
            degree = jnp.sum(adjacency, axis=1, keepdims=True)
            degree = jnp.maximum(degree, 1.0)
            
            # Message passing
            messages = jnp.dot(adjacency, nodes) / degree
            
            # Simple MLP layer
            hidden = jax.nn.relu(messages @ jnp.ones((nodes.shape[1], 64)))
            output = hidden @ jnp.ones((64, nodes.shape[1]))
            
            return output
        
        nodes = data['nodes']
        adjacency = data['adjacency']
        edges = data.get('edges', jnp.array([]))
        
        return forward_pass(nodes, edges, adjacency)
    
    def _gpu_gradient_computation(self, data: Dict[str, Any]) -> Dict[str, jnp.ndarray]:
        """GPU-accelerated gradient computation."""
        @jax.jit
        def compute_gradients(params, inputs, targets):
            def loss_fn(p):
                # Simple loss computation
                pred = inputs @ p
                return jnp.mean((pred - targets) ** 2)
            
            return jax.grad(loss_fn)(params)
        
        params = data['parameters']
        inputs = data['inputs']
        targets = data['targets']
        
        gradients = compute_gradients(params, inputs, targets)
        return {'gradients': gradients}
    
    def _cpu_ml_task(self, task: ComputeTask) -> Any:
        """Execute ML task on CPU."""
        start_time = time.time()
        
        try:
            if task.task_type == 'graph_forward':
                result = self._cpu_graph_forward(task.data)
            elif task.task_type == 'gradient_computation':
                result = self._cpu_gradient_computation(task.data)
            elif task.task_type == 'parameter_update':
                result = self._cpu_parameter_update(task.data)
            else:
                result = {'status': 'unknown_task_type'}
            
            duration = time.time() - start_time
            self.metrics_collector.record('cpu_task_duration', duration)
            
            return result
            
        except Exception as e:
            logger.error(f"CPU ML task {task.task_id} failed: {e}")
            raise DistributedComputeError(f"ML task failed: {e}")
    
    def _cpu_graph_forward(self, data: Dict[str, Any]) -> np.ndarray:
        """CPU graph forward pass."""
        nodes = np.array(data['nodes'])
        adjacency = np.array(data['adjacency'])
        
        # Simple graph convolution
        degree = np.sum(adjacency, axis=1, keepdims=True)
        degree = np.maximum(degree, 1.0)
        
        # Message passing
        messages = np.dot(adjacency, nodes) / degree
        
        # Apply non-linearity
        output = np.maximum(0, messages)  # ReLU activation
        
        return output
    
    def _cpu_gradient_computation(self, data: Dict[str, Any]) -> Dict[str, np.ndarray]:
        """CPU gradient computation."""
        # Simplified gradient computation
        parameters = data['parameters']
        inputs = data['inputs']
        targets = data['targets']
        
        # Simple linear regression gradient
        predictions = np.dot(inputs, parameters)
        error = predictions - targets
        gradients = np.dot(inputs.T, error) / len(targets)
        
        return {'gradients': gradients}
    
    def _cpu_parameter_update(self, data: Dict[str, Any]) -> Dict[str, np.ndarray]:
        """CPU parameter update."""
        parameters = data['parameters']
        gradients = data['gradients']
        learning_rate = data.get('learning_rate', 0.01)
        
        updated_params = parameters - learning_rate * gradients
        
        return {'parameters': updated_params}
    
    def _data_processing_task(self, task: ComputeTask) -> Any:
        """Execute data processing task."""
        start_time = time.time()
        
        try:
            if task.task_type == 'metric_aggregation':
                result = self._aggregate_metrics(task.data)
            elif task.task_type == 'data_processing':
                result = self._process_data(task.data)
            else:
                result = {'status': 'processed'}
            
            duration = time.time() - start_time
            self.metrics_collector.record('data_task_duration', duration)
            
            return result
            
        except Exception as e:
            logger.error(f"Data task {task.task_id} failed: {e}")
            raise DistributedComputeError(f"Data task failed: {e}")
    
    def _aggregate_metrics(self, data: List[Dict[str, float]]) -> Dict[str, float]:
        """Aggregate metrics from multiple sources."""
        if not data:
            return {}
        
        # Collect all metric names
        all_keys = set()
        for item in data:
            all_keys.update(item.keys())
        
        # Aggregate each metric
        aggregated = {}
        for key in all_keys:
            values = [item[key] for item in data if key in item]
            if values:
                aggregated[key] = {
                    'mean': np.mean(values),
                    'std': np.std(values),
                    'min': np.min(values),
                    'max': np.max(values),
                    'count': len(values)
                }
        
        return aggregated
    
    def _process_data(self, data: Any) -> Any:
        """Generic data processing."""
        if isinstance(data, list):
            # Process list items in parallel
            return [self._process_single_item(item) for item in data]
        else:
            return self._process_single_item(data)
    
    def _process_single_item(self, item: Any) -> Any:
        """Process a single data item."""
        if isinstance(item, dict) and 'transform' in item:
            # Apply transformation
            transform = item['transform']
            data = item['data']
            
            if transform == 'normalize':
                return (data - np.mean(data)) / (np.std(data) + 1e-8)
            elif transform == 'scale':
                factor = item.get('factor', 1.0)
                return data * factor
            else:
                return data
        
        return item
    
    def _generic_task(self, task: ComputeTask) -> Any:
        """Execute generic compute task."""
        start_time = time.time()
        
        try:
            # Generic task execution
            result = {'task_id': task.task_id, 'status': 'completed', 'data': task.data}
            
            duration = time.time() - start_time
            self.metrics_collector.record('generic_task_duration', duration)
            
            return result
            
        except Exception as e:
            logger.error(f"Generic task {task.task_id} failed: {e}")
            raise DistributedComputeError(f"Generic task failed: {e}")
    
    def get_task_result(self, task_id: str, timeout: float = None) -> Any:
        """Get result of a submitted task."""
        if task_id not in self.active_tasks:
            if task_id in self.completed_tasks:
                return self.completed_tasks[task_id]['result']
            else:
                raise ValueError(f"Task {task_id} not found")
        
        task_info = self.active_tasks[task_id]
        future = task_info['future']
        
        try:
            result = future.result(timeout=timeout)
            
            # Move to completed tasks
            self.completed_tasks[task_id] = {
                'task': task_info['task'],
                'result': result,
                'duration': time.time() - task_info['start_time'],
                'node': task_info['node']
            }
            
            del self.active_tasks[task_id]
            
            return result
            
        except Exception as e:
            logger.error(f"Task {task_id} execution failed: {e}")
            del self.active_tasks[task_id]
            raise DistributedComputeError(f"Task execution failed: {e}")
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get current system status."""
        return {
            'active_tasks': len(self.active_tasks),
            'completed_tasks': len(self.completed_tasks),
            'compute_nodes': len(self.compute_nodes),
            'thread_pool_active': self.thread_pool._threads,
            'process_pool_active': len(self.process_pool._processes) if hasattr(self.process_pool, '_processes') else 0,
            'gpu_enabled': self.enable_gpu,
            'max_workers': self.max_workers
        }
    
    def shutdown(self) -> None:
        """Shutdown distributed compute manager."""
        logger.info("Shutting down distributed compute manager...")
        
        # Wait for active tasks to complete (with timeout)
        for task_id in list(self.active_tasks.keys()):
            try:
                self.get_task_result(task_id, timeout=30.0)
            except Exception as e:
                logger.warning(f"Task {task_id} did not complete cleanly: {e}")
        
        # Shutdown thread and process pools
        self.thread_pool.shutdown(wait=True)
        self.process_pool.shutdown(wait=True)
        
        # Shutdown metrics collector
        if hasattr(self.metrics_collector, 'shutdown'):
            self.metrics_collector.shutdown()
        
        logger.info("Distributed compute manager shutdown complete")

class LoadBalancer:
    """Load balancer for distributed compute nodes."""
    
    def __init__(self, algorithm: str = 'least_loaded'):
        self.algorithm = algorithm
        self.node_loads = {}
        self.last_selection = {}
    
    def select_node(self, task: ComputeTask, available_nodes: List[ComputeNode]) -> str:
        """Select optimal node for task execution."""
        if not available_nodes:
            raise DistributedComputeError("No available compute nodes")
        
        active_nodes = [node for node in available_nodes if node.is_active]
        if not active_nodes:
            raise DistributedComputeError("No active compute nodes")
        
        if self.algorithm == 'least_loaded':
            return self._select_least_loaded(active_nodes)
        elif self.algorithm == 'round_robin':
            return self._select_round_robin(active_nodes)
        elif self.algorithm == 'resource_aware':
            return self._select_resource_aware(task, active_nodes)
        else:
            # Default to first available node
            return active_nodes[0].node_id
    
    def _select_least_loaded(self, nodes: List[ComputeNode]) -> str:
        """Select node with lowest load factor."""
        min_load = float('inf')
        selected_node = nodes[0].node_id
        
        for node in nodes:
            if node.load_factor < min_load:
                min_load = node.load_factor
                selected_node = node.node_id
        
        return selected_node
    
    def _select_round_robin(self, nodes: List[ComputeNode]) -> str:
        """Select node using round-robin algorithm."""
        node_ids = [node.node_id for node in nodes]
        last_idx = self.last_selection.get('round_robin', -1)
        next_idx = (last_idx + 1) % len(node_ids)
        self.last_selection['round_robin'] = next_idx
        return node_ids[next_idx]
    
    def _select_resource_aware(self, task: ComputeTask, nodes: List[ComputeNode]) -> str:
        """Select node based on resource requirements."""
        # Score nodes based on available resources
        best_score = -1
        selected_node = nodes[0].node_id
        
        for node in nodes:
            score = self._calculate_node_score(task, node)
            if score > best_score:
                best_score = score
                selected_node = node.node_id
        
        return selected_node
    
    def _calculate_node_score(self, task: ComputeTask, node: ComputeNode) -> float:
        """Calculate compatibility score between task and node."""
        score = 0.0
        
        # Prefer nodes with lower load
        score += (1.0 - node.load_factor) * 0.4
        
        # Task-specific scoring
        if task.task_type in ['graph_forward', 'gradient_computation'] and node.gpu_count > 0:
            score += 0.3  # GPU tasks prefer GPU nodes
        
        if task.task_type == 'data_processing' and node.memory_gb > 16:
            score += 0.2  # Memory-intensive tasks prefer high-memory nodes
        
        # CPU capacity
        score += min(node.cpu_count / 8.0, 1.0) * 0.1
        
        return score
    
    def update_node_load(self, node_id: str, load_factor: float) -> None:
        """Update load factor for a node."""
        self.node_loads[node_id] = load_factor

# Global distributed compute manager instance
_global_compute_manager = None

def get_distributed_compute_manager(**kwargs) -> DistributedComputeManager:
    """Get or create global distributed compute manager."""
    global _global_compute_manager
    if _global_compute_manager is None:
        _global_compute_manager = DistributedComputeManager(**kwargs)
    return _global_compute_manager