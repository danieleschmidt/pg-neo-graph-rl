"""
Auto-scaling and load balancing for federated graph RL systems.
Dynamically adjusts resources based on workload and performance metrics.
"""

import asyncio
import time
import threading
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict, deque
import math

import jax.numpy as jnp

from ..utils.exceptions import ResourceError, GraphRLError
from ..utils.logging import get_logger
from ..monitoring.alerting import HealthMonitor, MetricThreshold

logger = get_logger(__name__)


class ScalingDirection(Enum):
    """Scaling direction options."""
    UP = "up"
    DOWN = "down"
    STABLE = "stable"


class LoadBalancingStrategy(Enum):
    """Load balancing strategies."""
    ROUND_ROBIN = "round_robin"
    LEAST_LOADED = "least_loaded"
    WEIGHTED_ROUND_ROBIN = "weighted_round_robin"
    HASH_BASED = "hash_based"
    ADAPTIVE = "adaptive"


@dataclass
class WorkerNode:
    """Represents a worker node in the system."""
    worker_id: str
    capacity: int
    current_load: int = 0
    last_heartbeat: float = field(default_factory=time.time)
    active: bool = True
    performance_score: float = 1.0
    weight: float = 1.0
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def load_percentage(self) -> float:
        """Calculate load as percentage of capacity."""
        return (self.current_load / self.capacity) * 100 if self.capacity > 0 else 0
    
    @property
    def available_capacity(self) -> int:
        """Calculate available capacity."""
        return max(0, self.capacity - self.current_load)
    
    def is_healthy(self, heartbeat_timeout: float = 60.0) -> bool:
        """Check if worker is healthy based on heartbeat."""
        return (time.time() - self.last_heartbeat) < heartbeat_timeout and self.active


@dataclass
class ScalingMetrics:
    """Metrics for auto-scaling decisions."""
    avg_load_percentage: float
    max_load_percentage: float
    queue_length: int
    response_time_p95: float
    throughput: float
    error_rate: float
    timestamp: float = field(default_factory=time.time)


@dataclass
class ScalingRule:
    """Rule for auto-scaling decisions."""
    name: str
    metric_name: str
    threshold_up: float
    threshold_down: float
    min_workers: int
    max_workers: int
    cooldown_seconds: float = 300.0
    evaluation_window: int = 5
    enabled: bool = True


class AutoScaler:
    """
    Automatic scaling system for federated learning workers.
    """
    
    def __init__(self, 
                 initial_workers: int = 2,
                 min_workers: int = 1,
                 max_workers: int = 10,
                 target_load_percentage: float = 70.0):
        self.min_workers = min_workers
        self.max_workers = max_workers
        self.target_load_percentage = target_load_percentage
        
        # Worker management
        self.workers: Dict[str, WorkerNode] = {}
        self.worker_counter = 0
        self.lock = threading.RLock()
        
        # Metrics and monitoring
        self.metrics_history: deque = deque(maxlen=100)
        self.scaling_history: List[Dict[str, Any]] = []
        self.last_scaling_time = 0
        
        # Scaling rules
        self.scaling_rules: List[ScalingRule] = []
        self._setup_default_scaling_rules()
        
        # Callbacks
        self.scale_up_callback: Optional[Callable[[int], List[str]]] = None
        self.scale_down_callback: Optional[Callable[[List[str]], None]] = None
        
        # Monitoring
        self.is_running = False
        self.monitor_interval = 30.0  # seconds
        
        logger.info(f"AutoScaler initialized: min={min_workers}, max={max_workers}, target_load={target_load_percentage}%")
    
    def _setup_default_scaling_rules(self):
        """Setup default scaling rules."""
        self.scaling_rules = [
            ScalingRule(
                name="load_based_scaling",
                metric_name="avg_load_percentage",
                threshold_up=80.0,
                threshold_down=50.0,
                min_workers=self.min_workers,
                max_workers=self.max_workers,
                cooldown_seconds=300.0
            ),
            ScalingRule(
                name="queue_based_scaling",
                metric_name="queue_length",
                threshold_up=10.0,
                threshold_down=2.0,
                min_workers=self.min_workers,
                max_workers=self.max_workers,
                cooldown_seconds=180.0
            ),
            ScalingRule(
                name="response_time_scaling",
                metric_name="response_time_p95",
                threshold_up=5.0,  # 5 seconds
                threshold_down=1.0,  # 1 second
                min_workers=self.min_workers,
                max_workers=self.max_workers,
                cooldown_seconds=240.0
            )
        ]
    
    def add_worker(self, 
                   worker_id: str, 
                   capacity: int = 100,
                   weight: float = 1.0,
                   metadata: Optional[Dict[str, Any]] = None) -> None:
        """Add a worker to the pool."""
        with self.lock:
            worker = WorkerNode(
                worker_id=worker_id,
                capacity=capacity,
                weight=weight,
                metadata=metadata or {}
            )
            self.workers[worker_id] = worker
            logger.info(f"Added worker: {worker_id} (capacity: {capacity})")
    
    def remove_worker(self, worker_id: str) -> bool:
        """Remove a worker from the pool."""
        with self.lock:
            if worker_id in self.workers:
                del self.workers[worker_id]
                logger.info(f"Removed worker: {worker_id}")
                return True
            return False
    
    def update_worker_load(self, worker_id: str, current_load: int) -> None:
        """Update worker's current load."""
        with self.lock:
            if worker_id in self.workers:
                self.workers[worker_id].current_load = current_load
                self.workers[worker_id].last_heartbeat = time.time()
    
    def update_worker_performance(self, worker_id: str, performance_score: float) -> None:
        """Update worker's performance score."""
        with self.lock:
            if worker_id in self.workers:
                self.workers[worker_id].performance_score = performance_score
    
    def get_scaling_metrics(self) -> ScalingMetrics:
        """Calculate current scaling metrics."""
        with self.lock:
            active_workers = [w for w in self.workers.values() if w.is_healthy()]
            
            if not active_workers:
                return ScalingMetrics(
                    avg_load_percentage=0.0,
                    max_load_percentage=0.0,
                    queue_length=0,
                    response_time_p95=0.0,
                    throughput=0.0,
                    error_rate=0.0
                )
            
            # Calculate load metrics
            load_percentages = [w.load_percentage for w in active_workers]
            avg_load = sum(load_percentages) / len(load_percentages)
            max_load = max(load_percentages) if load_percentages else 0.0
            
            # Queue length (simplified - would be actual queue in real implementation)
            total_queue = sum(max(0, w.current_load - w.capacity) for w in active_workers)
            
            # Performance metrics (simplified - would be from actual monitoring)
            avg_performance = sum(w.performance_score for w in active_workers) / len(active_workers)
            response_time_p95 = 2.0 / avg_performance if avg_performance > 0 else 5.0
            throughput = sum(w.capacity * w.performance_score for w in active_workers)
            error_rate = max(0.0, 1.0 - avg_performance) * 100
            
            return ScalingMetrics(
                avg_load_percentage=avg_load,
                max_load_percentage=max_load,
                queue_length=total_queue,
                response_time_p95=response_time_p95,
                throughput=throughput,
                error_rate=error_rate
            )
    
    def evaluate_scaling_decision(self, metrics: ScalingMetrics) -> Tuple[ScalingDirection, int, str]:
        """
        Evaluate whether scaling is needed.
        
        Returns:
            Tuple of (direction, amount, reason)
        """
        current_time = time.time()
        active_workers = len([w for w in self.workers.values() if w.is_healthy()])
        
        # Check cooldown period
        if current_time - self.last_scaling_time < 180.0:  # 3 minute global cooldown
            return ScalingDirection.STABLE, 0, "Cooldown period active"
        
        # Evaluate each scaling rule
        for rule in self.scaling_rules:
            if not rule.enabled:
                continue
            
            # Check rule-specific cooldown
            if current_time - self.last_scaling_time < rule.cooldown_seconds:
                continue
            
            metric_value = getattr(metrics, rule.metric_name, 0.0)
            
            # Scale up conditions
            if (metric_value > rule.threshold_up and 
                active_workers < rule.max_workers):
                
                # Calculate scale amount based on metric severity
                severity = (metric_value - rule.threshold_up) / rule.threshold_up
                scale_amount = max(1, min(3, int(severity * 2 + 1)))  # 1-3 workers
                scale_amount = min(scale_amount, rule.max_workers - active_workers)
                
                return (
                    ScalingDirection.UP, 
                    scale_amount,
                    f"Rule '{rule.name}': {rule.metric_name}={metric_value:.2f} > {rule.threshold_up}"
                )
            
            # Scale down conditions
            elif (metric_value < rule.threshold_down and 
                  active_workers > rule.min_workers):
                
                # Calculate scale amount (more conservative for scale down)
                scale_amount = 1  # Always scale down by 1
                scale_amount = min(scale_amount, active_workers - rule.min_workers)
                
                return (
                    ScalingDirection.DOWN,
                    scale_amount,
                    f"Rule '{rule.name}': {rule.metric_name}={metric_value:.2f} < {rule.threshold_down}"
                )
        
        return ScalingDirection.STABLE, 0, "No scaling rules triggered"
    
    def execute_scaling(self, direction: ScalingDirection, amount: int, reason: str) -> bool:
        """Execute scaling decision."""
        if direction == ScalingDirection.STABLE or amount == 0:
            return True
        
        current_time = time.time()
        
        try:
            if direction == ScalingDirection.UP:
                # Scale up
                new_workers = []
                for i in range(amount):
                    worker_id = f"auto_worker_{self.worker_counter}"
                    self.worker_counter += 1
                    
                    # Call scale up callback if available
                    if self.scale_up_callback:
                        callback_workers = self.scale_up_callback(1)
                        if callback_workers:
                            worker_id = callback_workers[0]
                    
                    self.add_worker(worker_id, capacity=100)
                    new_workers.append(worker_id)
                
                logger.info(f"Scaled UP by {amount} workers: {new_workers}. Reason: {reason}")
                
            elif direction == ScalingDirection.DOWN:
                # Scale down - remove least loaded workers
                with self.lock:
                    workers_to_remove = sorted(
                        [(w.worker_id, w.load_percentage) for w in self.workers.values() if w.is_healthy()],
                        key=lambda x: x[1]  # Sort by load percentage
                    )[:amount]
                
                removed_workers = []
                for worker_id, _ in workers_to_remove:
                    # Call scale down callback if available
                    if self.scale_down_callback:
                        self.scale_down_callback([worker_id])
                    
                    if self.remove_worker(worker_id):
                        removed_workers.append(worker_id)
                
                logger.info(f"Scaled DOWN by {len(removed_workers)} workers: {removed_workers}. Reason: {reason}")
            
            # Record scaling event
            self.last_scaling_time = current_time
            self.scaling_history.append({
                "timestamp": current_time,
                "direction": direction.value,
                "amount": amount,
                "reason": reason,
                "workers_before": len(self.workers),
                "workers_after": len([w for w in self.workers.values() if w.is_healthy()])
            })
            
            return True
            
        except Exception as e:
            logger.error(f"Scaling execution failed: {e}")
            return False
    
    async def start_monitoring(self):
        """Start the auto-scaling monitoring loop."""
        self.is_running = True
        logger.info("Starting auto-scaling monitoring")
        
        while self.is_running:
            try:
                # Get current metrics
                metrics = self.get_scaling_metrics()
                self.metrics_history.append(metrics)
                
                # Evaluate scaling decision
                direction, amount, reason = self.evaluate_scaling_decision(metrics)
                
                # Execute scaling if needed
                if direction != ScalingDirection.STABLE:
                    self.execute_scaling(direction, amount, reason)
                
                # Log current status
                active_workers = len([w for w in self.workers.values() if w.is_healthy()])
                logger.debug(
                    f"Auto-scaling status: {active_workers} workers, "
                    f"avg_load={metrics.avg_load_percentage:.1f}%, "
                    f"decision={direction.value}"
                )
                
                await asyncio.sleep(self.monitor_interval)
                
            except Exception as e:
                logger.error(f"Auto-scaling monitoring error: {e}")
                await asyncio.sleep(10)  # Brief pause on error
    
    def stop_monitoring(self):
        """Stop auto-scaling monitoring."""
        self.is_running = False
        logger.info("Stopped auto-scaling monitoring")
    
    def get_status(self) -> Dict[str, Any]:
        """Get comprehensive auto-scaling status."""
        with self.lock:
            active_workers = [w for w in self.workers.values() if w.is_healthy()]
            
            return {
                "total_workers": len(self.workers),
                "active_workers": len(active_workers),
                "worker_details": {
                    w.worker_id: {
                        "capacity": w.capacity,
                        "current_load": w.current_load,
                        "load_percentage": w.load_percentage,
                        "performance_score": w.performance_score,
                        "healthy": w.is_healthy()
                    }
                    for w in self.workers.values()
                },
                "current_metrics": self.get_scaling_metrics(),
                "scaling_rules": [
                    {
                        "name": rule.name,
                        "metric": rule.metric_name,
                        "threshold_up": rule.threshold_up,
                        "threshold_down": rule.threshold_down,
                        "enabled": rule.enabled
                    }
                    for rule in self.scaling_rules
                ],
                "recent_scaling_events": self.scaling_history[-5:],
                "is_monitoring": self.is_running
            }


class LoadBalancer:
    """
    Load balancer for distributing work across federated learning workers.
    """
    
    def __init__(self, 
                 strategy: LoadBalancingStrategy = LoadBalancingStrategy.ADAPTIVE,
                 health_check_interval: float = 30.0):
        self.strategy = strategy
        self.health_check_interval = health_check_interval
        
        # Worker management
        self.workers: Dict[str, WorkerNode] = {}
        self.lock = threading.RLock()
        
        # Round robin state
        self.round_robin_index = 0
        
        # Adaptive strategy state
        self.worker_response_times: Dict[str, deque] = defaultdict(lambda: deque(maxlen=10))
        self.worker_error_counts: Dict[str, int] = defaultdict(int)
        
        logger.info(f"LoadBalancer initialized with strategy: {strategy.value}")
    
    def add_worker(self, worker: WorkerNode) -> None:
        """Add worker to load balancer."""
        with self.lock:
            self.workers[worker.worker_id] = worker
            logger.info(f"Added worker to load balancer: {worker.worker_id}")
    
    def remove_worker(self, worker_id: str) -> bool:
        """Remove worker from load balancer."""
        with self.lock:
            if worker_id in self.workers:
                del self.workers[worker_id]
                logger.info(f"Removed worker from load balancer: {worker_id}")
                return True
            return False
    
    def select_worker(self, task_key: Optional[str] = None) -> Optional[WorkerNode]:
        """Select optimal worker based on strategy."""
        with self.lock:
            healthy_workers = [w for w in self.workers.values() if w.is_healthy()]
            
            if not healthy_workers:
                return None
            
            if self.strategy == LoadBalancingStrategy.ROUND_ROBIN:
                return self._round_robin_selection(healthy_workers)
            
            elif self.strategy == LoadBalancingStrategy.LEAST_LOADED:
                return self._least_loaded_selection(healthy_workers)
            
            elif self.strategy == LoadBalancingStrategy.WEIGHTED_ROUND_ROBIN:
                return self._weighted_round_robin_selection(healthy_workers)
            
            elif self.strategy == LoadBalancingStrategy.HASH_BASED:
                return self._hash_based_selection(healthy_workers, task_key)
            
            elif self.strategy == LoadBalancingStrategy.ADAPTIVE:
                return self._adaptive_selection(healthy_workers)
            
            else:
                return healthy_workers[0]  # Fallback
    
    def _round_robin_selection(self, workers: List[WorkerNode]) -> WorkerNode:
        """Round robin worker selection."""
        worker = workers[self.round_robin_index % len(workers)]
        self.round_robin_index += 1
        return worker
    
    def _least_loaded_selection(self, workers: List[WorkerNode]) -> WorkerNode:
        """Select worker with lowest load percentage."""
        return min(workers, key=lambda w: w.load_percentage)
    
    def _weighted_round_robin_selection(self, workers: List[WorkerNode]) -> WorkerNode:
        """Weighted round robin based on worker weights."""
        # Create weighted list
        weighted_workers = []
        for worker in workers:
            weight = max(1, int(worker.weight * 10))  # Scale weights
            weighted_workers.extend([worker] * weight)
        
        if weighted_workers:
            worker = weighted_workers[self.round_robin_index % len(weighted_workers)]
            self.round_robin_index += 1
            return worker
        
        return workers[0]
    
    def _hash_based_selection(self, workers: List[WorkerNode], task_key: Optional[str]) -> WorkerNode:
        """Hash-based consistent worker selection."""
        if task_key is None:
            return workers[0]
        
        # Simple hash-based selection
        hash_value = hash(task_key) % len(workers)
        return workers[hash_value]
    
    def _adaptive_selection(self, workers: List[WorkerNode]) -> WorkerNode:
        """Adaptive selection based on performance metrics."""
        # Calculate composite scores for each worker
        scores = []
        
        for worker in workers:
            # Base score from load (lower load = higher score)
            load_score = max(0, 100 - worker.load_percentage) / 100
            
            # Performance score
            perf_score = worker.performance_score
            
            # Response time score (average of recent response times)
            response_times = list(self.worker_response_times[worker.worker_id])
            if response_times:
                avg_response_time = sum(response_times) / len(response_times)
                response_score = max(0, 1.0 - (avg_response_time / 10.0))  # Normalize to 0-1
            else:
                response_score = 1.0
            
            # Error rate score
            error_count = self.worker_error_counts[worker.worker_id]
            error_score = max(0, 1.0 - (error_count / 100.0))  # Normalize errors
            
            # Composite score (weighted combination)
            composite_score = (
                load_score * 0.3 +
                perf_score * 0.3 +
                response_score * 0.25 +
                error_score * 0.15
            )
            
            scores.append((worker, composite_score))
        
        # Select worker with highest score
        return max(scores, key=lambda x: x[1])[0]
    
    def record_response_time(self, worker_id: str, response_time: float) -> None:
        """Record response time for adaptive selection."""
        if worker_id in self.workers:
            self.worker_response_times[worker_id].append(response_time)
    
    def record_error(self, worker_id: str) -> None:
        """Record error for adaptive selection."""
        if worker_id in self.workers:
            self.worker_error_counts[worker_id] += 1
    
    def get_load_distribution(self) -> Dict[str, float]:
        """Get current load distribution across workers."""
        with self.lock:
            return {
                worker_id: worker.load_percentage
                for worker_id, worker in self.workers.items()
                if worker.is_healthy()
            }


def create_auto_scaling_system(min_workers: int = 2, 
                              max_workers: int = 10) -> Tuple[AutoScaler, LoadBalancer]:
    """
    Create complete auto-scaling system with load balancer.
    
    Args:
        min_workers: Minimum number of workers
        max_workers: Maximum number of workers
        
    Returns:
        Tuple of (AutoScaler, LoadBalancer)
    """
    auto_scaler = AutoScaler(
        min_workers=min_workers,
        max_workers=max_workers,
        target_load_percentage=70.0
    )
    
    load_balancer = LoadBalancer(strategy=LoadBalancingStrategy.ADAPTIVE)
    
    # Initialize with minimum workers
    for i in range(min_workers):
        worker_id = f"initial_worker_{i}"
        worker = WorkerNode(worker_id=worker_id, capacity=100)
        auto_scaler.add_worker(worker_id, capacity=100)
        load_balancer.add_worker(worker)
    
    logger.info(f"Auto-scaling system created with {min_workers} initial workers")
    return auto_scaler, load_balancer