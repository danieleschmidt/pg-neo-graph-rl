"""Advanced metrics collection and analysis for federated learning."""
import threading
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

import jax.numpy as jnp
import numpy as np

from ..core.types import GraphState
from ..utils.logging import get_logger, get_performance_logger

logger = get_logger(__name__)
perf_logger = get_performance_logger(__name__)

@dataclass
class MetricSnapshot:
    """Single metric measurement with metadata."""
    name: str
    value: float
    timestamp: float
    agent_id: Optional[int] = None
    episode: Optional[int] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class AggregatedMetrics:
    """Statistical summary of metric values."""
    mean: float
    std: float
    min: float
    max: float
    median: float
    count: int
    percentiles: Dict[int, float] = field(default_factory=dict)

class AdvancedMetricsCollector:
    """Real-time metrics collection with statistical analysis."""

    def __init__(self, window_size: int = 1000, enable_persistence: bool = True):
        self.window_size = window_size
        self.enable_persistence = enable_persistence

        # Thread-safe metric storage
        self._metrics = defaultdict(lambda: deque(maxlen=window_size))
        self._lock = threading.RLock()

        # Performance tracking
        self._start_time = time.time()
        self._episode_start_times = {}

        # Alert thresholds
        self._alert_thresholds = {}
        self._alert_callbacks = []

        # Background analysis thread
        self._analysis_thread = None
        self._stop_analysis = threading.Event()

        if enable_persistence:
            self._start_background_analysis()

    def record(self, name: str, value: float, agent_id: Optional[int] = None,
               episode: Optional[int] = None, **metadata) -> None:
        """Record a metric value."""
        snapshot = MetricSnapshot(
            name=name,
            value=value,
            timestamp=time.time(),
            agent_id=agent_id,
            episode=episode,
            metadata=metadata
        )

        with self._lock:
            self._metrics[name].append(snapshot)

        # Check for alerts
        self._check_alerts(name, value)

        # Log high-frequency metrics at debug level
        if name in ['step_reward', 'loss', 'gradient_norm']:
            perf_logger.debug(f"Metric {name}: {value:.4f}")
        else:
            logger.info(f"Recorded {name}: {value:.4f}")

    def get_recent_values(self, name: str, n: int = None) -> List[float]:
        """Get recent values for a metric."""
        with self._lock:
            snapshots = list(self._metrics[name])

        if n is not None:
            snapshots = snapshots[-n:]

        return [s.value for s in snapshots]

    def get_aggregated(self, name: str, n: int = None) -> Optional[AggregatedMetrics]:
        """Get statistical summary of recent metric values."""
        values = self.get_recent_values(name, n)

        if not values:
            return None

        values_array = np.array(values)

        return AggregatedMetrics(
            mean=float(np.mean(values_array)),
            std=float(np.std(values_array)),
            min=float(np.min(values_array)),
            max=float(np.max(values_array)),
            median=float(np.median(values_array)),
            count=len(values),
            percentiles={
                25: float(np.percentile(values_array, 25)),
                75: float(np.percentile(values_array, 75)),
                90: float(np.percentile(values_array, 90)),
                95: float(np.percentile(values_array, 95)),
                99: float(np.percentile(values_array, 99))
            }
        )

    def get_trend(self, name: str, window: int = 100) -> Dict[str, float]:
        """Analyze trend in metric values."""
        values = self.get_recent_values(name, window)

        if len(values) < 2:
            return {'slope': 0.0, 'r_squared': 0.0, 'is_increasing': False}

        x = np.arange(len(values))
        y = np.array(values)

        # Linear regression
        A = np.vstack([x, np.ones(len(x))]).T
        slope, intercept = np.linalg.lstsq(A, y, rcond=None)[0]

        # R-squared
        y_pred = slope * x + intercept
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        r_squared = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0

        return {
            'slope': float(slope),
            'r_squared': float(r_squared),
            'is_increasing': slope > 0,
            'trend_strength': abs(float(r_squared))
        }

    def set_alert_threshold(self, name: str, min_val: float = None,
                           max_val: float = None, callback: Callable = None) -> None:
        """Set alert thresholds for a metric."""
        self._alert_thresholds[name] = {
            'min': min_val,
            'max': max_val,
            'callback': callback
        }

        logger.info(f"Alert threshold set for {name}: [{min_val}, {max_val}]")

    def add_alert_callback(self, callback: Callable[[str, float], None]) -> None:
        """Add global alert callback."""
        self._alert_callbacks.append(callback)

    def _check_alerts(self, name: str, value: float) -> None:
        """Check if metric value triggers any alerts."""
        if name in self._alert_thresholds:
            threshold = self._alert_thresholds[name]

            alert_triggered = False
            alert_message = ""

            if threshold['min'] is not None and value < threshold['min']:
                alert_triggered = True
                alert_message = f"Metric {name} below threshold: {value} < {threshold['min']}"

            elif threshold['max'] is not None and value > threshold['max']:
                alert_triggered = True
                alert_message = f"Metric {name} above threshold: {value} > {threshold['max']}"

            if alert_triggered:
                logger.warning(alert_message)

                # Call metric-specific callback
                if threshold['callback']:
                    try:
                        threshold['callback'](name, value)
                    except Exception as e:
                        logger.error(f"Alert callback failed: {e}")

                # Call global callbacks
                for callback in self._alert_callbacks:
                    try:
                        callback(name, value)
                    except Exception as e:
                        logger.error(f"Global alert callback failed: {e}")

    def start_episode_timing(self, episode: int, agent_id: Optional[int] = None) -> None:
        """Start timing an episode."""
        key = (episode, agent_id)
        self._episode_start_times[key] = time.time()

    def end_episode_timing(self, episode: int, agent_id: Optional[int] = None) -> float:
        """End timing an episode and record duration."""
        key = (episode, agent_id)

        if key not in self._episode_start_times:
            logger.warning(f"Episode timing not started for {key}")
            return 0.0

        duration = time.time() - self._episode_start_times[key]
        del self._episode_start_times[key]

        self.record('episode_duration', duration, agent_id=agent_id, episode=episode)
        return duration

    def get_system_metrics(self) -> Dict[str, float]:
        """Get current system performance metrics."""
        import gc

        import psutil

        try:
            cpu_percent = psutil.cpu_percent()
            memory = psutil.virtual_memory()

            return {
                'cpu_usage_percent': cpu_percent,
                'memory_usage_percent': memory.percent,
                'memory_available_gb': memory.available / (1024**3),
                'gc_objects': len(gc.get_objects()),
                'uptime_hours': (time.time() - self._start_time) / 3600
            }
        except ImportError:
            logger.warning("psutil not available for system metrics")
            return {
                'uptime_hours': (time.time() - self._start_time) / 3600
            }

    def get_graph_metrics(self, state: GraphState) -> Dict[str, float]:
        """Extract graph-specific metrics."""
        num_nodes = state.nodes.shape[0]
        num_edges = state.edges.shape[0]

        # Graph connectivity metrics
        adj_matrix = state.adjacency
        degree_sequence = jnp.sum(adj_matrix, axis=1)

        metrics = {
            'num_nodes': float(num_nodes),
            'num_edges': float(num_edges),
            'edge_density': float(num_edges) / (num_nodes * (num_nodes - 1) / 2) if num_nodes > 1 else 0.0,
            'avg_degree': float(jnp.mean(degree_sequence)),
            'max_degree': float(jnp.max(degree_sequence)),
            'degree_variance': float(jnp.var(degree_sequence))
        }

        # Node feature statistics
        if state.nodes.size > 0:
            node_norms = jnp.linalg.norm(state.nodes, axis=1)
            metrics.update({
                'avg_node_norm': float(jnp.mean(node_norms)),
                'max_node_norm': float(jnp.max(node_norms)),
                'node_feature_variance': float(jnp.mean(jnp.var(state.nodes, axis=0)))
            })

        return metrics

    def _start_background_analysis(self) -> None:
        """Start background thread for metric analysis."""
        def analysis_loop():
            while not self._stop_analysis.wait(30):  # Run every 30 seconds
                try:
                    self._periodic_analysis()
                except Exception as e:
                    logger.error(f"Background analysis error: {e}")

        self._analysis_thread = threading.Thread(target=analysis_loop, daemon=True)
        self._analysis_thread.start()
        logger.info("Started background metrics analysis")

    def _periodic_analysis(self) -> None:
        """Periodic analysis of collected metrics."""
        with self._lock:
            metric_names = list(self._metrics.keys())

        for name in metric_names:
            if len(self._metrics[name]) < 10:
                continue

            # Detect anomalies
            recent_values = self.get_recent_values(name, 50)
            if self._detect_anomaly(recent_values):
                logger.warning(f"Anomaly detected in metric {name}")

            # Check for stagnation
            if self._detect_stagnation(name):
                logger.warning(f"Metric {name} appears to be stagnating")

    def _detect_anomaly(self, values: List[float], threshold: float = 3.0) -> bool:
        """Detect statistical anomalies using z-score."""
        if len(values) < 10:
            return False

        mean = np.mean(values)
        std = np.std(values)

        if std == 0:
            return False

        latest_value = values[-1]
        z_score = abs((latest_value - mean) / std)

        return z_score > threshold

    def _detect_stagnation(self, name: str, threshold: float = 0.01) -> bool:
        """Detect if a metric has stagnated."""
        trend = self.get_trend(name, window=50)

        # Consider stagnated if slope is very small and trend is strong
        return (abs(trend['slope']) < threshold and
                trend['trend_strength'] > 0.8)

    def export_metrics(self, format: str = 'dict') -> Any:
        """Export all metrics in specified format."""
        with self._lock:
            data = {}
            for name, snapshots in self._metrics.items():
                data[name] = {
                    'values': [s.value for s in snapshots],
                    'timestamps': [s.timestamp for s in snapshots],
                    'metadata': [s.metadata for s in snapshots]
                }

        if format == 'dict':
            return data
        elif format == 'json':
            import json
            return json.dumps(data, default=str)
        else:
            raise ValueError(f"Unsupported export format: {format}")

    def shutdown(self) -> None:
        """Shutdown the metrics collector."""
        if self._analysis_thread and self._analysis_thread.is_alive():
            self._stop_analysis.set()
            self._analysis_thread.join(timeout=5)
            logger.info("Metrics collector shutdown complete")

class FederatedMetricsAggregator:
    """Aggregate metrics across multiple federated agents."""

    def __init__(self, num_agents: int):
        self.num_agents = num_agents
        self.agent_collectors = {}
        self.global_collector = AdvancedMetricsCollector(window_size=10000)

    def register_agent(self, agent_id: int, collector: AdvancedMetricsCollector) -> None:
        """Register an agent's metrics collector."""
        self.agent_collectors[agent_id] = collector
        logger.info(f"Registered agent {agent_id} for federated metrics")

    def aggregate_metric(self, metric_name: str) -> AggregatedMetrics:
        """Aggregate a metric across all agents."""
        all_values = []

        for agent_id, collector in self.agent_collectors.items():
            agent_values = collector.get_recent_values(metric_name, n=100)
            all_values.extend(agent_values)

        if not all_values:
            return None

        values_array = np.array(all_values)

        aggregated = AggregatedMetrics(
            mean=float(np.mean(values_array)),
            std=float(np.std(values_array)),
            min=float(np.min(values_array)),
            max=float(np.max(values_array)),
            median=float(np.median(values_array)),
            count=len(all_values),
            percentiles={
                25: float(np.percentile(values_array, 25)),
                50: float(np.percentile(values_array, 50)),
                75: float(np.percentile(values_array, 75)),
                90: float(np.percentile(values_array, 90)),
                95: float(np.percentile(values_array, 95)),
                99: float(np.percentile(values_array, 99))
            }
        )

        # Record aggregated metric
        self.global_collector.record(f"federated_{metric_name}", aggregated.mean)

        return aggregated

    def get_convergence_metrics(self) -> Dict[str, float]:
        """Calculate convergence metrics across agents."""
        reward_values = []
        loss_values = []

        for agent_id, collector in self.agent_collectors.items():
            agent_rewards = collector.get_recent_values('episode_reward', n=10)
            agent_losses = collector.get_recent_values('loss', n=50)

            if agent_rewards:
                reward_values.append(np.mean(agent_rewards))
            if agent_losses:
                loss_values.append(np.mean(agent_losses))

        metrics = {
            'num_active_agents': len(self.agent_collectors),
            'reward_convergence_std': float(np.std(reward_values)) if reward_values else 0.0,
            'loss_convergence_std': float(np.std(loss_values)) if loss_values else 0.0
        }

        # Calculate coefficient of variation for convergence assessment
        if reward_values and np.mean(reward_values) != 0:
            metrics['reward_cv'] = metrics['reward_convergence_std'] / abs(np.mean(reward_values))
        else:
            metrics['reward_cv'] = float('inf')

        return metrics
