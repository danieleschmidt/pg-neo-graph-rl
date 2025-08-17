"""Real-time monitoring dashboard for federated learning systems."""
import time
import threading
import json
from typing import Dict, List, Any, Optional
from collections import defaultdict, deque
from dataclasses import dataclass, asdict
import numpy as np

from ..utils.logging import get_logger
from .advanced_metrics import AdvancedMetricsCollector, FederatedMetricsAggregator
from ..utils.circuit_breaker import get_circuit_breaker_manager

logger = get_logger(__name__)

@dataclass
class DashboardConfig:
    """Dashboard configuration."""
    update_interval: float = 1.0  # seconds
    history_length: int = 1000
    alert_cooldown: float = 60.0  # seconds between same alerts
    enable_predictions: bool = True
    max_agents_display: int = 100

class RealTimeDashboard:
    """Real-time monitoring dashboard for federated learning."""
    
    def __init__(self, config: DashboardConfig = None):
        self.config = config or DashboardConfig()
        
        # Core components
        self.metrics_aggregator: Optional[FederatedMetricsAggregator] = None
        self.circuit_manager = get_circuit_breaker_manager()
        
        # Dashboard state
        self.active = False
        self.last_update = 0.0
        self.update_thread: Optional[threading.Thread] = None
        self.stop_event = threading.Event()
        
        # Alert management
        self.alert_history = deque(maxlen=1000)
        self.last_alert_time = defaultdict(float)
        
        # Performance tracking
        self.performance_history = defaultdict(lambda: deque(maxlen=self.config.history_length))
        
        # Dashboard data
        self.dashboard_data = {
            'system': {},
            'federated': {},
            'agents': {},
            'alerts': [],
            'circuit_breakers': {},
            'predictions': {},
            'metadata': {
                'last_update': 0,
                'update_interval': self.config.update_interval,
                'active_agents': 0
            }
        }
        
        logger.info("Real-time dashboard initialized")
    
    def start(self, metrics_aggregator: FederatedMetricsAggregator) -> None:
        """Start the real-time dashboard."""
        if self.active:
            logger.warning("Dashboard already active")
            return
            
        self.metrics_aggregator = metrics_aggregator
        self.active = True
        self.stop_event.clear()
        
        # Start update thread
        self.update_thread = threading.Thread(target=self._update_loop, daemon=True)
        self.update_thread.start()
        
        logger.info("Real-time dashboard started")
    
    def stop(self) -> None:
        """Stop the real-time dashboard."""
        if not self.active:
            return
            
        self.active = False
        self.stop_event.set()
        
        if self.update_thread and self.update_thread.is_alive():
            self.update_thread.join(timeout=5.0)
            
        logger.info("Real-time dashboard stopped")
    
    def _update_loop(self) -> None:
        """Main update loop for dashboard."""
        while self.active and not self.stop_event.wait(self.config.update_interval):
            try:
                self._update_dashboard_data()
            except Exception as e:
                logger.error(f"Dashboard update error: {e}")
    
    def _update_dashboard_data(self) -> None:
        """Update all dashboard data."""
        current_time = time.time()
        
        # Update system metrics
        self._update_system_metrics()
        
        # Update federated learning metrics
        self._update_federated_metrics()
        
        # Update agent-specific metrics
        self._update_agent_metrics()
        
        # Update circuit breaker status
        self._update_circuit_breakers()
        
        # Check for alerts
        self._check_alerts()
        
        # Generate predictions if enabled
        if self.config.enable_predictions:
            self._update_predictions()
        
        # Update metadata
        self.dashboard_data['metadata'].update({
            'last_update': current_time,
            'active_agents': len(self.metrics_aggregator.agent_collectors) if self.metrics_aggregator else 0
        })
        
        self.last_update = current_time
    
    def _update_system_metrics(self) -> None:
        """Update system-wide metrics."""
        try:
            import psutil
            
            # CPU and memory
            cpu_percent = psutil.cpu_percent(interval=0.1)
            memory = psutil.virtual_memory()
            disk = psutil.disk_usage('/')
            
            # Network I/O
            net_io = psutil.net_io_counters()
            
            system_data = {
                'cpu_usage': cpu_percent,
                'memory_usage': memory.percent,
                'memory_available_gb': memory.available / (1024**3),
                'disk_usage': disk.percent,
                'network_bytes_sent': net_io.bytes_sent,
                'network_bytes_recv': net_io.bytes_recv,
                'timestamp': time.time()
            }
            
            # Track history
            for key, value in system_data.items():
                if key != 'timestamp':
                    self.performance_history[f'system_{key}'].append(value)
            
            self.dashboard_data['system'] = system_data
            
        except ImportError:
            logger.warning("psutil not available for system metrics")
            self.dashboard_data['system'] = {'error': 'psutil not available'}
        except Exception as e:
            logger.error(f"System metrics error: {e}")
            self.dashboard_data['system'] = {'error': str(e)}
    
    def _update_federated_metrics(self) -> None:
        """Update federated learning metrics."""
        if not self.metrics_aggregator:
            return
            
        try:
            # Key federated metrics
            convergence_metrics = self.metrics_aggregator.get_convergence_metrics()
            
            # Aggregate important metrics across agents
            key_metrics = ['episode_reward', 'loss', 'gradient_norm', 'episode_duration']
            aggregated_data = {}
            
            for metric in key_metrics:
                agg = self.metrics_aggregator.aggregate_metric(metric)
                if agg:
                    aggregated_data[metric] = {
                        'mean': agg.mean,
                        'std': agg.std,
                        'min': agg.min,
                        'max': agg.max,
                        'count': agg.count
                    }
                    
                    # Track history
                    self.performance_history[f'fed_{metric}_mean'].append(agg.mean)
            
            federated_data = {
                'convergence': convergence_metrics,
                'aggregated_metrics': aggregated_data,
                'global_stats': {
                    'total_episodes': sum(len(collector.get_recent_values('episode_reward', 1000)) 
                                        for collector in self.metrics_aggregator.agent_collectors.values()),
                    'active_agents': len(self.metrics_aggregator.agent_collectors)
                },
                'timestamp': time.time()
            }
            
            self.dashboard_data['federated'] = federated_data
            
        except Exception as e:
            logger.error(f"Federated metrics error: {e}")
            self.dashboard_data['federated'] = {'error': str(e)}
    
    def _update_agent_metrics(self) -> None:
        """Update individual agent metrics."""
        if not self.metrics_aggregator:
            return
            
        try:
            agent_data = {}
            
            for agent_id, collector in self.metrics_aggregator.agent_collectors.items():
                if len(agent_data) >= self.config.max_agents_display:
                    break
                    
                # Recent performance
                recent_rewards = collector.get_recent_values('episode_reward', 10)
                recent_losses = collector.get_recent_values('loss', 50)
                
                agent_metrics = {
                    'id': agent_id,
                    'recent_reward': np.mean(recent_rewards) if recent_rewards else 0.0,
                    'reward_trend': collector.get_trend('episode_reward', 50),
                    'recent_loss': np.mean(recent_losses) if recent_losses else 0.0,
                    'episode_count': len(collector.get_recent_values('episode_reward', 1000)),
                    'last_update': time.time()
                }
                
                # Health status
                if recent_rewards:
                    agent_metrics['status'] = 'active'
                    if np.mean(recent_rewards) < -1000:  # Threshold for poor performance
                        agent_metrics['status'] = 'struggling'
                else:
                    agent_metrics['status'] = 'inactive'
                
                agent_data[agent_id] = agent_metrics
            
            self.dashboard_data['agents'] = agent_data
            
        except Exception as e:
            logger.error(f"Agent metrics error: {e}")
            self.dashboard_data['agents'] = {'error': str(e)}
    
    def _update_circuit_breakers(self) -> None:
        """Update circuit breaker status."""
        try:
            breaker_summary = self.circuit_manager.get_health_summary()
            all_stats = self.circuit_manager.get_all_stats()
            
            circuit_data = {
                'summary': breaker_summary,
                'individual': {
                    name: {
                        'state': breaker.state.value,
                        'total_requests': stats.total_requests,
                        'failed_requests': stats.failed_requests,
                        'success_rate': (stats.successful_requests / max(stats.total_requests, 1)) * 100,
                        'consecutive_failures': stats.consecutive_failures
                    }
                    for name, (breaker, stats) in zip(all_stats.keys(), 
                                                     [(self.circuit_manager.get_breaker(name), stats) 
                                                      for name, stats in all_stats.items()])
                    if breaker is not None
                },
                'timestamp': time.time()
            }
            
            self.dashboard_data['circuit_breakers'] = circuit_data
            
        except Exception as e:
            logger.error(f"Circuit breaker update error: {e}")
            self.dashboard_data['circuit_breakers'] = {'error': str(e)}
    
    def _check_alerts(self) -> None:
        """Check for alert conditions."""
        current_time = time.time()
        new_alerts = []
        
        try:
            # System alerts
            system_data = self.dashboard_data.get('system', {})
            if 'cpu_usage' in system_data and system_data['cpu_usage'] > 90:
                self._add_alert('HIGH_CPU', f"CPU usage: {system_data['cpu_usage']:.1f}%", current_time)
            
            if 'memory_usage' in system_data and system_data['memory_usage'] > 85:
                self._add_alert('HIGH_MEMORY', f"Memory usage: {system_data['memory_usage']:.1f}%", current_time)
            
            # Federated learning alerts
            fed_data = self.dashboard_data.get('federated', {})
            convergence = fed_data.get('convergence', {})
            
            if convergence.get('reward_cv', 0) > 2.0:  # High coefficient of variation
                self._add_alert('POOR_CONVERGENCE', f"Reward CV: {convergence.get('reward_cv', 0):.2f}", current_time)
            
            # Circuit breaker alerts
            cb_data = self.dashboard_data.get('circuit_breakers', {})
            summary = cb_data.get('summary', {})
            
            if summary.get('open_breakers', 0) > 0:
                self._add_alert('CIRCUIT_BREAKERS_OPEN', 
                              f"{summary['open_breakers']} circuit breakers open", current_time)
            
            # Agent alerts
            agent_data = self.dashboard_data.get('agents', {})
            if isinstance(agent_data, dict):
                struggling_agents = [aid for aid, data in agent_data.items() 
                                   if isinstance(data, dict) and data.get('status') == 'struggling']
                
                if len(struggling_agents) > len(agent_data) * 0.3:  # >30% struggling
                    self._add_alert('MANY_STRUGGLING_AGENTS', 
                                  f"{len(struggling_agents)} agents struggling", current_time)
        
        except Exception as e:
            logger.error(f"Alert check error: {e}")
    
    def _add_alert(self, alert_type: str, message: str, timestamp: float) -> None:
        """Add an alert if not in cooldown period."""
        last_alert_time = self.last_alert_time.get(alert_type, 0)
        
        if timestamp - last_alert_time >= self.config.alert_cooldown:
            alert = {
                'type': alert_type,
                'message': message,
                'timestamp': timestamp,
                'severity': self._get_alert_severity(alert_type)
            }
            
            self.alert_history.append(alert)
            self.last_alert_time[alert_type] = timestamp
            
            # Keep only recent alerts in dashboard
            recent_alerts = [a for a in self.alert_history 
                           if timestamp - a['timestamp'] < 3600]  # Last hour
            self.dashboard_data['alerts'] = recent_alerts[-50:]  # Max 50 alerts
            
            logger.warning(f"ALERT [{alert_type}]: {message}")
    
    def _get_alert_severity(self, alert_type: str) -> str:
        """Get severity level for alert type."""
        high_severity = ['CIRCUIT_BREAKERS_OPEN', 'SYSTEM_FAILURE']
        medium_severity = ['HIGH_CPU', 'HIGH_MEMORY', 'POOR_CONVERGENCE']
        
        if alert_type in high_severity:
            return 'high'
        elif alert_type in medium_severity:
            return 'medium'
        else:
            return 'low'
    
    def _update_predictions(self) -> None:
        """Update performance predictions."""
        try:
            predictions = {}
            
            # Predict system resource usage
            cpu_history = list(self.performance_history['system_cpu_usage'])
            if len(cpu_history) >= 10:
                predictions['cpu_usage_10min'] = self._simple_prediction(cpu_history, 10)
            
            # Predict convergence rate
            reward_history = list(self.performance_history['fed_episode_reward_mean'])
            if len(reward_history) >= 20:
                predictions['reward_trend'] = self._simple_prediction(reward_history, 5)
            
            self.dashboard_data['predictions'] = predictions
            
        except Exception as e:
            logger.error(f"Prediction error: {e}")
            self.dashboard_data['predictions'] = {'error': str(e)}
    
    def _simple_prediction(self, history: List[float], steps: int) -> Dict[str, float]:
        """Simple linear prediction."""
        if len(history) < 3:
            return {'predicted_value': history[-1] if history else 0.0, 'confidence': 0.0}
        
        x = np.arange(len(history))
        y = np.array(history)
        
        # Linear regression
        A = np.vstack([x, np.ones(len(x))]).T
        slope, intercept = np.linalg.lstsq(A, y, rcond=None)[0]
        
        # Predict future value
        future_x = len(history) + steps - 1
        predicted_value = slope * future_x + intercept
        
        # Simple confidence based on recent variance
        recent_variance = np.var(history[-10:]) if len(history) >= 10 else np.var(history)
        confidence = max(0.0, 1.0 - (recent_variance / max(abs(predicted_value), 1.0)))
        
        return {
            'predicted_value': float(predicted_value),
            'confidence': float(min(confidence, 1.0)),
            'slope': float(slope)
        }
    
    def get_dashboard_data(self) -> Dict[str, Any]:
        """Get current dashboard data."""
        return self.dashboard_data.copy()
    
    def export_dashboard_data(self, format: str = 'json') -> str:
        """Export dashboard data in specified format."""
        data = self.get_dashboard_data()
        
        if format == 'json':
            return json.dumps(data, indent=2, default=str)
        else:
            raise ValueError(f"Unsupported export format: {format}")
    
    def get_alert_summary(self) -> Dict[str, Any]:
        """Get summary of recent alerts."""
        current_time = time.time()
        recent_alerts = [a for a in self.alert_history 
                        if current_time - a['timestamp'] < 3600]
        
        summary = {
            'total_alerts_1h': len(recent_alerts),
            'high_severity': len([a for a in recent_alerts if a['severity'] == 'high']),
            'medium_severity': len([a for a in recent_alerts if a['severity'] == 'medium']),
            'low_severity': len([a for a in recent_alerts if a['severity'] == 'low']),
            'most_recent': recent_alerts[-1] if recent_alerts else None
        }
        
        return summary