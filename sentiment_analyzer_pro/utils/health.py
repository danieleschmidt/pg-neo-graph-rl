"""Health monitoring and system diagnostics."""

import asyncio
import logging
import time
import psutil
import torch
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict
from enum import Enum

from ..utils.exceptions import ResourceError


logger = logging.getLogger(__name__)


class HealthStatus(Enum):
    """Health status levels."""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    CRITICAL = "critical"


@dataclass
class HealthCheck:
    """Individual health check result."""
    name: str
    status: HealthStatus
    message: str
    details: Dict[str, Any]
    execution_time: float
    timestamp: float


@dataclass
class SystemMetrics:
    """System resource metrics."""
    cpu_percent: float
    memory_percent: float
    memory_used_gb: float
    memory_total_gb: float
    disk_percent: float
    gpu_available: bool
    gpu_memory_used_gb: float
    gpu_memory_total_gb: float
    load_average: List[float]
    uptime: float


class HealthMonitor:
    """
    Comprehensive health monitoring system.
    
    Features:
    - System resource monitoring
    - Model health checks
    - Performance metrics tracking
    - Alert thresholds
    - Automated diagnostics
    """
    
    def __init__(
        self,
        cpu_threshold: float = 80.0,
        memory_threshold: float = 85.0,
        disk_threshold: float = 90.0,
        gpu_memory_threshold: float = 90.0,
        response_time_threshold: float = 5.0
    ):
        """
        Initialize health monitor.
        
        Args:
            cpu_threshold: CPU usage threshold for warnings
            memory_threshold: Memory usage threshold for warnings
            disk_threshold: Disk usage threshold for warnings
            gpu_memory_threshold: GPU memory threshold for warnings
            response_time_threshold: Response time threshold for warnings
        """
        self.cpu_threshold = cpu_threshold
        self.memory_threshold = memory_threshold
        self.disk_threshold = disk_threshold
        self.gpu_memory_threshold = gpu_memory_threshold
        self.response_time_threshold = response_time_threshold
        
        self.start_time = time.time()
        self.check_history: List[HealthCheck] = []
        
        logger.info("HealthMonitor initialized")
    
    async def run_health_check(
        self,
        analyzer=None,
        include_model_check: bool = True,
        include_performance_check: bool = True
    ) -> Dict[str, Any]:
        """
        Run comprehensive health check.
        
        Args:
            analyzer: Sentiment analyzer instance to check
            include_model_check: Whether to include model health check
            include_performance_check: Whether to include performance tests
            
        Returns:
            Complete health report
        """
        start_time = time.time()
        checks = []
        
        # System resources check
        system_check = await self._check_system_resources()
        checks.append(system_check)
        
        # Model health check
        if include_model_check and analyzer:
            model_check = await self._check_model_health(analyzer)
            checks.append(model_check)
        
        # Performance check
        if include_performance_check and analyzer:
            perf_check = await self._check_performance(analyzer)
            checks.append(perf_check)
        
        # Memory check
        memory_check = await self._check_memory_usage()
        checks.append(memory_check)
        
        # GPU check
        gpu_check = await self._check_gpu_health()
        checks.append(gpu_check)
        
        # Determine overall status
        overall_status = self._determine_overall_status(checks)
        
        # Store checks
        self.check_history.extend(checks)
        
        # Keep only last 100 checks
        if len(self.check_history) > 100:
            self.check_history = self.check_history[-100:]
        
        health_report = {
            "overall_status": overall_status.value,
            "timestamp": time.time(),
            "execution_time": time.time() - start_time,
            "checks": [asdict(check) for check in checks],
            "summary": self._create_summary(checks),
            "recommendations": self._generate_recommendations(checks),
            "system_metrics": self._get_system_metrics()
        }
        
        logger.info(f"Health check completed: {overall_status.value} ({len(checks)} checks)")
        return health_report
    
    async def _check_system_resources(self) -> HealthCheck:
        """Check system resource utilization."""
        start_time = time.time()
        
        try:
            # CPU usage
            cpu_percent = psutil.cpu_percent(interval=1)
            
            # Memory usage
            memory = psutil.virtual_memory()
            memory_percent = memory.percent
            
            # Disk usage
            disk = psutil.disk_usage('/')
            disk_percent = disk.percent
            
            details = {
                "cpu_percent": cpu_percent,
                "memory_percent": memory_percent,
                "memory_used_gb": round(memory.used / (1024**3), 2),
                "memory_total_gb": round(memory.total / (1024**3), 2),
                "disk_percent": disk_percent,
                "disk_free_gb": round(disk.free / (1024**3), 2)
            }
            
            # Determine status
            if (cpu_percent > self.cpu_threshold or 
                memory_percent > self.memory_threshold or 
                disk_percent > self.disk_threshold):
                status = HealthStatus.DEGRADED
                message = "System resources under pressure"
            else:
                status = HealthStatus.HEALTHY
                message = "System resources normal"
            
        except Exception as e:
            status = HealthStatus.UNHEALTHY
            message = f"Failed to check system resources: {e}"
            details = {"error": str(e)}
        
        return HealthCheck(
            name="system_resources",
            status=status,
            message=message,
            details=details,
            execution_time=time.time() - start_time,
            timestamp=time.time()
        )
    
    async def _check_model_health(self, analyzer) -> HealthCheck:
        """Check model health and functionality."""
        start_time = time.time()
        
        try:
            # Test model prediction
            test_text = "This is a test message for health check"
            result = await analyzer.analyze_async(test_text)
            
            details = {
                "model_loaded": True,
                "test_prediction_label": result.label,
                "test_prediction_confidence": result.confidence,
                "test_processing_time": result.processing_time,
                "model_info": analyzer.get_model_info()
            }
            
            # Check prediction quality
            if result.processing_time > self.response_time_threshold:
                status = HealthStatus.DEGRADED
                message = f"Model response time high: {result.processing_time:.3f}s"
            elif result.confidence < 0.1:
                status = HealthStatus.DEGRADED
                message = "Model prediction confidence low"
            else:
                status = HealthStatus.HEALTHY
                message = "Model functioning normally"
                
        except Exception as e:
            status = HealthStatus.UNHEALTHY
            message = f"Model health check failed: {e}"
            details = {"error": str(e), "model_loaded": False}
        
        return HealthCheck(
            name="model_health",
            status=status,
            message=message,
            details=details,
            execution_time=time.time() - start_time,
            timestamp=time.time()
        )
    
    async def _check_performance(self, analyzer) -> HealthCheck:
        """Check system performance with batch processing."""
        start_time = time.time()
        
        try:
            # Performance test with batch
            test_texts = [
                "This is a positive test message",
                "This is a negative test message", 
                "This is a neutral test message"
            ] * 5  # 15 texts total
            
            perf_start = time.time()
            results = await analyzer.analyze_batch_async(test_texts)
            perf_time = time.time() - perf_start
            
            avg_time_per_text = perf_time / len(test_texts)
            throughput = len(test_texts) / perf_time
            
            details = {
                "batch_size": len(test_texts),
                "total_time": perf_time,
                "avg_time_per_text": avg_time_per_text,
                "throughput_texts_per_sec": throughput,
                "successful_predictions": len(results)
            }
            
            # Performance thresholds
            if avg_time_per_text > 1.0:  # 1 second per text is slow
                status = HealthStatus.DEGRADED
                message = f"Performance degraded: {avg_time_per_text:.3f}s per text"
            elif throughput < 5:  # Less than 5 texts per second
                status = HealthStatus.DEGRADED
                message = f"Low throughput: {throughput:.1f} texts/sec"
            else:
                status = HealthStatus.HEALTHY
                message = f"Performance good: {throughput:.1f} texts/sec"
                
        except Exception as e:
            status = HealthStatus.UNHEALTHY
            message = f"Performance check failed: {e}"
            details = {"error": str(e)}
        
        return HealthCheck(
            name="performance",
            status=status,
            message=message,
            details=details,
            execution_time=time.time() - start_time,
            timestamp=time.time()
        )
    
    async def _check_memory_usage(self) -> HealthCheck:
        """Check memory usage patterns."""
        start_time = time.time()
        
        try:
            process = psutil.Process()
            memory_info = process.memory_info()
            memory_percent = process.memory_percent()
            
            details = {
                "process_memory_mb": round(memory_info.rss / (1024**2), 2),
                "process_memory_percent": memory_percent,
                "memory_info": {
                    "rss": memory_info.rss,
                    "vms": memory_info.vms
                }
            }
            
            # Check for memory issues
            if memory_percent > 50:  # Process using more than 50% of system memory
                status = HealthStatus.DEGRADED
                message = f"High memory usage: {memory_percent:.1f}%"
            else:
                status = HealthStatus.HEALTHY
                message = f"Memory usage normal: {memory_percent:.1f}%"
                
        except Exception as e:
            status = HealthStatus.UNHEALTHY
            message = f"Memory check failed: {e}"
            details = {"error": str(e)}
        
        return HealthCheck(
            name="memory_usage",
            status=status,
            message=message,
            details=details,
            execution_time=time.time() - start_time,
            timestamp=time.time()
        )
    
    async def _check_gpu_health(self) -> HealthCheck:
        """Check GPU availability and health."""
        start_time = time.time()
        
        try:
            gpu_available = torch.cuda.is_available()
            
            details = {
                "gpu_available": gpu_available,
                "gpu_count": torch.cuda.device_count() if gpu_available else 0
            }
            
            if gpu_available:
                # Get GPU memory info
                for i in range(torch.cuda.device_count()):
                    props = torch.cuda.get_device_properties(i)
                    memory_allocated = torch.cuda.memory_allocated(i)
                    memory_reserved = torch.cuda.memory_reserved(i)
                    
                    details[f"gpu_{i}"] = {
                        "name": props.name,
                        "memory_total_gb": round(props.total_memory / (1024**3), 2),
                        "memory_allocated_gb": round(memory_allocated / (1024**3), 2),
                        "memory_reserved_gb": round(memory_reserved / (1024**3), 2),
                        "memory_utilization": round(memory_allocated / props.total_memory * 100, 2)
                    }
                    
                    # Check GPU memory usage
                    memory_util = memory_allocated / props.total_memory * 100
                    if memory_util > self.gpu_memory_threshold:
                        status = HealthStatus.DEGRADED
                        message = f"GPU {i} memory high: {memory_util:.1f}%"
                    else:
                        status = HealthStatus.HEALTHY
                        message = "GPU resources healthy"
            else:
                status = HealthStatus.HEALTHY  # CPU-only is fine
                message = "GPU not available, using CPU"
                
        except Exception as e:
            status = HealthStatus.DEGRADED
            message = f"GPU check failed: {e}"
            details = {"error": str(e), "gpu_available": False}
        
        return HealthCheck(
            name="gpu_health",
            status=status,
            message=message,
            details=details,
            execution_time=time.time() - start_time,
            timestamp=time.time()
        )
    
    def _determine_overall_status(self, checks: List[HealthCheck]) -> HealthStatus:
        """Determine overall system health status."""
        if not checks:
            return HealthStatus.UNHEALTHY
        
        status_counts = {status: 0 for status in HealthStatus}
        for check in checks:
            status_counts[check.status] += 1
        
        # Priority order: critical > unhealthy > degraded > healthy
        if status_counts[HealthStatus.CRITICAL] > 0:
            return HealthStatus.CRITICAL
        elif status_counts[HealthStatus.UNHEALTHY] > 0:
            return HealthStatus.UNHEALTHY
        elif status_counts[HealthStatus.DEGRADED] > 0:
            return HealthStatus.DEGRADED
        else:
            return HealthStatus.HEALTHY
    
    def _create_summary(self, checks: List[HealthCheck]) -> Dict[str, Any]:
        """Create summary of health checks."""
        status_counts = {status.value: 0 for status in HealthStatus}
        total_execution_time = 0
        
        for check in checks:
            status_counts[check.status.value] += 1
            total_execution_time += check.execution_time
        
        return {
            "total_checks": len(checks),
            "status_breakdown": status_counts,
            "total_execution_time": round(total_execution_time, 3),
            "avg_execution_time": round(total_execution_time / len(checks), 3) if checks else 0
        }
    
    def _generate_recommendations(self, checks: List[HealthCheck]) -> List[str]:
        """Generate recommendations based on health checks."""
        recommendations = []
        
        for check in checks:
            if check.status in [HealthStatus.DEGRADED, HealthStatus.UNHEALTHY]:
                
                if check.name == "system_resources":
                    if check.details.get("cpu_percent", 0) > self.cpu_threshold:
                        recommendations.append("Consider reducing batch size or adding CPU resources")
                    if check.details.get("memory_percent", 0) > self.memory_threshold:
                        recommendations.append("Consider adding RAM or optimizing memory usage")
                    if check.details.get("disk_percent", 0) > self.disk_threshold:
                        recommendations.append("Free up disk space or add storage")
                
                elif check.name == "model_health":
                    recommendations.append("Check model configuration and restart if necessary")
                
                elif check.name == "performance":
                    throughput = check.details.get("throughput_texts_per_sec", 0)
                    if throughput < 5:
                        recommendations.append("Consider GPU acceleration or model optimization")
                
                elif check.name == "gpu_health":
                    recommendations.append("Monitor GPU memory usage and clear cache if needed")
        
        if not recommendations:
            recommendations.append("System is operating normally")
        
        return recommendations
    
    def _get_system_metrics(self) -> Dict[str, Any]:
        """Get current system metrics."""
        try:
            # CPU and memory
            cpu_percent = psutil.cpu_percent()
            memory = psutil.virtual_memory()
            disk = psutil.disk_usage('/')
            
            # Load average (Unix-like systems)
            try:
                load_avg = list(psutil.getloadavg())
            except (AttributeError, OSError):
                load_avg = [0.0, 0.0, 0.0]
            
            # GPU metrics
            gpu_memory_used = 0
            gpu_memory_total = 0
            if torch.cuda.is_available():
                for i in range(torch.cuda.device_count()):
                    props = torch.cuda.get_device_properties(i)
                    gpu_memory_used += torch.cuda.memory_allocated(i)
                    gpu_memory_total += props.total_memory
            
            return {
                "cpu_percent": cpu_percent,
                "memory_percent": memory.percent,
                "memory_used_gb": round(memory.used / (1024**3), 2),
                "memory_total_gb": round(memory.total / (1024**3), 2),
                "disk_percent": disk.percent,
                "gpu_available": torch.cuda.is_available(),
                "gpu_memory_used_gb": round(gpu_memory_used / (1024**3), 2),
                "gpu_memory_total_gb": round(gpu_memory_total / (1024**3), 2),
                "load_average": load_avg,
                "uptime": time.time() - self.start_time
            }
            
        except Exception as e:
            logger.error(f"Error getting system metrics: {e}")
            return {"error": str(e)}
    
    def get_health_history(self, limit: int = 20) -> List[Dict[str, Any]]:
        """Get recent health check history."""
        return [asdict(check) for check in self.check_history[-limit:]]