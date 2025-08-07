"""Memory management and optimization utilities."""

import gc
import logging
import time
import threading
from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass
import psutil
import torch


logger = logging.getLogger(__name__)


@dataclass
class MemoryStats:
    """Memory usage statistics."""
    process_memory_mb: float
    process_memory_percent: float
    system_memory_percent: float
    system_available_gb: float
    gpu_memory_used_gb: float
    gpu_memory_total_gb: float
    gpu_memory_percent: float
    timestamp: float


class MemoryOptimizer:
    """
    Advanced memory optimization for sentiment analysis models.
    
    Features:
    - Automatic garbage collection
    - GPU memory management
    - Memory pool optimization
    - Emergency cleanup procedures
    - Memory pressure detection
    """
    
    def __init__(
        self,
        cleanup_threshold: float = 0.85,
        emergency_threshold: float = 0.95,
        cleanup_interval: int = 300,
        enable_auto_cleanup: bool = True
    ):
        """
        Initialize memory optimizer.
        
        Args:
            cleanup_threshold: Memory usage threshold to trigger cleanup
            emergency_threshold: Emergency cleanup threshold
            cleanup_interval: Automatic cleanup interval in seconds
            enable_auto_cleanup: Enable automatic cleanup
        """
        self.cleanup_threshold = cleanup_threshold
        self.emergency_threshold = emergency_threshold
        self.cleanup_interval = cleanup_interval
        self.enable_auto_cleanup = enable_auto_cleanup
        
        self.last_cleanup = time.time()
        self.cleanup_history: List[float] = []
        
        # Start background cleanup if enabled
        if enable_auto_cleanup:
            self._start_background_cleanup()
        
        logger.info(f"MemoryOptimizer initialized with cleanup threshold: {cleanup_threshold}")
    
    def optimize_memory(self, aggressive: bool = False) -> MemoryStats:
        """
        Optimize memory usage.
        
        Args:
            aggressive: Perform aggressive optimization
            
        Returns:
            Memory statistics after optimization
        """
        start_time = time.time()
        initial_stats = self.get_memory_stats()
        
        # Python garbage collection
        collected = gc.collect()
        logger.debug(f"Garbage collection freed {collected} objects")
        
        # GPU memory optimization
        if torch.cuda.is_available():
            self._optimize_gpu_memory(aggressive)
        
        # Aggressive cleanup if requested
        if aggressive:
            self._aggressive_cleanup()
        
        final_stats = self.get_memory_stats()
        
        # Log optimization results
        memory_freed = initial_stats.process_memory_mb - final_stats.process_memory_mb
        optimization_time = time.time() - start_time
        
        logger.info(
            f"Memory optimization completed: "
            f"freed {memory_freed:.1f}MB in {optimization_time:.3f}s"
        )
        
        self.last_cleanup = time.time()
        self.cleanup_history.append(optimization_time)
        
        # Keep only last 100 cleanup times
        if len(self.cleanup_history) > 100:
            self.cleanup_history = self.cleanup_history[-100:]
        
        return final_stats
    
    def _optimize_gpu_memory(self, aggressive: bool = False) -> None:
        """Optimize GPU memory usage."""
        if not torch.cuda.is_available():
            return
        
        for device_id in range(torch.cuda.device_count()):
            # Get memory info before cleanup
            before_allocated = torch.cuda.memory_allocated(device_id)
            before_reserved = torch.cuda.memory_reserved(device_id)
            
            # Clear cache
            torch.cuda.empty_cache()
            
            # Aggressive cleanup: force garbage collection and synchronization
            if aggressive:
                torch.cuda.synchronize(device_id)
                gc.collect()
                torch.cuda.empty_cache()
            
            # Get memory info after cleanup
            after_allocated = torch.cuda.memory_allocated(device_id)
            after_reserved = torch.cuda.memory_reserved(device_id)
            
            freed_mb = (before_reserved - after_reserved) / (1024**2)
            
            logger.debug(
                f"GPU {device_id} memory optimization: "
                f"freed {freed_mb:.1f}MB reserved memory"
            )
    
    def _aggressive_cleanup(self) -> None:
        """Perform aggressive memory cleanup."""
        # Multiple rounds of garbage collection
        for _ in range(3):
            gc.collect()
        
        # Force garbage collection of all generations
        for generation in range(gc.get_count().__len__()):
            gc.collect(generation)
        
        # Clear GPU memory multiple times
        if torch.cuda.is_available():
            for _ in range(2):
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
        
        logger.info("Aggressive memory cleanup completed")
    
    def check_memory_pressure(self) -> bool:
        """
        Check if system is under memory pressure.
        
        Returns:
            True if memory pressure detected
        """
        stats = self.get_memory_stats()
        
        # Check system memory
        if stats.system_memory_percent > self.cleanup_threshold * 100:
            return True
        
        # Check process memory
        if stats.process_memory_percent > self.cleanup_threshold * 100:
            return True
        
        # Check GPU memory
        if stats.gpu_memory_percent > self.cleanup_threshold * 100:
            return True
        
        return False
    
    def emergency_cleanup(self) -> MemoryStats:
        """
        Perform emergency memory cleanup.
        
        Returns:
            Memory statistics after emergency cleanup
        """
        logger.warning("Performing emergency memory cleanup")
        
        # Aggressive cleanup
        stats = self.optimize_memory(aggressive=True)
        
        # Additional emergency measures
        if torch.cuda.is_available():
            # Clear all GPU tensors
            torch.cuda.empty_cache()
            
            # Reset GPU memory pool if available
            if hasattr(torch.cuda, 'reset_peak_memory_stats'):
                for device_id in range(torch.cuda.device_count()):
                    torch.cuda.reset_peak_memory_stats(device_id)
        
        # Force immediate garbage collection
        for _ in range(5):
            gc.collect()
        
        logger.warning("Emergency memory cleanup completed")
        return stats
    
    def get_memory_stats(self) -> MemoryStats:
        """Get current memory usage statistics."""
        try:
            # Process memory
            process = psutil.Process()
            process_memory = process.memory_info()
            process_memory_mb = process_memory.rss / (1024**2)
            process_memory_percent = process.memory_percent()
            
            # System memory
            system_memory = psutil.virtual_memory()
            system_memory_percent = system_memory.percent
            system_available_gb = system_memory.available / (1024**3)
            
            # GPU memory
            gpu_memory_used_gb = 0
            gpu_memory_total_gb = 0
            gpu_memory_percent = 0
            
            if torch.cuda.is_available():
                for device_id in range(torch.cuda.device_count()):
                    allocated = torch.cuda.memory_allocated(device_id)
                    props = torch.cuda.get_device_properties(device_id)
                    
                    gpu_memory_used_gb += allocated / (1024**3)
                    gpu_memory_total_gb += props.total_memory / (1024**3)
                
                if gpu_memory_total_gb > 0:
                    gpu_memory_percent = (gpu_memory_used_gb / gpu_memory_total_gb) * 100
            
            return MemoryStats(
                process_memory_mb=process_memory_mb,
                process_memory_percent=process_memory_percent,
                system_memory_percent=system_memory_percent,
                system_available_gb=system_available_gb,
                gpu_memory_used_gb=gpu_memory_used_gb,
                gpu_memory_total_gb=gpu_memory_total_gb,
                gpu_memory_percent=gpu_memory_percent,
                timestamp=time.time()
            )
            
        except Exception as e:
            logger.error(f"Error getting memory stats: {e}")
            return MemoryStats(
                process_memory_mb=0,
                process_memory_percent=0,
                system_memory_percent=0,
                system_available_gb=0,
                gpu_memory_used_gb=0,
                gpu_memory_total_gb=0,
                gpu_memory_percent=0,
                timestamp=time.time()
            )
    
    def _start_background_cleanup(self) -> None:
        """Start background memory cleanup thread."""
        def cleanup_worker():
            while self.enable_auto_cleanup:
                try:
                    time.sleep(self.cleanup_interval)
                    
                    if self.check_memory_pressure():
                        self.optimize_memory()
                    
                except Exception as e:
                    logger.error(f"Background cleanup error: {e}")
        
        cleanup_thread = threading.Thread(target=cleanup_worker, daemon=True)
        cleanup_thread.start()
        logger.info("Background memory cleanup thread started")
    
    def get_optimization_stats(self) -> Dict[str, Any]:
        """Get memory optimization statistics."""
        current_stats = self.get_memory_stats()
        
        avg_cleanup_time = 0
        if self.cleanup_history:
            avg_cleanup_time = sum(self.cleanup_history) / len(self.cleanup_history)
        
        return {
            "current_memory": {
                "process_memory_mb": current_stats.process_memory_mb,
                "system_memory_percent": current_stats.system_memory_percent,
                "gpu_memory_percent": current_stats.gpu_memory_percent
            },
            "thresholds": {
                "cleanup_threshold": self.cleanup_threshold,
                "emergency_threshold": self.emergency_threshold
            },
            "optimization_stats": {
                "total_cleanups": len(self.cleanup_history),
                "avg_cleanup_time": avg_cleanup_time,
                "last_cleanup": self.last_cleanup,
                "time_since_last_cleanup": time.time() - self.last_cleanup
            },
            "memory_pressure": self.check_memory_pressure()
        }


class MemoryManager:
    """
    High-level memory management for sentiment analysis systems.
    
    Features:
    - Automatic memory monitoring
    - Proactive optimization
    - Memory leak detection
    - Resource allocation strategies
    """
    
    def __init__(
        self,
        optimizer: Optional[MemoryOptimizer] = None,
        monitoring_interval: int = 60,
        enable_monitoring: bool = True
    ):
        """
        Initialize memory manager.
        
        Args:
            optimizer: Memory optimizer instance
            monitoring_interval: Monitoring interval in seconds
            enable_monitoring: Enable continuous monitoring
        """
        self.optimizer = optimizer or MemoryOptimizer()
        self.monitoring_interval = monitoring_interval
        self.enable_monitoring = enable_monitoring
        
        self.memory_history: List[MemoryStats] = []
        self.leak_detection_baseline: Optional[MemoryStats] = None
        
        if enable_monitoring:
            self._start_memory_monitoring()
        
        logger.info("MemoryManager initialized")
    
    def optimize_for_batch_processing(self, batch_size: int, model_size: str = "base") -> Dict[str, Any]:
        """
        Optimize memory settings for batch processing.
        
        Args:
            batch_size: Target batch size
            model_size: Model size ("base", "large", "distil")
            
        Returns:
            Optimization recommendations
        """
        current_stats = self.optimizer.get_memory_stats()
        
        # Memory requirements estimation
        memory_per_sample = {
            "distil": 50,  # MB per sample
            "base": 80,
            "large": 150
        }
        
        estimated_memory_mb = batch_size * memory_per_sample.get(model_size, 80)
        available_memory_mb = current_stats.system_available_gb * 1024
        
        recommendations = {
            "current_memory_stats": current_stats,
            "estimated_requirement_mb": estimated_memory_mb,
            "available_memory_mb": available_memory_mb,
            "optimization_applied": []
        }
        
        # Check if optimization needed
        if estimated_memory_mb > available_memory_mb * 0.8:
            # Proactive cleanup
            self.optimizer.optimize_memory(aggressive=True)
            recommendations["optimization_applied"].append("aggressive_cleanup")
            
            # Recommend smaller batch size
            max_safe_batch = int(available_memory_mb * 0.7 / memory_per_sample.get(model_size, 80))
            recommendations["recommended_batch_size"] = max(1, max_safe_batch)
            recommendations["optimization_applied"].append("batch_size_reduction")
        else:
            recommendations["recommended_batch_size"] = batch_size
        
        return recommendations
    
    def detect_memory_leaks(self, baseline_samples: int = 10) -> Dict[str, Any]:
        """
        Detect potential memory leaks.
        
        Args:
            baseline_samples: Number of samples to establish baseline
            
        Returns:
            Memory leak analysis
        """
        if len(self.memory_history) < baseline_samples * 2:
            return {"status": "insufficient_data", "message": "Need more memory samples"}
        
        # Establish baseline
        baseline = self.memory_history[:baseline_samples]
        recent = self.memory_history[-baseline_samples:]
        
        baseline_avg = sum(s.process_memory_mb for s in baseline) / len(baseline)
        recent_avg = sum(s.process_memory_mb for s in recent) / len(recent)
        
        memory_growth = recent_avg - baseline_avg
        growth_rate = memory_growth / baseline_avg if baseline_avg > 0 else 0
        
        analysis = {
            "baseline_memory_mb": baseline_avg,
            "recent_memory_mb": recent_avg,
            "memory_growth_mb": memory_growth,
            "growth_rate_percent": growth_rate * 100,
            "samples_analyzed": len(self.memory_history),
            "leak_detected": False,
            "recommendations": []
        }
        
        # Leak detection thresholds
        if growth_rate > 0.1:  # 10% growth
            analysis["leak_detected"] = True
            analysis["severity"] = "high" if growth_rate > 0.25 else "medium"
            analysis["recommendations"].append("investigate_memory_leak")
            analysis["recommendations"].append("increase_cleanup_frequency")
        
        elif growth_rate > 0.05:  # 5% growth
            analysis["leak_detected"] = True
            analysis["severity"] = "low"
            analysis["recommendations"].append("monitor_memory_usage")
        
        return analysis
    
    def _start_memory_monitoring(self) -> None:
        """Start continuous memory monitoring."""
        def monitoring_worker():
            while self.enable_monitoring:
                try:
                    stats = self.optimizer.get_memory_stats()
                    self.memory_history.append(stats)
                    
                    # Keep only last 1000 samples
                    if len(self.memory_history) > 1000:
                        self.memory_history = self.memory_history[-1000:]
                    
                    # Check for emergency situations
                    if stats.system_memory_percent > 95:
                        logger.critical("Critical memory usage detected!")
                        self.optimizer.emergency_cleanup()
                    
                    time.sleep(self.monitoring_interval)
                    
                except Exception as e:
                    logger.error(f"Memory monitoring error: {e}")
        
        monitor_thread = threading.Thread(target=monitoring_worker, daemon=True)
        monitor_thread.start()
        logger.info("Memory monitoring thread started")
    
    def get_memory_report(self) -> Dict[str, Any]:
        """Get comprehensive memory report."""
        current_stats = self.optimizer.get_memory_stats()
        optimization_stats = self.optimizer.get_optimization_stats()
        
        # Calculate trends if we have history
        trend_analysis = {"status": "no_data"}
        if len(self.memory_history) >= 5:
            recent_samples = self.memory_history[-5:]
            memory_values = [s.process_memory_mb for s in recent_samples]
            
            # Simple trend calculation
            if len(memory_values) >= 2:
                trend = memory_values[-1] - memory_values[0]
                trend_analysis = {
                    "status": "calculated",
                    "trend_mb": trend,
                    "trend_direction": "increasing" if trend > 0 else "decreasing" if trend < 0 else "stable"
                }
        
        return {
            "current_stats": current_stats,
            "optimization_stats": optimization_stats,
            "trend_analysis": trend_analysis,
            "memory_history_size": len(self.memory_history),
            "monitoring_enabled": self.enable_monitoring,
            "leak_analysis": self.detect_memory_leaks() if len(self.memory_history) >= 20 else None
        }
    
    def cleanup_on_exit(self) -> None:
        """Cleanup memory before shutdown."""
        logger.info("Performing final memory cleanup...")
        self.enable_monitoring = False
        self.optimizer.optimize_memory(aggressive=True)
        logger.info("Final memory cleanup completed")