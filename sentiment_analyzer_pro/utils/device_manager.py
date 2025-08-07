"""Device management utilities for optimal model performance."""

import logging
import platform
from typing import Optional, Dict, Any
try:
    import torch
except ImportError:
    torch = None
import psutil


logger = logging.getLogger(__name__)


class DeviceManager:
    """
    Intelligent device management for PyTorch models.
    
    Features:
    - Automatic device detection
    - GPU memory monitoring
    - Performance optimization
    - Fallback strategies
    """
    
    def __init__(self):
        """Initialize device manager."""
        self.available_devices = self._detect_devices()
        self.optimal_device = self._select_optimal_device()
        
        logger.info(f"DeviceManager initialized. Available devices: {self.available_devices}")
        logger.info(f"Optimal device selected: {self.optimal_device}")
    
    def _detect_devices(self) -> Dict[str, Dict[str, Any]]:
        """Detect all available compute devices."""
        devices = {}
        
        # CPU information
        devices["cpu"] = {
            "available": True,
            "cores": psutil.cpu_count(logical=True),
            "memory_gb": round(psutil.virtual_memory().total / (1024**3), 2),
            "arch": platform.machine()
        }
        
        # CUDA GPU information
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                props = torch.cuda.get_device_properties(i)
                devices[f"cuda:{i}"] = {
                    "available": True,
                    "name": props.name,
                    "memory_gb": round(props.total_memory / (1024**3), 2),
                    "compute_capability": f"{props.major}.{props.minor}",
                    "multiprocessor_count": props.multi_processor_count
                }
        
        # Apple Metal Performance Shaders (MPS)
        if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            devices["mps"] = {
                "available": True,
                "name": "Apple Metal Performance Shaders",
                "arch": platform.machine()
            }
        
        return devices
    
    def _select_optimal_device(self) -> str:
        """Select the optimal device for sentiment analysis."""
        # Priority order: CUDA > MPS > CPU
        
        # Check for CUDA GPUs
        cuda_devices = [name for name in self.available_devices.keys() if name.startswith("cuda")]
        if cuda_devices:
            # Select GPU with most memory
            best_cuda = max(cuda_devices, key=lambda x: self.available_devices[x]["memory_gb"])
            return best_cuda
        
        # Check for Apple MPS
        if "mps" in self.available_devices:
            return "mps"
        
        # Fallback to CPU
        return "cpu"
    
    def get_device(self, device_preference: Optional[str] = None) -> torch.device:
        """
        Get PyTorch device based on preference and availability.
        
        Args:
            device_preference: Preferred device ("auto", "cpu", "cuda", "mps", or specific like "cuda:0")
            
        Returns:
            PyTorch device object
        """
        if device_preference is None or device_preference == "auto":
            device_str = self.optimal_device
        else:
            device_str = device_preference
        
        # Validate device availability
        if device_str not in self.available_devices:
            logger.warning(f"Requested device {device_str} not available. Using {self.optimal_device}")
            device_str = self.optimal_device
        
        return torch.device(device_str)
    
    def get_memory_info(self, device: Optional[torch.device] = None) -> Dict[str, float]:
        """Get memory information for a device."""
        if device is None:
            device = torch.device(self.optimal_device)
        
        memory_info = {}
        
        if device.type == "cuda":
            # GPU memory
            memory_info.update({
                "allocated_gb": torch.cuda.memory_allocated(device) / (1024**3),
                "cached_gb": torch.cuda.memory_reserved(device) / (1024**3),
                "total_gb": torch.cuda.get_device_properties(device).total_memory / (1024**3)
            })
        elif device.type == "cpu":
            # System memory
            mem = psutil.virtual_memory()
            memory_info.update({
                "used_gb": mem.used / (1024**3),
                "available_gb": mem.available / (1024**3),
                "total_gb": mem.total / (1024**3),
                "percent": mem.percent
            })
        
        return memory_info
    
    def clear_cache(self, device: Optional[torch.device] = None) -> None:
        """Clear cache for the specified device."""
        if device is None:
            device = torch.device(self.optimal_device)
        
        if device.type == "cuda":
            torch.cuda.empty_cache()
            logger.info(f"Cleared CUDA cache for {device}")
        elif device.type == "mps":
            if hasattr(torch.mps, 'empty_cache'):
                torch.mps.empty_cache()
                logger.info(f"Cleared MPS cache for {device}")
    
    def optimize_for_inference(self, device: Optional[torch.device] = None) -> Dict[str, Any]:
        """Optimize device settings for inference."""
        if device is None:
            device = torch.device(self.optimal_device)
        
        optimizations = {
            "device": str(device),
            "optimizations_applied": []
        }
        
        if device.type == "cuda":
            # Enable TensorFloat-32 for better performance on Ampere GPUs
            if torch.cuda.get_device_capability(device)[0] >= 8:  # Ampere or newer
                torch.backends.cuda.matmul.allow_tf32 = True
                torch.backends.cudnn.allow_tf32 = True
                optimizations["optimizations_applied"].extend(["tf32_enabled"])
            
            # Set optimal CUDNN settings
            torch.backends.cudnn.benchmark = True
            optimizations["optimizations_applied"].append("cudnn_benchmark")
            
        elif device.type == "cpu":
            # Optimize CPU threading
            if hasattr(torch, 'set_num_threads'):
                num_threads = min(torch.get_num_threads(), psutil.cpu_count(logical=False))
                torch.set_num_threads(num_threads)
                optimizations["optimizations_applied"].append(f"cpu_threads_{num_threads}")
        
        logger.info(f"Applied optimizations: {optimizations['optimizations_applied']}")
        return optimizations
    
    def get_device_recommendations(self) -> Dict[str, str]:
        """Get device recommendations based on system capabilities."""
        recommendations = {}
        
        # Memory recommendations
        if "cuda:0" in self.available_devices:
            gpu_memory = self.available_devices["cuda:0"]["memory_gb"]
            if gpu_memory >= 8:
                recommendations["batch_size"] = "Use batch_size 32-64 for optimal throughput"
            elif gpu_memory >= 4:
                recommendations["batch_size"] = "Use batch_size 16-32 to avoid OOM"
            else:
                recommendations["batch_size"] = "Use batch_size 8-16, consider CPU fallback"
        
        # Model size recommendations
        cpu_memory = self.available_devices["cpu"]["memory_gb"]
        if cpu_memory < 8:
            recommendations["model_size"] = "Use DistilBERT or other lightweight models"
        elif cpu_memory < 16:
            recommendations["model_size"] = "BERT-base models recommended"
        else:
            recommendations["model_size"] = "BERT-large models supported"
        
        # Performance recommendations
        if self.optimal_device.startswith("cuda"):
            recommendations["performance"] = "GPU acceleration available - optimal for production"
        elif self.optimal_device == "mps":
            recommendations["performance"] = "Apple Silicon acceleration available"
        else:
            recommendations["performance"] = "CPU only - consider GPU for better performance"
        
        return recommendations
    
    def health_check(self) -> Dict[str, Any]:
        """Perform device health check."""
        health_status = {
            "timestamp": torch.utils.data._utils.IS_WINDOWS,  # Simple timestamp alternative
            "overall_status": "healthy",
            "devices": {},
            "warnings": [],
            "errors": []
        }
        
        # Check each device
        for device_name, device_info in self.available_devices.items():
            device_health = {"available": device_info["available"]}
            
            try:
                device_obj = torch.device(device_name)
                
                # Test basic tensor operations
                test_tensor = torch.randn(10, 10, device=device_obj)
                result = torch.mm(test_tensor, test_tensor.T)
                device_health["test_passed"] = True
                
                # Memory check
                memory_info = self.get_memory_info(device_obj)
                device_health["memory"] = memory_info
                
                # Check for memory pressure
                if device_name.startswith("cuda"):
                    if memory_info.get("cached_gb", 0) > memory_info.get("total_gb", 1) * 0.8:
                        health_status["warnings"].append(f"{device_name}: High memory usage")
                
            except Exception as e:
                device_health["test_passed"] = False
                device_health["error"] = str(e)
                health_status["errors"].append(f"{device_name}: {str(e)}")
                health_status["overall_status"] = "degraded"
            
            health_status["devices"][device_name] = device_health
        
        return health_status