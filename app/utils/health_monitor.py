"""
System Health Monitor
Tracks CPU, GPU, memory, and audio health
"""
import psutil
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    torch = None
from typing import Dict, Optional
import logging

logger = logging.getLogger(__name__)

class HealthMonitor:
    """
    Monitors system resources and audio pipeline health
    Provides warnings when resources are constrained
    """
    
    def __init__(self):
        self.cuda_available = TORCH_AVAILABLE and torch is not None and torch.cuda.is_available()
        self.process = psutil.Process()
        
        logger.info(f"Health monitor initialized: CUDA={self.cuda_available}")
    
    def get_cpu_stats(self) -> Dict:
        """Get CPU usage statistics"""
        return {
            "percent": self.process.cpu_percent(interval=0.1),
            "threads": self.process.num_threads(),
            "system_percent": psutil.cpu_percent(interval=0.1)
        }
    
    def get_memory_stats(self) -> Dict:
        """Get memory usage statistics"""
        mem_info = self.process.memory_info()
        system_mem = psutil.virtual_memory()
        
        return {
            "process_mb": mem_info.rss / (1024 * 1024),
            "system_percent": system_mem.percent,
            "available_mb": system_mem.available / (1024 * 1024)
        }
    
    def get_gpu_stats(self) -> Optional[Dict]:
        """Get GPU usage statistics"""
        if not self.cuda_available or not TORCH_AVAILABLE or torch is None:
            return None
        
        try:
            gpu_mem = torch.cuda.memory_allocated() / (1024 * 1024)
            gpu_mem_reserved = torch.cuda.memory_reserved() / (1024 * 1024)
            gpu_name = torch.cuda.get_device_name(0)
            
            return {
                "name": gpu_name,
                "memory_allocated_mb": gpu_mem,
                "memory_reserved_mb": gpu_mem_reserved,
                "device_count": torch.cuda.device_count()
            }
        except Exception as e:
            logger.error(f"Failed to get GPU stats: {e}")
            return None
    
    def get_health_report(self) -> Dict:
        """
        Get complete health report
        
        Returns:
            Dictionary with all health metrics
        """
        report = {
            "cpu": self.get_cpu_stats(),
            "memory": self.get_memory_stats(),
            "gpu": self.get_gpu_stats()
        }
        
        # Check for warnings
        warnings = []
        
        if report["cpu"]["system_percent"] > 80:
            warnings.append("High CPU usage")
        
        if report["memory"]["system_percent"] > 85:
            warnings.append("High memory usage")
        
        if report["gpu"] and report["gpu"]["memory_allocated_mb"] > 6000:
            warnings.append("High GPU memory usage")
        
        report["warnings"] = warnings
        
        return report
    
    def log_health(self):
        """Log current health status"""
        report = self.get_health_report()
        
        logger.info(f"CPU: {report['cpu']['percent']:.1f}% | "
                   f"RAM: {report['memory']['process_mb']:.0f}MB "
                   f"({report['memory']['system_percent']:.1f}%)")
        
        if report["gpu"]:
            logger.info(f"GPU: {report['gpu']['name']} | "
                       f"VRAM: {report['gpu']['memory_allocated_mb']:.0f}MB")
        
        if report["warnings"]:
            logger.warning(f"Health warnings: {', '.join(report['warnings'])}")