"""
VoiceForge-Nextgen - GPU Memory Manager
File: app/core/gpu_memory_manager.py

Purpose:
    Dynamic GPU memory management
    Track VRAM usage, auto-evict models when needed

Dependencies:
    - torch (CUDA memory)
    - psutil (system memory)

Data Flow:
    Model load → Check VRAM → Evict if needed → Allocate

Usage:
    manager = GPUMemoryManager(max_vram_gb=8.0)
    if manager.can_allocate(model_size):
        # Safe to load model
"""

import torch
import logging
from typing import Dict, Optional, List, Tuple
from collections import OrderedDict
import time

logger = logging.getLogger("GPUMemoryManager")


class GPUMemoryManager:
    """
    Manages GPU VRAM allocation for models
    
    Features:
        - Track per-model VRAM usage
        - Auto-evict LRU models when VRAM low
        - Alert on fragmentation
        - Mixed precision support
    """
    
    def __init__(
        self,
        max_vram_gb: float = 8.0,
        threshold_percent: float = 90.0,
        enable_fp16: bool = True
    ):
        """
        Initialize GPU memory manager
        
        Args:
            max_vram_gb: Maximum VRAM to use (GB)
            threshold_percent: Auto-evict threshold (%)
            enable_fp16: Enable FP16 for inference
        """
        self.max_vram_bytes = int(max_vram_gb * 1024 ** 3)
        self.threshold_percent = threshold_percent
        self.enable_fp16 = enable_fp16
        
        # Track allocated models: {model_id: (size_bytes, last_access)}
        self.allocations: OrderedDict[str, Tuple[int, float]] = OrderedDict()
        
        # Check CUDA availability
        self.cuda_available = torch.cuda.is_available()
        if not self.cuda_available:
            logger.warning("CUDA not available, GPU manager disabled")
            return
        
        # Get device properties
        self.device = torch.device('cuda:0')
        self.device_name = torch.cuda.get_device_name(0)
        self.total_vram = torch.cuda.get_device_properties(0).total_memory
        
        logger.info(
            f"GPUMemoryManager initialized: "
            f"device={self.device_name}, "
            f"total_vram={self.total_vram / 1024**3:.2f}GB, "
            f"max_usage={max_vram_gb:.1f}GB, "
            f"fp16={enable_fp16}"
        )
    
    def can_allocate(self, size_bytes: int) -> bool:
        """
        Check if we can allocate size_bytes
        
        Args:
            size_bytes: Required size
            
        Returns:
            True if allocation possible
        """
        if not self.cuda_available:
            return True  # No GPU, always "ok"
        
        current_usage = self.get_current_usage()
        available = self.max_vram_bytes - current_usage
        
        if size_bytes <= available:
            return True
        
        logger.warning(
            f"Insufficient VRAM: need {size_bytes / 1024**2:.1f}MB, "
            f"available {available / 1024**2:.1f}MB"
        )
        
        return False
    
    def allocate(self, model_id: str, size_bytes: int) -> bool:
        """
        Register model allocation
        
        Args:
            model_id: Model identifier
            size_bytes: Model size in bytes
            
        Returns:
            True if allocated successfully
        """
        if not self.cuda_available:
            return True
        
        # Check if already allocated
        if model_id in self.allocations:
            # Update access time
            size, _ = self.allocations[model_id]
            self.allocations[model_id] = (size, time.time())
            self.allocations.move_to_end(model_id)
            return True
        
        # Check if we need to evict
        while not self.can_allocate(size_bytes):
            if not self._evict_lru():
                logger.error("Cannot evict more models")
                return False
        
        # Register allocation
        self.allocations[model_id] = (size_bytes, time.time())
        self.allocations.move_to_end(model_id)
        
        logger.info(
            f"Allocated {size_bytes / 1024**2:.1f}MB for {model_id} "
            f"({len(self.allocations)} models in VRAM)"
        )
        
        return True
    
    def deallocate(self, model_id: str) -> bool:
        """
        Deallocate model
        
        Args:
            model_id: Model identifier
            
        Returns:
            True if deallocated
        """
        if model_id in self.allocations:
            size, _ = self.allocations.pop(model_id)
            logger.info(f"Deallocated {size / 1024**2:.1f}MB from {model_id}")
            
            # Clear CUDA cache
            if self.cuda_available:
                torch.cuda.empty_cache()
            
            return True
        
        return False
    
    def _evict_lru(self) -> bool:
        """
        Evict least recently used model
        
        Returns:
            True if evicted successfully
        """
        if not self.allocations:
            return False
        
        # Get LRU model (first in OrderedDict)
        lru_model_id = next(iter(self.allocations))
        size, last_access = self.allocations[lru_model_id]
        
        logger.warning(
            f"Evicting LRU model: {lru_model_id} "
            f"({size / 1024**2:.1f}MB, "
            f"last_access={time.time() - last_access:.1f}s ago)"
        )
        
        # Note: Actual model unloading should be done by ModelCache
        # We just track the allocation here
        self.deallocate(lru_model_id)
        
        return True
    
    def get_current_usage(self) -> int:
        """
        Get current VRAM usage
        
        Returns:
            Used VRAM in bytes
        """
        if not self.cuda_available:
            return 0
        
        return torch.cuda.memory_allocated(self.device)
    
    def get_usage_percent(self) -> float:
        """
        Get VRAM usage as percentage
        
        Returns:
            Usage percentage (0-100)
        """
        if not self.cuda_available:
            return 0.0
        
        current = self.get_current_usage()
        return (current / self.max_vram_bytes) * 100
    
    def should_evict(self) -> bool:
        """Check if we should evict models"""
        return self.get_usage_percent() > self.threshold_percent
    
    def get_stats(self) -> Dict:
        """
        Get memory statistics
        
        Returns:
            Dictionary with stats
        """
        if not self.cuda_available:
            return {'cuda_available': False}
        
        current = self.get_current_usage()
        reserved = torch.cuda.memory_reserved(self.device)
        
        return {
            'cuda_available': True,
            'device': self.device_name,
            'total_vram': self.total_vram,
            'max_vram': self.max_vram_bytes,
            'current_usage': current,
            'reserved': reserved,
            'usage_percent': self.get_usage_percent(),
            'num_models': len(self.allocations),
            'models': list(self.allocations.keys())
        }
    
    def clear_cache(self):
        """Clear CUDA cache"""
        if self.cuda_available:
            torch.cuda.empty_cache()
            logger.debug("CUDA cache cleared")
    
    def optimize_memory(self):
        """Run memory optimization"""
        if not self.cuda_available:
            return
        
        # Clear cache
        self.clear_cache()
        
        # Evict if needed
        while self.should_evict():
            if not self._evict_lru():
                break
        
        logger.info(f"Memory optimized: {self.get_usage_percent():.1f}% used")


# Singleton instance
_gpu_manager_instance: Optional[GPUMemoryManager] = None


def get_gpu_manager(
    max_vram_gb: float = 8.0,
    **kwargs
) -> GPUMemoryManager:
    """Get singleton GPU memory manager"""
    global _gpu_manager_instance
    
    if _gpu_manager_instance is None:
        _gpu_manager_instance = GPUMemoryManager(
            max_vram_gb=max_vram_gb,
            **kwargs
        )
    
    return _gpu_manager_instance