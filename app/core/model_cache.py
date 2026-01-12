"""
Multi-Model Hot Cache
Pre-load and cache multiple RVC models in GPU/CPU memory
With 24GB RAM, allocate 8GB for model cache
CORRECTNESS: Model validation before caching
STABILITY: LRU eviction when cache full
PERFORMANCE: Zero-latency model switching
"""
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    torch = None

from pathlib import Path
from typing import Dict, Optional, Any
from collections import OrderedDict
import logging
import time

from .model_loader import ModelLoader

logger = logging.getLogger(__name__)

class ModelCacheEntry:
    """Cached model with metadata"""
    
    def __init__(
        self,
        model_id: str,
        model_data: Dict[str, Any],
        device: str = "cuda"
    ):
        self.model_id = model_id
        self.model_data = model_data
        self.device = device
        
        # Calculate memory usage
        self.memory_bytes = 0
        if "model" in model_data and TORCH_AVAILABLE and torch is not None:
            model = model_data["model"]
            if isinstance(model, torch.nn.Module):
                # Estimate model size
                param_count = sum(p.numel() for p in model.parameters())
                self.memory_bytes = param_count * 4  # float32 = 4 bytes
            elif isinstance(model, dict):
                # State dict
                self.memory_bytes = sum(t.numel() * 4 for t in model.values() if isinstance(t, torch.Tensor))
        
        # Add index size if present
        if "index" in model_data and model_data["index"] is not None:
            # FAISS index size estimation
            self.memory_bytes += 100 * 1024 * 1024  # ~100MB estimate
        
        self.last_access = time.time()
        self.access_count = 0
        self.is_warmed_up = False
    
    def validate(self) -> bool:
        """
        Validate model integrity
        
        CORRECTNESS: Ensure model is usable
        """
        if "model" not in self.model_data and "checkpoint" not in self.model_data:
            logger.error(f"Model {self.model_id} missing model data")
            return False
        
        # Check if model is on correct device
        if "model" in self.model_data and TORCH_AVAILABLE and torch is not None:
            model = self.model_data["model"]
            if isinstance(model, torch.nn.Module):
                # Check device
                try:
                    first_param = next(model.parameters())
                    if first_param.device.type != self.device:
                        logger.warning(f"Model {self.model_id} on wrong device")
                        return False
                except StopIteration:
                    pass
        
        return True

class ModelCache:
    """
    LRU cache for RVC models
    
    With 24GB RAM, allocate 8GB for models:
    - Each model: ~50-200MB (depends on architecture)
    - Can cache 40-160 models
    """
    
    def __init__(
        self,
        max_cache_size_gb: float = 8.0,
        device: str = "cuda"
    ):
        self.max_cache_size_bytes = int(max_cache_size_gb * (1024**3))
        self.device = device
        
        # Verify device
        if device == "cuda":
            if not TORCH_AVAILABLE or torch is None:
                logger.warning("PyTorch not available, using CPU")
                self.device = "cpu"
            elif not torch.cuda.is_available():
                logger.warning("CUDA requested but not available, using CPU")
                self.device = "cpu"
        
        # LRU cache using OrderedDict
        self.cache: OrderedDict[str, ModelCacheEntry] = OrderedDict()
        self.current_memory_bytes = 0
        
        # Model loader
        self.model_loader = ModelLoader(device=self.device)
        
        # Statistics
        self.hits = 0
        self.misses = 0
        self.evictions = 0
        self.validation_failures = 0
        self.load_count = 0
        
        logger.info(f"ModelCache initialized: max_size={max_cache_size_gb}GB, device={self.device}")
    
    def get(self, model_id: str) -> Optional[Dict[str, Any]]:
        """
        Get cached model
        
        CORRECTNESS: Validates before returning
        PERFORMANCE: O(1) lookup
        
        Args:
            model_id: Model identifier (path or name)
            
        Returns:
            Model data dict or None if not cached
        """
        if model_id in self.cache:
            # Move to end (LRU update)
            entry = self.cache.pop(model_id)
            self.cache[model_id] = entry
            
            entry.last_access = time.time()
            entry.access_count += 1
            self.hits += 1
            
            # CORRECTNESS: Validate before returning
            if not entry.validate():
                logger.error(f"Cached model failed validation: {model_id}")
                self.validation_failures += 1
                self._remove_entry(model_id)
                return None
            
            return entry.model_data
        else:
            self.misses += 1
            return None
    
    def put(
        self,
        model_id: str,
        model_data: Dict[str, Any],
        warmup: bool = False
    ) -> bool:
        """
        Cache model
        
        STABILITY: Handles cache overflow gracefully
        CORRECTNESS: Validates before caching
        
        Args:
            model_id: Model identifier
            model_data: Model data from ModelLoader
            warmup: If True, run inference once to warm up GPU
            
        Returns:
            True if cached successfully
        """
        # Create entry
        entry = ModelCacheEntry(
            model_id=model_id,
            model_data=model_data,
            device=self.device
        )
        
        # CORRECTNESS: Validate before caching
        if not entry.validate():
            logger.error("Model failed validation, not caching")
            return False
        
        # Evict until we have space
        while (self.current_memory_bytes + entry.memory_bytes > self.max_cache_size_bytes
               and len(self.cache) > 0):
            self._evict_lru()
        
        # Check if this model fits at all
        if entry.memory_bytes > self.max_cache_size_bytes:
            logger.warning(f"Model too large: {entry.memory_bytes/(1024**3):.2f}GB")
            return False
        
        # Add to cache
        if model_id in self.cache:
            # Update existing
            old_entry = self.cache[model_id]
            self.current_memory_bytes -= old_entry.memory_bytes
        
        self.cache[model_id] = entry
        self.current_memory_bytes += entry.memory_bytes
        
        # Warmup if requested
        if warmup and not entry.is_warmed_up:
            self._warmup_model(entry)
            entry.is_warmed_up = True
        
        return True
    
    def load_and_cache(
        self,
        pth_path: str,
        index_path: Optional[str] = None,
        warmup: bool = False
    ) -> Optional[Dict[str, Any]]:
        """
        Load model from disk and cache
        
        Args:
            pth_path: Path to .pth file
            index_path: Optional path to .index file
            warmup: If True, warm up GPU
            
        Returns:
            Model data or None on failure
        """
        model_id = str(Path(pth_path).absolute())
        
        # Check cache first
        cached = self.get(model_id)
        if cached is not None:
            return cached
        
        # Load from disk
        try:
            logger.info(f"Loading model: {Path(pth_path).name}")
            model_data = self.model_loader.load_model(pth_path, index_path)
            
            # Validate loaded model
            if not self.model_loader.validate_model(model_data):
                logger.error("Loaded model failed validation")
                return None
            
            # Cache it
            if self.put(model_id, model_data, warmup=warmup):
                self.load_count += 1
                return model_data
            else:
                logger.warning("Failed to cache model (cache full)")
                return model_data  # Return anyway, just not cached
                
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            return None
    
    def _evict_lru(self):
        """Evict least recently used model"""
        if not self.cache:
            return
        
        # OrderedDict: first item is LRU
        model_id, entry = self.cache.popitem(last=False)
        self._remove_entry(model_id)
        self.evictions += 1
        
        logger.info(f"Evicted model: {model_id}")
    
    def _remove_entry(self, model_id: str):
        """Remove entry and free memory"""
        if model_id not in self.cache:
            return
        
        entry = self.cache[model_id]
        self.current_memory_bytes -= entry.memory_bytes
        
        # Clear GPU memory if CUDA
        if self.device == "cuda" and TORCH_AVAILABLE and torch is not None and "model" in entry.model_data:
            model = entry.model_data.get("model")
            if isinstance(model, torch.nn.Module):
                del model
                torch.cuda.empty_cache()
        
        del self.cache[model_id]
    
    def _warmup_model(self, entry: ModelCacheEntry):
        """Warm up model with dummy inference"""
        if not TORCH_AVAILABLE or torch is None:
            return
        
        try:
            model = entry.model_data.get("model")
            if isinstance(model, torch.nn.Module):
                # Create dummy input
                dummy_input = torch.randn(1, 256, device=self.device)
                
                # Run inference
                with torch.no_grad():
                    try:
                        _ = model(dummy_input)
                    except Exception:
                        # Some models have different input shapes
                        pass
                
                logger.debug(f"Warmed up model: {entry.model_id}")
        except Exception as e:
            logger.warning(f"Failed to warm up model: {e}")
    
    def get_stats(self) -> dict:
        """Get cache statistics"""
        return {
            "cached_models": len(self.cache),
            "memory_used_gb": self.current_memory_bytes / (1024**3),
            "memory_max_gb": self.max_cache_size_bytes / (1024**3),
            "memory_percent": (self.current_memory_bytes / self.max_cache_size_bytes) * 100,
            "hits": self.hits,
            "misses": self.misses,
            "hit_rate": self.hits / (self.hits + self.misses) if (self.hits + self.misses) > 0 else 0,
            "evictions": self.evictions,
            "load_count": self.load_count,
            "validation_failures": self.validation_failures,
            "device": self.device
        }
    
    def clear(self):
        """Clear entire cache"""
        # Free all models
        for model_id in list(self.cache.keys()):
            self._remove_entry(model_id)
        
        self.cache.clear()
        self.current_memory_bytes = 0
        
        if self.device == "cuda" and TORCH_AVAILABLE and torch is not None:
            torch.cuda.empty_cache()
        
        logger.info("Model cache cleared")

