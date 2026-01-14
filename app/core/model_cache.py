"""
Multi-Model Hot Cache
Pre-load and cache multiple RVC models in GPU/CPU memory
With 24GB RAM, allocate 8GB for model cache
CORRECTNESS: Model validation before caching
STABILITY: LRU eviction when cache full
PERFORMANCE: Zero-latency model switching
"""

# =========================
# Existing imports (GIỮ NGUYÊN)
# =========================
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

# =========================
# NEW IMPORTS (THEO CLAUDE)
# =========================
# NOTE: ADDED – GPU memory management & profiling
from app.core.gpu_memory_manager import get_gpu_manager
from app.core.memory_profiler import get_profiler

logger = logging.getLogger(__name__)


# ============================================================
# ModelCacheEntry (GIỮ NGUYÊN – KHÔNG CHẠM)
# ============================================================
class ModelCacheEntry:
    """Cached model with metadata"""

    def __init__(self, model_id: str, model_data: Dict[str, Any], device: str = "cuda"):
        self.model_id = model_id
        self.model_data = model_data
        self.device = device

        # Calculate memory usage
        self.memory_bytes = 0
        if "model" in model_data and TORCH_AVAILABLE and torch is not None:
            model = model_data["model"]
            if isinstance(model, torch.nn.Module):
                param_count = sum(p.numel() for p in model.parameters())
                self.memory_bytes = param_count * 4  # float32
            elif isinstance(model, dict):
                self.memory_bytes = sum(
                    t.numel() * 4 for t in model.values() if isinstance(t, torch.Tensor)
                )

        # Index size (rough estimate)
        if "index" in model_data and model_data["index"] is not None:
            self.memory_bytes += 100 * 1024 * 1024  # ~100MB

        self.last_access = time.time()
        self.access_count = 0
        self.is_warmed_up = False

    def validate(self) -> bool:
        if "model" not in self.model_data and "checkpoint" not in self.model_data:
            logger.error(f"Model {self.model_id} missing model data")
            return False

        if "model" in self.model_data and TORCH_AVAILABLE and torch is not None:
            model = self.model_data["model"]
            if isinstance(model, torch.nn.Module):
                try:
                    first_param = next(model.parameters())
                    if first_param.device.type != self.device:
                        logger.warning(f"Model {self.model_id} on wrong device")
                        return False
                except StopIteration:
                    pass

        return True


# ============================================================
# ModelCache (ĐÃ TÍCH HỢP GPU MANAGER + PROFILER)
# ============================================================
class ModelCache:
    """
    LRU cache for RVC models
    """

    def __init__(self, max_cache_size_gb: float = 8.0, device: str = "cuda"):
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

        # LRU cache
        self.cache: OrderedDict[str, ModelCacheEntry] = OrderedDict()
        self.current_memory_bytes = 0

        # Model loader
        self.model_loader = ModelLoader(device=self.device)

        # =========================
        # NEW: GPU MEMORY MANAGER
        # =========================
        # NOTE: ADDED – theo Claude Update 4
        self.gpu_manager = get_gpu_manager(max_vram_gb=max_cache_size_gb)

        # =========================
        # NEW: MEMORY PROFILER
        # =========================
        # NOTE: ADDED – theo Claude Update 4
        self.profiler = get_profiler()

        # Statistics
        self.hits = 0
        self.misses = 0
        self.evictions = 0
        self.validation_failures = 0
        self.load_count = 0

        logger.info(
            f"ModelCache initialized: max_size={max_cache_size_gb}GB, device={self.device}"
        )
        logger.info(f"ModelCache with GPU management enabled")

    # ========================================================
    # GET (GIỮ NGUYÊN)
    # ========================================================
    def get(self, model_id: str) -> Optional[Dict[str, Any]]:
        if model_id in self.cache:
            entry = self.cache.pop(model_id)
            self.cache[model_id] = entry

            entry.last_access = time.time()
            entry.access_count += 1
            self.hits += 1

            if not entry.validate():
                logger.error(f"Cached model failed validation: {model_id}")
                self.validation_failures += 1
                self._remove_entry(model_id)
                return None

            return entry.model_data

        self.misses += 1
        return None

    # ========================================================
    # PUT (ĐÃ TÍCH HỢP GPU MANAGER + PROFILER)
    # ========================================================
    def put(
        self, model_id: str, model_data: Dict[str, Any], warmup: bool = False
    ) -> bool:
        """
        Cache model with GPU memory management
        """

        # =========================
        # NEW: Estimate model size
        # =========================
        model_size = self._estimate_model_size(model_data)

        # =========================
        # NEW: Check GPU allocation
        # =========================
        if not self.gpu_manager.can_allocate(model_size):
            logger.warning(f"Insufficient VRAM for {model_id}")

            while not self.gpu_manager.can_allocate(model_size):
                if not self._evict_lru():
                    logger.error("Cannot free enough VRAM")
                    return False

        # =========================
        # NEW: Profile loading
        # =========================
        with self.profiler.profile(f"load_model_{model_id}"):

            entry = ModelCacheEntry(
                model_id=model_id, model_data=model_data, device=self.device
            )

            if not entry.validate():
                logger.error("Model failed validation, not caching")
                return False

            while (
                self.current_memory_bytes + entry.memory_bytes
                > self.max_cache_size_bytes
                and len(self.cache) > 0
            ):
                self._evict_lru()

            if entry.memory_bytes > self.max_cache_size_bytes:
                logger.warning(
                    f"Model too large: {entry.memory_bytes / (1024 ** 3):.2f}GB"
                )
                return False

            if model_id in self.cache:
                old_entry = self.cache[model_id]
                self.current_memory_bytes -= old_entry.memory_bytes

            self.cache[model_id] = entry
            self.current_memory_bytes += entry.memory_bytes

            if warmup and not entry.is_warmed_up:
                self._warmup_model(entry)
                entry.is_warmed_up = True

            # =========================
            # NEW: Register GPU allocation
            # =========================
            self.gpu_manager.allocate(model_id, model_size)

        return True

    # ========================================================
    # LOAD & CACHE (GIỮ NGUYÊN)
    # ========================================================
    def load_and_cache(
        self, pth_path: str, index_path: Optional[str] = None, warmup: bool = False
    ) -> Optional[Dict[str, Any]]:
        model_id = str(Path(pth_path).absolute())

        cached = self.get(model_id)
        if cached is not None:
            return cached

        try:
            logger.info(f"Loading model: {Path(pth_path).name}")
            model_data = self.model_loader.load_model(pth_path, index_path)

            if not self.model_loader.validate_model(model_data):
                logger.error("Loaded model failed validation")
                return None

            if self.put(model_id, model_data, warmup=warmup):
                self.load_count += 1
                return model_data

            logger.warning("Failed to cache model (cache full)")
            return model_data

        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            return None

    # ========================================================
    # EVICT LRU (UPDATED – GPU DEALLOCATE)
    # ========================================================
    def _evict_lru(self) -> bool:
        if not self.cache:
            return False

        model_id, entry = self.cache.popitem(last=False)
        self._remove_entry(model_id)

        # =========================
        # NEW: GPU deallocation
        # =========================
        self.gpu_manager.deallocate(model_id)
        self.gpu_manager.clear_cache()

        self.evictions += 1
        logger.info(f"Evicted model: {model_id}")

        return True

    # ========================================================
    # REMOVE ENTRY (GIỮ NGUYÊN)
    # ========================================================
    def _remove_entry(self, model_id: str):
        if model_id not in self.cache:
            return

        entry = self.cache[model_id]
        self.current_memory_bytes -= entry.memory_bytes

        if (
            self.device == "cuda"
            and TORCH_AVAILABLE
            and torch is not None
            and "model" in entry.model_data
        ):
            model = entry.model_data.get("model")
            if isinstance(model, torch.nn.Module):
                del model
                torch.cuda.empty_cache()

        del self.cache[model_id]

    # ========================================================
    # WARMUP (GIỮ NGUYÊN)
    # ========================================================
    def _warmup_model(self, entry: ModelCacheEntry):
        if not TORCH_AVAILABLE or torch is None:
            return

        try:
            model = entry.model_data.get("model")
            if isinstance(model, torch.nn.Module):
                dummy_input = torch.randn(1, 256, device=self.device)
                with torch.no_grad():
                    try:
                        _ = model(dummy_input)
                    except Exception:
                        pass
        except Exception as e:
            logger.warning(f"Failed to warm up model: {e}")

    # ========================================================
    # NEW: MODEL SIZE ESTIMATION (CLAUDE)
    # ========================================================
    def _estimate_model_size(self, model_data: dict) -> int:
        size = 0

        if "model" in model_data:
            model = model_data["model"]
            if hasattr(model, "parameters"):
                for param in model.parameters():
                    size += param.nelement() * param.element_size()

        if "index" in model_data and model_data["index"] is not None:
            try:
                index = model_data["index"]
                size += index.ntotal * index.d * 4
            except Exception:
                pass

        return size

    # ========================================================
    # STATS & CLEAR (GIỮ NGUYÊN)
    # ========================================================
    def get_stats(self) -> dict:
        return {
            "cached_models": len(self.cache),
            "memory_used_gb": self.current_memory_bytes / (1024**3),
            "memory_max_gb": self.max_cache_size_bytes / (1024**3),
            "memory_percent": (
                (self.current_memory_bytes / self.max_cache_size_bytes) * 100
                if self.max_cache_size_bytes > 0
                else 0
            ),
            "hits": self.hits,
            "misses": self.misses,
            "hit_rate": (
                (self.hits / (self.hits + self.misses))
                if (self.hits + self.misses) > 0
                else 0
            ),
            "evictions": self.evictions,
            "load_count": self.load_count,
            "validation_failures": self.validation_failures,
            "device": self.device,
        }

    def clear(self):
        for model_id in list(self.cache.keys()):
            self._remove_entry(model_id)

        self.cache.clear()
        self.current_memory_bytes = 0

        if self.device == "cuda" and TORCH_AVAILABLE and torch is not None:
            torch.cuda.empty_cache()

        logger.info("Model cache cleared")
