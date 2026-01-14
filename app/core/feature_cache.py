"""
Predictive Feature Cache
Pre-compute F0, HuBERT features ahead of inference
With 24GB RAM, allocate 4GB for feature caching
CORRECTNESS: Feature validation
STABILITY: Bounded cache size
PERFORMANCE: Zero-latency feature retrieval
"""

import numpy as np
from typing import Dict, Optional, Tuple
from collections import OrderedDict
import hashlib
import logging

# =========================
# NEW (Claude Update 3)
# =========================
from app.core.feature_extractor import F0Extractor, HubertExtractor

# =========================

logger = logging.getLogger(__name__)


class FeatureCacheEntry:
    """Cached feature set for audio chunk"""

    def __init__(
        self,
        audio_hash: str,
        f0: np.ndarray,
        voiced_mask: Optional[np.ndarray] = None,
        hubert_features: Optional[np.ndarray] = None,
    ):
        self.audio_hash = audio_hash
        self.f0 = f0
        self.voiced_mask = voiced_mask
        self.hubert_features = hubert_features

        # Calculate memory usage
        self.memory_bytes = f0.nbytes
        if voiced_mask is not None:
            self.memory_bytes += voiced_mask.nbytes
        if hubert_features is not None:
            self.memory_bytes += hubert_features.nbytes

        self.access_count = 0

    def validate(self) -> bool:
        """
        Validate feature integrity

        CORRECTNESS: Ensure features are usable
        """
        if self.f0 is None or self.f0.size == 0:
            return False

        if not np.isfinite(self.f0).all():
            logger.error("Non-finite F0 values")
            return False

        if self.voiced_mask is not None:
            if self.voiced_mask.shape[0] != self.f0.shape[0]:
                logger.error("F0 and voiced mask shape mismatch")
                return False

        if self.hubert_features is not None:
            if not np.isfinite(self.hubert_features).all():
                logger.error("Non-finite HuBERT features")
                return False

        return True


class FeatureCache:
    """
    LRU cache for extracted audio features
    """

    def __init__(self, max_cache_size_gb: float = 4.0):
        self.max_cache_size_bytes = int(max_cache_size_gb * (1024**3))

        self.cache: OrderedDict[str, FeatureCacheEntry] = OrderedDict()
        self.current_memory_bytes = 0

        self.hits = 0
        self.misses = 0
        self.evictions = 0
        self.validation_failures = 0

        # =========================
        # NEW (Claude Update 3)
        # =========================
        self.device = "cpu"
        self.f0_extractor = None
        self.hubert_extractor = None
        # =========================

        logger.info(f"FeatureCache initialized: max_size={max_cache_size_gb}GB")

    def _compute_hash(self, audio: np.ndarray) -> str:
        if audio.size > 2000:
            samples = np.concatenate([audio[:1000], audio[-1000:]])
        else:
            samples = audio

        audio_bytes = samples.tobytes()
        hash_obj = hashlib.blake2b(audio_bytes, digest_size=16)
        shape_str = f"{audio.shape}_{audio.dtype}".encode()
        hash_obj.update(shape_str)

        return hash_obj.hexdigest()

    def get(self, audio: np.ndarray) -> Optional[FeatureCacheEntry]:
        audio_hash = self._compute_hash(audio)

        if audio_hash in self.cache:
            entry = self.cache.pop(audio_hash)
            self.cache[audio_hash] = entry

            entry.access_count += 1
            self.hits += 1

            if not entry.validate():
                self.validation_failures += 1
                del self.cache[audio_hash]
                self.current_memory_bytes -= entry.memory_bytes
                return None

            return entry

        self.misses += 1
        return None

    def put(
        self,
        audio: np.ndarray,
        f0: np.ndarray,
        voiced_mask: Optional[np.ndarray] = None,
        hubert_features: Optional[np.ndarray] = None,
    ) -> bool:
        audio_hash = self._compute_hash(audio)

        entry = FeatureCacheEntry(
            audio_hash=audio_hash,
            f0=f0,
            voiced_mask=voiced_mask,
            hubert_features=hubert_features,
        )

        if not entry.validate():
            return False

        while (
            self.current_memory_bytes + entry.memory_bytes > self.max_cache_size_bytes
            and len(self.cache) > 0
        ):
            self._evict_lru()

        if entry.memory_bytes > self.max_cache_size_bytes:
            return False

        if audio_hash in self.cache:
            old_entry = self.cache[audio_hash]
            self.current_memory_bytes -= old_entry.memory_bytes

        self.cache[audio_hash] = entry
        self.current_memory_bytes += entry.memory_bytes

        return True

    def _evict_lru(self):
        if not self.cache:
            return

        _, entry = self.cache.popitem(last=False)
        self.current_memory_bytes -= entry.memory_bytes
        self.evictions += 1

    # ======================================================
    # NEW (Claude Update 3) – ADD METHODS ONLY
    # ======================================================

    def _init_extractors(self):
        """Initialize extractors lazily"""
        if self.f0_extractor is None:
            self.f0_extractor = F0Extractor(method="harvest", sr=48000)

        if self.hubert_extractor is None:
            self.hubert_extractor = HubertExtractor(device=self.device)

    def get_or_extract_f0(
        self, audio: np.ndarray, method: str = "harvest"
    ) -> np.ndarray:
        cached = self.get(audio)
        if cached is not None and cached.f0 is not None:
            return cached.f0

        self._init_extractors()

        if self.f0_extractor.method != method:
            self.f0_extractor = F0Extractor(method=method, sr=48000)

        f0 = self.f0_extractor.extract(audio)
        self.put(audio, f0=f0)

        return f0

    def get_or_extract_hubert(self, audio: np.ndarray) -> np.ndarray:
        cached = self.get(audio)
        if cached is not None and cached.hubert_features is not None:
            return cached.hubert_features

        self._init_extractors()
        features = self.hubert_extractor.extract(audio, sr=48000)

        if cached:
            self.put(
                audio,
                f0=cached.f0,
                voiced_mask=cached.voiced_mask,
                hubert_features=features,
            )
        else:
            dummy_f0 = np.zeros(features.shape[0], dtype=np.float32)
            self.put(audio, f0=dummy_f0, hubert_features=features)

        return features

    # ======================================================

    def get_stats(self) -> dict:
        return {
            "cached_entries": len(self.cache),
            "memory_used_gb": self.current_memory_bytes / (1024**3),
            "memory_max_gb": self.max_cache_size_bytes / (1024**3),
            "memory_percent": (self.current_memory_bytes / self.max_cache_size_bytes)
            * 100,
            "hits": self.hits,
            "misses": self.misses,
            "hit_rate": (
                self.hits / (self.hits + self.misses)
                if (self.hits + self.misses) > 0
                else 0
            ),
            "evictions": self.evictions,
            "validation_failures": self.validation_failures,
        }

    def clear(self):
        self.cache.clear()
        self.current_memory_bytes = 0
        logger.info("Feature cache cleared")
