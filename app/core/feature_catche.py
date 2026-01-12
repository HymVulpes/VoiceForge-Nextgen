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

logger = logging.getLogger(__name__)

class FeatureCacheEntry:
    """Cached feature set for audio chunk"""
    
    def __init__(
        self,
        audio_hash: str,
        f0: np.ndarray,
        voiced_mask: Optional[np.ndarray] = None,
        hubert_features: Optional[np.ndarray] = None
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
        # Check F0
        if self.f0 is None or self.f0.size == 0:
            return False
        
        if not np.isfinite(self.f0).all():
            logger.error("Non-finite F0 values")
            return False
        
        # Check voiced mask if present
        if self.voiced_mask is not None:
            if self.voiced_mask.shape[0] != self.f0.shape[0]:
                logger.error("F0 and voiced mask shape mismatch")
                return False
        
        # Check HuBERT features if present
        if self.hubert_features is not None:
            if not np.isfinite(self.hubert_features).all():
                logger.error("Non-finite HuBERT features")
                return False
        
        return True

class FeatureCache:
    """
    LRU cache for extracted audio features
    
    With 24GB RAM, allocate 4GB for features:
    - Each entry: ~1-10MB (depends on audio length)
    - Can cache 400-4000 feature sets
    """
    
    def __init__(self, max_cache_size_gb: float = 4.0):
        self.max_cache_size_bytes = int(max_cache_size_gb * (1024**3))
        
        # LRU cache using OrderedDict
        self.cache: OrderedDict[str, FeatureCacheEntry] = OrderedDict()
        self.current_memory_bytes = 0
        
        # Statistics
        self.hits = 0
        self.misses = 0
        self.evictions = 0
        self.validation_failures = 0
        
        logger.info(f"FeatureCache initialized: max_size={max_cache_size_gb}GB")
    
    def _compute_hash(self, audio: np.ndarray) -> str:
        """
        Compute deterministic hash for audio chunk
        
        CORRECTNESS: Same audio = same hash
        
        Args:
            audio: Audio samples
            
        Returns:
            Hexadecimal hash string
        """
        # Use first/last 1000 samples + length for fast hash
        if audio.size > 2000:
            samples = np.concatenate([audio[:1000], audio[-1000:]])
        else:
            samples = audio
        
        # Convert to bytes and hash
        audio_bytes = samples.tobytes()
        hash_obj = hashlib.blake2b(audio_bytes, digest_size=16)
        
        # Include shape info
        shape_str = f"{audio.shape}_{audio.dtype}".encode()
        hash_obj.update(shape_str)
        
        return hash_obj.hexdigest()
    
    def get(self, audio: np.ndarray) -> Optional[FeatureCacheEntry]:
        """
        Get cached features for audio
        
        CORRECTNESS: Validates before returning
        PERFORMANCE: O(1) lookup
        
        Args:
            audio: Audio samples to lookup
            
        Returns:
            FeatureCacheEntry or None if not cached
        """
        audio_hash = self._compute_hash(audio)
        
        if audio_hash in self.cache:
            # Move to end (LRU update)
            entry = self.cache.pop(audio_hash)
            self.cache[audio_hash] = entry
            
            entry.access_count += 1
            self.hits += 1
            
            # CORRECTNESS: Validate before returning
            if not entry.validate():
                logger.error(f"Cached features failed validation: {audio_hash}")
                self.validation_failures += 1
                del self.cache[audio_hash]
                self.current_memory_bytes -= entry.memory_bytes
                return None
            
            return entry
        else:
            self.misses += 1
            return None
    
    def put(
        self,
        audio: np.ndarray,
        f0: np.ndarray,
        voiced_mask: Optional[np.ndarray] = None,
        hubert_features: Optional[np.ndarray] = None
    ) -> bool:
        """
        Cache features for audio
        
        STABILITY: Handles cache overflow gracefully
        CORRECTNESS: Validates before caching
        
        Args:
            audio: Audio samples
            f0: Fundamental frequency
            voiced_mask: Optional voiced/unvoiced mask
            hubert_features: Optional HuBERT features
            
        Returns:
            True if cached successfully
        """
        audio_hash = self._compute_hash(audio)
        
        # Create entry
        entry = FeatureCacheEntry(
            audio_hash=audio_hash,
            f0=f0,
            voiced_mask=voiced_mask,
            hubert_features=hubert_features
        )
        
        # CORRECTNESS: Validate before caching
        if not entry.validate():
            logger.error("Features failed validation, not caching")
            return False
        
        # Evict until we have space
        while (self.current_memory_bytes + entry.memory_bytes > self.max_cache_size_bytes
               and len(self.cache) > 0):
            self._evict_lru()
        
        # Check if this entry fits at all
        if entry.memory_bytes > self.max_cache_size_bytes:
            logger.warning(f"Feature entry too large: {entry.memory_bytes/(1024**2):.1f}MB")
            return False
        
        # Add to cache
        if audio_hash in self.cache:
            # Update existing
            old_entry = self.cache[audio_hash]
            self.current_memory_bytes -= old_entry.memory_bytes
        
        self.cache[audio_hash] = entry
        self.current_memory_bytes += entry.memory_bytes
        
        return True
    
    def _evict_lru(self):
        """Evict least recently used entry"""
        if not self.cache:
            return
        
        # OrderedDict: first item is LRU
        audio_hash, entry = self.cache.popitem(last=False)
        self.current_memory_bytes -= entry.memory_bytes
        self.evictions += 1
    
    def get_stats(self) -> dict:
        """Get cache statistics"""
        return {
            "cached_entries": len(self.cache),
            "memory_used_gb": self.current_memory_bytes / (1024**3),
            "memory_max_gb": self.max_cache_size_bytes / (1024**3),
            "memory_percent": (self.current_memory_bytes / self.max_cache_size_bytes) * 100,
            "hits": self.hits,
            "misses": self.misses,
            "hit_rate": self.hits / (self.hits + self.misses) if (self.hits + self.misses) > 0 else 0,
            "evictions": self.evictions,
            "validation_failures": self.validation_failures
        }
    
    def clear(self):
        """Clear entire cache"""
        self.cache.clear()
        self.current_memory_bytes = 0
        logger.info("Feature cache cleared")


class PredictiveFeatureExtractor:
    """
    Predictive feature extraction with caching
    Pre-computes features ahead of inference
    """
    
    def __init__(
        self,
        feature_cache: FeatureCache,
        f0_method: str = "harvest"
    ):
        self.cache = feature_cache
        self.f0_method = f0_method
        
        # Import feature extraction modules (lazy)
        self._f0_extractor = None
        self._hubert_model = None
    
    def extract_f0(self, audio: np.ndarray, sample_rate: int = 48000) -> Tuple[np.ndarray, np.ndarray]:
        """
        Extract F0 with caching
        
        CORRECTNESS: Validates output
        PERFORMANCE: Cache hit = zero-latency
        
        Args:
            audio: Audio samples (float32, mono)
            sample_rate: Sample rate
            
        Returns:
            (f0, voiced_mask)
        """
        # Check cache first
        cached = self.cache.get(audio)
        if cached is not None:
            return cached.f0, cached.voiced_mask
        
        # Extract F0 (MOCK - replace with actual implementation)
        logger.debug(f"Extracting F0 ({self.f0_method}): {audio.shape[0]} samples")
        
        # MOCK: Generate dummy F0
        num_frames = len(audio) // 256  # Typical hop size
        f0 = np.random.uniform(80, 400, num_frames).astype(np.float32)
        voiced_mask = (f0 > 0).astype(np.float32)
        
        # CORRECTNESS: Validate
        if not np.isfinite(f0).all():
            logger.error("F0 extraction produced non-finite values")
            f0 = np.nan_to_num(f0)
        
        # Cache result
        self.cache.put(audio, f0=f0, voiced_mask=voiced_mask)
        
        return f0, voiced_mask
    
    def extract_hubert(self, audio: np.ndarray, sample_rate: int = 16000) -> np.ndarray:
        """
        Extract HuBERT features with caching
        
        Args:
            audio: Audio samples
            sample_rate: Sample rate (HuBERT expects 16kHz)
            
        Returns:
            HuBERT features
        """
        # Check cache
        cached = self.cache.get(audio)
        if cached is not None and cached.hubert_features is not None:
            return cached.hubert_features
        
        # Extract HuBERT (MOCK)
        logger.debug(f"Extracting HuBERT: {audio.shape[0]} samples")
        
        # MOCK: Generate dummy features
        num_frames = len(audio) // 320  # Typical HuBERT hop
        features = np.random.randn(num_frames, 256).astype(np.float32)
        
        # Cache result (retrieve existing entry if has F0)
        cached_entry = self.cache.get(audio)
        if cached_entry:
            self.cache.put(
                audio,
                f0=cached_entry.f0,
                voiced_mask=cached_entry.voiced_mask,
                hubert_features=features
            )
        else:
            # Create new entry with just HuBERT
            dummy_f0 = np.zeros(num_frames, dtype=np.float32)
            self.cache.put(audio, f0=dummy_f0, hubert_features=features)
        
        return features