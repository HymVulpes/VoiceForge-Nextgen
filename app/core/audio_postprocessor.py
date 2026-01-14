"""
VoiceForge-Nextgen - Audio Postprocessor
File: app/core/audio_postprocessor.py

Purpose:
    Post-process audio after RVC inference
    De-emphasis, smoothing, limiting

Dependencies:
    - numpy (array operations)
    - scipy (signal processing)

Data Flow:
    RVC output → de-emphasis → smooth → limit → denormalize → output

Usage:
    postprocessor = AudioPostprocessor()
    output = postprocessor.process(inference_output)
"""

import numpy as np
from scipy import signal
import logging

logger = logging.getLogger("AudioPostprocessor")


class AudioPostprocessor:
    """
    Post-processes audio after RVC inference
    
    Steps:
        1. De-emphasis (reverse pre-emphasis)
        2. Smooth transitions
        3. Apply limiter (prevent clipping)
        4. Denormalize (if needed)
        5. Final validation
    """
    
    def __init__(
        self,
        de_emphasis: float = 0.97,
        apply_limiter: bool = True,
        limiter_threshold: float = 0.95,
        smooth_transitions: bool = True,
        fade_samples: int = 100
    ):
        """
        Initialize postprocessor
        
        Args:
            de_emphasis: De-emphasis coefficient (should match pre-emphasis)
            apply_limiter: Apply soft limiter
            limiter_threshold: Limiter threshold (0-1)
            smooth_transitions: Smooth audio transitions
            fade_samples: Fade length for smoothing
        """
        self.de_emphasis = de_emphasis
        self.apply_limiter = apply_limiter
        self.limiter_threshold = limiter_threshold
        self.smooth_transitions = smooth_transitions
        self.fade_samples = fade_samples
        
        # Previous chunk for crossfade
        self.prev_chunk = None
    
    def process(self, audio: np.ndarray) -> np.ndarray:
        """
        Process audio through post-processing pipeline
        
        Args:
            audio: Inference output (float32, -1 to 1)
            
        Returns:
            Post-processed audio
        """
        if audio.size == 0:
            logger.warning("Empty audio input")
            return audio
        
        # Ensure float32
        if audio.dtype != np.float32:
            audio = audio.astype(np.float32)
        
        # Ensure 1D
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        
        # Step 1: De-emphasis
        if self.de_emphasis > 0:
            audio = self._apply_de_emphasis(audio, self.de_emphasis)
        
        # Step 2: Smooth transitions (crossfade with previous chunk)
        if self.smooth_transitions and self.prev_chunk is not None:
            audio = self._crossfade(self.prev_chunk, audio)
        
        # Step 3: Apply limiter
        if self.apply_limiter:
            audio = self._apply_limiter(audio, self.limiter_threshold)
        
        # Step 4: Final validation
        audio = self._validate_output(audio)
        
        # Store for next crossfade
        if self.smooth_transitions and len(audio) > self.fade_samples:
            self.prev_chunk = audio[-self.fade_samples:].copy()
        
        return audio
    
    def _apply_de_emphasis(
        self,
        audio: np.ndarray,
        coeff: float
    ) -> np.ndarray:
        """
        Apply de-emphasis filter (reverse of pre-emphasis)
        
        Args:
            audio: Input audio
            coeff: De-emphasis coefficient
            
        Returns:
            De-emphasized audio
        """
        if coeff <= 0 or coeff >= 1:
            return audio
        
        # y[n] = x[n] + coeff * y[n-1]
        de_emphasized = np.zeros_like(audio)
        de_emphasized[0] = audio[0]
        
        for i in range(1, len(audio)):
            de_emphasized[i] = audio[i] + coeff * de_emphasized[i - 1]
        
        return de_emphasized.astype(np.float32)
    
    def _crossfade(
        self,
        prev: np.ndarray,
        current: np.ndarray
    ) -> np.ndarray:
        """
        Crossfade between previous and current chunk
        
        Args:
            prev: Previous chunk tail
            current: Current chunk
            
        Returns:
            Crossfaded audio
        """
        fade_len = min(len(prev), len(current), self.fade_samples)
        
        if fade_len == 0:
            return current
        
        # Create fade curves
        fade_out = np.linspace(1, 0, fade_len)
        fade_in = np.linspace(0, 1, fade_len)
        
        # Apply crossfade
        result = current.copy()
        result[:fade_len] = (
            prev[:fade_len] * fade_out +
            current[:fade_len] * fade_in
        )
        
        return result
    
    def _apply_limiter(
        self,
        audio: np.ndarray,
        threshold: float
    ) -> np.ndarray:
        """
        Apply soft limiter to prevent clipping
        
        Args:
            audio: Input audio
            threshold: Limiter threshold (0-1)
            
        Returns:
            Limited audio
        """
        # Soft clipping using tanh
        peak = np.abs(audio).max()
        
        if peak > threshold:
            # Scale to threshold, then apply soft clipping
            scale = threshold / peak
            scaled = audio * scale
            
            # Apply tanh for soft knee
            limited = np.tanh(scaled / threshold) * threshold
            
            return limited.astype(np.float32)
        
        return audio
    
    def _validate_output(self, audio: np.ndarray) -> np.ndarray:
        """
        Validate output audio
        
        Args:
            audio: Output audio
            
        Returns:
            Validated audio
        """
        # Check for NaN/Inf
        if not np.isfinite(audio).all():
            logger.error("Non-finite values in output, cleaning...")
            audio = np.nan_to_num(audio, nan=0.0, posinf=0.0, neginf=0.0)
        
        # Check for clipping
        peak = np.abs(audio).max()
        if peak > 1.0:
            logger.warning(f"Output clipping detected: peak={peak:.3f}, scaling down")
            audio = audio / peak * 0.95
        
        # Check for silence
        rms = np.sqrt(np.mean(audio ** 2))
        if rms < 1e-6:
            logger.warning("Output is silent")
        
        return audio
    
    def reset(self):
        """Reset state (clear previous chunk)"""
        self.prev_chunk = None
    
    def get_params(self) -> dict:
        """Get postprocessor parameters"""
        return {
            'de_emphasis': self.de_emphasis,
            'apply_limiter': self.apply_limiter,
            'limiter_threshold': self.limiter_threshold,
            'smooth_transitions': self.smooth_transitions,
            'fade_samples': self.fade_samples
        }