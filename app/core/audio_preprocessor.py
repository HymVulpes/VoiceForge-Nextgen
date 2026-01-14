"""
VoiceForge-Nextgen - Audio Preprocessor
File: app/core/audio_preprocessor.py

Purpose:
    Preprocess audio before RVC inference
    Resample, normalize, apply filters

Dependencies:
    - numpy (array operations)
    - scipy (signal processing)
    - librosa (audio utilities)

Data Flow:
    Raw audio → resample → normalize → pre-emphasis → trim silence → output

Usage:
    preprocessor = AudioPreprocessor(target_sr=48000)
    processed = preprocessor.process(audio)
"""

import numpy as np
from scipy import signal
from typing import Optional
import logging

logger = logging.getLogger("AudioPreprocessor")


class AudioPreprocessor:
    """
    Preprocesses audio for RVC inference
    
    Steps:
        1. Resample to target sample rate
        2. Normalize amplitude
        3. Apply pre-emphasis filter
        4. Trim/pad to fixed length (optional)
        5. Remove DC offset
    """
    
    def __init__(
        self,
        target_sr: int = 48000,
        normalize: bool = True,
        pre_emphasis: float = 0.97,
        trim_silence: bool = False,
        fixed_length: Optional[int] = None
    ):
        """
        Initialize preprocessor
        
        Args:
            target_sr: Target sample rate
            normalize: Apply normalization
            pre_emphasis: Pre-emphasis coefficient (0-1)
            trim_silence: Remove leading/trailing silence
            fixed_length: Fixed length in samples (pad/trim)
        """
        self.target_sr = target_sr
        self.normalize = normalize
        self.pre_emphasis = pre_emphasis
        self.trim_silence = trim_silence
        self.fixed_length = fixed_length
    
    def process(
        self,
        audio: np.ndarray,
        source_sr: Optional[int] = None
    ) -> np.ndarray:
        """
        Process audio through pipeline
        
        Args:
            audio: Input audio (float32, -1 to 1)
            source_sr: Source sample rate (for resampling)
            
        Returns:
            Processed audio
        """
        if audio.size == 0:
            logger.warning("Empty audio input")
            return audio
        
        # Ensure float32
        if audio.dtype != np.float32:
            audio = audio.astype(np.float32)
        
        # Ensure 1D
        if audio.ndim > 1:
            audio = audio.mean(axis=1)  # Mono conversion
        
        # Step 1: Resample if needed
        if source_sr is not None and source_sr != self.target_sr:
            audio = self._resample(audio, source_sr, self.target_sr)
        
        # Step 2: Remove DC offset
        audio = audio - np.mean(audio)
        
        # Step 3: Normalize
        if self.normalize:
            audio = self._normalize(audio)
        
        # Step 4: Pre-emphasis
        if self.pre_emphasis > 0:
            audio = self._apply_pre_emphasis(audio, self.pre_emphasis)
        
        # Step 5: Trim silence
        if self.trim_silence:
            audio = self._trim_silence(audio)
        
        # Step 6: Pad/trim to fixed length
        if self.fixed_length is not None:
            audio = self._pad_or_trim(audio, self.fixed_length)
        
        # Validate output
        if not np.isfinite(audio).all():
            logger.error("Non-finite values in processed audio")
            audio = np.nan_to_num(audio, nan=0.0, posinf=0.0, neginf=0.0)
        
        return audio
    
    def _resample(
        self,
        audio: np.ndarray,
        orig_sr: int,
        target_sr: int
    ) -> np.ndarray:
        """
        Resample audio using scipy
        
        Args:
            audio: Input audio
            orig_sr: Original sample rate
            target_sr: Target sample rate
            
        Returns:
            Resampled audio
        """
        if orig_sr == target_sr:
            return audio
        
        # Calculate new length
        num_samples = int(len(audio) * target_sr / orig_sr)
        
        # Use scipy resample (high quality)
        try:
            resampled = signal.resample(audio, num_samples)
            return resampled.astype(np.float32)
        except Exception as e:
            logger.error(f"Resample failed: {e}")
            return audio
    
    def _normalize(self, audio: np.ndarray, target_db: float = -20.0) -> np.ndarray:
        """
        Normalize audio to target dB
        
        Args:
            audio: Input audio
            target_db: Target dB level
            
        Returns:
            Normalized audio
        """
        # Calculate RMS
        rms = np.sqrt(np.mean(audio ** 2))
        
        if rms < 1e-6:
            logger.warning("Audio too quiet, skipping normalization")
            return audio
        
        # Calculate target RMS from dB
        target_rms = 10 ** (target_db / 20)
        
        # Scale
        scale = target_rms / rms
        normalized = audio * scale
        
        # Prevent clipping
        peak = np.abs(normalized).max()
        if peak > 1.0:
            normalized = normalized / peak * 0.95
        
        return normalized
    
    def _apply_pre_emphasis(
        self,
        audio: np.ndarray,
        coeff: float
    ) -> np.ndarray:
        """
        Apply pre-emphasis filter
        
        Args:
            audio: Input audio
            coeff: Pre-emphasis coefficient
            
        Returns:
            Filtered audio
        """
        if coeff <= 0 or coeff >= 1:
            return audio
        
        # y[n] = x[n] - coeff * x[n-1]
        emphasized = np.append(audio[0], audio[1:] - coeff * audio[:-1])
        return emphasized.astype(np.float32)
    
    def _trim_silence(
        self,
        audio: np.ndarray,
        threshold_db: float = -40.0
    ) -> np.ndarray:
        """
        Trim leading and trailing silence
        
        Args:
            audio: Input audio
            threshold_db: Silence threshold in dB
            
        Returns:
            Trimmed audio
        """
        # Convert to dB
        threshold = 10 ** (threshold_db / 20)
        
        # Find non-silent regions
        mask = np.abs(audio) > threshold
        
        if not mask.any():
            logger.warning("Entire audio is silence")
            return audio
        
        # Find start and end
        indices = np.where(mask)[0]
        start = max(0, indices[0] - int(0.1 * self.target_sr))  # 100ms padding
        end = min(len(audio), indices[-1] + int(0.1 * self.target_sr))
        
        return audio[start:end]
    
    def _pad_or_trim(self, audio: np.ndarray, target_length: int) -> np.ndarray:
        """
        Pad or trim audio to fixed length
        
        Args:
            audio: Input audio
            target_length: Target length in samples
            
        Returns:
            Fixed-length audio
        """
        current_length = len(audio)
        
        if current_length == target_length:
            return audio
        
        if current_length < target_length:
            # Pad with zeros
            padding = target_length - current_length
            return np.pad(audio, (0, padding), mode='constant')
        else:
            # Trim
            return audio[:target_length]
    
    def get_params(self) -> dict:
        """Get preprocessor parameters"""
        return {
            'target_sr': self.target_sr,
            'normalize': self.normalize,
            'pre_emphasis': self.pre_emphasis,
            'trim_silence': self.trim_silence,
            'fixed_length': self.fixed_length
        }