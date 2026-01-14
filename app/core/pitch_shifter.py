"""
VoiceForge-Nextgen - Pitch Shifter
File: app/core/pitch_shifter.py

Purpose:
    Pitch shifting algorithms for F0 manipulation
    Smooth F0 curves, interpolate unvoiced regions

Dependencies:
    - numpy (array operations)
    - scipy (signal processing)

Data Flow:
    F0 array → shift_semitones → smooth → interpolate_unvoiced → output F0

Usage:
    shifter = PitchShifter()
    shifted_f0 = shifter.shift(f0, semitones=12)
"""

import numpy as np
from scipy import signal, interpolate
import logging

logger = logging.getLogger("PitchShifter")


class PitchShifter:
    """
    Pitch shifting and F0 manipulation utilities
    
    Features:
        - Shift F0 by semitones
        - Smooth F0 curves
        - Interpolate unvoiced regions
        - Validate F0 range
    """
    
    # Valid F0 range (Hz)
    MIN_F0 = 50.0
    MAX_F0 = 1100.0
    
    def __init__(self):
        """Initialize pitch shifter"""
        pass
    
    def shift(
        self,
        f0: np.ndarray,
        semitones: float = 0.0,
        smooth_window: int = 5,
        interpolate_unvoiced: bool = True
    ) -> np.ndarray:
        """
        Shift F0 by semitones
        
        Args:
            f0: Input F0 curve (Hz, 0 = unvoiced)
            semitones: Shift amount in semitones (+/- 12)
            smooth_window: Smoothing window size
            interpolate_unvoiced: Fill unvoiced regions
            
        Returns:
            Shifted F0 curve
        """
        if f0.size == 0:
            return f0
        
        # Copy to avoid modifying original
        f0_shifted = f0.copy()
        
        # Step 1: Shift voiced regions
        if semitones != 0:
            f0_shifted = self._shift_semitones(f0_shifted, semitones)
        
        # Step 2: Interpolate unvoiced
        if interpolate_unvoiced:
            f0_shifted = self._interpolate_unvoiced(f0_shifted)
        
        # Step 3: Smooth
        if smooth_window > 1:
            f0_shifted = self._smooth_f0(f0_shifted, smooth_window)
        
        # Step 4: Validate range
        f0_shifted = self._validate_range(f0_shifted)
        
        return f0_shifted
    
    def _shift_semitones(
        self,
        f0: np.ndarray,
        semitones: float
    ) -> np.ndarray:
        """
        Shift F0 by semitones (exponential)
        
        Args:
            f0: Input F0
            semitones: Shift amount
            
        Returns:
            Shifted F0
        """
        # f_new = f_old * 2^(semitones/12)
        ratio = 2 ** (semitones / 12.0)
        
        # Only shift voiced regions (f0 > 0)
        voiced_mask = f0 > 0
        f0_shifted = f0.copy()
        f0_shifted[voiced_mask] *= ratio
        
        return f0_shifted
    
    def _interpolate_unvoiced(self, f0: np.ndarray) -> np.ndarray:
        """
        Interpolate unvoiced regions (f0 == 0)
        
        Args:
            f0: Input F0
            
        Returns:
            Interpolated F0
        """
        if not (f0 == 0).any():
            return f0  # No unvoiced regions
        
        # Find voiced regions
        voiced_mask = f0 > 0
        
        if not voiced_mask.any():
            return f0  # All unvoiced, cannot interpolate
        
        # Get indices
        voiced_indices = np.where(voiced_mask)[0]
        all_indices = np.arange(len(f0))
        
        # Interpolate
        try:
            f = interpolate.interp1d(
                voiced_indices,
                f0[voiced_indices],
                kind='linear',
                bounds_error=False,
                fill_value=(f0[voiced_indices[0]], f0[voiced_indices[-1]])
            )
            
            f0_interpolated = f0.copy()
            f0_interpolated = f(all_indices)
            
            return f0_interpolated.astype(np.float32)
            
        except Exception as e:
            logger.warning(f"Interpolation failed: {e}")
            return f0
    
    def _smooth_f0(self, f0: np.ndarray, window: int) -> np.ndarray:
        """
        Smooth F0 curve using median filter
        
        Args:
            f0: Input F0
            window: Window size (odd number)
            
        Returns:
            Smoothed F0
        """
        if window < 3:
            return f0
        
        # Ensure odd window
        if window % 2 == 0:
            window += 1
        
        # Apply median filter only to voiced regions
        voiced_mask = f0 > 0
        
        if not voiced_mask.any():
            return f0
        
        try:
            f0_smoothed = f0.copy()
            f0_smoothed[voiced_mask] = signal.medfilt(
                f0[voiced_mask],
                kernel_size=min(window, voiced_mask.sum())
            )
            
            return f0_smoothed.astype(np.float32)
            
        except Exception as e:
            logger.warning(f"Smoothing failed: {e}")
            return f0
    
    def _validate_range(self, f0: np.ndarray) -> np.ndarray:
        """
        Validate F0 is within valid range
        
        Args:
            f0: Input F0
            
        Returns:
            Validated F0
        """
        f0_validated = f0.copy()
        
        # Clip voiced regions to valid range
        voiced_mask = f0_validated > 0
        
        if voiced_mask.any():
            f0_validated[voiced_mask] = np.clip(
                f0_validated[voiced_mask],
                self.MIN_F0,
                self.MAX_F0
            )
            
            # Log warnings
            too_low = (f0[voiced_mask] < self.MIN_F0).sum()
            too_high = (f0[voiced_mask] > self.MAX_F0).sum()
            
            if too_low > 0:
                logger.warning(f"{too_low} F0 values below {self.MIN_F0} Hz (clipped)")
            if too_high > 0:
                logger.warning(f"{too_high} F0 values above {self.MAX_F0} Hz (clipped)")
        
        return f0_validated
    
    def get_f0_stats(self, f0: np.ndarray) -> dict:
        """
        Get F0 statistics
        
        Args:
            f0: Input F0
            
        Returns:
            Dictionary with statistics
        """
        voiced_mask = f0 > 0
        
        if not voiced_mask.any():
            return {
                'mean': 0.0,
                'std': 0.0,
                'min': 0.0,
                'max': 0.0,
                'voiced_ratio': 0.0
            }
        
        voiced_f0 = f0[voiced_mask]
        
        return {
            'mean': float(np.mean(voiced_f0)),
            'std': float(np.std(voiced_f0)),
            'min': float(np.min(voiced_f0)),
            'max': float(np.max(voiced_f0)),
            'voiced_ratio': float(voiced_mask.sum() / len(f0))
        }


def shift_f0(
    f0: np.ndarray,
    semitones: float = 0.0,
    smooth: bool = True
) -> np.ndarray:
    """
    Convenience function to shift F0
    
    Args:
        f0: Input F0 curve
        semitones: Shift amount
        smooth: Apply smoothing
        
    Returns:
        Shifted F0
    """
    shifter = PitchShifter()
    return shifter.shift(
        f0,
        semitones=semitones,
        smooth_window=5 if smooth else 1,
        interpolate_unvoiced=True
    )