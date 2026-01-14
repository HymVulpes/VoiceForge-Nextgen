"""
VoiceForge-Nextgen - Audio Quality Tests
File: tests/test_audio_quality.py

Purpose:
    Validate audio quality metrics
    Ensure no degradation in pipeline

Dependencies:
    - pytest
    - numpy
    - scipy (signal metrics)

Usage:
    pytest tests/test_audio_quality.py -v
"""

import pytest
import numpy as np
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT_DIR))

from app.core.audio_preprocessor import AudioPreprocessor
from app.core.audio_postprocessor import AudioPostprocessor


class TestAudioMetrics:
    """Test audio quality metrics"""
    
    def calculate_snr(self, signal: np.ndarray, noise: np.ndarray) -> float:
        """
        Calculate Signal-to-Noise Ratio
        
        Args:
            signal: Original signal
            noise: Noise (difference)
            
        Returns:
            SNR in dB
        """
        signal_power = np.mean(signal ** 2)
        noise_power = np.mean(noise ** 2)
        
        if noise_power < 1e-10:
            return float('inf')
        
        snr = 10 * np.log10(signal_power / noise_power)
        return snr
    
    def test_snr_passthrough(self):
        """Test SNR for passthrough (should be very high)"""
        audio = np.random.randn(48000).astype(np.float32) * 0.1
        
        # Passthrough
        output = audio.copy()
        
        # Calculate SNR
        noise = output - audio
        snr = self.calculate_snr(audio, noise)
        
        # Should be infinite (perfect copy)
        assert snr > 100  # Very high SNR
    
    def test_snr_with_processing(self):
        """Test SNR after preprocessing"""
        audio = np.random.randn(48000).astype(np.float32) * 0.1
        
        preprocessor = AudioPreprocessor(target_sr=48000, normalize=False)
        processed = preprocessor.process(audio, source_sr=48000)
        
        # Calculate SNR
        noise = processed - audio
        snr = self.calculate_snr(audio, noise)
        
        # Should still be high (> 40 dB)
        assert snr > 40
    
    def test_no_clipping(self):
        """Test output has no clipping"""
        # Create loud audio
        audio = np.random.randn(48000).astype(np.float32) * 0.9
        
        postprocessor = AudioPostprocessor(apply_limiter=True)
        output = postprocessor.process(audio)
        
        # Check no clipping
        assert np.abs(output).max() <= 1.0
        assert (output >= -1.0).all()
        assert (output <= 1.0).all()
    
    def test_no_nan_inf(self):
        """Test output has no NaN or Inf"""
        audio = np.random.randn(48000).astype(np.float32) * 0.1
        
        preprocessor = AudioPreprocessor(target_sr=48000)
        processed = preprocessor.process(audio, source_sr=48000)
        
        # Check finite
        assert np.isfinite(processed).all()
        assert not np.isnan(processed).any()
        assert not np.isinf(processed).any()
    
    def test_frequency_response(self):
        """Test frequency response preservation"""
        sr = 48000
        duration = 1.0
        
        # Generate test tones at different frequencies
        freqs = [100, 440, 1000, 5000]  # Hz
        
        for freq in freqs:
            # Generate tone
            t = np.linspace(0, duration, int(sr * duration))
            audio = np.sin(2 * np.pi * freq * t).astype(np.float32) * 0.5
            
            # Process
            preprocessor = AudioPreprocessor(target_sr=sr, normalize=False)
            processed = preprocessor.process(audio, source_sr=sr)
            
            # FFT to check frequency
            fft_orig = np.fft.rfft(audio)
            fft_proc = np.fft.rfft(processed)
            
            freqs_axis = np.fft.rfftfreq(len(audio), 1/sr)
            
            # Find peak frequency
            peak_orig = freqs_axis[np.argmax(np.abs(fft_orig))]
            peak_proc = freqs_axis[np.argmax(np.abs(fft_proc))]
            
            # Should be similar
            assert abs(peak_orig - peak_proc) < 10  # Hz


class TestDynamicRange:
    """Test dynamic range preservation"""
    
    def test_dynamic_range_preservation(self):
        """Test dynamic range is preserved"""
        # Create audio with wide dynamic range
        quiet = np.random.randn(24000).astype(np.float32) * 0.01
        loud = np.random.randn(24000).astype(np.float32) * 0.5
        audio = np.concatenate([quiet, loud])
        
        preprocessor = AudioPreprocessor(target_sr=48000, normalize=False)
        processed = preprocessor.process(audio, source_sr=48000)
        
        # Calculate RMS for each half
        rms_orig_quiet = np.sqrt(np.mean(audio[:24000] ** 2))
        rms_orig_loud = np.sqrt(np.mean(audio[24000:] ** 2))
        
        rms_proc_quiet = np.sqrt(np.mean(processed[:24000] ** 2))
        rms_proc_loud = np.sqrt(np.mean(processed[24000:] ** 2))
        
        # Ratio should be similar
        ratio_orig = rms_orig_loud / rms_orig_quiet
        ratio_proc = rms_proc_loud / rms_proc_quiet
        
        assert abs(ratio_orig - ratio_proc) / ratio_orig < 0.3  # Within 30%


class TestSilenceHandling:
    """Test silence handling"""
    
    def test_silence_passthrough(self):
        """Test silence is preserved"""
        audio = np.zeros(48000, dtype=np.float32)
        
        preprocessor = AudioPreprocessor(target_sr=48000, normalize=False)
        processed = preprocessor.process(audio, source_sr=48000)
        
        # Should still be silent
        rms = np.sqrt(np.mean(processed ** 2))
        assert rms < 1e-6
    
    def test_silence_detection(self):
        """Test silence detection"""
        # Create audio with silence
        audio = np.concatenate([
            np.random.randn(24000).astype(np.float32) * 0.1,
            np.zeros(24000, dtype=np.float32)
        ])
        
        # Check RMS
        rms_first_half = np.sqrt(np.mean(audio[:24000] ** 2))
        rms_second_half = np.sqrt(np.mean(audio[24000:] ** 2))
        
        assert rms_first_half > 0.01
        assert rms_second_half < 1e-6


class TestLatencyQuality:
    """Test quality vs latency tradeoffs"""
    
    def test_processing_adds_minimal_distortion(self):
        """Test processing doesn't add significant distortion"""
        audio = np.random.randn(48000).astype(np.float32) * 0.1
        
        # Full pipeline
        preprocessor = AudioPreprocessor(target_sr=48000)
        postprocessor = AudioPostprocessor()
        
        processed = preprocessor.process(audio, source_sr=48000)
        final = postprocessor.process(processed)
        
        # Calculate correlation
        correlation = np.corrcoef(audio[:len(final)], final)[0, 1]
        
        # Should be highly correlated
        assert correlation > 0.95


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])