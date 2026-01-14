"""
VoiceForge-Nextgen - End-to-End Pipeline Tests
File: tests/test_e2e_pipeline.py

Purpose:
    Integration tests for complete audio pipeline
    Test from audio input to final output

Dependencies:
    - pytest
    - numpy
    - app.core.* (all pipeline components)

Usage:
    pytest tests/test_e2e_pipeline.py -v
"""

import pytest
import numpy as np
import sys
from pathlib import Path

# Add app to path
ROOT_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT_DIR))

from app.core.audio_preprocessor import AudioPreprocessor
from app.core.audio_postprocessor import AudioPostprocessor
from app.core.pitch_shifter import PitchShifter
from app.core.feature_extractor import F0Extractor
from app.core.model_scanner import ModelScanner


class TestPreprocessing:
    """Test audio preprocessing"""
    
    def test_preprocessor_creation(self):
        """Test preprocessor can be created"""
        preprocessor = AudioPreprocessor(target_sr=48000)
        assert preprocessor is not None
        assert preprocessor.target_sr == 48000
    
    def test_preprocess_audio(self):
        """Test audio preprocessing"""
        # Create test audio (1 second, 48kHz)
        audio = np.random.randn(48000).astype(np.float32) * 0.1
        
        preprocessor = AudioPreprocessor(target_sr=48000, normalize=True)
        processed = preprocessor.process(audio, source_sr=48000)
        
        # Check output
        assert processed is not None
        assert processed.dtype == np.float32
        assert len(processed) == len(audio)
        assert np.isfinite(processed).all()
    
    def test_preprocess_resample(self):
        """Test resampling"""
        # Create 16kHz audio
        audio = np.random.randn(16000).astype(np.float32) * 0.1
        
        preprocessor = AudioPreprocessor(target_sr=48000)
        processed = preprocessor.process(audio, source_sr=16000)
        
        # Should be resampled to 48kHz
        assert len(processed) == 48000
    
    def test_preprocess_normalize(self):
        """Test normalization"""
        # Create quiet audio
        audio = np.random.randn(48000).astype(np.float32) * 0.01
        
        preprocessor = AudioPreprocessor(target_sr=48000, normalize=True)
        processed = preprocessor.process(audio, source_sr=48000)
        
        # RMS should be higher after normalization
        original_rms = np.sqrt(np.mean(audio ** 2))
        processed_rms = np.sqrt(np.mean(processed ** 2))
        
        assert processed_rms > original_rms


class TestPostprocessing:
    """Test audio postprocessing"""
    
    def test_postprocessor_creation(self):
        """Test postprocessor can be created"""
        postprocessor = AudioPostprocessor()
        assert postprocessor is not None
    
    def test_postprocess_audio(self):
        """Test audio postprocessing"""
        audio = np.random.randn(48000).astype(np.float32) * 0.5
        
        postprocessor = AudioPostprocessor(apply_limiter=True)
        processed = postprocessor.process(audio)
        
        # Check output
        assert processed is not None
        assert processed.dtype == np.float32
        assert np.isfinite(processed).all()
        assert np.abs(processed).max() <= 1.0
    
    def test_limiter(self):
        """Test limiter prevents clipping"""
        # Create audio with clipping
        audio = np.random.randn(48000).astype(np.float32) * 2.0
        
        postprocessor = AudioPostprocessor(apply_limiter=True, limiter_threshold=0.95)
        processed = postprocessor.process(audio)
        
        # Should be limited
        assert np.abs(processed).max() <= 0.95


class TestPitchShifting:
    """Test pitch shifting"""
    
    def test_pitch_shifter_creation(self):
        """Test pitch shifter can be created"""
        shifter = PitchShifter()
        assert shifter is not None
    
    def test_shift_f0(self):
        """Test F0 shifting"""
        # Create test F0 curve (440 Hz = A4)
        f0 = np.full(100, 440.0, dtype=np.float32)
        
        shifter = PitchShifter()
        shifted = shifter.shift(f0, semitones=12)  # Up one octave
        
        # Should be 880 Hz (A5)
        assert shifted is not None
        assert np.allclose(shifted[shifted > 0], 880.0, rtol=0.01)
    
    def test_interpolate_unvoiced(self):
        """Test unvoiced interpolation"""
        # F0 with gaps (unvoiced regions)
        f0 = np.array([440, 0, 0, 0, 440], dtype=np.float32)
        
        shifter = PitchShifter()
        interpolated = shifter.shift(f0, semitones=0, interpolate_unvoiced=True)
        
        # Gaps should be filled
        assert (interpolated > 0).all()


class TestFeatureExtraction:
    """Test feature extraction"""
    
    def test_f0_extractor_creation(self):
        """Test F0 extractor can be created"""
        extractor = F0Extractor(method='harvest', sr=48000)
        assert extractor is not None
    
    def test_extract_f0(self):
        """Test F0 extraction"""
        # Create test audio (sine wave at 440 Hz)
        sr = 48000
        duration = 1.0
        freq = 440.0
        
        t = np.linspace(0, duration, int(sr * duration))
        audio = np.sin(2 * np.pi * freq * t).astype(np.float32)
        
        extractor = F0Extractor(method='harvest', sr=sr)
        f0 = extractor.extract(audio)
        
        # Check output
        assert f0 is not None
        assert len(f0) > 0
        assert f0.dtype == np.float32
        
        # Mean F0 should be close to 440 Hz
        voiced_f0 = f0[f0 > 0]
        if len(voiced_f0) > 0:
            mean_f0 = np.mean(voiced_f0)
            assert 400 < mean_f0 < 480  # Allow some tolerance


class TestModelScanning:
    """Test model scanning"""
    
    def test_scanner_creation(self):
        """Test scanner can be created"""
        scanner = ModelScanner(base_path=Path("SampleVoice"))
        assert scanner is not None
    
    def test_scan_directory(self):
        """Test directory scanning"""
        scanner = ModelScanner(base_path=Path("SampleVoice"))
        result = scanner.scan_directory()
        
        # Check result structure
        assert result is not None
        assert hasattr(result, 'total_found')
        assert hasattr(result, 'valid_models')
        assert hasattr(result, 'invalid_models')


class TestPipelineIntegration:
    """Test complete pipeline integration"""
    
    def test_full_pipeline_passthrough(self):
        """Test complete pipeline with passthrough (no AI)"""
        # Create test audio
        audio = np.random.randn(48000).astype(np.float32) * 0.1
        
        # Preprocess
        preprocessor = AudioPreprocessor(target_sr=48000)
        preprocessed = preprocessor.process(audio, source_sr=48000)
        
        # Mock inference (passthrough)
        output = preprocessed * 0.95
        
        # Postprocess
        postprocessor = AudioPostprocessor()
        final = postprocessor.process(output)
        
        # Validate
        assert final is not None
        assert final.dtype == np.float32
        assert len(final) == len(audio)
        assert np.isfinite(final).all()
    
    def test_latency_measurement(self):
        """Test latency measurement"""
        import time
        
        audio = np.random.randn(4800).astype(np.float32) * 0.1  # 100ms at 48kHz
        
        preprocessor = AudioPreprocessor(target_sr=48000)
        postprocessor = AudioPostprocessor()
        
        # Measure
        start = time.time()
        
        preprocessed = preprocessor.process(audio, source_sr=48000)
        output = preprocessed * 0.95  # Mock inference
        final = postprocessor.process(output)
        
        latency_ms = (time.time() - start) * 1000
        
        # Should be fast (< 10ms for 100ms audio)
        assert latency_ms < 10.0
        
        print(f"\nPipeline latency: {latency_ms:.2f}ms")


# Pytest configuration
@pytest.fixture
def test_audio():
    """Fixture for test audio"""
    return np.random.randn(48000).astype(np.float32) * 0.1


@pytest.fixture
def test_f0():
    """Fixture for test F0 curve"""
    return np.full(100, 440.0, dtype=np.float32)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])