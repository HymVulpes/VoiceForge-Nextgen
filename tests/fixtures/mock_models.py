"""
VoiceForge-Nextgen - Mock Models for Testing
File: tests/fixtures/mock_models.py

Purpose:
    Mock RVC models for testing without real model files
    Provide dummy models for unit tests

Dependencies:
    - torch
    - numpy

Usage:
    from tests.fixtures.mock_models import create_mock_model
    
    model_data = create_mock_model()
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, Any, Optional
from pathlib import Path


class MockRVCModel(nn.Module):
    """
    Mock RVC model for testing
    
    Mimics RVC model interface but returns passthrough audio
    """
    
    def __init__(
        self,
        hidden_dim: int = 768,
        output_dim: int = 1,
        sample_rate: int = 48000
    ):
        """
        Initialize mock model
        
        Args:
            hidden_dim: Hidden dimension (HuBERT features)
            output_dim: Output dimension
            sample_rate: Sample rate
        """
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.sample_rate = sample_rate
        
        # Dummy layers (just for structure)
        self.encoder = nn.Linear(hidden_dim, hidden_dim)
        self.decoder = nn.Linear(hidden_dim, output_dim)
        
        # Config
        self.config = {
            'sample_rate': sample_rate,
            'f0_method': 'harvest',
            'version': 'v2',
            'hidden_dim': hidden_dim
        }
    
    def forward(
        self,
        hubert_features: torch.Tensor,
        f0: torch.Tensor,
        protect: float = 0.33
    ) -> torch.Tensor:
        """
        Forward pass (mock inference)
        
        Args:
            hubert_features: HuBERT features (B, T, 768)
            f0: F0 curve (B, T)
            protect: Protection value
            
        Returns:
            Audio waveform (B, T)
        """
        # Simple passthrough with slight modification
        # In real RVC, this would be complex voice conversion
        
        batch_size, seq_len, _ = hubert_features.shape
        
        # Mock processing
        encoded = self.encoder(hubert_features)
        output = self.decoder(encoded).squeeze(-1)
        
        # Add slight variation based on F0
        f0_normalized = f0 / 440.0  # Normalize around A4
        output = output * (0.95 + 0.1 * f0_normalized)
        
        return output
    
    def infer(
        self,
        audio: np.ndarray,
        f0: np.ndarray,
        hubert: np.ndarray,
        **kwargs
    ) -> np.ndarray:
        """
        Inference method (numpy interface)
        
        Args:
            audio: Input audio
            f0: F0 curve
            hubert: HuBERT features
            
        Returns:
            Output audio
        """
        # Convert to tensors
        hubert_tensor = torch.from_numpy(hubert).float().unsqueeze(0)
        f0_tensor = torch.from_numpy(f0).float().unsqueeze(0)
        
        # Forward pass
        with torch.no_grad():
            output_tensor = self.forward(hubert_tensor, f0_tensor)
        
        # Convert back to numpy
        output = output_tensor.squeeze(0).cpu().numpy()
        
        return output.astype(np.float32)


class MockFAISSIndex:
    """
    Mock FAISS index for testing
    """
    
    def __init__(self, d: int = 768, ntotal: int = 1000):
        """
        Initialize mock index
        
        Args:
            d: Feature dimension
            ntotal: Number of vectors
        """
        self.d = d
        self.ntotal = ntotal
        
        # Mock data
        self.data = np.random.randn(ntotal, d).astype(np.float32)
    
    def search(self, queries: np.ndarray, k: int = 1):
        """
        Mock search
        
        Args:
            queries: Query vectors (N, d)
            k: Number of neighbors
            
        Returns:
            (distances, indices)
        """
        n_queries = len(queries)
        
        # Return random indices
        distances = np.random.rand(n_queries, k).astype(np.float32)
        indices = np.random.randint(0, self.ntotal, (n_queries, k))
        
        return distances, indices
    
    def reconstruct_batch(self, indices: np.ndarray) -> np.ndarray:
        """
        Mock reconstruct
        
        Args:
            indices: Vector indices
            
        Returns:
            Reconstructed vectors
        """
        # Return mock vectors
        return self.data[indices % self.ntotal]


def create_mock_model(
    sample_rate: int = 48000,
    with_index: bool = True
) -> Dict[str, Any]:
    """
    Create mock model data
    
    Args:
        sample_rate: Model sample rate
        with_index: Include FAISS index
        
    Returns:
        Model data dictionary
    """
    model = MockRVCModel(sample_rate=sample_rate)
    model.eval()
    
    model_data = {
        'model': model,
        'config': model.config,
        'sample_rate': sample_rate,
        'f0_method': 'harvest',
        'version': 'v2',
        'size': estimate_model_size(model)
    }
    
    if with_index:
        model_data['index'] = MockFAISSIndex()
    else:
        model_data['index'] = None
    
    return model_data


def create_mock_checkpoint(
    save_path: Optional[Path] = None
) -> Dict[str, Any]:
    """
    Create mock RVC checkpoint file
    
    Args:
        save_path: Path to save checkpoint (optional)
        
    Returns:
        Checkpoint dictionary
    """
    model = MockRVCModel()
    
    checkpoint = {
        'weight': model.state_dict(),
        'config': model.config,
        'version': 'v2',
        'sr': 48000,
        'f0': 'harvest'
    }
    
    if save_path is not None:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(checkpoint, save_path)
        print(f"Mock checkpoint saved to: {save_path}")
    
    return checkpoint


def estimate_model_size(model: nn.Module) -> int:
    """
    Estimate model size in bytes
    
    Args:
        model: PyTorch model
        
    Returns:
        Size in bytes
    """
    total_size = 0
    
    for param in model.parameters():
        total_size += param.nelement() * param.element_size()
    
    for buffer in model.buffers():
        total_size += buffer.nelement() * buffer.element_size()
    
    return total_size


def create_test_audio(
    duration_s: float = 1.0,
    sample_rate: int = 48000,
    frequency: float = 440.0
) -> np.ndarray:
    """
    Create test audio (sine wave)
    
    Args:
        duration_s: Duration in seconds
        sample_rate: Sample rate
        frequency: Frequency in Hz
        
    Returns:
        Audio array
    """
    num_samples = int(duration_s * sample_rate)
    t = np.linspace(0, duration_s, num_samples)
    
    audio = np.sin(2 * np.pi * frequency * t).astype(np.float32)
    
    return audio


def create_test_f0(
    num_frames: int = 100,
    base_freq: float = 440.0,
    add_vibrato: bool = False
) -> np.ndarray:
    """
    Create test F0 curve
    
    Args:
        num_frames: Number of frames
        base_freq: Base frequency
        add_vibrato: Add vibrato effect
        
    Returns:
        F0 array
    """
    f0 = np.full(num_frames, base_freq, dtype=np.float32)
    
    if add_vibrato:
        # Add vibrato (5Hz, ±5% depth)
        t = np.linspace(0, num_frames / 100, num_frames)  # Assume 100 fps
        vibrato = base_freq * 0.05 * np.sin(2 * np.pi * 5 * t)
        f0 = f0 + vibrato
    
    return f0


def create_test_hubert_features(
    num_frames: int = 100,
    dim: int = 768
) -> np.ndarray:
    """
    Create test HuBERT features
    
    Args:
        num_frames: Number of frames
        dim: Feature dimension
        
    Returns:
        Feature array (num_frames, dim)
    """
    # Random features with some structure
    features = np.random.randn(num_frames, dim).astype(np.float32)
    
    # Normalize
    features = features / np.linalg.norm(features, axis=1, keepdims=True)
    
    return features


# Pytest fixtures
def pytest_configure():
    """Register custom markers"""
    import pytest
    
    pytest.register_assert_rewrite('tests.fixtures.mock_models')


# Example usage
if __name__ == "__main__":
    print("Creating mock models for testing...")
    
    # Create mock model
    model_data = create_mock_model()
    print(f"✓ Mock model created: {estimate_model_size(model_data['model']) / 1024:.1f} KB")
    
    # Create test data
    audio = create_test_audio(duration_s=1.0)
    f0 = create_test_f0(num_frames=100)
    hubert = create_test_hubert_features(num_frames=100)
    
    print(f"✓ Test audio: {len(audio)} samples")
    print(f"✓ Test F0: {len(f0)} frames")
    print(f"✓ Test HuBERT: {hubert.shape}")
    
    # Test inference
    model = model_data['model']
    output = model.infer(audio, f0, hubert)
    
    print(f"✓ Mock inference: {len(output)} samples")
    print("\nMock models ready for testing!")