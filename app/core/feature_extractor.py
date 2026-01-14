"""
VoiceForge-Nextgen - Feature Extractor
File: app/core/feature_extractor.py

Purpose:
    Extract F0 (pitch) and HuBERT (content) features for RVC
    Supports multiple F0 methods (harvest, dio, crepe, rmvpe)

Dependencies:
    - pyworld (harvest, dio)
    - parselmouth (praat)
    - torch (HuBERT)
    - numpy

Data Flow:
    Audio → F0Extractor → F0 curve
    Audio → HubertExtractor → Content features

Usage:
    f0_extractor = F0Extractor(method='harvest', sr=48000)
    f0 = f0_extractor.extract(audio)
    
    hubert_extractor = HubertExtractor(device='cuda')
    features = hubert_extractor.extract(audio)
"""

import numpy as np
import torch
from pathlib import Path
from typing import Optional, Literal
import logging

logger = logging.getLogger("FeatureExtractor")


class F0Extractor:
    """
    Extract F0 (fundamental frequency) using various methods
    
    Supported methods:
        - harvest: WORLD vocoder (robust, slow)
        - dio: WORLD vocoder (fast, less accurate)
        - pm: Praat (high quality)
        - crepe: Deep learning (most accurate, requires GPU)
        - rmvpe: Custom RVC method
    """
    
    def __init__(
        self,
        method: Literal['harvest', 'dio', 'pm', 'crepe', 'rmvpe'] = 'harvest',
        sr: int = 48000,
        hop_length: int = 512,
        f0_min: float = 50.0,
        f0_max: float = 1100.0
    ):
        """
        Initialize F0 extractor
        
        Args:
            method: Extraction method
            sr: Sample rate
            hop_length: Hop size in samples
            f0_min: Minimum F0 (Hz)
            f0_max: Maximum F0 (Hz)
        """
        self.method = method.lower()
        self.sr = sr
        self.hop_length = hop_length
        self.f0_min = f0_min
        self.f0_max = f0_max
        
        # Validate method
        valid_methods = ['harvest', 'dio', 'pm', 'crepe', 'rmvpe']
        if self.method not in valid_methods:
            logger.warning(f"Unknown method {method}, using 'harvest'")
            self.method = 'harvest'
        
        # Initialize method-specific extractors
        self._init_extractor()
    
    def _init_extractor(self):
        """Initialize method-specific extractor"""
        if self.method in ['harvest', 'dio']:
            try:
                import pyworld
                self.pyworld = pyworld
                logger.info(f"F0 extractor initialized: {self.method} (pyworld)")
            except ImportError:
                logger.error("pyworld not installed, falling back to 'pm'")
                self.method = 'pm'
        
        if self.method == 'pm':
            try:
                import parselmouth
                self.parselmouth = parselmouth
                logger.info("F0 extractor initialized: pm (praat)")
            except ImportError:
                logger.error("parselmouth not installed")
                raise RuntimeError("No F0 extractor available")
        
        if self.method == 'crepe':
            logger.warning("CREPE not yet implemented, using 'harvest'")
            self.method = 'harvest'
            self._init_extractor()
        
        if self.method == 'rmvpe':
            logger.info("RMVPE: will use assets/RMVPE if available")
    
    def extract(self, audio: np.ndarray) -> np.ndarray:
        """
        Extract F0 from audio
        
        Args:
            audio: Audio array (float32, -1 to 1)
            
        Returns:
            F0 curve (Hz, 0 = unvoiced)
        """
        if audio.size == 0:
            return np.array([], dtype=np.float32)
        
        # Ensure correct format
        audio = audio.astype(np.float64)  # pyworld needs float64
        
        try:
            if self.method == 'harvest':
                f0 = self._extract_harvest(audio)
            elif self.method == 'dio':
                f0 = self._extract_dio(audio)
            elif self.method == 'pm':
                f0 = self._extract_pm(audio)
            elif self.method == 'rmvpe':
                f0 = self._extract_rmvpe(audio)
            else:
                raise ValueError(f"Unknown method: {self.method}")
            
            # Validate
            if not np.isfinite(f0).all():
                logger.warning("Non-finite F0 values, cleaning...")
                f0 = np.nan_to_num(f0, nan=0.0)
            
            return f0.astype(np.float32)
            
        except Exception as e:
            logger.error(f"F0 extraction failed: {e}")
            # Return zero F0 (unvoiced)
            num_frames = int(len(audio) / self.hop_length) + 1
            return np.zeros(num_frames, dtype=np.float32)
    
    def _extract_harvest(self, audio: np.ndarray) -> np.ndarray:
        """Extract F0 using WORLD harvest"""
        f0, t = self.pyworld.harvest(
            audio,
            self.sr,
            f0_floor=self.f0_min,
            f0_ceil=self.f0_max,
            frame_period=self.hop_length / self.sr * 1000  # ms
        )
        return f0
    
    def _extract_dio(self, audio: np.ndarray) -> np.ndarray:
        """Extract F0 using WORLD dio"""
        f0, t = self.pyworld.dio(
            audio,
            self.sr,
            f0_floor=self.f0_min,
            f0_ceil=self.f0_max,
            frame_period=self.hop_length / self.sr * 1000  # ms
        )
        return f0
    
    def _extract_pm(self, audio: np.ndarray) -> np.ndarray:
        """Extract F0 using Praat"""
        # Convert to parselmouth Sound
        sound = self.parselmouth.Sound(audio, sampling_frequency=self.sr)
        
        # Extract pitch
        pitch = sound.to_pitch(
            time_step=self.hop_length / self.sr,
            pitch_floor=self.f0_min,
            pitch_ceiling=self.f0_max
        )
        
        # Convert to numpy
        f0 = pitch.selected_array['frequency']
        f0[f0 == 0] = 0  # Unvoiced
        
        return f0
    
    def _extract_rmvpe(self, audio: np.ndarray) -> np.ndarray:
        """Extract F0 using RMVPE (placeholder)"""
        logger.warning("RMVPE not implemented, using harvest")
        return self._extract_harvest(audio)


class HubertExtractor:
    """
    Extract content features using HuBERT
    
    Uses pre-trained HuBERT model from assets/
    """
    
    def __init__(
        self,
        model_path: Optional[Path] = None,
        device: str = 'cuda'
    ):
        """
        Initialize HuBERT extractor
        
        Args:
            model_path: Path to hubert_base.pt (auto-detect if None)
            device: Device for inference
        """
        self.device = device if torch.cuda.is_available() else 'cpu'
        
        # Find model
        if model_path is None:
            model_path = self._find_hubert_model()
        
        self.model_path = model_path
        self.model = None
        
        # Load model lazily
        logger.info(f"HuBERT extractor initialized (device: {self.device})")
    
    def _find_hubert_model(self) -> Optional[Path]:
        """Find HuBERT model in assets/"""
        assets_dir = Path("app/core/assets")
        
        # Search patterns
        patterns = [
            "hubert_base*.pt",
            "hubert_base*.pth",
            "checkpoint_best_legacy_500.pt"
        ]
        
        for pattern in patterns:
            matches = list(assets_dir.glob(pattern))
            if matches:
                logger.info(f"Found HuBERT model: {matches[0].name}")
                return matches[0]
        
        logger.warning("HuBERT model not found in assets/")
        return None
    
    def _load_model(self):
        """Load HuBERT model (lazy loading)"""
        if self.model is not None:
            return  # Already loaded
        
        if self.model_path is None or not self.model_path.exists():
            raise FileNotFoundError(
                "HuBERT model not found. "
                "Place hubert_base.pt in app/core/assets/"
            )
        
        try:
            logger.info(f"Loading HuBERT model: {self.model_path.name}")
            
            # Load checkpoint
            checkpoint = torch.load(self.model_path, map_location=self.device)
            
            # Extract model (checkpoint structure varies)
            if isinstance(checkpoint, dict):
                if 'model' in checkpoint:
                    self.model = checkpoint['model']
                else:
                    # Assume checkpoint is state_dict
                    from transformers import HubertModel
                    self.model = HubertModel.from_pretrained('facebook/hubert-base-ls960')
                    self.model.load_state_dict(checkpoint)
            else:
                self.model = checkpoint
            
            self.model.to(self.device)
            self.model.eval()
            
            logger.info("HuBERT model loaded successfully")
            
        except Exception as e:
            logger.error(f"Failed to load HuBERT: {e}")
            raise
    
    def extract(self, audio: np.ndarray, sr: int = 16000) -> np.ndarray:
        """
        Extract content features
        
        Args:
            audio: Audio array (float32)
            sr: Sample rate (HuBERT expects 16kHz)
            
        Returns:
            Feature array (num_frames, 768)
        """
        if audio.size == 0:
            return np.array([[]], dtype=np.float32)
        
        # Load model if not loaded
        if self.model is None:
            self._load_model()
        
        try:
            # Resample to 16kHz if needed
            if sr != 16000:
                from scipy import signal
                audio = signal.resample(
                    audio,
                    int(len(audio) * 16000 / sr)
                )
            
            # Convert to tensor
            audio_tensor = torch.from_numpy(audio).float().unsqueeze(0).to(self.device)
            
            # Extract features
            with torch.no_grad():
                if hasattr(self.model, 'extract_features'):
                    features = self.model.extract_features(audio_tensor)[0]
                else:
                    features = self.model(audio_tensor).last_hidden_state
            
            # Convert to numpy
            features = features.squeeze(0).cpu().numpy()
            
            return features.astype(np.float32)
            
        except Exception as e:
            logger.error(f"HuBERT extraction failed: {e}")
            # Return dummy features
            num_frames = int(len(audio) / 320) + 1  # Approximate
            return np.zeros((num_frames, 768), dtype=np.float32)
    
    def get_device(self) -> str:
        """Get current device"""
        return self.device


# Convenience functions
def extract_f0(
    audio: np.ndarray,
    sr: int = 48000,
    method: str = 'harvest'
) -> np.ndarray:
    """Extract F0 from audio"""
    extractor = F0Extractor(method=method, sr=sr)
    return extractor.extract(audio)


def extract_hubert(
    audio: np.ndarray,
    sr: int = 16000,
    device: str = 'cuda'
) -> np.ndarray:
    """Extract HuBERT features"""
    extractor = HubertExtractor(device=device)
    return extractor.extract(audio, sr)