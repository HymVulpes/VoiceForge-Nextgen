"""
RVC Model Loader with Validation
Loads .pth checkpoint and optional .index file
"""

try:
    import torch

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    torch = None

from pathlib import Path
from typing import Optional, Dict, Any
import logging

logger = logging.getLogger(__name__)


class ModelLoader:
    """
    Loads and validates RVC models
    STATELESS - no inference logic here
    """

    def __init__(self, device: str = "cuda"):
        """
        Args:
            device: 'cuda' or 'cpu'
        """
        self.device = device

        # Verify CUDA availability
        if device == "cuda":
            if not TORCH_AVAILABLE or torch is None:
                logger.warning("PyTorch not available, using CPU")
                self.device = "cpu"
            elif not torch.cuda.is_available():
                logger.warning("CUDA requested but not available, falling back to CPU")
                self.device = "cpu"

        logger.info(f"ModelLoader initialized: device={self.device}")

    # ============================================================
    # 🔴 REPLACED METHOD (OLD load_model REMOVED COMPLETELY)
    # 🔴 NEW load_model IMPLEMENTED FROM CLAUDE
    # ============================================================
    def load_model(
        self, pth_path: str, index_path: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Load RVC model with enhanced metadata extraction
        """
        if not TORCH_AVAILABLE or torch is None:
            raise ImportError(
                "PyTorch is required to load models. Install with: pip install torch"
            )

        pth_file = Path(pth_path)
        if not pth_file.exists():
            raise FileNotFoundError(f"Model file not found: {pth_path}")

        try:
            # Load checkpoint (RVC uses pickle objects)
            checkpoint = torch.load(
                pth_path, map_location=self.device, weights_only=False
            )

            # Parse config safely (supports multiple RVC formats)
            config = self._parse_config(checkpoint)

            # Create and load model
            model = self._create_model_from_config(config)
            model.load_state_dict(checkpoint.get("weight", checkpoint))
            model.to(self.device)
            model.eval()

            # Load FAISS index if provided
            index = None
            if index_path is not None:
                index_file = Path(index_path)
                if index_file.exists():
                    index = self._load_faiss_index(index_file)

            return {
                "model": model,
                "index": index,
                "config": config,
                "sample_rate": config.get("sample_rate", 48000),
                "f0_method": config.get("f0_method", "harvest"),
                "version": config.get("version", "v2"),
            }

        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise

    # ============================================================
    # 🟢 NEW METHODS ADDED (DID NOT EXIST IN ORIGINAL CODE)
    # ============================================================
    def _parse_config(self, checkpoint: Dict) -> Dict:
        """Parse model config from checkpoint"""
        config = checkpoint.get("config", {})

        # Old RVC format: config as list
        if isinstance(config, list):
            return {
                "sample_rate": config[0] if len(config) > 0 else 48000,
                "version": "v1",
            }

        return {
            "sample_rate": checkpoint.get("sr", config.get("sample_rate", 48000)),
            "f0_method": checkpoint.get("f0", config.get("f0_method", "harvest")),
            "version": checkpoint.get("version", "v2"),
        }

    def _create_model_from_config(self, config: Dict):
        """Create model instance from config (placeholder)"""
        logger.warning("Using placeholder model creation")

        import torch.nn as nn

        class DummyRVCModel(nn.Module):
            def forward(self, hubert, f0, protect=0.33):
                return torch.zeros_like(hubert[:, :, 0])

        return DummyRVCModel()

    def _load_faiss_index(self, index_path: Path):
        """Load FAISS index"""
        try:
            import faiss

            index = faiss.read_index(str(index_path))
            logger.info(f"✓ FAISS index loaded: {index.ntotal} vectors")
            return index
        except ImportError:
            logger.warning("faiss not installed, index disabled")
            return None
        except Exception as e:
            logger.error(f"Failed to load index: {e}")
            return None

    # ============================================================
    # 🟡 validate_model() GIỮ NGUYÊN LOGIC, NHƯNG PHẢI PHÙ HỢP OUTPUT
    # ============================================================
    def validate_model(self, model_data: Dict[str, Any]) -> bool:
        """
        Validate loaded model structure
        """
        required_keys = ["model", "config"]

        for key in required_keys:
            if key not in model_data:
                logger.error(f"Missing required key: {key}")
                return False

        logger.info("Model validation passed")
        return True
