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
    
    def load_model(
        self,
        pth_path: str,
        index_path: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Load RVC model checkpoint
        
        Args:
            pth_path: Path to .pth checkpoint file
            index_path: Optional path to .index FAISS file
            
        Returns:
            Dictionary containing:
                - model: Model state dict or module
                - config: Model configuration
                - index: FAISS index (if provided)
                - metadata: Additional info
        """
        pth_file = Path(pth_path)
        
        # Validate file exists
        if not pth_file.exists():
            raise FileNotFoundError(f"Model file not found: {pth_path}")
        
        logger.info(f"Loading model: {pth_file.name}")
        
        if not TORCH_AVAILABLE or torch is None:
            raise ImportError("PyTorch is required to load models. Install with: pip install torch")
        
        try:
            # Load checkpoint
            checkpoint = torch.load(pth_path, map_location=self.device)
            
            # Extract model components
            # RVC models typically store: 'model', 'config', 'weight', etc.
            result = {
                "checkpoint": checkpoint,
                "metadata": {
                    "pth_path": str(pth_file.absolute()),
                    "file_size_mb": pth_file.stat().st_size / (1024 * 1024),
                    "device": self.device
                }
            }
            
            # Check checkpoint structure
            if isinstance(checkpoint, dict):
                if "config" in checkpoint:
                    result["config"] = checkpoint["config"]
                    logger.info(f"Model config found: {checkpoint['config']}")
                
                if "weight" in checkpoint:
                    result["model"] = checkpoint["weight"]
                elif "model" in checkpoint:
                    result["model"] = checkpoint["model"]
                else:
                    logger.warning("No 'model' or 'weight' key in checkpoint")
                    result["model"] = checkpoint
            else:
                logger.warning(f"Checkpoint is not a dict: {type(checkpoint)}")
                result["model"] = checkpoint
            
            # Load FAISS index if provided
            if index_path:
                index_file = Path(index_path)
                if index_file.exists():
                    result["index"] = self._load_index(index_path)
                    logger.info(f"Loaded FAISS index: {index_file.name}")
                else:
                    logger.warning(f"Index file not found: {index_path}")
                    result["index"] = None
            else:
                result["index"] = None
            
            logger.info(f"Model loaded successfully: {pth_file.name}")
            return result
            
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise
    
    def _load_index(self, index_path: str) -> Optional[Any]:
        """
        Load FAISS index file
        
        Returns:
            FAISS index object or None
        """
        try:
            import faiss
            index = faiss.read_index(index_path)
            logger.debug(f"FAISS index loaded: ntotal={index.ntotal}")
            return index
        except ImportError:
            logger.warning("FAISS not installed, index will not be used")
            return None
        except Exception as e:
            logger.error(f"Failed to load FAISS index: {e}")
            return None
    
    def validate_model(self, model_data: Dict[str, Any]) -> bool:
        """
        Validate loaded model structure
        
        Args:
            model_data: Output from load_model()
            
        Returns:
            True if valid
        """
        required_keys = ["checkpoint", "metadata"]
        
        for key in required_keys:
            if key not in model_data:
                logger.error(f"Missing required key: {key}")
                return False
        
        # Check model component exists
        if "model" not in model_data and "checkpoint" not in model_data:
            logger.error("No model data found in checkpoint")
            return False
        
        logger.info("Model validation passed")
        return True