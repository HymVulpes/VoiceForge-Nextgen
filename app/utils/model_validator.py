"""
VoiceForge-Nextgen - Model Validator
File: app/utils/model_validator.py

Purpose:
    Validate RVC model file integrity
    Check PyTorch checkpoint format
    Verify required keys and dimensions

Dependencies:
    - torch (checkpoint loading)
    - pathlib (file operations)
    - logging (error reporting)

Usage:
    validator = ModelValidator()
    is_valid, error = validator.validate_model(pth_path, index_path)
"""

import torch
from pathlib import Path
from typing import Tuple, Optional, Dict, Any
import logging

logger = logging.getLogger("ModelValidator")


class ModelValidator:
    """
    Validates RVC model files for correctness and integrity
    
    Checks:
        - File exists and readable
        - PyTorch checkpoint format
        - Required keys present
        - Index file compatibility
        - No corruption
    """
    
    # Required keys in RVC checkpoint
    REQUIRED_KEYS_V1 = ['weight', 'config', 'version']
    REQUIRED_KEYS_V2 = ['weight', 'config', 'version', 'sr', 'f0']
    
    def __init__(self):
        self.validation_cache = {}  # Cache validation results
    
    def validate_model(
        self,
        pth_path: Path,
        index_path: Optional[Path] = None
    ) -> Tuple[bool, Optional[str]]:
        """
        Validate a model file
        
        Args:
            pth_path: Path to .pth file
            index_path: Path to .index file (optional)
            
        Returns:
            (is_valid, error_message)
            - (True, None) if valid
            - (False, error_message) if invalid
        """
        # Check cache
        cache_key = str(pth_path)
        if cache_key in self.validation_cache:
            return self.validation_cache[cache_key]
        
        try:
            # Step 1: File exists
            if not pth_path.exists():
                error = f"File not found: {pth_path}"
                self.validation_cache[cache_key] = (False, error)
                return False, error
            
            # Step 2: File readable
            if not pth_path.is_file():
                error = f"Not a file: {pth_path}"
                self.validation_cache[cache_key] = (False, error)
                return False, error
            
            # Step 3: Load checkpoint
            try:
                checkpoint = torch.load(
                    pth_path,
                    map_location='cpu',
                    weights_only=False  # Allow pickle for RVC models
                )
            except Exception as e:
                error = f"Failed to load checkpoint: {e}"
                self.validation_cache[cache_key] = (False, error)
                return False, error
            
            # Step 4: Validate checkpoint structure
            is_valid, error = self._validate_checkpoint(checkpoint)
            if not is_valid:
                self.validation_cache[cache_key] = (False, error)
                return False, error
            
            # Step 5: Validate index if provided
            if index_path is not None:
                is_valid, error = self._validate_index(index_path, checkpoint)
                if not is_valid:
                    self.validation_cache[cache_key] = (False, error)
                    return False, error
            
            # All checks passed
            logger.info(f"✓ Model validated: {pth_path.name}")
            self.validation_cache[cache_key] = (True, None)
            return True, None
            
        except Exception as e:
            error = f"Validation error: {e}"
            logger.error(error)
            self.validation_cache[cache_key] = (False, error)
            return False, error
    
    def _validate_checkpoint(
        self,
        checkpoint: Dict[str, Any]
    ) -> Tuple[bool, Optional[str]]:
        """
        Validate checkpoint structure
        
        Args:
            checkpoint: Loaded PyTorch checkpoint
            
        Returns:
            (is_valid, error_message)
        """
        if not isinstance(checkpoint, dict):
            return False, "Checkpoint is not a dictionary"
        
        # Detect version
        version = checkpoint.get('version', 'v1')
        required_keys = self.REQUIRED_KEYS_V2 if version == 'v2' else self.REQUIRED_KEYS_V1
        
        # Check required keys
        missing_keys = [k for k in required_keys if k not in checkpoint]
        if missing_keys:
            return False, f"Missing required keys: {missing_keys}"
        
        # Validate weight
        if 'weight' in checkpoint:
            weight = checkpoint['weight']
            if not isinstance(weight, dict):
                return False, "weight is not a dictionary"
            
            # Check for common RVC weight keys
            expected_weight_keys = ['emb_g.weight', 'enc_p.emb.weight']
            has_keys = any(k in weight for k in expected_weight_keys)
            if not has_keys:
                logger.warning("Unusual weight structure (might be custom RVC)")
        
        # Validate config
        if 'config' in checkpoint:
            config = checkpoint['config']
            if not isinstance(config, (dict, list)):
                return False, "config is not dict or list"
        
        return True, None
    
    def _validate_index(
        self,
        index_path: Path,
        checkpoint: Dict[str, Any]
    ) -> Tuple[bool, Optional[str]]:
        """
        Validate index file
        
        Args:
            index_path: Path to .index file
            checkpoint: Model checkpoint
            
        Returns:
            (is_valid, error_message)
        """
        try:
            # Check file exists
            if not index_path.exists():
                return False, f"Index file not found: {index_path}"
            
            # Try to load with faiss (if available)
            try:
                import faiss
                index = faiss.read_index(str(index_path))
                
                # Validate dimensions match model
                # (RVC typically uses 256 or 768 dimensions)
                if index.d not in [256, 768]:
                    logger.warning(
                        f"Unusual index dimension: {index.d} "
                        f"(expected 256 or 768)"
                    )
                
                logger.debug(f"Index validated: {index.ntotal} vectors, {index.d}D")
                return True, None
                
            except ImportError:
                # faiss not available, just check file is readable
                if index_path.stat().st_size == 0:
                    return False, "Index file is empty"
                return True, None
                
        except Exception as e:
            return False, f"Index validation error: {e}"
    
    def extract_metadata(self, pth_path: Path) -> Optional[Dict[str, Any]]:
        """
        Extract metadata from model checkpoint
        
        Args:
            pth_path: Path to .pth file
            
        Returns:
            Dictionary with metadata or None if failed
        """
        try:
            checkpoint = torch.load(pth_path, map_location='cpu', weights_only=False)
            
            metadata = {
                'version': checkpoint.get('version', 'v1'),
                'sample_rate': checkpoint.get('sr', checkpoint.get('sample_rate', 48000)),
                'f0_method': checkpoint.get('f0', checkpoint.get('f0_method', 'harvest')),
            }
            
            # Try to extract from config
            if 'config' in checkpoint:
                config = checkpoint['config']
                if isinstance(config, dict):
                    metadata['sample_rate'] = config.get('sample_rate', metadata['sample_rate'])
                    metadata['f0_method'] = config.get('f0_method', metadata['f0_method'])
            
            return metadata
            
        except Exception as e:
            logger.error(f"Failed to extract metadata: {e}")
            return None
    
    def clear_cache(self):
        """Clear validation cache"""
        self.validation_cache.clear()
        logger.debug("Validation cache cleared")


# Singleton instance
_validator_instance = None

def get_validator() -> ModelValidator:
    """Get singleton validator instance"""
    global _validator_instance
    if _validator_instance is None:
        _validator_instance = ModelValidator()
    return _validator_instance