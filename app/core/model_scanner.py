"""
VoiceForge-Nextgen - Model Scanner
File: app/core/model_scanner.py

Purpose:
    Scan SampleVoice/ directory for RVC models
    Validate and extract metadata
    Update database with discovered models

Dependencies:
    - pathlib (file scanning)
    - app.core.model_info (ModelInfo, ModelScanResult)
    - app.utils.model_validator (ModelValidator)
    - app.db.repository (VoiceModelRepository)

Data Flow:
    SampleVoice/*.pth → scan() → validate() → extract_metadata() → Database

Usage:
    scanner = ModelScanner(
        base_path=Path("SampleVoice"),
        db_repo=voice_model_repo
    )
    result = scanner.scan_and_update()
"""

import time
from pathlib import Path
from typing import List, Optional, Set
import logging

from app.core.model_info import ModelInfo, ModelScanResult
from app.utils.model_validator import get_validator

logger = logging.getLogger("ModelScanner")


class ModelScanner:
    """
    Scans directory for RVC models and updates database
    
    Features:
        - Recursive directory scanning
        - Automatic .pth + .index pairing
        - Validation of model files
        - Metadata extraction
        - Database synchronization
    """
    
    def __init__(
        self,
        base_path: Path,
        db_repo=None,  # VoiceModelRepository
        recursive: bool = True
    ):
        """
        Initialize scanner
        
        Args:
            base_path: Root directory to scan (e.g., SampleVoice/)
            db_repo: Database repository instance
            recursive: Scan subdirectories
        """
        self.base_path = Path(base_path)
        self.db_repo = db_repo
        self.recursive = recursive
        self.validator = get_validator()
        
        if not self.base_path.exists():
            logger.warning(f"Base path does not exist: {self.base_path}")
            self.base_path.mkdir(parents=True, exist_ok=True)
    
    def scan_directory(self) -> ModelScanResult:
        """
        Scan directory for models
        
        Returns:
            ModelScanResult with all discovered models
        """
        start_time = time.time()
        logger.info(f"Scanning for models in: {self.base_path}")
        
        result = ModelScanResult()
        
        # Find all .pth files
        pattern = "**/*.pth" if self.recursive else "*.pth"
        pth_files = list(self.base_path.glob(pattern))
        
        logger.info(f"Found {len(pth_files)} .pth files")
        result.total_found = len(pth_files)
        
        for pth_path in pth_files:
            try:
                # Find matching .index file
                index_path = self._find_index_file(pth_path)
                
                # Validate model
                is_valid, error = self.validator.validate_model(pth_path, index_path)
                
                if not is_valid:
                    logger.warning(f"✗ Invalid model: {pth_path.name} - {error}")
                    result.invalid_models.append((pth_path, error))
                    continue
                
                # Create ModelInfo
                model_info = ModelInfo.from_files(pth_path, index_path)
                
                # Extract metadata
                metadata = self.validator.extract_metadata(pth_path)
                if metadata:
                    model_info.sample_rate = metadata.get('sample_rate', 48000)
                    model_info.f0_method = metadata.get('f0_method', 'harvest')
                    model_info.version = metadata.get('version', 'v2')
                
                result.valid_models.append(model_info)
                logger.info(f"✓ Valid model: {model_info.name}")
                
            except Exception as e:
                logger.error(f"Error processing {pth_path.name}: {e}")
                result.invalid_models.append((pth_path, str(e)))
        
        result.scan_time = time.time() - start_time
        logger.info(result.summary())
        
        return result
    
    def _find_index_file(self, pth_path: Path) -> Optional[Path]:
        """
        Find matching .index file for a .pth file
        
        Args:
            pth_path: Path to .pth file
            
        Returns:
            Path to .index file or None
        """
        # Try exact name match
        index_path = pth_path.with_suffix('.index')
        if index_path.exists():
            return index_path
        
        # Try added_ prefix (common RVC naming)
        index_name = f"added_{pth_path.stem}.index"
        index_path = pth_path.parent / index_name
        if index_path.exists():
            return index_path
        
        # Try without prefix
        index_name = pth_path.stem.replace('added_', '') + '.index'
        index_path = pth_path.parent / index_name
        if index_path.exists():
            return index_path
        
        logger.debug(f"No index file found for {pth_path.name}")
        return None
    
    def scan_and_update(self) -> int:
        """
        Scan directory and update database
        
        Returns:
            Number of models added/updated
        """
        if self.db_repo is None:
            logger.error("No database repository provided")
            return 0
        
        # Scan directory
        result = self.scan_directory()
        
        if not result.valid_models:
            logger.warning("No valid models found")
            return 0
        
        # Update database
        try:
            num_updated = self._update_database(result.valid_models)
            logger.info(f"Database updated: {num_updated} models")
            return num_updated
            
        except Exception as e:
            logger.error(f"Failed to update database: {e}")
            return 0
    
    def _update_database(self, models: List[ModelInfo]) -> int:
        """
        Update database with scanned models
        
        Args:
            models: List of valid models
            
        Returns:
            Number of models updated
        """
        # Get existing model IDs
        existing_models = self.db_repo.get_all_models()
        existing_ids = {m.model_id for m in existing_models}
        
        # Get scanned model IDs
        scanned_ids = {m.model_id for m in models}
        
        # Models to add/update
        count = 0
        for model in models:
            try:
                # Convert to dict for database
                model_dict = model.to_dict()
                
                if model.model_id in existing_ids:
                    # Update existing
                    self.db_repo.update_model(model.model_id, model_dict)
                    logger.debug(f"Updated: {model.name}")
                else:
                    # Add new
                    self.db_repo.add_model(model_dict)
                    logger.debug(f"Added: {model.name}")
                
                count += 1
                
            except Exception as e:
                logger.error(f"Failed to update {model.model_id}: {e}")
        
        # Mark missing models as invalid
        missing_ids = existing_ids - scanned_ids
        if missing_ids:
            logger.info(f"Marking {len(missing_ids)} missing models as invalid")
            for model_id in missing_ids:
                try:
                    self.db_repo.update_model(model_id, {'is_valid': False})
                except Exception as e:
                    logger.error(f"Failed to mark {model_id} as invalid: {e}")
        
        return count
    
    def get_model_summary(self) -> dict:
        """
        Get summary of scanned models
        
        Returns:
            Dictionary with statistics
        """
        result = self.scan_directory()
        
        return {
            'total_found': result.total_found,
            'valid': len(result.valid_models),
            'invalid': len(result.invalid_models),
            'success_rate': result.success_rate(),
            'scan_time': result.scan_time,
            'models': [
                {
                    'id': m.model_id,
                    'name': m.name,
                    'size_mb': m.file_size / 1024 / 1024,
                    'sample_rate': m.sample_rate,
                    'version': m.version
                }
                for m in result.valid_models
            ]
        }


def scan_models(
    base_path: Path = Path("SampleVoice"),
    db_repo=None
) -> ModelScanResult:
    """
    Convenience function to scan models
    
    Args:
        base_path: Directory to scan
        db_repo: Database repository
        
    Returns:
        ModelScanResult
    """
    scanner = ModelScanner(base_path, db_repo)
    return scanner.scan_directory()