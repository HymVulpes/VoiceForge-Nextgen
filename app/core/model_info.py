"""
VoiceForge-Nextgen - Model Information Data Class
File: app/core/model_info.py

Purpose:
    Data class for RVC model metadata
    Used by ModelScanner to represent discovered models

Dependencies:
    - pydantic (validation)
    - pathlib (file paths)
    - datetime (timestamps)

Usage:
    model = ModelInfo(
        model_id="alice_v1",
        name="Alice Voice",
        pth_path=Path("SampleVoice/alice.pth"),
        index_path=Path("SampleVoice/alice.index")
    )
"""

from pathlib import Path
from datetime import datetime
from typing import Optional
from pydantic import BaseModel, Field, validator


class ModelInfo(BaseModel):
    """
    Metadata for a single RVC model
    
    Attributes:
        model_id: Unique identifier (generated from filename)
        name: Display name
        pth_path: Path to .pth file
        index_path: Path to .index file (optional)
        sample_rate: Model sample rate (default 48000)
        f0_method: F0 extraction method (harvest/dio/crepe)
        version: RVC version (v1/v2)
        file_size: Size of .pth file in bytes
        created_at: File creation timestamp
        is_valid: Validation status
    """
    
    model_id: str = Field(..., description="Unique model identifier")
    name: str = Field(..., description="Human-readable name")
    pth_path: Path = Field(..., description="Path to .pth file")
    index_path: Optional[Path] = Field(None, description="Path to .index file")
    
    sample_rate: int = Field(48000, description="Model sample rate")
    f0_method: str = Field("harvest", description="F0 extraction method")
    version: str = Field("v2", description="RVC version")
    
    file_size: int = Field(0, description="File size in bytes")
    created_at: datetime = Field(default_factory=datetime.now)
    is_valid: bool = Field(True, description="Validation status")
    
    class Config:
        arbitrary_types_allowed = True
        json_encoders = {
            Path: str,
            datetime: lambda v: v.isoformat()
        }
    
    @validator('pth_path')
    def pth_must_exist(cls, v):
        """Validate .pth file exists"""
        if not v.exists():
            raise ValueError(f"PTH file not found: {v}")
        if not v.suffix == '.pth':
            raise ValueError(f"Not a .pth file: {v}")
        return v
    
    @validator('index_path')
    def index_optional(cls, v):
        """Validate .index file if provided"""
        if v is not None:
            if not v.exists():
                raise ValueError(f"Index file not found: {v}")
            if not v.suffix == '.index':
                raise ValueError(f"Not a .index file: {v}")
        return v
    
    @validator('f0_method')
    def f0_valid(cls, v):
        """Validate F0 method"""
        valid_methods = ['harvest', 'dio', 'crepe', 'rmvpe', 'pm']
        if v.lower() not in valid_methods:
            raise ValueError(f"Invalid f0_method: {v}. Must be one of {valid_methods}")
        return v.lower()
    
    @validator('version')
    def version_valid(cls, v):
        """Validate RVC version"""
        valid_versions = ['v1', 'v2']
        if v.lower() not in valid_versions:
            raise ValueError(f"Invalid version: {v}. Must be one of {valid_versions}")
        return v.lower()
    
    def to_dict(self) -> dict:
        """Convert to dictionary for database storage"""
        return {
            'model_id': self.model_id,
            'name': self.name,
            'pth_path': str(self.pth_path),
            'index_path': str(self.index_path) if self.index_path else None,
            'sample_rate': self.sample_rate,
            'f0_method': self.f0_method,
            'version': self.version,
            'file_size': self.file_size,
            'created_at': self.created_at,
            'is_valid': self.is_valid
        }
    
    @classmethod
    def from_files(
        cls,
        pth_path: Path,
        index_path: Optional[Path] = None
    ) -> "ModelInfo":
        """
        Create ModelInfo from file paths
        
        Args:
            pth_path: Path to .pth file
            index_path: Path to .index file (optional)
            
        Returns:
            ModelInfo instance
        """
        # Generate ID from filename
        model_id = pth_path.stem
        
        # Generate display name (capitalize, replace underscores)
        name = model_id.replace('_', ' ').title()
        
        # Get file size
        file_size = pth_path.stat().st_size
        
        # Get creation time
        created_at = datetime.fromtimestamp(pth_path.stat().st_mtime)
        
        return cls(
            model_id=model_id,
            name=name,
            pth_path=pth_path,
            index_path=index_path,
            file_size=file_size,
            created_at=created_at
        )
    
    def __repr__(self) -> str:
        return (
            f"ModelInfo(id={self.model_id}, "
            f"name={self.name}, "
            f"size={self.file_size / 1024 / 1024:.1f}MB, "
            f"valid={self.is_valid})"
        )


class ModelScanResult(BaseModel):
    """
    Result of a model scanning operation
    
    Attributes:
        total_found: Total models found
        valid_models: List of valid models
        invalid_models: List of invalid models
        scan_time: Time taken to scan
    """
    
    total_found: int = 0
    valid_models: list[ModelInfo] = []
    invalid_models: list[tuple[Path, str]] = []  # (path, error_message)
    scan_time: float = 0.0
    
    class Config:
        arbitrary_types_allowed = True
    
    def success_rate(self) -> float:
        """Calculate success rate"""
        if self.total_found == 0:
            return 0.0
        return len(self.valid_models) / self.total_found * 100
    
    def summary(self) -> str:
        """Generate summary string"""
        return (
            f"Scan Results:\n"
            f"  Total found: {self.total_found}\n"
            f"  Valid: {len(self.valid_models)}\n"
            f"  Invalid: {len(self.invalid_models)}\n"
            f"  Success rate: {self.success_rate():.1f}%\n"
            f"  Scan time: {self.scan_time:.2f}s"
        )