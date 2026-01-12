"""
Runtime Context - Unique run_id per execution
All debugging anchored to this context
"""
import uuid
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any
import json

@dataclass
class StageStatus:
    """Status of a processing stage"""
    name: str
    start_time: Optional[float] = None
    end_time: Optional[float] = None
    latency_ms: Optional[float] = None
    status: str = "PENDING"  # PENDING, RUNNING, OK, WARN, FAIL
    error_message: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

@dataclass
class DebugContext:
    """
    Complete runtime context for debugging
    Created once per app launch, persists through entire session
    """
    run_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    start_timestamp: datetime = field(default_factory=datetime.utcnow)
    
    # Paths
    root_dir: Optional[Path] = None
    sample_voice_dir: Optional[Path] = None
    logs_dir: Optional[Path] = None
    snapshots_dir: Optional[Path] = None
    
    # Model information
    active_model_id: Optional[int] = None
    model_pth_path: Optional[str] = None
    model_index_path: Optional[str] = None
    
    # Audio configuration
    input_device_index: int = -1
    output_device_index: int = -1
    input_device_name: str = "Unknown"
    output_device_name: str = "Unknown"
    sample_rate: int = 48000
    buffer_size: int = 256
    
    # Processing configuration
    f0_method: str = "harvest"  # harvest, crepe, pm
    target_latency_ms: float = 20.0
    
    # Stage tracking
    stages: Dict[str, StageStatus] = field(default_factory=dict)
    
    def __post_init__(self):
        """Initialize stages"""
        stage_names = ["INPUT", "PREPROCESS", "INFERENCE", "POSTPROCESS", "OUTPUT"]
        for name in stage_names:
            self.stages[name] = StageStatus(name=name)
    
    def start_stage(self, stage_name: str, timestamp: float):
        """Mark stage as started"""
        if stage_name in self.stages:
            self.stages[stage_name].start_time = timestamp
            self.stages[stage_name].status = "RUNNING"
    
    def end_stage(self, stage_name: str, timestamp: float, status: str = "OK", error: Optional[str] = None):
        """Mark stage as completed"""
        if stage_name in self.stages:
            stage = self.stages[stage_name]
            stage.end_time = timestamp
            stage.status = status
            stage.error_message = error
            if stage.start_time:
                stage.latency_ms = (timestamp - stage.start_time) * 1000
    
    def to_dict(self) -> Dict[str, Any]:
        """Serialize for snapshots"""
        return {
            "run_id": self.run_id,
            "start_timestamp": self.start_timestamp.isoformat(),
            "paths": {
                "root_dir": str(self.root_dir) if self.root_dir else None,
                "sample_voice_dir": str(self.sample_voice_dir) if self.sample_voice_dir else None,
                "logs_dir": str(self.logs_dir) if self.logs_dir else None,
                "snapshots_dir": str(self.snapshots_dir) if self.snapshots_dir else None,
            },
            "model": {
                "active_model_id": self.active_model_id,
                "pth_path": self.model_pth_path,
                "index_path": self.model_index_path,
            },
            "audio": {
                "input_device_index": self.input_device_index,
                "output_device_index": self.output_device_index,
                "input_device_name": self.input_device_name,
                "output_device_name": self.output_device_name,
                "sample_rate": self.sample_rate,
                "buffer_size": self.buffer_size,
            },
            "processing": {
                "f0_method": self.f0_method,
                "target_latency_ms": self.target_latency_ms,
            },
            "stages": {name: stage.to_dict() for name, stage in self.stages.items()}
        }
    
    def save_snapshot(self, reason: str, additional_data: Optional[Dict] = None):
        """Save context snapshot to file"""
        if not self.snapshots_dir:
            return None
            
        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        filename = f"snapshot_{self.run_id[:8]}_{timestamp}_{reason}.json"
        filepath = self.snapshots_dir / filename
        
        snapshot = {
            "reason": reason,
            "timestamp": datetime.utcnow().isoformat(),
            "context": self.to_dict(),
            "additional_data": additional_data or {}
        }
        
        filepath.write_text(json.dumps(snapshot, indent=2))
        return str(filepath)