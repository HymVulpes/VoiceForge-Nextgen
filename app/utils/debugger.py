"""
Snapshot Debugger
Captures system state snapshots for offline debugging
"""
import json
import numpy as np
from pathlib import Path
from typing import Optional, Dict, Any
from datetime import datetime
import logging
import traceback

logger = logging.getLogger(__name__)

class SnapshotDebugger:
    """
    Captures snapshots of system state for debugging
    
    Snapshots include:
    - Full runtime context
    - Audio buffers (if available)
    - Error information
    - System state
    """
    
    def __init__(self, snapshots_dir: Path):
        """
        Args:
            snapshots_dir: Directory to save snapshots
        """
        self.snapshots_dir = Path(snapshots_dir)
        self.snapshots_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"SnapshotDebugger initialized: {self.snapshots_dir}")
    
    def capture_snapshot(
        self,
        run_id: str,
        reason: str,
        stage: str,
        context: Dict[str, Any],
        error: Optional[Exception] = None,
        additional_data: Optional[Dict] = None,
        audio_input: Optional[np.ndarray] = None,
        audio_output: Optional[np.ndarray] = None
    ) -> Optional[str]:
        """
        Capture a snapshot of system state
        
        Args:
            run_id: Current run identifier
            reason: Reason for snapshot (e.g., "error", "latency_spike")
            stage: Processing stage where snapshot was taken
            context: Runtime context dictionary
            error: Exception if any
            additional_data: Additional data to include
            audio_input: Input audio buffer (optional)
            audio_output: Output audio buffer (optional)
            
        Returns:
            Path to snapshot directory or None on failure
        """
        try:
            # Create snapshot directory
            timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
            snapshot_name = f"snapshot_{run_id[:8]}_{timestamp}_{reason}"
            snapshot_dir = self.snapshots_dir / snapshot_name
            snapshot_dir.mkdir(parents=True, exist_ok=True)
            
            # Prepare metadata
            metadata = {
                "run_id": run_id,
                "timestamp": datetime.utcnow().isoformat(),
                "reason": reason,
                "stage": stage,
                "context": context,
                "additional_data": additional_data or {}
            }
            
            # Add error information if present
            if error:
                metadata["error"] = {
                    "type": type(error).__name__,
                    "message": str(error),
                    "stack_trace": traceback.format_exc()
                }
            
            # Save metadata
            metadata_path = snapshot_dir / "metadata.json"
            with open(metadata_path, 'w', encoding='utf-8') as f:
                json.dump(metadata, f, indent=2, default=str)
            
            # Save audio buffers if provided
            if audio_input is not None:
                audio_input_path = snapshot_dir / "audio_input.npy"
                np.save(audio_input_path, audio_input)
                logger.debug(f"Saved input audio: {audio_input_path}")
            
            if audio_output is not None:
                audio_output_path = snapshot_dir / "audio_output.npy"
                np.save(audio_output_path, audio_output)
                logger.debug(f"Saved output audio: {audio_output_path}")
            
            logger.info(f"Snapshot captured: {snapshot_name}")
            return str(snapshot_dir)
            
        except Exception as e:
            logger.error(f"Failed to capture snapshot: {e}")
            return None
    
    def load_snapshot(self, snapshot_path: str) -> Optional[Dict[str, Any]]:
        """
        Load a snapshot from disk
        
        Args:
            snapshot_path: Path to snapshot directory
            
        Returns:
            Dictionary with snapshot data or None on failure
        """
        try:
            snapshot_dir = Path(snapshot_path)
            
            # Load metadata
            metadata_path = snapshot_dir / "metadata.json"
            if not metadata_path.exists():
                logger.error(f"Metadata not found: {metadata_path}")
                return None
            
            with open(metadata_path, 'r', encoding='utf-8') as f:
                snapshot_data = json.load(f)
            
            # Load audio buffers if present
            audio_input_path = snapshot_dir / "audio_input.npy"
            if audio_input_path.exists():
                snapshot_data["audio_input"] = np.load(audio_input_path)
            
            audio_output_path = snapshot_dir / "audio_output.npy"
            if audio_output_path.exists():
                snapshot_data["audio_output"] = np.load(audio_output_path)
            
            return snapshot_data
            
        except Exception as e:
            logger.error(f"Failed to load snapshot: {e}")
            return None


