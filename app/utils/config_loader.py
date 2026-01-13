"""
Config Loader - Load config.yml properly
File: app/utils/config_loader.py
"""
import yaml
from pathlib import Path
from typing import Dict, Any
import logging

logger = logging.getLogger(__name__)

class ConfigLoader:
    """Load and validate configuration"""
    
    DEFAULT_CONFIG = {
        "resources": {
            "total_ram_gb": 24,
            "allocation": {
                "audio_buffers_gb": 2,
                "model_cache_gb": 8,
                "feature_cache_gb": 4
            }
        },
        "audio": {
            "sample_rate": 48000,
            "buffer_size": 256,
            "channels": 1,
            "validate_devices_on_startup": True
        },
        "processing": {
            "f0_method": "harvest",
            "target_latency_ms": 20.0,
            "failsafe": {
                "max_consecutive_errors": 5,
                "auto_recover": True
            }
        },
        "debug": {
            "snapshots": {
                "on_error": True,
                "on_latency_spike": True
            }
        }
    }

    def __init__(self, config_path: Path):
        self.config_path = config_path
        self.config = self.DEFAULT_CONFIG.copy()

    def load(self) -> Dict[str, Any]:
        """Load config from file or use defaults"""
        if not self.config_path.exists():
            logger.warning(f"Config file not found: {self.config_path}. Using defaults.")
            return self.config
            
        try:
            with open(self.config_path, 'r', encoding='utf-8') as f:
                loaded_config = yaml.safe_load(f)
                if loaded_config:
                    self._merge_config(loaded_config)
            logger.info(f"Config loaded successfully from {self.config_path}")
            return self.config
        except Exception as e:
            logger.error(f"Failed to load config: {e}. Using defaults.")
            return self.config

    def _merge_config(self, loaded: Dict[str, Any]):
        """Recursively merge loaded config with defaults"""
        for key, value in loaded.items():
            if key in self.config and isinstance(value, dict) and isinstance(self.config[key], dict):
                self.config[key].update(value)
            else:
                self.config[key] = value