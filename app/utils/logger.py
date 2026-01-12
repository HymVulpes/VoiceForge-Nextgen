"""
Structured Logging Configuration
Color-coded console + persistent file logs
"""
import logging
import colorlog
from pathlib import Path
from datetime import datetime

def setup_logger(logs_dir: Path, run_id: str) -> logging.Logger:
    """
    Configure structured logging with color output
    
    Args:
        logs_dir: Directory for log files
        run_id: Current run identifier
        
    Returns:
        Configured logger
    """
    logs_dir.mkdir(parents=True, exist_ok=True)
    
    # Create logger
    logger = logging.getLogger("VoiceForge")
    logger.setLevel(logging.DEBUG)
    logger.handlers.clear()  # Remove existing handlers
    
    # Console handler with colors
    console_handler = colorlog.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_format = colorlog.ColoredFormatter(
        "%(log_color)s%(asctime)s [%(levelname)8s] %(name)s:%(lineno)d - %(message)s%(reset)s",
        datefmt="%H:%M:%S",
        log_colors={
            'DEBUG': 'cyan',
            'INFO': 'green',
            'WARNING': 'yellow',
            'ERROR': 'red',
            'CRITICAL': 'red,bg_white',
        }
    )
    console_handler.setFormatter(console_format)
    logger.addHandler(console_handler)
    
    # File handler - persistent debug log
    debug_log_path = logs_dir / "debug.log"
    file_handler = logging.FileHandler(debug_log_path, mode='a', encoding='utf-8')
    file_handler.setLevel(logging.DEBUG)
    file_format = logging.Formatter(
        "%(asctime)s [%(levelname)8s] [%(name)s:%(lineno)d] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    file_handler.setFormatter(file_format)
    logger.addHandler(file_handler)
    
    # Run-specific log file
    timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    run_log_path = logs_dir / f"run_{run_id[:8]}_{timestamp}.log"
    run_handler = logging.FileHandler(run_log_path, mode='w', encoding='utf-8')
    run_handler.setLevel(logging.DEBUG)
    run_handler.setFormatter(file_format)
    logger.addHandler(run_handler)
    
    logger.info(f"Logging initialized: run_id={run_id}")
    logger.info(f"Debug log: {debug_log_path}")
    logger.info(f"Run log: {run_log_path}")
    
    return logger