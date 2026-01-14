"""
VoiceForge-Nextgen - Metrics Widget
File: app/gui/metrics_widget.py

Purpose:
    Display real-time system metrics
    CPU, GPU, RAM, latency, audio stats

Dependencies:
    - PyQt6
    - psutil (system metrics)
    - torch (GPU metrics)

Usage:
    metrics = MetricsWidget()
    metrics.update_metrics(cpu=12.5, ram=8450, ...)
"""

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel,
    QGroupBox, QProgressBar, QGridLayout
)
from PyQt6.QtCore import QTimer
import psutil
import logging

logger = logging.getLogger("MetricsWidget")


class MetricsWidget(QWidget):
    """
    Real-time metrics display widget
    
    Displays:
        - CPU usage
        - RAM usage
        - GPU usage (if available)
        - VRAM usage
        - Latency
        - Audio stats
    """
    
    def __init__(self):
        """Initialize metrics widget"""
        super().__init__()
        
        self._setup_ui()
        
        # Auto-update timer
        self.update_timer = QTimer()
        self.update_timer.timeout.connect(self._auto_update)
        
        # Check GPU availability
        self.gpu_available = False
        try:
            import torch
            self.gpu_available = torch.cuda.is_available()
        except ImportError:
            pass
    
    def _setup_ui(self):
        """Setup UI components"""
        layout = QVBoxLayout(self)
        
        # Group box
        group = QGroupBox("System Metrics")
        group_layout = QGridLayout(group)
        
        # CPU
        group_layout.addWidget(QLabel("CPU:"), 0, 0)
        self.cpu_bar = QProgressBar()
        self.cpu_bar.setMaximum(100)
        self.cpu_label = QLabel("0%")
        group_layout.addWidget(self.cpu_bar, 0, 1)
        group_layout.addWidget(self.cpu_label, 0, 2)
        
        # RAM
        group_layout.addWidget(QLabel("RAM:"), 1, 0)
        self.ram_bar = QProgressBar()
        self.ram_bar.setMaximum(100)
        self.ram_label = QLabel("0 MB")
        group_layout.addWidget(self.ram_bar, 1, 1)
        group_layout.addWidget(self.ram_label, 1, 2)
        
        # GPU
        group_layout.addWidget(QLabel("GPU:"), 2, 0)
        self.gpu_bar = QProgressBar()
        self.gpu_bar.setMaximum(100)
        self.gpu_label = QLabel("N/A")
        group_layout.addWidget(self.gpu_bar, 2, 1)
        group_layout.addWidget(self.gpu_label, 2, 2)
        
        # VRAM
        group_layout.addWidget(QLabel("VRAM:"), 3, 0)
        self.vram_bar = QProgressBar()
        self.vram_bar.setMaximum(100)
        self.vram_label = QLabel("N/A")
        group_layout.addWidget(self.vram_bar, 3, 1)
        group_layout.addWidget(self.vram_label, 3, 2)
        
        layout.addWidget(group)
        
        # Audio stats group
        audio_group = QGroupBox("Audio Statistics")
        audio_layout = QGridLayout(audio_group)
        
        # Latency
        audio_layout.addWidget(QLabel("Latency:"), 0, 0)
        self.latency_label = QLabel("-- ms")
        self.latency_label.setStyleSheet("font-weight: bold;")
        audio_layout.addWidget(self.latency_label, 0, 1)
        
        # Buffer
        audio_layout.addWidget(QLabel("Buffer:"), 1, 0)
        self.buffer_label = QLabel("-- / --")
        audio_layout.addWidget(self.buffer_label, 1, 1)
        
        # Errors
        audio_layout.addWidget(QLabel("Errors:"), 2, 0)
        self.errors_label = QLabel("0")
        audio_layout.addWidget(self.errors_label, 2, 1)
        
        layout.addWidget(audio_group)
        
        # Model info group
        model_group = QGroupBox("Model Status")
        model_layout = QVBoxLayout(model_group)
        
        self.model_name_label = QLabel("No model loaded")
        self.model_name_label.setStyleSheet("font-weight: bold;")
        
        self.model_info_label = QLabel("")
        
        model_layout.addWidget(self.model_name_label)
        model_layout.addWidget(self.model_info_label)
        
        layout.addWidget(model_group)
        
        layout.addStretch()
    
    def start_auto_update(self, interval_ms: int = 1000):
        """
        Start automatic metric updates
        
        Args:
            interval_ms: Update interval in milliseconds
        """
        self.update_timer.start(interval_ms)
        logger.info(f"Auto-update started: {interval_ms}ms interval")
    
    def stop_auto_update(self):
        """Stop automatic updates"""
        self.update_timer.stop()
        logger.info("Auto-update stopped")
    
    def _auto_update(self):
        """Auto-update system metrics"""
        try:
            # CPU
            cpu_percent = psutil.cpu_percent(interval=0.1)
            self.update_cpu(cpu_percent)
            
            # RAM
            mem = psutil.virtual_memory()
            self.update_ram(mem.used / 1024 / 1024, mem.percent)
            
            # GPU
            if self.gpu_available:
                try:
                    import torch
                    
                    gpu_util = 0  # Would need nvidia-ml-py for real GPU util
                    vram_used = torch.cuda.memory_allocated(0) / 1024 / 1024
                    vram_total = torch.cuda.get_device_properties(0).total_memory / 1024 / 1024
                    vram_percent = (vram_used / vram_total) * 100
                    
                    self.update_gpu(gpu_util)
                    self.update_vram(vram_used, vram_percent)
                    
                except Exception as e:
                    logger.debug(f"GPU update failed: {e}")
            
        except Exception as e:
            logger.error(f"Auto-update failed: {e}")
    
    def update_cpu(self, percent: float):
        """Update CPU metrics"""
        self.cpu_bar.setValue(int(percent))
        self.cpu_label.setText(f"{percent:.1f}%")
        
        # Color coding
        if percent > 80:
            self.cpu_bar.setStyleSheet("QProgressBar::chunk { background-color: #d9534f; }")
        elif percent > 60:
            self.cpu_bar.setStyleSheet("QProgressBar::chunk { background-color: #f0ad4e; }")
        else:
            self.cpu_bar.setStyleSheet("QProgressBar::chunk { background-color: #5cb85c; }")
    
    def update_ram(self, used_mb: float, percent: float):
        """Update RAM metrics"""
        self.ram_bar.setValue(int(percent))
        self.ram_label.setText(f"{used_mb:.0f} MB ({percent:.1f}%)")
        
        # Color coding
        if percent > 85:
            self.ram_bar.setStyleSheet("QProgressBar::chunk { background-color: #d9534f; }")
        elif percent > 70:
            self.ram_bar.setStyleSheet("QProgressBar::chunk { background-color: #f0ad4e; }")
        else:
            self.ram_bar.setStyleSheet("QProgressBar::chunk { background-color: #5cb85c; }")
    
    def update_gpu(self, percent: float):
        """Update GPU metrics"""
        self.gpu_bar.setValue(int(percent))
        self.gpu_label.setText(f"{percent:.1f}%")
    
    def update_vram(self, used_mb: float, percent: float):
        """Update VRAM metrics"""
        self.vram_bar.setValue(int(percent))
        self.vram_label.setText(f"{used_mb:.0f} MB ({percent:.1f}%)")
        
        # Color coding
        if percent > 90:
            self.vram_bar.setStyleSheet("QProgressBar::chunk { background-color: #d9534f; }")
        elif percent > 75:
            self.vram_bar.setStyleSheet("QProgressBar::chunk { background-color: #f0ad4e; }")
        else:
            self.vram_bar.setStyleSheet("QProgressBar::chunk { background-color: #5cb85c; }")
    
    def update_latency(self, latency_ms: float, warning_threshold: float = 15.0):
        """
        Update latency display
        
        Args:
            latency_ms: Latency in milliseconds
            warning_threshold: Threshold for warning color
        """
        self.latency_label.setText(f"{latency_ms:.2f} ms")
        
        # Color coding
        if latency_ms > warning_threshold * 2:
            self.latency_label.setStyleSheet("color: #d9534f; font-weight: bold;")
        elif latency_ms > warning_threshold:
            self.latency_label.setStyleSheet("color: #f0ad4e; font-weight: bold;")
        else:
            self.latency_label.setStyleSheet("color: #5cb85c; font-weight: bold;")
    
    def update_buffer(self, used: int, total: int):
        """Update buffer usage"""
        self.buffer_label.setText(f"{used} / {total}")
    
    def update_errors(self, count: int):
        """Update error count"""
        self.errors_label.setText(str(count))
        
        if count > 0:
            self.errors_label.setStyleSheet("color: #d9534f; font-weight: bold;")
        else:
            self.errors_label.setStyleSheet("color: white;")
    
    def update_model_info(self, name: str, info: str = ""):
        """
        Update model information
        
        Args:
            name: Model name
            info: Additional info
        """
        self.model_name_label.setText(name)
        self.model_info_label.setText(info)
    
    def clear_model_info(self):
        """Clear model information"""
        self.model_name_label.setText("No model loaded")
        self.model_info_label.setText("")