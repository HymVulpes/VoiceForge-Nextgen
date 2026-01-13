"""
VoiceForge-Nextgen GUI - Inspired by Dubbing AI
File: app/gui/main_window.py (TẠO MỚI)
"""
import sys
from pathlib import Path
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QComboBox, QLabel, QSlider, QGroupBox, QProgressBar
)
from PyQt6.QtCore import Qt, QTimer, pyqtSignal
from PyQt6.QtGui import QFont, QIcon
import logging

logger = logging.getLogger(__name__)

class StatusWidget(QWidget):
    """Real-time status display"""
    def __init__(self):
        super().__init__()
        layout = QVBoxLayout()
        
        # Status indicator
        self.status_label = QLabel("● READY")
        self.status_label.setStyleSheet("color: #00ff00; font-size: 14px; font-weight: bold;")
        
        # Metrics
        self.latency_label = QLabel("Latency: --ms")
        self.cpu_label = QLabel("CPU: --%")
        self.ram_label = QLabel("RAM: --GB/24GB")
        self.gpu_label = QLabel("GPU: --%")
        self.vram_label = QLabel("VRAM: --GB/8GB")
        
        for label in [self.latency_label, self.cpu_label, self.ram_label, 
                      self.gpu_label, self.vram_label]:
            label.setStyleSheet("color: #ffffff; font-size: 12px;")
        
        layout.addWidget(self.status_label)
        layout.addWidget(self.latency_label)
        layout.addWidget(self.cpu_label)
        layout.addWidget(self.ram_label)
        layout.addWidget(self.gpu_label)
        layout.addWidget(self.vram_label)
        
        self.setLayout(layout)
    
    def update_status(self, status: str, color: str = "#00ff00"):
        """Update status indicator"""
        self.status_label.setText(f"● {status}")
        self.status_label.setStyleSheet(f"color: {color}; font-size: 14px; font-weight: bold;")
    
    def update_metrics(self, metrics: dict):
        """Update performance metrics"""
        self.latency_label.setText(f"Latency: {metrics.get('latency', 0):.1f}ms")
        self.cpu_label.setText(f"CPU: {metrics.get('cpu', 0):.1f}%")
        self.ram_label.setText(f"RAM: {metrics.get('ram', 0):.1f}GB/24GB")
        self.gpu_label.setText(f"GPU: {metrics.get('gpu', 0):.1f}%")
        self.vram_label.setText(f"VRAM: {metrics.get('vram', 0):.1f}GB/8GB")


class VoiceForgeWindow(QMainWindow):
    """Main GUI Window - Dubbing AI Style"""
    
    # Signals
    start_processing = pyqtSignal()
    stop_processing = pyqtSignal()
    
    def __init__(self, app_instance=None):
        super().__init__()
        self.app_instance = app_instance
        self.is_processing = False
        
        self.init_ui()
        self.setup_timer()
    
    def init_ui(self):
        """Initialize UI components"""
        self.setWindowTitle("🎙️ VoiceForge-Nextgen")
        self.setGeometry(100, 100, 600, 700)
        self.setStyleSheet("""
            QMainWindow {
                background-color: #1e1e1e;
            }
            QGroupBox {
                color: #ffffff;
                border: 2px solid #3c3c3c;
                border-radius: 5px;
                margin-top: 10px;
                font-weight: bold;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 5px;
            }
            QLabel {
                color: #ffffff;
            }
            QComboBox, QPushButton {
                background-color: #2d2d2d;
                color: #ffffff;
                border: 1px solid #3c3c3c;
                border-radius: 3px;
                padding: 5px;
            }
            QComboBox:hover, QPushButton:hover {
                background-color: #3c3c3c;
            }
            QPushButton:pressed {
                background-color: #4c4c4c;
            }
            QSlider::groove:horizontal {
                background: #3c3c3c;
                height: 8px;
                border-radius: 4px;
            }
            QSlider::handle:horizontal {
                background: #007acc;
                width: 18px;
                margin: -5px 0;
                border-radius: 9px;
            }
        """)
        
        # Central widget
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)
        main_layout.setSpacing(10)
        main_layout.setContentsMargins(20, 20, 20, 20)
        
        # 1. Audio Devices Section
        devices_group = self.create_devices_section()
        main_layout.addWidget(devices_group)
        
        # 2. Voice Model Section
        model_group = self.create_model_section()
        main_layout.addWidget(model_group)
        
        # 3. Settings Section
        settings_group = self.create_settings_section()
        main_layout.addWidget(settings_group)
        
        # 4. Status Section
        status_group = self.create_status_section()
        main_layout.addWidget(status_group)
        
        # 5. Control Buttons
        controls_layout = self.create_controls()
        main_layout.addLayout(controls_layout)
        
        main_layout.addStretch()
    
    def create_devices_section(self):
        """Create audio devices section"""
        group = QGroupBox("🎤 Audio Devices")
        layout = QHBoxLayout()
        
        # Input device
        input_layout = QVBoxLayout()
        input_layout.addWidget(QLabel("Input:"))
        self.input_combo = QComboBox()
        self.input_combo.addItems(["Default Microphone", "USB Microphone"])
        input_layout.addWidget(self.input_combo)
        layout.addLayout(input_layout)
        
        # Output device
        output_layout = QVBoxLayout()
        output_layout.addWidget(QLabel("Output:"))
        self.output_combo = QComboBox()
        self.output_combo.addItems(["CABLE Input (VB-Audio Virtual Cable)", "Speakers"])
        output_layout.addWidget(self.output_combo)
        layout.addLayout(output_layout)
        
        group.setLayout(layout)
        return group
    
    def create_model_section(self):
        """Create voice model section"""
        group = QGroupBox("🤖 Voice Model")
        layout = QHBoxLayout()
        
        # Model selector
        model_layout = QVBoxLayout()
        model_layout.addWidget(QLabel("Model:"))
        self.model_combo = QComboBox()
        self.model_combo.addItems(["No models loaded", "Scan SampleVoice folder..."])
        model_layout.addWidget(self.model_combo)
        layout.addLayout(model_layout)
        
        # Refresh button
        self.refresh_btn = QPushButton("🔄 Refresh")
        self.refresh_btn.setFixedWidth(100)
        self.refresh_btn.clicked.connect(self.refresh_models)
        layout.addWidget(self.refresh_btn, alignment=Qt.AlignmentFlag.AlignBottom)
        
        group.setLayout(layout)
        return group
    
    def create_settings_section(self):
        """Create settings section"""
        group = QGroupBox("⚙️ Settings")
        layout = QVBoxLayout()
        
        # Pitch slider
        pitch_layout = QHBoxLayout()
        pitch_layout.addWidget(QLabel("Pitch:"))
        self.pitch_slider = QSlider(Qt.Orientation.Horizontal)
        self.pitch_slider.setRange(-12, 12)
        self.pitch_slider.setValue(0)
        self.pitch_slider.setTickPosition(QSlider.TickPosition.TicksBelow)
        self.pitch_slider.setTickInterval(1)
        pitch_layout.addWidget(self.pitch_slider)
        self.pitch_value_label = QLabel("0")
        self.pitch_value_label.setFixedWidth(30)
        pitch_layout.addWidget(self.pitch_value_label)
        layout.addLayout(pitch_layout)
        self.pitch_slider.valueChanged.connect(
            lambda v: self.pitch_value_label.setText(str(v))
        )
        
        # Formant slider
        formant_layout = QHBoxLayout()
        formant_layout.addWidget(QLabel("Formant:"))
        self.formant_slider = QSlider(Qt.Orientation.Horizontal)
        self.formant_slider.setRange(0, 10)
        self.formant_slider.setValue(5)
        formant_layout.addWidget(self.formant_slider)
        self.formant_value_label = QLabel("5")
        self.formant_value_label.setFixedWidth(30)
        formant_layout.addWidget(self.formant_value_label)
        layout.addLayout(formant_layout)
        self.formant_slider.valueChanged.connect(
            lambda v: self.formant_value_label.setText(str(v))
        )
        
        # Volume slider
        volume_layout = QHBoxLayout()
        volume_layout.addWidget(QLabel("Volume:"))
        self.volume_slider = QSlider(Qt.Orientation.Horizontal)
        self.volume_slider.setRange(0, 100)
        self.volume_slider.setValue(100)
        volume_layout.addWidget(self.volume_slider)
        self.volume_value_label = QLabel("100")
        self.volume_value_label.setFixedWidth(30)
        volume_layout.addWidget(self.volume_value_label)
        layout.addLayout(volume_layout)
        self.volume_slider.valueChanged.connect(
            lambda v: self.volume_value_label.setText(str(v))
        )
        
        group.setLayout(layout)
        return group
    
    def create_status_section(self):
        """Create status section"""
        group = QGroupBox("📊 Status")
        layout = QVBoxLayout()
        
        self.status_widget = StatusWidget()
        layout.addWidget(self.status_widget)
        
        group.setLayout(layout)
        return group
    
    def create_controls(self):
        """Create control buttons"""
        layout = QHBoxLayout()
        layout.addStretch()
        
        # Start button
        self.start_btn = QPushButton("▶ START")
        self.start_btn.setFixedSize(120, 40)
        self.start_btn.setStyleSheet("""
            QPushButton {
                background-color: #00aa00;
                color: white;
                font-size: 14px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #00cc00;
            }
        """)
        self.start_btn.clicked.connect(self.on_start)
        layout.addWidget(self.start_btn)
        
        # Stop button
        self.stop_btn = QPushButton("⏹ STOP")
        self.stop_btn.setFixedSize(120, 40)
        self.stop_btn.setEnabled(False)
        self.stop_btn.setStyleSheet("""
            QPushButton {
                background-color: #aa0000;
                color: white;
                font-size: 14px;
                font-weight: bold;
            }
            QPushButton:hover:enabled {
                background-color: #cc0000;
            }
            QPushButton:disabled {
                background-color: #3c3c3c;
            }
        """)
        self.stop_btn.clicked.connect(self.on_stop)
        layout.addWidget(self.stop_btn)
        
        layout.addStretch()
        return layout
    
    def setup_timer(self):
        """Setup update timer"""
        self.update_timer = QTimer()
        self.update_timer.timeout.connect(self.update_ui)
        self.update_timer.start(1000)  # Update every 1 second
    
    def update_ui(self):
        """Update UI with current metrics"""
        if not self.app_instance:
            return
        
        try:
            # Get metrics from app
            metrics = {
                'latency': 12.3,
                'cpu': 15.2,
                'ram': 8.2,
                'gpu': 25.0,
                'vram': 1.2
            }
            
            # TODO: Get real metrics from self.app_instance
            # metrics = self.app_instance.get_metrics()
            
            self.status_widget.update_metrics(metrics)
            
        except Exception as e:
            logger.error(f"Failed to update UI: {e}")
    
    def refresh_models(self):
        """Refresh voice models list"""
        logger.info("Refreshing models...")
        # TODO: Scan SampleVoice folder
        self.model_combo.clear()
        self.model_combo.addItems(["No models found"])
    
    def on_start(self):
        """Start processing"""
        logger.info("Starting voice processing...")
        self.is_processing = True
        self.start_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)
        self.status_widget.update_status("PROCESSING", "#00ff00")
        
        self.start_processing.emit()
    
    def on_stop(self):
        """Stop processing"""
        logger.info("Stopping voice processing...")
        self.is_processing = False
        self.start_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self.status_widget.update_status("STOPPED", "#ff0000")
        
        self.stop_processing.emit()


def main():
    """Standalone GUI test"""
    app = QApplication(sys.argv)
    
    # Set application style
    app.setStyle("Fusion")
    
    window = VoiceForgeWindow()
    window.show()
    
    sys.exit(app.exec())


if __name__ == "__main__":
    main()