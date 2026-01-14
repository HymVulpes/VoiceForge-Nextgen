"""
VoiceForge-Nextgen - Base Window
File: app/gui/base_window.py

Purpose:
    Base window class with common GUI utilities
    Provides consistent styling and behavior

Dependencies:
    - PyQt6

Usage:
    class MyWindow(BaseWindow):
        def __init__(self):
            super().__init__(title="My Window")
"""

from PyQt6.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QMessageBox
)
from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtGui import QFont, QPalette, QColor
import logging

logger = logging.getLogger("BaseWindow")


class BaseWindow(QMainWindow):
    """
    Base window with common functionality
    
    Features:
        - Consistent styling
        - Error handling
        - Status messages
        - Window management
    """
    
    def __init__(
        self,
        title: str = "VoiceForge-Nextgen",
        width: int = 1000,
        height: int = 700
    ):
        """
        Initialize base window
        
        Args:
            title: Window title
            width: Window width
            height: Window height
        """
        super().__init__()
        
        self.setWindowTitle(title)
        self.setMinimumSize(width, height)
        
        # Setup styling
        self._setup_style()
        
        # Central widget
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        
        # Main layout
        self.main_layout = QVBoxLayout(self.central_widget)
        self.main_layout.setContentsMargins(10, 10, 10, 10)
        self.main_layout.setSpacing(10)
        
        logger.debug(f"BaseWindow initialized: {title}")
    
    def _setup_style(self):
        """Setup window styling"""
        # Set font
        font = QFont("Segoe UI", 9)
        self.setFont(font)
        
        # Set dark theme
        self._set_dark_theme()
    
    def _set_dark_theme(self):
        """Apply dark theme"""
        palette = QPalette()
        
        # Colors
        bg_color = QColor(45, 45, 48)
        fg_color = QColor(255, 255, 255)
        highlight_color = QColor(0, 122, 204)
        
        palette.setColor(QPalette.ColorRole.Window, bg_color)
        palette.setColor(QPalette.ColorRole.WindowText, fg_color)
        palette.setColor(QPalette.ColorRole.Base, QColor(30, 30, 30))
        palette.setColor(QPalette.ColorRole.AlternateBase, bg_color)
        palette.setColor(QPalette.ColorRole.Text, fg_color)
        palette.setColor(QPalette.ColorRole.Button, bg_color)
        palette.setColor(QPalette.ColorRole.ButtonText, fg_color)
        palette.setColor(QPalette.ColorRole.Highlight, highlight_color)
        palette.setColor(QPalette.ColorRole.HighlightedText, fg_color)
        
        self.setPalette(palette)
        
        # Additional styling
        self.setStyleSheet("""
            QMainWindow {
                background-color: #2d2d30;
            }
            QPushButton {
                background-color: #3e3e42;
                border: 1px solid #555;
                padding: 5px 15px;
                border-radius: 3px;
                color: white;
            }
            QPushButton:hover {
                background-color: #4e4e52;
            }
            QPushButton:pressed {
                background-color: #007acc;
            }
            QPushButton:disabled {
                background-color: #2d2d30;
                color: #666;
            }
            QLabel {
                color: white;
            }
            QLineEdit, QSpinBox, QDoubleSpinBox, QComboBox {
                background-color: #1e1e1e;
                border: 1px solid #555;
                padding: 3px;
                color: white;
            }
            QGroupBox {
                border: 1px solid #555;
                margin-top: 10px;
                padding-top: 10px;
                color: white;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                subcontrol-position: top left;
                padding: 0 3px;
            }
        """)
    
    def show_error(self, title: str, message: str):
        """
        Show error dialog
        
        Args:
            title: Error title
            message: Error message
        """
        QMessageBox.critical(self, title, message)
        logger.error(f"{title}: {message}")
    
    def show_warning(self, title: str, message: str):
        """
        Show warning dialog
        
        Args:
            title: Warning title
            message: Warning message
        """
        QMessageBox.warning(self, title, message)
        logger.warning(f"{title}: {message}")
    
    def show_info(self, title: str, message: str):
        """
        Show info dialog
        
        Args:
            title: Info title
            message: Info message
        """
        QMessageBox.information(self, title, message)
        logger.info(f"{title}: {message}")
    
    def confirm(self, title: str, message: str) -> bool:
        """
        Show confirmation dialog
        
        Args:
            title: Dialog title
            message: Confirmation message
            
        Returns:
            True if confirmed
        """
        reply = QMessageBox.question(
            self,
            title,
            message,
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
        )
        
        return reply == QMessageBox.StandardButton.Yes
    
    def set_status(self, message: str, timeout: int = 3000):
        """
        Set status bar message
        
        Args:
            message: Status message
            timeout: Timeout in milliseconds
        """
        if hasattr(self, 'statusBar'):
            self.statusBar().showMessage(message, timeout)
    
    def closeEvent(self, event):
        """Handle window close event"""
        # Ask for confirmation
        if self.confirm("Exit", "Are you sure you want to exit?"):
            logger.info("Window closed by user")
            event.accept()
        else:
            event.ignore()