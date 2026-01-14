"""
VoiceForge-Nextgen - Model Selector Widget
File: app/gui/model_selector_widget.py

Purpose:
    Widget for selecting and managing RVC models
    Lists available models, shows details, allows switching

Dependencies:
    - PyQt6
    - app.db.repository (for model data)

Usage:
    selector = ModelSelectorWidget(db_repo)
    selector.model_changed.connect(on_model_changed)
"""

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel,
    QPushButton, QListWidget, QListWidgetItem,
    QGroupBox, QLineEdit
)
from PyQt6.QtCore import pyqtSignal, Qt
import logging

logger = logging.getLogger("ModelSelector")


class ModelSelectorWidget(QWidget):
    """
    Model selection and management widget
    
    Signals:
        model_changed(model_id: str): Emitted when model is selected
        refresh_requested(): Emitted when refresh is requested
    """
    
    model_changed = pyqtSignal(str)
    refresh_requested = pyqtSignal()
    
    def __init__(self, db_repo=None):
        """
        Initialize model selector
        
        Args:
            db_repo: Database repository instance
        """
        super().__init__()
        
        self.db_repo = db_repo
        self.current_model_id = None
        
        self._setup_ui()
        
        # Load initial models
        if self.db_repo:
            self.refresh_models()
    
    def _setup_ui(self):
        """Setup UI components"""
        layout = QVBoxLayout(self)
        
        # Group box
        group = QGroupBox("Model Selection")
        group_layout = QVBoxLayout(group)
        
        # Search bar
        search_layout = QHBoxLayout()
        search_label = QLabel("Search:")
        self.search_input = QLineEdit()
        self.search_input.setPlaceholderText("Type to filter models...")
        self.search_input.textChanged.connect(self._on_search_changed)
        
        search_layout.addWidget(search_label)
        search_layout.addWidget(self.search_input)
        
        group_layout.addLayout(search_layout)
        
        # Model list
        self.model_list = QListWidget()
        self.model_list.setMinimumHeight(200)
        self.model_list.itemClicked.connect(self._on_model_clicked)
        self.model_list.itemDoubleClicked.connect(self._on_model_double_clicked)
        
        group_layout.addWidget(self.model_list)
        
        # Model info
        info_layout = QVBoxLayout()
        
        self.info_name = QLabel("No model selected")
        self.info_name.setStyleSheet("font-weight: bold; font-size: 11pt;")
        
        self.info_details = QLabel("")
        self.info_details.setWordWrap(True)
        
        info_layout.addWidget(self.info_name)
        info_layout.addWidget(self.info_details)
        
        group_layout.addLayout(info_layout)
        
        # Buttons
        button_layout = QHBoxLayout()
        
        self.refresh_button = QPushButton("🔄 Refresh")
        self.refresh_button.clicked.connect(self._on_refresh_clicked)
        
        self.load_button = QPushButton("Load Model")
        self.load_button.setEnabled(False)
        self.load_button.clicked.connect(self._on_load_clicked)
        
        button_layout.addWidget(self.refresh_button)
        button_layout.addStretch()
        button_layout.addWidget(self.load_button)
        
        group_layout.addLayout(button_layout)
        
        layout.addWidget(group)
    
    def refresh_models(self):
        """Refresh model list from database"""
        if not self.db_repo:
            logger.warning("No database repository provided")
            return
        
        try:
            # Clear list
            self.model_list.clear()
            
            # Get models from database
            models = self.db_repo.get_all_models()
            
            logger.info(f"Found {len(models)} models in database")
            
            # Add to list
            for model in models:
                if not model.is_valid:
                    continue  # Skip invalid models
                
                # Create list item
                item = QListWidgetItem(model.name)
                item.setData(Qt.ItemDataRole.UserRole, model.model_id)
                
                # Add metadata as tooltip
                tooltip = (
                    f"ID: {model.model_id}\n"
                    f"Version: {model.version}\n"
                    f"Sample Rate: {model.sample_rate}Hz\n"
                    f"F0 Method: {model.f0_method}\n"
                    f"Size: {model.file_size / 1024 / 1024:.1f}MB"
                )
                item.setToolTip(tooltip)
                
                self.model_list.addItem(item)
            
            # Update count
            count = self.model_list.count()
            logger.info(f"Model list updated: {count} valid models")
            
        except Exception as e:
            logger.error(f"Failed to refresh models: {e}")
    
    def _on_search_changed(self, text: str):
        """Handle search text change"""
        # Filter model list
        for i in range(self.model_list.count()):
            item = self.model_list.item(i)
            
            if text.lower() in item.text().lower():
                item.setHidden(False)
            else:
                item.setHidden(True)
    
    def _on_model_clicked(self, item: QListWidgetItem):
        """Handle model item clicked"""
        model_id = item.data(Qt.ItemDataRole.UserRole)
        
        # Get model details
        if self.db_repo:
            try:
                model = self.db_repo.get_model_by_id(model_id)
                
                if model:
                    # Update info
                    self.info_name.setText(model.name)
                    
                    details = (
                        f"Version: {model.version} | "
                        f"Sample Rate: {model.sample_rate}Hz\n"
                        f"F0 Method: {model.f0_method} | "
                        f"Size: {model.file_size / 1024 / 1024:.1f}MB\n"
                        f"Path: {model.pth_path}"
                    )
                    self.info_details.setText(details)
                    
                    # Enable load button
                    self.load_button.setEnabled(True)
                    self.current_model_id = model_id
                
            except Exception as e:
                logger.error(f"Failed to get model details: {e}")
    
    def _on_model_double_clicked(self, item: QListWidgetItem):
        """Handle model item double-clicked (load immediately)"""
        self._on_model_clicked(item)
        self._on_load_clicked()
    
    def _on_refresh_clicked(self):
        """Handle refresh button clicked"""
        logger.info("Refreshing model list...")
        self.refresh_requested.emit()
        self.refresh_models()
    
    def _on_load_clicked(self):
        """Handle load button clicked"""
        if self.current_model_id:
            logger.info(f"Loading model: {self.current_model_id}")
            self.model_changed.emit(self.current_model_id)
    
    def get_selected_model_id(self) -> str:
        """Get currently selected model ID"""
        return self.current_model_id
    
    def select_model(self, model_id: str):
        """
        Programmatically select a model
        
        Args:
            model_id: Model ID to select
        """
        for i in range(self.model_list.count()):
            item = self.model_list.item(i)
            
            if item.data(Qt.ItemDataRole.UserRole) == model_id:
                self.model_list.setCurrentItem(item)
                self._on_model_clicked(item)
                break