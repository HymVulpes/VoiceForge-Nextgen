import sys
import signal
import logging
from pathlib import Path
from PyQt6.QtCore import QThread, pyqtSignal

# Import nội bộ
from app.utils.config_loader import ConfigLoader

# Khởi tạo log
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

ROOT_DIR = Path(__file__).parent.parent

class BackendWorker(QThread):
    """Worker thread to run backend processing without blocking GUI"""
    error_occurred = pyqtSignal(str)

    def __init__(self, backend_instance):
        super().__init__()
        self.backend = backend_instance

    def run(self):
        try:
            if not self.backend.initialize():
                self.error_occurred.emit("Backend initialization failed")
                return
            
            # Bắt đầu vòng lặp xử lý âm thanh (Golden Path)
            if not self.backend.start_golden_path():
                self.error_occurred.emit("Failed to start Golden Path")
        except Exception as e:
            logger.error(f"Worker Error: {e}", exc_info=True)
            self.error_occurred.emit(str(e))

class VoiceForgeAppV2:
    def __init__(self):
        # Khởi tạo Config
        config_path = ROOT_DIR / "config.yml"
        self.config_loader = ConfigLoader(config_path)
        self.config = self.config_loader.load()
        
        # Gán các giá trị từ config
        self.target_latency_ms = self.config["processing"]["target_latency_ms"]
        self.sample_rate = self.config["audio"]["sample_rate"]
        self.buffer_size = self.config["audio"]["buffer_size"]
        
        self._running = False
        logger.info("VoiceForgeAppV2 Initialized")

    def initialize(self):
        # Logic khởi tạo (load models, v.v.)
        logger.info("Initializing backend resources...")
        # ADD TO app/main_v2.py in initialize() method

# After buffer pool initialization:

        logger.info("Scanning models in SampleVoice/...")
        from app.core.model_scanner import ModelScanner

        scanner = ModelScanner(
            base_path=Path("SampleVoice"),
            db_repo=self.voice_model_repo
        )

        try:
            num_models = scanner.scan_and_update()
            logger.info(f"✓ Found and validated {num_models} models")
            
            if num_models == 0:
                logger.warning("No models found - place .pth files in SampleVoice/")
            
        except Exception as e:
            logger.error(f"Model scanning failed: {e}")
            # Continue anyway - not critical for Golden Path
        return True

    def start_golden_path(self):
        self._running = True
        logger.info("Golden Path started.")
        while self._running:
            # Logic xử lý audio chính ở đây
            pass
        return True

    def stop(self):
        self._running = False
        logger.info("Backend stopped.")

    def cleanup(self):
        logger.info("Cleaning up resources...")

# ============================================================
# GUI INTEGRATION
# ============================================================

def create_gui_app():
    """Create GUI application with backend integration using Threading"""
    from PyQt6.QtWidgets import QApplication
    from app.gui.main_window import VoiceForgeWindow

    qt_app = QApplication(sys.argv)
    qt_app.setStyle("Fusion")

    # 1. Khởi tạo Backend
    backend = VoiceForgeAppV2()
    
    # 2. Khởi tạo Worker Thread
    worker = BackendWorker(backend)

    # 3. Khởi tạo Window
    window = VoiceForgeWindow(app_instance=backend)

    # 4. Kết nối Signals (Quan trọng)
    window.start_processing.connect(worker.start) # Chạy worker thay vì chạy trực tiếp
    window.stop_processing.connect(backend.stop)
    
    worker.error_occurred.connect(lambda msg: logger.error(f"GUI Alert: {msg}"))

    window.show()
    return qt_app.exec()

def main():
    """Main entry point - CLI or GUI"""
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--gui', action='store_true', help='Launch GUI mode')
    args = parser.parse_args()

    if args.gui:
        return create_gui_app()
    else:
        # Chế độ CLI gốc
        app = VoiceForgeAppV2()
        
        def signal_handler(sig, frame):
            logger.info("Signal received, shutting down...")
            app.stop()
            app.cleanup()
            sys.exit(0)

        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)

        try:
            if not app.initialize():
                sys.exit(1)
            app.start_golden_path()
        except Exception as e:
            logger.critical(f"Fatal error: {e}")
            sys.exit(1)
        finally:
            app.cleanup()

if __name__ == "__main__":
    main()