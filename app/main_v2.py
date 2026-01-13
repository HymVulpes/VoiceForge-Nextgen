"""
VoiceForge-Nextgen Main V2 - RAM-Optimized
Priority: Correctness → Stability → Performance → Debug

24GB RAM Allocation:
- System: 4GB
- Audio Buffers: 2GB (pre-allocated)
- Model Cache: 8GB (multi-model hot cache)
- Feature Cache: 4GB (F0, HuBERT)
- Working Memory: 4GB
- Safety Margin: 2GB
"""
import sys
from pathlib import Path
import logging
import signal
import time
import psutil

# Add app directory to path
ROOT_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT_DIR))

from app.db.base import DatabaseManager
from app.db.repository import AudioConfigRepository
from app.utils.runtime_context import DebugContext
from app.utils.logger import setup_logger
from app.utils.debugger import SnapshotDebugger
from app.utils.profiler import Profiler
from app.utils.health_monitor import HealthMonitor

from app.audio.device_manager import DeviceManager
from app.audio.buffer_pool import BufferPool
from app.audio.audio_stream_v2 import AudioStreamV2

from app.core.model_cache import ModelCache
from app.core.feature_cache import FeatureCache

logger = logging.getLogger(__name__)


class VoiceForgeAppV2:
    """
    RAM-Optimized Application with Correctness First
    """

    def __init__(self):
        self.context = DebugContext()

        self.context.root_dir = ROOT_DIR
        self.context.sample_voice_dir = ROOT_DIR / "SampleVoice"
        self.context.logs_dir = ROOT_DIR / "logs"
        self.context.snapshots_dir = self.context.logs_dir / "snapshots"

        self.db_manager: DatabaseManager = None
        self.device_manager: DeviceManager = None
        self.debugger: SnapshotDebugger = None
        self.profiler: Profiler = None
        self.health_monitor: HealthMonitor = None

        self.buffer_pool: BufferPool = None
        self.model_cache: ModelCache = None
        self.feature_cache: FeatureCache = None

        self.audio_stream: AudioStreamV2 = None

        self.running = False
        self.last_integrity_check = 0.0
        self.integrity_check_interval = 60.0

        self.monitoring_mode = "minimal"

    def _check_system_resources(self) -> bool:
        logger.info("Checking system resources...")

        mem = psutil.virtual_memory()
        total_gb = mem.total / (1024**3)
        available_gb = mem.available / (1024**3)

        logger.info(f"System RAM: {total_gb:.1f}GB total, {available_gb:.1f}GB available")

        if available_gb < 10:
            logger.error("Insufficient available RAM")
            return False

        disk = psutil.disk_usage(str(ROOT_DIR))
        free_gb = disk.free / (1024**3)
        logger.info(f"Disk space: {free_gb:.1f}GB free")

        return True

    def initialize(self) -> bool:
        try:
            setup_logger(self.context.logs_dir, self.context.run_id)

            logger.info("=" * 80)
            logger.info("VoiceForge-Nextgen V2 (RAM-Optimized)")
            logger.info(f"Run ID: {self.context.run_id}")
            logger.info("=" * 80)

            if not self._check_system_resources():
                return False

            self.db_manager = DatabaseManager(self.context.root_dir / "voiceforge.db")
            self.db_manager.initialize()

            self.debugger = SnapshotDebugger(self.context.snapshots_dir)
            self.profiler = Profiler(target_latency_ms=20.0)
            self.health_monitor = HealthMonitor()

            self.buffer_pool = BufferPool(
                slot_size_samples=480000,
                num_slots=100,
                channels=1
            )

            if not self.buffer_pool.validate_integrity():
                return False

            self.model_cache = ModelCache(max_cache_size_gb=8.0, device="cuda")
            self.feature_cache = FeatureCache(max_cache_size_gb=4.0)

            self.device_manager = DeviceManager()
            self.device_manager.initialize()

            session = self.db_manager.get_session()
            audio_config = AudioConfigRepository.get_or_create(session)

            vac = self.device_manager.find_virtual_cable()
            if not vac:
                logger.error("Virtual Audio Cable NOT FOUND")
                return False

            AudioConfigRepository.update(
                session,
                virtual_output_index=vac.index,
                output_device_name=vac.name
            )

            default_input = self.device_manager.get_default_input()
            if not default_input:
                logger.error("No input device found")
                return False

            AudioConfigRepository.update(
                session,
                input_device_index=default_input.index,
                input_device_name=default_input.name
            )

            self.context.input_device_index = default_input.index
            self.context.output_device_index = vac.index
            self.context.input_device_name = default_input.name
            self.context.output_device_name = vac.name
            self.context.sample_rate = audio_config.sample_rate
            self.context.buffer_size = audio_config.buffer_size

            session.close()
            return True

        except Exception as e:
            logger.critical(f"Initialization failed: {e}", exc_info=True)
            self.cleanup()
            return False

    def start_golden_path(self) -> bool:
        logger.info("Starting GOLDEN PATH (no AI)")

        self.monitoring_mode = "minimal"

        self.audio_stream = AudioStreamV2(
            input_device_index=self.context.input_device_index,
            output_device_index=self.context.output_device_index,
            sample_rate=self.context.sample_rate,
            buffer_size=self.context.buffer_size,
            buffer_pool=self.buffer_pool,
            pyaudio_instance=self.device_manager.p
        )

        self.audio_stream.set_processing_callback(None)

        if not self.audio_stream.start():
            return False

        self.running = True
        self._monitor_loop()
        return True

    # ===================== FIXED PART =====================
    def _monitor_loop(self):
        """Monitor với exit condition rõ ràng"""

        monitor_interval = {
            "full": 5.0,
            "minimal": 30.0,
            "off": None
        }.get(self.monitoring_mode, 30.0)

        last_monitor = time.time()
        loop_counter = 0
        max_iterations = 10000

        while self.running and loop_counter < max_iterations:
            try:
                time.sleep(0.1)
                loop_counter += 1

                if self.audio_stream and getattr(self.audio_stream, "input_callback_count", 0) > 0:
                    loop_counter = 0

                now = time.time()

                if monitor_interval and now - last_monitor >= monitor_interval:
                    self._log_system_status()
                    last_monitor = now

                if now - self.last_integrity_check >= 60.0:
                    self._integrity_check()
                    self.last_integrity_check = now

            except KeyboardInterrupt:
                self.running = False
                break
            except Exception as e:
                logger.error(f"Monitor error: {e}", exc_info=True)
                time.sleep(1.0)

        if loop_counter >= max_iterations:
            logger.warning("Monitor loop reached safety limit")
    # =====================================================

    def _log_system_status(self):
        logger.info("System running normally")

    def _integrity_check(self):
        if not self.buffer_pool.validate_integrity():
            logger.critical("Buffer pool corruption detected")
            if self.audio_stream:
                self.audio_stream._activate_failsafe("buffer_pool_corruption")

    def stop(self):
        self.running = False
        if self.audio_stream:
            self.audio_stream.stop()

    def cleanup(self):
        if self.audio_stream:
            self.audio_stream.dispose()
        if self.device_manager:
            self.device_manager.dispose()
        if self.db_manager:
            self.db_manager.dispose()


def main():
    app = VoiceForgeAppV2()

    def handle_signal(sig, frame):
        app.stop()
        app.cleanup()
        sys.exit(0)

    signal.signal(signal.SIGINT, handle_signal)
    signal.signal(signal.SIGTERM, handle_signal)

    if not app.initialize():
        sys.exit(1)

    if not app.start_golden_path():
        sys.exit(1)

    app.cleanup()


if __name__ == "__main__":
    main()
