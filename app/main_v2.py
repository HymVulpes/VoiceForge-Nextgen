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
    
    Guarantees:
    1. CORRECTNESS: All operations validated
    2. STABILITY: Never crashes, always failsafe
    3. PERFORMANCE: Zero-copy, pre-allocated
    4. DEBUG: Full observability
    """
    
    def __init__(self):
        # Runtime context
        self.context = DebugContext()
        
        # Paths
        self.context.root_dir = ROOT_DIR
        self.context.sample_voice_dir = ROOT_DIR / "SampleVoice"
        self.context.logs_dir = ROOT_DIR / "logs"
        self.context.snapshots_dir = self.context.logs_dir / "snapshots"
        
        # Core components
        self.db_manager: DatabaseManager = None
        self.device_manager: DeviceManager = None
        self.debugger: SnapshotDebugger = None
        self.profiler: Profiler = None
        self.health_monitor: HealthMonitor = None
        
        # RAM-optimized components
        self.buffer_pool: BufferPool = None
        self.model_cache: ModelCache = None
        self.feature_cache: FeatureCache = None
        
        # Audio stream
        self.audio_stream: AudioStreamV2 = None
        
        # Running state
        self.running = False
        
        # Integrity check timer
        self.last_integrity_check = 0.0
        self.integrity_check_interval = 60.0  # Check every 60 seconds

        # Diagnostic/monitoring mode: full|minimal|off
        # Golden Path should be minimal; AI active can be switched to full/event-based later
        self.monitoring_mode = "minimal"
    
    def _check_system_resources(self) -> bool:
        """
        CORRECTNESS: Validate system has sufficient resources
        
        Returns:
            True if resources adequate
        """
        logger.info("Checking system resources...")
        
        # Check RAM
        mem = psutil.virtual_memory()
        total_gb = mem.total / (1024**3)
        available_gb = mem.available / (1024**3)
        
        logger.info(f"System RAM: {total_gb:.1f}GB total, {available_gb:.1f}GB available")
        
        if total_gb < 20:
            logger.warning(f"System has only {total_gb:.1f}GB RAM (24GB recommended)")
        
        if available_gb < 10:
            logger.error(f"Insufficient available RAM: {available_gb:.1f}GB (need 10GB+)")
            return False
        
        # Check disk space
        disk = psutil.disk_usage(str(ROOT_DIR))
        free_gb = disk.free / (1024**3)
        
        logger.info(f"Disk space: {free_gb:.1f}GB free")
        
        if free_gb < 5:
            logger.warning(f"Low disk space: {free_gb:.1f}GB (5GB+ recommended)")
        
        return True
    
    def initialize(self) -> bool:
        """
        Initialize all components with validation
        
        CORRECTNESS: Returns False on any failure
        STABILITY: Cleans up on failure
        
        Returns:
            True if initialization successful
        """
        try:
            # Stage 1: Logging
            logger_instance = setup_logger(self.context.logs_dir, self.context.run_id)
            logger.info("=" * 80)
            logger.info("VoiceForge-Nextgen V2 (RAM-Optimized)")
            logger.info(f"Run ID: {self.context.run_id}")
            logger.info("Priority: Correctness → Stability → Performance → Debug")
            logger.info("=" * 80)
            
            # Stage 2: System resources check
            if not self._check_system_resources():
                logger.error("System resource check failed")
                return False
            
            # Stage 3: Database
            logger.info("Initializing database...")
            db_path = self.context.root_dir / "voiceforge.db"
            self.db_manager = DatabaseManager(db_path)
            self.db_manager.initialize()
            
            # Stage 4: Debug infrastructure
            logger.info("Initializing debug infrastructure...")
            self.debugger = SnapshotDebugger(self.context.snapshots_dir)
            self.profiler = Profiler(target_latency_ms=20.0)
            self.health_monitor = HealthMonitor()
            
            # Stage 5: Pre-allocate buffer pool (2GB)
            logger.info("Pre-allocating buffer pool (2GB)...")
            self.buffer_pool = BufferPool(
                slot_size_samples=480000,  # 10 seconds @ 48kHz
                num_slots=100,  # ~2GB total
                channels=1
            )
            
            # CORRECTNESS: Validate buffer pool
            if not self.buffer_pool.validate_integrity():
                logger.error("Buffer pool integrity check failed")
                return False
            
            # Stage 6: Model cache (8GB)
            logger.info("Initializing model cache (8GB)...")
            self.model_cache = ModelCache(
                max_cache_size_gb=8.0,
                device="cuda"
            )
            
            # Stage 7: Feature cache (4GB)
            logger.info("Initializing feature cache (4GB)...")
            self.feature_cache = FeatureCache(max_cache_size_gb=4.0)
            
            # Stage 8: Audio devices
            logger.info("Enumerating audio devices...")
            self.device_manager = DeviceManager()
            self.device_manager.initialize()
            
            # List devices
            logger.info("Available INPUT devices:")
            for dev in self.device_manager.get_input_devices():
                logger.info(f"  {dev}")
            
            logger.info("Available OUTPUT devices:")
            for dev in self.device_manager.get_output_devices():
                logger.info(f"  {dev}")
            
            # Stage 9: Load audio config
            session = self.db_manager.get_session()
            audio_config = AudioConfigRepository.get_or_create(session)
            
            # Find Virtual Audio Cable
            vac_device = self.device_manager.find_virtual_cable()
            if vac_device:
                audio_config.virtual_output_index = vac_device.index
                audio_config.output_device_name = vac_device.name
                AudioConfigRepository.update(
                    session,
                    virtual_output_index=vac_device.index,
                    output_device_name=vac_device.name
                )
                logger.info(f"Virtual Audio Cable: {vac_device}")
            else:
                logger.error("Virtual Audio Cable NOT FOUND")
                logger.error("Please install VB-Audio Virtual Cable")
                session.close()
                return False
            
            # Get default input
            default_input = self.device_manager.get_default_input()
            if default_input:
                audio_config.input_device_index = default_input.index
                audio_config.input_device_name = default_input.name
                AudioConfigRepository.update(
                    session,
                    input_device_index=default_input.index,
                    input_device_name=default_input.name
                )
            else:
                logger.error("No input device found")
                session.close()
                return False
            
            # Update context BEFORE closing session
            self.context.input_device_index = audio_config.input_device_index
            self.context.output_device_index = audio_config.virtual_output_index
            self.context.input_device_name = audio_config.input_device_name
            self.context.output_device_name = audio_config.output_device_name
            self.context.sample_rate = audio_config.sample_rate
            self.context.buffer_size = audio_config.buffer_size
            
            session.close()
            
            logger.info(f"Audio config: IN={audio_config.input_device_name}, OUT={audio_config.output_device_name}")
            
            # Stage 10: Validate device configuration
            logger.info("Validating audio device configuration...")
            input_valid, input_error = self.device_manager.validate_device(
                audio_config.input_device_index,
                audio_config.sample_rate,
                1,  # mono
                is_input=True
            )
            
            if not input_valid:
                logger.error(f"Input device validation failed: {input_error}")
                return False
            
            output_valid, output_error = self.device_manager.validate_device(
                audio_config.virtual_output_index,
                audio_config.sample_rate,
                1,
                is_input=False
            )
            
            if not output_valid:
                logger.error(f"Output device validation failed: {output_error}")
                return False
            
            logger.info("✓ All initialization checks passed")
            return True
            
        except Exception as e:
            logger.critical(f"Initialization failed: {e}", exc_info=True)
            self.cleanup()
            return False
    
    def start_golden_path(self) -> bool:
        """
        Start Golden Path: Mic → VAC (no AI)
        CORRECTNESS: Must always work as baseline
        
        Returns:
            True if started successfully
        """
        logger.info("=" * 80)
        logger.info("Starting GOLDEN PATH mode (no AI processing)")
        logger.info("This is the baseline - must ALWAYS work")
        logger.info("=" * 80)
        
        # Golden Path uses minimal monitoring to avoid jitter
        self.monitoring_mode = "minimal"
        
        try:
            # Create audio stream V2
            self.audio_stream = AudioStreamV2(
                input_device_index=self.context.input_device_index,
                output_device_index=self.context.output_device_index,
                sample_rate=self.context.sample_rate,
                buffer_size=self.context.buffer_size,
                buffer_pool=self.buffer_pool,
                pyaudio_instance=self.device_manager.p
            )
            
            # Set to Golden Path (no processing)
            self.audio_stream.set_processing_callback(None)
            
            # Start stream
            if not self.audio_stream.start():
                logger.error("Failed to start audio stream")
                return False
            
            self.running = True
            logger.info("✓ Golden Path ACTIVE")
            logger.info("Speak into microphone - audio should route to Virtual Audio Cable")
            logger.info("Press Ctrl+C to stop")
            
            # Monitor loop
            self._monitor_loop()
            
            return True
            
        except KeyboardInterrupt:
            logger.info("Interrupted by user")
            return True
        except Exception as e:
            logger.error(f"Golden Path failed: {e}", exc_info=True)
            
            # Capture snapshot
            self.debugger.capture_snapshot(
                run_id=self.context.run_id,
                reason="golden_path_failure",
                stage="AUDIO_START",
                context=self.context.to_dict(),
                error=e
            )
            
            return False
    
    def _monitor_loop(self):
        """
        Monitor system health and log statistics
        
        STABILITY: Detects issues early
        DEBUG: Full observability
        """
        # Monitoring policy based on mode
        monitor_interval = {
            "full": 5.0,
            "minimal": 30.0,  # less frequent to reduce jitter
            "off": None
        }.get(self.monitoring_mode, 30.0)
        
        integrity_interval = {
            "full": 60.0,
            "minimal": None,  # disable periodic integrity in minimal
            "off": None
        }.get(self.monitoring_mode, None)
        
        last_monitor = time.time()
        
        # If monitoring is off, keep a lightweight wait loop to allow stop()
        if monitor_interval is None and integrity_interval is None:
            while self.running:
                time.sleep(0.5)
            return
        
        while self.running:
            try:
                time.sleep(0.1)
                
                current_time = time.time()
                
                # Periodic monitoring
                if monitor_interval is not None and current_time - last_monitor >= monitor_interval:
                    self._log_system_status()
                    last_monitor = current_time
                
                # Periodic integrity checks
                if integrity_interval is not None and current_time - self.last_integrity_check >= integrity_interval:
                    self._integrity_check()
                    self.last_integrity_check = current_time
                
            except KeyboardInterrupt:
                logger.info("Monitor loop interrupted")
                self.running = False
                break
            except Exception as e:
                logger.error(f"Monitor loop error: {e}", exc_info=True)
    
    def _log_system_status(self):
        """Log comprehensive system status"""
        logger.info("-" * 60)
        
        # Audio stream stats
        if self.audio_stream:
            stats = self.audio_stream.get_stats()
            logger.info(f"Audio Stream: state={stats['state']}, "
                       f"in_callbacks={stats['input_callbacks']}, "
                       f"out_writes={stats['output_writes']}, "
                       f"errors={stats['errors']}")
            
            if stats['failsafe_mode']:
                logger.warning("⚠️  FAILSAFE MODE ACTIVE")
        
        # Buffer pool stats
        pool_stats = self.buffer_pool.get_usage_stats()
        logger.info(f"Buffer Pool: {pool_stats['in_use']}/{pool_stats['total_slots']} "
                   f"({pool_stats['usage_percent']:.1f}%), "
                   f"max_usage={pool_stats['max_usage']}")
        
        # Model cache stats
        model_stats = self.model_cache.get_stats()
        logger.info(f"Model Cache: {model_stats['cached_models']} models, "
                   f"{model_stats['memory_used_gb']:.2f}GB/"
                   f"{model_stats['memory_max_gb']:.2f}GB")
        
        # Feature cache stats
        feature_stats = self.feature_cache.get_stats()
        logger.info(f"Feature Cache: {feature_stats['cached_entries']} entries, "
                   f"hit_rate={feature_stats['hit_rate']:.1%}")
        
        # System health
        self.health_monitor.log_health()
        
        logger.info("-" * 60)
    
    def _integrity_check(self):
        """
        Periodic integrity check
        
        CORRECTNESS: Detect memory corruption early
        """
        logger.info("Running integrity check...")
        
        # Check buffer pool
        if not self.buffer_pool.validate_integrity():
            logger.critical("❌ Buffer pool integrity check FAILED")
            
            # Capture snapshot
            self.debugger.capture_snapshot(
                run_id=self.context.run_id,
                reason="buffer_pool_corruption",
                stage="INTEGRITY_CHECK",
                context=self.context.to_dict(),
                additional_data=self.buffer_pool.get_usage_stats()
            )
            
            # STABILITY: Don't crash, but activate failsafe
            if self.audio_stream:
                self.audio_stream._activate_failsafe("buffer_pool_corruption")
        else:
            logger.info("✓ Buffer pool integrity OK")
    
    def stop(self):
        """Stop application gracefully"""
        logger.info("Stopping application...")
        self.running = False
        
        if self.audio_stream:
            self.audio_stream.stop()
        
        # Log final statistics
        logger.info("=" * 80)
        logger.info("FINAL STATISTICS")
        logger.info("=" * 80)
        self._log_system_status()
        
        if self.profiler:
            self.profiler.log_report()
    
    def cleanup(self):
        """Cleanup all resources"""
        logger.info("Cleaning up resources...")
        
        if self.audio_stream:
            self.audio_stream.dispose()
        
        if self.device_manager:
            self.device_manager.dispose()
        
        if self.db_manager:
            self.db_manager.dispose()
        
        logger.info("Cleanup complete")

def main():
    """Main entry point"""
    app = VoiceForgeAppV2()
    
    # Signal handlers
    def signal_handler(sig, frame):
        logger.info("Signal received, shutting down...")
        app.stop()
        app.cleanup()
        sys.exit(0)
    
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    try:
        # Initialize
        if not app.initialize():
            logger.critical("Initialization failed - exiting")
            sys.exit(1)
        
        # Start Golden Path
        if not app.start_golden_path():
            logger.critical("Golden Path failed - this should never happen")
            sys.exit(1)
        
    except Exception as e:
        logger.critical(f"Fatal error: {e}", exc_info=True)
        sys.exit(1)
    finally:
        app.cleanup()

if __name__ == "__main__":
    main()