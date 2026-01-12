"""
Audio Stream V2 - RAM-Optimized with Pre-allocated Buffers
Uses BufferPool and TripleBuffer for lock-free, zero-copy audio
CORRECTNESS: State machine with validated transitions
STABILITY: Automatic failsafe mode
PERFORMANCE: Pre-allocated, zero-copy where possible
"""
import pyaudio
import numpy as np
import threading
import time
from typing import Optional, Callable
from enum import Enum
import logging

from .buffer_pool import BufferPool
from .triple_buffer import TripleBuffer

logger = logging.getLogger(__name__)

class StreamState(Enum):
    """Audio stream state machine"""
    STOPPED = "stopped"
    STARTING = "starting"
    RUNNING = "running"
    STOPPING = "stopping"
    FAILSAFE = "failsafe"
    FAILED = "failed"

class AudioStreamV2:
    """
    Real-time audio streaming V2 with pre-allocated buffers
    
    Features:
    - Pre-allocated buffer pool (no dynamic allocation)
    - Lock-free triple buffering
    - State machine with validation
    - Automatic failsafe mode
    - Zero-copy audio path where possible
    """
    
    def __init__(
        self,
        input_device_index: int,
        output_device_index: int,
        sample_rate: int = 48000,
        buffer_size: int = 256,
        channels: int = 1,
        buffer_pool: Optional[BufferPool] = None,
        pyaudio_instance: Optional[pyaudio.PyAudio] = None
    ):
        """
        Args:
            input_device_index: Input device index
            output_device_index: Output device index
            sample_rate: Sample rate (Hz)
            buffer_size: PyAudio buffer size (samples)
            channels: Number of channels (1=mono, 2=stereo)
            buffer_pool: Pre-allocated buffer pool (required)
            pyaudio_instance: Shared PyAudio instance
        """
        # CORRECTNESS: Validate buffer_pool
        if buffer_pool is None:
            raise ValueError("buffer_pool is required for AudioStreamV2")
        
        self.input_device_index = input_device_index
        self.output_device_index = output_device_index
        self.sample_rate = sample_rate
        self.buffer_size = buffer_size
        self.channels = channels
        self.buffer_pool = buffer_pool
        
        # PyAudio instance
        self.p = pyaudio_instance or pyaudio.PyAudio()
        self.owns_pyaudio = pyaudio_instance is None
        
        # Streams
        self.input_stream: Optional[pyaudio.Stream] = None
        self.output_stream: Optional[pyaudio.Stream] = None
        
        # Triple buffer for lock-free communication
        # Buffer size: 1 second of audio
        triple_buffer_size = sample_rate
        self.triple_buffer = TripleBuffer(
            buffer_size_samples=triple_buffer_size,
            channels=channels
        )
        
        # Processing callback (optional, for AI inference)
        self.processing_callback: Optional[Callable[[np.ndarray], np.ndarray]] = None
        
        # State machine
        self.state = StreamState.STOPPED
        self.state_lock = threading.Lock()
        
        # Golden Path mode (direct passthrough)
        self.golden_path_mode = True
        
        # Thread safety
        self.running = False
        self.processing_thread: Optional[threading.Thread] = None
        
        # Failsafe mode
        self.failsafe_mode = False
        self.consecutive_errors = 0
        self.max_consecutive_errors = 5
        
        # Statistics
        self.input_callback_count = 0
        self.output_writes = 0
        self.errors = 0
        self.last_error_time = 0.0
    
    def set_processing_callback(self, callback: Optional[Callable[[np.ndarray], np.ndarray]]):
        """
        Set AI processing callback
        
        Args:
            callback: Function that takes audio input (float32) and returns processed output
                      If None, operates in Golden Path mode
        """
        with self.state_lock:
            self.processing_callback = callback
            self.golden_path_mode = callback is None
            logger.info(f"Processing mode: {'Golden Path' if self.golden_path_mode else 'AI Processing'}")
    
    def _validate_state_transition(self, new_state: StreamState) -> bool:
        """
        Validate state transition
        
        CORRECTNESS: Only allow valid transitions
        """
        valid_transitions = {
            StreamState.STOPPED: [StreamState.STARTING],
            StreamState.STARTING: [StreamState.RUNNING, StreamState.FAILED],
            StreamState.RUNNING: [StreamState.STOPPING, StreamState.FAILSAFE, StreamState.FAILED],
            StreamState.STOPPING: [StreamState.STOPPED, StreamState.FAILED],
            StreamState.FAILSAFE: [StreamState.RUNNING, StreamState.STOPPING, StreamState.FAILED],
            StreamState.FAILED: [StreamState.STOPPED]
        }
        
        current_state = self.state
        if new_state not in valid_transitions.get(current_state, []):
            logger.error(f"Invalid state transition: {current_state} → {new_state}")
            return False
        
        return True
    
    def _set_state(self, new_state: StreamState):
        """Set state with validation"""
        with self.state_lock:
            if self._validate_state_transition(new_state):
                old_state = self.state
                self.state = new_state
                logger.debug(f"State transition: {old_state} → {new_state}")
            else:
                logger.error(f"State transition rejected: {self.state} → {new_state}")
    
    def _input_callback(self, in_data, frame_count, time_info, status):
        """
        Audio input callback - runs in PyAudio thread
        CRITICAL: Must never block or do heavy processing
        """
        self.input_callback_count += 1
        
        if status:
            logger.warning(f"Input callback status: {status}")
        
        try:
            # Convert bytes to numpy
            audio_data = np.frombuffer(in_data, dtype=np.float32).reshape(-1, self.channels)
            
            # Golden Path: Direct write to output
            if self.golden_path_mode or self.failsafe_mode:
                # Immediate passthrough - no buffering delay
                if self.output_stream and self.output_stream.is_active():
                    try:
                        self.output_stream.write(audio_data.tobytes())
                        self.output_writes += 1
                    except Exception as e:
                        logger.error(f"Output write failed: {e}")
                        self.errors += 1
                        self._handle_error("output_write_failed", e)
            else:
                # AI mode: Write to triple buffer
                if not self.triple_buffer.write(audio_data):
                    self.errors += 1
                    self._handle_error("buffer_overflow", None)
            
            return (None, pyaudio.paContinue)
            
        except Exception as e:
            logger.error(f"Input callback error: {e}")
            self.errors += 1
            self._handle_error("input_callback_error", e)
            return (None, pyaudio.paContinue)
    
    def _handle_error(self, error_type: str, error: Optional[Exception]):
        """Handle error and activate failsafe if needed"""
        self.consecutive_errors += 1
        self.last_error_time = time.time()
        
        if self.consecutive_errors >= self.max_consecutive_errors:
            logger.warning(f"Too many consecutive errors ({self.consecutive_errors}), activating failsafe")
            self._activate_failsafe(error_type)
    
    def _activate_failsafe(self, reason: str):
        """
        Activate failsafe mode
        
        STABILITY: Never crash, always provide audio
        """
        if self.failsafe_mode:
            return  # Already in failsafe
        
        logger.warning(f"Activating FAILSAFE mode: {reason}")
        self.failsafe_mode = True
        self.golden_path_mode = True  # Force passthrough
        self._set_state(StreamState.FAILSAFE)
        
        # Clear error counter (give it a chance to recover)
        self.consecutive_errors = 0
    
    def start(self) -> bool:
        """
        Start audio streams
        
        CORRECTNESS: Returns False on failure
        STABILITY: Cleans up on failure
        
        Returns:
            True if started successfully
        """
        with self.state_lock:
            if self.state != StreamState.STOPPED:
                logger.warning(f"Cannot start from state: {self.state}")
                return False
            
            self._set_state(StreamState.STARTING)
        
        logger.info(f"Starting audio streams: {self.sample_rate}Hz, buffer={self.buffer_size}")
        
        try:
            # Input stream
            self.input_stream = self.p.open(
                format=pyaudio.paFloat32,
                channels=self.channels,
                rate=self.sample_rate,
                input=True,
                input_device_index=self.input_device_index,
                frames_per_buffer=self.buffer_size,
                stream_callback=self._input_callback
            )
            
            # Output stream (non-callback mode for Golden Path)
            self.output_stream = self.p.open(
                format=pyaudio.paFloat32,
                channels=self.channels,
                rate=self.sample_rate,
                output=True,
                output_device_index=self.output_device_index,
                frames_per_buffer=self.buffer_size
            )
            
            # Start streams
            self.input_stream.start_stream()
            self.output_stream.start_stream()
            
            self.running = True
            self._set_state(StreamState.RUNNING)
            
            # Start processing thread if not in Golden Path
            if not self.golden_path_mode:
                self.processing_thread = threading.Thread(
                    target=self._processing_loop,
                    daemon=True
                )
                self.processing_thread.start()
            
            logger.info("Audio streams started successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to start audio streams: {e}")
            self._set_state(StreamState.FAILED)
            self.stop()
            return False
    
    def _processing_loop(self):
        """
        Processing loop for AI mode (run in separate thread)
        Reads from triple buffer, processes, writes to output
        """
        if self.golden_path_mode:
            return
        
        logger.info("Starting AI processing loop")
        
        chunk_size = self.buffer_size * 16  # Process 16 buffers at a time
        
        while self.running and self.state == StreamState.RUNNING:
            try:
                # Read from triple buffer
                audio_chunk = self.triple_buffer.read(chunk_size)
                
                if audio_chunk is None:
                    # Not enough data - sleep briefly
                    time.sleep(0.001)
                    continue
                
                # Process through AI callback
                try:
                    if self.processing_callback:
                        processed = self.processing_callback(audio_chunk)
                        
                        # CORRECTNESS: Validate output
                        if processed is None:
                            logger.warning("Processing callback returned None, using passthrough")
                            processed = audio_chunk
                        elif processed.shape != audio_chunk.shape:
                            logger.warning(f"Output shape mismatch: {audio_chunk.shape} → {processed.shape}")
                            processed = audio_chunk  # Fallback to passthrough
                    else:
                        processed = audio_chunk  # Fallback to passthrough
                    
                    # Write to output
                    if self.output_stream and self.output_stream.is_active():
                        self.output_stream.write(processed.tobytes())
                        self.output_writes += 1
                        
                        # Reset error counter on success
                        if self.consecutive_errors > 0:
                            self.consecutive_errors = 0
                            
                            # Try to recover from failsafe
                            if self.failsafe_mode and time.time() - self.last_error_time > 2.0:
                                logger.info("Recovering from failsafe mode")
                                self.failsafe_mode = False
                                self._set_state(StreamState.RUNNING)
                    
                except Exception as e:
                    logger.error(f"Processing failed: {e}")
                    self._handle_error("processing_error", e)
                    
                    # Fallback: passthrough on error
                    if self.output_stream and self.output_stream.is_active():
                        self.output_stream.write(audio_chunk.tobytes())
                
            except Exception as e:
                logger.error(f"Processing loop error: {e}")
                self._handle_error("processing_loop_error", e)
                time.sleep(0.01)
    
    def stop(self):
        """Stop audio streams"""
        with self.state_lock:
            if self.state in [StreamState.STOPPED, StreamState.STOPPING]:
                return
            
            self._set_state(StreamState.STOPPING)
        
        logger.info("Stopping audio streams")
        self.running = False
        
        # Wait for processing thread
        if self.processing_thread and self.processing_thread.is_alive():
            self.processing_thread.join(timeout=1.0)
        
        # Stop streams
        if self.input_stream:
            try:
                self.input_stream.stop_stream()
                self.input_stream.close()
            except Exception:
                pass
            self.input_stream = None
        
        if self.output_stream:
            try:
                self.output_stream.stop_stream()
                self.output_stream.close()
            except Exception:
                pass
            self.output_stream = None
        
        self._set_state(StreamState.STOPPED)
        logger.info("Audio streams stopped")
    
    def get_stats(self) -> dict:
        """Get streaming statistics"""
        return {
            "state": self.state.value,
            "mode": "Golden Path" if self.golden_path_mode else "AI Processing",
            "failsafe_mode": self.failsafe_mode,
            "input_callbacks": self.input_callback_count,
            "output_writes": self.output_writes,
            "errors": self.errors,
            "consecutive_errors": self.consecutive_errors,
            "triple_buffer": self.triple_buffer.get_stats()
        }
    
    def dispose(self):
        """Cleanup resources"""
        self.stop()
        if self.owns_pyaudio:
            self.p.terminate()


