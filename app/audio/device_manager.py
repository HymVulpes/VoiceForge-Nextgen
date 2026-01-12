"""
Audio Device Manager
Runtime device enumeration and validation
"""
import pyaudio
from typing import List, Dict, Optional, Tuple
import logging

logger = logging.getLogger(__name__)

class AudioDevice:
    """Audio device information"""
    def __init__(self, index: int, info: dict):
        self.index = index
        self.name = info['name']
        self.max_input_channels = info['maxInputChannels']
        self.max_output_channels = info['maxOutputChannels']
        self.default_sample_rate = info['defaultSampleRate']
        self.is_input = self.max_input_channels > 0
        self.is_output = self.max_output_channels > 0
        
    def __repr__(self):
        device_type = []
        if self.is_input:
            device_type.append("INPUT")
        if self.is_output:
            device_type.append("OUTPUT")
        return f"[{self.index}] {self.name} ({'/'.join(device_type)})"

class DeviceManager:
    """
    Manages audio device enumeration and validation
    NEVER assumes devices exist - runtime validation required
    """
    
    def __init__(self):
        self.p = None
        self.devices: List[AudioDevice] = []
        
    def initialize(self):
        """Initialize PyAudio and enumerate devices"""
        try:
            self.p = pyaudio.PyAudio()
            self._enumerate_devices()
            logger.info(f"Found {len(self.devices)} audio devices")
        except Exception as e:
            logger.error(f"Failed to initialize PyAudio: {e}")
            raise
    
    def _enumerate_devices(self):
        """Enumerate all audio devices"""
        self.devices.clear()
        device_count = self.p.get_device_count()
        
        for i in range(device_count):
            try:
                info = self.p.get_device_info_by_index(i)
                device = AudioDevice(i, info)
                self.devices.append(device)
                logger.debug(f"Found device: {device}")
            except Exception as e:
                logger.warning(f"Failed to get info for device {i}: {e}")
    
    def get_input_devices(self) -> List[AudioDevice]:
        """Get all input devices"""
        return [d for d in self.devices if d.is_input]
    
    def get_output_devices(self) -> List[AudioDevice]:
        """Get all output devices"""
        return [d for d in self.devices if d.is_output]
    
    def find_device_by_name(self, name: str, is_input: bool = True) -> Optional[AudioDevice]:
        """
        Find device by partial name match
        
        Args:
            name: Device name or substring
            is_input: True for input, False for output
            
        Returns:
            AudioDevice or None
        """
        devices = self.get_input_devices() if is_input else self.get_output_devices()
        name_lower = name.lower()
        
        for device in devices:
            if name_lower in device.name.lower():
                return device
        return None
    
    def find_virtual_cable(self) -> Optional[AudioDevice]:
        """
        Find Virtual Audio Cable output device
        Searches for common VAC device names
        """
        vac_keywords = [
            "CABLE Input",  # VB-Audio Virtual Cable
            "VoiceMeeter Input",  # VoiceMeeter
            "Virtual Audio Cable",
            "VAC",
        ]
        
        output_devices = self.get_output_devices()
        for keyword in vac_keywords:
            for device in output_devices:
                if keyword.lower() in device.name.lower():
                    logger.info(f"Found Virtual Audio Cable: {device}")
                    return device
        
        logger.warning("No Virtual Audio Cable found - check installation")
        return None
    
    def get_default_input(self) -> Optional[AudioDevice]:
        """Get system default input device"""
        try:
            default_input_info = self.p.get_default_input_device_info()
            index = default_input_info['index']
            return next((d for d in self.devices if d.index == index), None)
        except Exception as e:
            logger.error(f"Failed to get default input: {e}")
            return None
    
    def validate_device(
        self,
        device_index: int,
        sample_rate: int,
        channels: int,
        is_input: bool
    ) -> Tuple[bool, Optional[str]]:
        """
        Validate device supports configuration
        
        Returns:
            (is_valid, error_message)
        """
        try:
            if not self.p:
                return False, "PyAudio not initialized"
            
            device_info = self.p.get_device_info_by_index(device_index)
            
            # Check channel support
            max_channels = device_info['maxInputChannels'] if is_input else device_info['maxOutputChannels']
            if max_channels < channels:
                return False, f"Device supports max {max_channels} channels, requested {channels}"
            
            # Try to open stream (will fail if config not supported)
            self.p.is_format_supported(
                sample_rate,
                input_device=device_index if is_input else None,
                output_device=None if is_input else device_index,
                input_channels=channels if is_input else 0,
                output_channels=0 if is_input else channels,
                input_format=pyaudio.paFloat32,
                output_format=pyaudio.paFloat32
            )
            
            return True, None
            
        except ValueError as e:
            return False, str(e)
        except Exception as e:
            return False, f"Validation failed: {e}"
    
    def dispose(self):
        """Cleanup PyAudio"""
        if self.p:
            self.p.terminate()
            logger.info("PyAudio terminated")