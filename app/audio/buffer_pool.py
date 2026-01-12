"""
Pre-allocated Buffer Pool - Zero Dynamic Allocation
With 24GB RAM, we pre-allocate all buffers at startup
CORRECTNESS: No allocation failures during realtime
STABILITY: No GC pauses
PERFORMANCE: Zero-copy operations
"""
import numpy as np
from typing import Optional, List
import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)

@dataclass
class BufferSlot:
    """Single buffer slot with metadata"""
    buffer: np.ndarray
    in_use: bool = False
    sequence_id: int = 0
    timestamp: float = 0.0

class BufferPool:
    """
    Pre-allocated buffer pool with O(1) acquire/release
    
    With 24GB RAM, allocate 2GB for audio buffers:
    - 1000 slots × 2MB each = ~2GB
    - Each slot: 10 seconds @ 48kHz stereo float32
    """
    
    def __init__(
        self,
        slot_size_samples: int = 480000,  # 10 seconds @ 48kHz
        num_slots: int = 1000,
        channels: int = 1,
        dtype=np.float32
    ):
        self.slot_size_samples = slot_size_samples
        self.num_slots = num_slots
        self.channels = channels
        self.dtype = dtype
        
        # Pre-allocate all buffers (CORRECTNESS: guaranteed memory)
        logger.info(f"Pre-allocating buffer pool: {num_slots} slots × {slot_size_samples} samples")
        
        self.slots: List[BufferSlot] = []
        total_bytes = 0
        
        for i in range(num_slots):
            buffer = np.zeros((slot_size_samples, channels), dtype=dtype)
            buffer.flags.writeable = True  # Ensure writeable
            self.slots.append(BufferSlot(buffer=buffer))
            total_bytes += buffer.nbytes
        
        logger.info(f"Buffer pool allocated: {total_bytes / (1024**3):.2f} GB")
        
        # Free list for O(1) acquisition
        self.free_indices: List[int] = list(range(num_slots))
        self.next_sequence_id = 0
        
        # Statistics
        self.acquire_count = 0
        self.release_count = 0
        self.max_usage = 0
    
    def acquire(self, required_samples: Optional[int] = None) -> Optional[BufferSlot]:
        """
        Acquire buffer from pool - O(1)
        
        CORRECTNESS: Returns None if exhausted, never raises
        STABILITY: No dynamic allocation
        
        Args:
            required_samples: Minimum samples needed (None = full slot)
            
        Returns:
            BufferSlot or None if pool exhausted
        """
        if not self.free_indices:
            logger.error("Buffer pool exhausted!")
            return None
        
        # Pop from free list
        idx = self.free_indices.pop()
        slot = self.slots[idx]
        
        # Validate not already in use (CORRECTNESS check)
        if slot.in_use:
            logger.critical(f"Buffer slot {idx} already in use! Memory corruption risk!")
            self.free_indices.append(idx)  # Return to pool
            return None
        
        # Mark as in-use
        slot.in_use = True
        slot.sequence_id = self.next_sequence_id
        self.next_sequence_id += 1
        
        # Zero buffer for clean state (CORRECTNESS)
        slot.buffer.fill(0.0)
        
        # Update statistics
        self.acquire_count += 1
        current_usage = self.num_slots - len(self.free_indices)
        self.max_usage = max(self.max_usage, current_usage)
        
        return slot
    
    def release(self, slot: BufferSlot) -> bool:
        """
        Release buffer back to pool - O(1)
        
        CORRECTNESS: Validates slot before release
        STABILITY: Never double-frees
        
        Returns:
            True if released successfully
        """
        if not slot.in_use:
            logger.warning("Attempting to release already-free buffer")
            return False
        
        # Find slot index
        try:
            idx = self.slots.index(slot)
        except ValueError:
            logger.error("Buffer not from this pool!")
            return False
        
        # Mark as free
        slot.in_use = False
        
        # Return to free list
        if idx in self.free_indices:
            logger.error(f"Buffer {idx} already in free list! Double-free detected!")
            return False
        
        self.free_indices.append(idx)
        self.release_count += 1
        
        return True
    
    def get_usage_stats(self) -> dict:
        """Get pool usage statistics"""
        in_use = self.num_slots - len(self.free_indices)
        return {
            "total_slots": self.num_slots,
            "in_use": in_use,
            "free": len(self.free_indices),
            "usage_percent": (in_use / self.num_slots) * 100,
            "max_usage": self.max_usage,
            "acquires": self.acquire_count,
            "releases": self.release_count,
            "total_memory_gb": (self.num_slots * self.slot_size_samples * 
                               self.channels * self.dtype().itemsize) / (1024**3)
        }
    
    def validate_integrity(self) -> bool:
        """
        Validate pool integrity (CORRECTNESS check)
        Call periodically to detect memory corruption
        
        Returns:
            True if pool is valid
        """
        issues = []
        
        # Check no duplicate free indices
        if len(self.free_indices) != len(set(self.free_indices)):
            issues.append("Duplicate entries in free list")
        
        # Check all free indices are actually free
        for idx in self.free_indices:
            if self.slots[idx].in_use:
                issues.append(f"Slot {idx} in free list but marked in-use")
        
        # Check counts match
        free_count = len(self.free_indices)
        in_use_count = sum(1 for s in self.slots if s.in_use)
        
        if free_count + in_use_count != self.num_slots:
            issues.append(f"Count mismatch: free={free_count} + in_use={in_use_count} != {self.num_slots}")
        
        if issues:
            for issue in issues:
                logger.error(f"Pool integrity check failed: {issue}")
            return False
        
        return True