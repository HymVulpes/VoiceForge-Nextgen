"""
VoiceForge-Nextgen - Memory Profiler
File: app/core/memory_profiler.py

Purpose:
    Profile memory usage during inference
    Track peak VRAM, detect fragmentation, analyze allocation patterns

Dependencies:
    - torch (CUDA memory stats)
    - psutil (system RAM)
    - time (timestamps)

Data Flow:
    Start profiling → Track allocations → End profiling → Generate report

Usage:
    profiler = MemoryProfiler()
    with profiler.profile('inference'):
        output = model.infer(audio)
    
    report = profiler.get_report('inference')
"""

import torch
import psutil
import time
import logging
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from collections import defaultdict
import json

logger = logging.getLogger("MemoryProfiler")


@dataclass
class MemorySnapshot:
    """Single memory snapshot"""
    timestamp: float
    cpu_ram_mb: float
    gpu_vram_mb: float
    gpu_allocated_mb: float
    gpu_reserved_mb: float
    gpu_cached_mb: float
    
    def to_dict(self) -> dict:
        return {
            'timestamp': self.timestamp,
            'cpu_ram_mb': self.cpu_ram_mb,
            'gpu_vram_mb': self.gpu_vram_mb,
            'gpu_allocated_mb': self.gpu_allocated_mb,
            'gpu_reserved_mb': self.gpu_reserved_mb,
            'gpu_cached_mb': self.gpu_cached_mb
        }


@dataclass
class ProfileSession:
    """Profiling session data"""
    name: str
    start_time: float
    end_time: Optional[float] = None
    snapshots: List[MemorySnapshot] = field(default_factory=list)
    peak_cpu_mb: float = 0.0
    peak_gpu_mb: float = 0.0
    
    def duration(self) -> float:
        """Get session duration"""
        if self.end_time is None:
            return time.time() - self.start_time
        return self.end_time - self.start_time
    
    def to_dict(self) -> dict:
        return {
            'name': self.name,
            'start_time': self.start_time,
            'end_time': self.end_time,
            'duration': self.duration(),
            'peak_cpu_mb': self.peak_cpu_mb,
            'peak_gpu_mb': self.peak_gpu_mb,
            'num_snapshots': len(self.snapshots),
            'snapshots': [s.to_dict() for s in self.snapshots]
        }


class MemoryProfiler:
    """
    Profile memory usage during model operations
    
    Features:
        - Track RAM and VRAM usage over time
        - Detect memory leaks
        - Identify fragmentation
        - Generate performance reports
    """
    
    def __init__(self, snapshot_interval: float = 0.1):
        """
        Initialize profiler
        
        Args:
            snapshot_interval: Interval between snapshots (seconds)
        """
        self.snapshot_interval = snapshot_interval
        
        # Active sessions
        self.sessions: Dict[str, ProfileSession] = {}
        
        # Completed sessions
        self.completed: List[ProfileSession] = []
        
        # Check CUDA
        self.cuda_available = torch.cuda.is_available()
        if not self.cuda_available:
            logger.warning("CUDA not available, GPU profiling disabled")
        
        # Get process for RAM tracking
        self.process = psutil.Process()
        
        logger.info(
            f"MemoryProfiler initialized: "
            f"cuda={self.cuda_available}, "
            f"interval={snapshot_interval}s"
        )
    
    def start_profile(self, name: str) -> bool:
        """
        Start profiling session
        
        Args:
            name: Session name
            
        Returns:
            True if started successfully
        """
        if name in self.sessions:
            logger.warning(f"Session '{name}' already active")
            return False
        
        session = ProfileSession(
            name=name,
            start_time=time.time()
        )
        
        # Take initial snapshot
        snapshot = self._take_snapshot()
        session.snapshots.append(snapshot)
        
        self.sessions[name] = session
        logger.debug(f"Started profiling: {name}")
        
        return True
    
    def record_snapshot(self, name: str) -> bool:
        """
        Record a snapshot for active session
        
        Args:
            name: Session name
            
        Returns:
            True if recorded
        """
        if name not in self.sessions:
            logger.warning(f"Session '{name}' not active")
            return False
        
        session = self.sessions[name]
        
        # Check if enough time has passed
        if session.snapshots:
            last_time = session.snapshots[-1].timestamp
            if time.time() - last_time < self.snapshot_interval:
                return False
        
        # Take snapshot
        snapshot = self._take_snapshot()
        session.snapshots.append(snapshot)
        
        # Update peaks
        session.peak_cpu_mb = max(session.peak_cpu_mb, snapshot.cpu_ram_mb)
        session.peak_gpu_mb = max(session.peak_gpu_mb, snapshot.gpu_vram_mb)
        
        return True
    
    def end_profile(self, name: str) -> Optional[ProfileSession]:
        """
        End profiling session
        
        Args:
            name: Session name
            
        Returns:
            Completed session or None
        """
        if name not in self.sessions:
            logger.warning(f"Session '{name}' not found")
            return None
        
        session = self.sessions.pop(name)
        session.end_time = time.time()
        
        # Take final snapshot
        snapshot = self._take_snapshot()
        session.snapshots.append(snapshot)
        
        # Update peaks
        session.peak_cpu_mb = max(session.peak_cpu_mb, snapshot.cpu_ram_mb)
        session.peak_gpu_mb = max(session.peak_gpu_mb, snapshot.gpu_vram_mb)
        
        self.completed.append(session)
        
        logger.info(
            f"Completed profiling: {name} "
            f"(duration={session.duration():.2f}s, "
            f"peak_cpu={session.peak_cpu_mb:.1f}MB, "
            f"peak_gpu={session.peak_gpu_mb:.1f}MB)"
        )
        
        return session
    
    def _take_snapshot(self) -> MemorySnapshot:
        """Take a memory snapshot"""
        # CPU RAM
        mem_info = self.process.memory_info()
        cpu_ram_mb = mem_info.rss / 1024 / 1024
        
        # GPU VRAM
        gpu_vram_mb = 0.0
        gpu_allocated_mb = 0.0
        gpu_reserved_mb = 0.0
        gpu_cached_mb = 0.0
        
        if self.cuda_available:
            try:
                gpu_allocated_mb = torch.cuda.memory_allocated(0) / 1024 / 1024
                gpu_reserved_mb = torch.cuda.memory_reserved(0) / 1024 / 1024
                
                # Get memory stats
                stats = torch.cuda.memory_stats(0)
                gpu_cached_mb = stats.get('inactive_split_bytes.all.current', 0) / 1024 / 1024
                
                gpu_vram_mb = gpu_allocated_mb
                
            except Exception as e:
                logger.debug(f"Failed to get GPU stats: {e}")
        
        return MemorySnapshot(
            timestamp=time.time(),
            cpu_ram_mb=cpu_ram_mb,
            gpu_vram_mb=gpu_vram_mb,
            gpu_allocated_mb=gpu_allocated_mb,
            gpu_reserved_mb=gpu_reserved_mb,
            gpu_cached_mb=gpu_cached_mb
        )
    
    def get_report(self, name: str) -> Optional[Dict[str, Any]]:
        """
        Get profiling report for a session
        
        Args:
            name: Session name
            
        Returns:
            Report dictionary or None
        """
        # Find session in completed
        session = None
        for s in self.completed:
            if s.name == name:
                session = s
                break
        
        if session is None:
            logger.warning(f"No completed session: {name}")
            return None
        
        # Generate report
        report = {
            'session': session.to_dict(),
            'analysis': self._analyze_session(session)
        }
        
        return report
    
    def _analyze_session(self, session: ProfileSession) -> dict:
        """
        Analyze session for insights
        
        Args:
            session: Profiling session
            
        Returns:
            Analysis dictionary
        """
        if len(session.snapshots) < 2:
            return {'error': 'Insufficient snapshots'}
        
        # Calculate memory growth
        first_snap = session.snapshots[0]
        last_snap = session.snapshots[-1]
        
        cpu_growth_mb = last_snap.cpu_ram_mb - first_snap.cpu_ram_mb
        gpu_growth_mb = last_snap.gpu_vram_mb - first_snap.gpu_vram_mb
        
        # Detect potential leak (growth > 10% of peak)
        leak_threshold_cpu = session.peak_cpu_mb * 0.1
        leak_threshold_gpu = session.peak_gpu_mb * 0.1
        
        potential_leak_cpu = cpu_growth_mb > leak_threshold_cpu
        potential_leak_gpu = gpu_growth_mb > leak_threshold_gpu
        
        # Calculate average usage
        avg_cpu = sum(s.cpu_ram_mb for s in session.snapshots) / len(session.snapshots)
        avg_gpu = sum(s.gpu_vram_mb for s in session.snapshots) / len(session.snapshots)
        
        return {
            'duration_s': session.duration(),
            'num_snapshots': len(session.snapshots),
            'cpu': {
                'peak_mb': session.peak_cpu_mb,
                'avg_mb': avg_cpu,
                'growth_mb': cpu_growth_mb,
                'potential_leak': potential_leak_cpu
            },
            'gpu': {
                'peak_mb': session.peak_gpu_mb,
                'avg_mb': avg_gpu,
                'growth_mb': gpu_growth_mb,
                'potential_leak': potential_leak_gpu
            },
            'warnings': []
        }
    
    def profile(self, name: str):
        """
        Context manager for profiling
        
        Usage:
            with profiler.profile('inference'):
                model.infer(audio)
        """
        return ProfileContext(self, name)
    
    def get_all_reports(self) -> List[dict]:
        """Get reports for all completed sessions"""
        return [self.get_report(s.name) for s in self.completed]
    
    def export_to_file(self, filepath: str):
        """Export all reports to JSON file"""
        reports = self.get_all_reports()
        
        with open(filepath, 'w') as f:
            json.dump(reports, f, indent=2)
        
        logger.info(f"Profiling reports exported to {filepath}")
    
    def clear_completed(self):
        """Clear completed sessions"""
        self.completed.clear()
        logger.debug("Cleared completed profiling sessions")


class ProfileContext:
    """Context manager for profiling"""
    
    def __init__(self, profiler: MemoryProfiler, name: str):
        self.profiler = profiler
        self.name = name
    
    def __enter__(self):
        self.profiler.start_profile(self.name)
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.profiler.end_profile(self.name)
        return False


# Singleton instance
_profiler_instance: Optional[MemoryProfiler] = None


def get_profiler(snapshot_interval: float = 0.1) -> MemoryProfiler:
    """Get singleton memory profiler"""
    global _profiler_instance
    
    if _profiler_instance is None:
        _profiler_instance = MemoryProfiler(snapshot_interval=snapshot_interval)
    
    return _profiler_instance