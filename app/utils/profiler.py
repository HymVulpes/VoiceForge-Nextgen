"""
Performance Profiler
Per-stage latency tracking and bottleneck detection
"""
import time
from typing import Dict, List, Optional
from dataclasses import dataclass, field
from datetime import datetime
import logging

logger = logging.getLogger(__name__)

@dataclass
class StageProfile:
    """Performance profile for a stage"""
    name: str
    latencies: List[float] = field(default_factory=list)
    max_latency: float = 0.0
    min_latency: float = float('inf')
    avg_latency: float = 0.0
    call_count: int = 0
    
    def update(self, latency_ms: float):
        """Update statistics with new measurement"""
        self.latencies.append(latency_ms)
        self.call_count += 1
        self.max_latency = max(self.max_latency, latency_ms)
        self.min_latency = min(self.min_latency, latency_ms)
        self.avg_latency = sum(self.latencies) / len(self.latencies)
        
        # Keep only last 1000 measurements
        if len(self.latencies) > 1000:
            self.latencies = self.latencies[-1000:]

class Profiler:
    """
    Performance profiler for realtime pipeline
    Tracks per-stage latencies and detects bottlenecks
    """
    
    def __init__(self, target_latency_ms: float = 20.0):
        self.target_latency_ms = target_latency_ms
        self.stages: Dict[str, StageProfile] = {}
        self._active_timers: Dict[str, float] = {}
    
    def start_stage(self, stage_name: str):
        """Start timing a stage"""
        self._active_timers[stage_name] = time.perf_counter()
    
    def end_stage(self, stage_name: str) -> Optional[float]:
        """
        End timing a stage
        
        Returns:
            Latency in milliseconds
        """
        if stage_name not in self._active_timers:
            logger.warning(f"Stage {stage_name} was not started")
            return None
        
        start_time = self._active_timers.pop(stage_name)
        latency_ms = (time.perf_counter() - start_time) * 1000
        
        # Create stage profile if doesn't exist
        if stage_name not in self.stages:
            self.stages[stage_name] = StageProfile(name=stage_name)
        
        # Update statistics
        self.stages[stage_name].update(latency_ms)
        
        # Warn if exceeds target
        if latency_ms > self.target_latency_ms:
            logger.warning(f"Stage {stage_name} exceeded target: {latency_ms:.2f}ms > {self.target_latency_ms}ms")
        
        return latency_ms
    
    def get_bottlenecks(self) -> List[str]:
        """
        Identify bottleneck stages
        
        Returns:
            List of stage names that exceed target latency
        """
        bottlenecks = []
        for name, profile in self.stages.items():
            if profile.avg_latency > self.target_latency_ms:
                bottlenecks.append(name)
        return bottlenecks
    
    def get_report(self) -> Dict:
        """
        Generate performance report
        
        Returns:
            Dictionary with profiling data
        """
        report = {
            "target_latency_ms": self.target_latency_ms,
            "stages": {}
        }
        
        total_latency = 0.0
        for name, profile in self.stages.items():
            report["stages"][name] = {
                "avg_ms": round(profile.avg_latency, 2),
                "min_ms": round(profile.min_latency, 2),
                "max_ms": round(profile.max_latency, 2),
                "call_count": profile.call_count
            }
            total_latency += profile.avg_latency
        
        report["total_avg_latency_ms"] = round(total_latency, 2)
        report["bottlenecks"] = self.get_bottlenecks()
        
        return report
    
    def log_report(self):
        """Log performance report"""
        report = self.get_report()
        
        logger.info("=" * 60)
        logger.info("PERFORMANCE REPORT")
        logger.info(f"Target Latency: {report['target_latency_ms']}ms")
        logger.info(f"Total Avg Latency: {report['total_avg_latency_ms']}ms")
        logger.info("-" * 60)
        
        for stage_name, metrics in report["stages"].items():
            logger.info(f"{stage_name:15} | Avg: {metrics['avg_ms']:6.2f}ms | "
                       f"Min: {metrics['min_ms']:6.2f}ms | Max: {metrics['max_ms']:6.2f}ms | "
                       f"Calls: {metrics['call_count']}")
        
        if report["bottlenecks"]:
            logger.warning(f"Bottlenecks detected: {', '.join(report['bottlenecks'])}")
        
        logger.info("=" * 60)