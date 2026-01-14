"""
VoiceForge-Nextgen - Latency Benchmark
File: tests/benchmark_latency.py

Purpose:
    Measure end-to-end latency of the audio pipeline
    Generate performance report

Dependencies:
    - time (measurements)
    - numpy
    - json (report export)

Usage:
    python tests/benchmark_latency.py
    
Output:
    logs/latency_report.json
"""

import time
import numpy as np
import json
import sys
from pathlib import Path
from typing import Dict, List
from dataclasses import dataclass, asdict

# Add app to path
ROOT_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT_DIR))

from app.core.audio_preprocessor import AudioPreprocessor
from app.core.audio_postprocessor import AudioPostprocessor
from app.core.feature_extractor import F0Extractor
from app.core.pitch_shifter import PitchShifter


@dataclass
class LatencyMeasurement:
    """Single latency measurement"""
    component: str
    duration_ms: float
    iterations: int
    avg_ms: float
    min_ms: float
    max_ms: float
    std_ms: float


class LatencyBenchmark:
    """
    Benchmark pipeline latency
    
    Measures:
        - Preprocessing latency
        - F0 extraction latency
        - Pitch shifting latency
        - Postprocessing latency
        - End-to-end latency
    """
    
    def __init__(self, chunk_size_ms: float = 100.0, sample_rate: int = 48000):
        """
        Initialize benchmark
        
        Args:
            chunk_size_ms: Audio chunk size in milliseconds
            sample_rate: Sample rate
        """
        self.chunk_size_ms = chunk_size_ms
        self.sample_rate = sample_rate
        self.chunk_samples = int(sample_rate * chunk_size_ms / 1000)
        
        # Results
        self.results: List[LatencyMeasurement] = []
        
        print(f"Latency Benchmark Initialized")
        print(f"  Chunk size: {chunk_size_ms}ms ({self.chunk_samples} samples)")
        print(f"  Sample rate: {sample_rate}Hz")
        print("=" * 60)
    
    def generate_test_audio(self) -> np.ndarray:
        """Generate test audio chunk"""
        return np.random.randn(self.chunk_samples).astype(np.float32) * 0.1
    
    def measure_component(
        self,
        name: str,
        func,
        iterations: int = 100
    ) -> LatencyMeasurement:
        """
        Measure component latency
        
        Args:
            name: Component name
            func: Function to measure
            iterations: Number of iterations
            
        Returns:
            LatencyMeasurement
        """
        print(f"\nMeasuring: {name}")
        
        latencies = []
        
        # Warmup
        for _ in range(10):
            func()
        
        # Measure
        for i in range(iterations):
            start = time.perf_counter()
            func()
            end = time.perf_counter()
            
            latency_ms = (end - start) * 1000
            latencies.append(latency_ms)
            
            if (i + 1) % 20 == 0:
                print(f"  Progress: {i+1}/{iterations}")
        
        # Calculate statistics
        latencies_arr = np.array(latencies)
        
        measurement = LatencyMeasurement(
            component=name,
            duration_ms=sum(latencies),
            iterations=iterations,
            avg_ms=float(np.mean(latencies_arr)),
            min_ms=float(np.min(latencies_arr)),
            max_ms=float(np.max(latencies_arr)),
            std_ms=float(np.std(latencies_arr))
        )
        
        self.results.append(measurement)
        
        print(f"  ✓ Avg: {measurement.avg_ms:.3f}ms")
        print(f"    Min: {measurement.min_ms:.3f}ms")
        print(f"    Max: {measurement.max_ms:.3f}ms")
        print(f"    Std: {measurement.std_ms:.3f}ms")
        
        return measurement
    
    def benchmark_preprocessing(self) -> LatencyMeasurement:
        """Benchmark preprocessing"""
        preprocessor = AudioPreprocessor(target_sr=self.sample_rate)
        audio = self.generate_test_audio()
        
        def process():
            return preprocessor.process(audio, source_sr=self.sample_rate)
        
        return self.measure_component("Preprocessing", process)
    
    def benchmark_f0_extraction(self, method: str = 'harvest') -> LatencyMeasurement:
        """Benchmark F0 extraction"""
        extractor = F0Extractor(method=method, sr=self.sample_rate)
        audio = self.generate_test_audio()
        
        def extract():
            return extractor.extract(audio)
        
        return self.measure_component(f"F0 Extraction ({method})", extract)
    
    def benchmark_pitch_shifting(self) -> LatencyMeasurement:
        """Benchmark pitch shifting"""
        shifter = PitchShifter()
        f0 = np.full(100, 440.0, dtype=np.float32)
        
        def shift():
            return shifter.shift(f0, semitones=12)
        
        return self.measure_component("Pitch Shifting", shift)
    
    def benchmark_postprocessing(self) -> LatencyMeasurement:
        """Benchmark postprocessing"""
        postprocessor = AudioPostprocessor()
        audio = self.generate_test_audio()
        
        def process():
            return postprocessor.process(audio)
        
        return self.measure_component("Postprocessing", process)
    
    def benchmark_end_to_end(self) -> LatencyMeasurement:
        """Benchmark complete pipeline"""
        preprocessor = AudioPreprocessor(target_sr=self.sample_rate)
        postprocessor = AudioPostprocessor()
        audio = self.generate_test_audio()
        
        def pipeline():
            # Preprocess
            processed = preprocessor.process(audio, source_sr=self.sample_rate)
            
            # Mock inference (passthrough)
            output = processed * 0.95
            
            # Postprocess
            final = postprocessor.process(output)
            
            return final
        
        return self.measure_component("End-to-End Pipeline", pipeline)
    
    def run_all_benchmarks(self):
        """Run all benchmarks"""
        print("\n" + "=" * 60)
        print("Starting Full Benchmark Suite")
        print("=" * 60)
        
        # Run benchmarks
        self.benchmark_preprocessing()
        self.benchmark_f0_extraction('harvest')
        self.benchmark_pitch_shifting()
        self.benchmark_postprocessing()
        self.benchmark_end_to_end()
        
        print("\n" + "=" * 60)
        print("Benchmark Complete!")
        print("=" * 60)
    
    def generate_report(self) -> Dict:
        """
        Generate benchmark report
        
        Returns:
            Report dictionary
        """
        report = {
            'metadata': {
                'chunk_size_ms': self.chunk_size_ms,
                'sample_rate': self.sample_rate,
                'chunk_samples': self.chunk_samples
            },
            'measurements': [asdict(m) for m in self.results],
            'summary': self._generate_summary()
        }
        
        return report
    
    def _generate_summary(self) -> Dict:
        """Generate summary statistics"""
        if not self.results:
            return {}
        
        total_latency = sum(m.avg_ms for m in self.results if m.component != "End-to-End Pipeline")
        
        e2e = next((m for m in self.results if m.component == "End-to-End Pipeline"), None)
        
        summary = {
            'total_component_latency_ms': total_latency,
            'measured_e2e_latency_ms': e2e.avg_ms if e2e else None,
            'overhead_ms': (e2e.avg_ms - total_latency) if e2e else None
        }
        
        return summary
    
    def print_summary(self):
        """Print benchmark summary"""
        summary = self._generate_summary()
        
        print("\n" + "=" * 60)
        print("BENCHMARK SUMMARY")
        print("=" * 60)
        
        for result in self.results:
            print(f"{result.component:30s} | {result.avg_ms:8.3f}ms")
        
        print("-" * 60)
        
        if 'total_component_latency_ms' in summary:
            print(f"{'Total Component Latency':30s} | {summary['total_component_latency_ms']:8.3f}ms")
        
        if summary.get('measured_e2e_latency_ms'):
            print(f"{'Measured E2E Latency':30s} | {summary['measured_e2e_latency_ms']:8.3f}ms")
        
        if summary.get('overhead_ms'):
            print(f"{'Overhead':30s} | {summary['overhead_ms']:8.3f}ms")
        
        print("=" * 60)
        
        # Target evaluation
        target_latency = 15.0  # ms
        e2e = next((m for m in self.results if m.component == "End-to-End Pipeline"), None)
        
        if e2e:
            if e2e.avg_ms <= target_latency:
                print(f"\n✓ PASS: Latency {e2e.avg_ms:.2f}ms <= target {target_latency}ms")
            else:
                print(f"\n✗ FAIL: Latency {e2e.avg_ms:.2f}ms > target {target_latency}ms")
    
    def export_report(self, filepath: str = "logs/latency_report.json"):
        """Export report to JSON file"""
        report = self.generate_report()
        
        # Ensure directory exists
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        
        with open(filepath, 'w') as f:
            json.dump(report, f, indent=2)
        
        print(f"\n✓ Report exported to: {filepath}")


def main():
    """Main benchmark entry point"""
    # Create benchmark
    benchmark = LatencyBenchmark(
        chunk_size_ms=100.0,  # 100ms chunks
        sample_rate=48000
    )
    
    # Run all benchmarks
    benchmark.run_all_benchmarks()
    
    # Print summary
    benchmark.print_summary()
    
    # Export report
    benchmark.export_report()


if __name__ == "__main__":
    main()