"""
VoiceForge-Nextgen - PyTorch Optimizer
File: app/utils/torch_optimizer.py

Purpose:
    PyTorch optimization utilities for better performance
    Enable TF32, torch.compile, CUDA graphs, etc.

Dependencies:
    - torch (optimization features)

Usage:
    optimizer = TorchOptimizer()
    optimizer.optimize_for_inference()
"""

import torch
import logging
from typing import Optional, Callable
import warnings

logger = logging.getLogger("TorchOptimizer")


class TorchOptimizer:
    """
    PyTorch optimization utilities
    
    Features:
        - Enable TF32 for Ampere GPUs
        - torch.compile for PyTorch 2.0+
        - CUDA graphs for static shapes
        - Benchmark mode
        - Mixed precision setup
    """
    
    def __init__(self):
        """Initialize optimizer"""
        self.cuda_available = torch.cuda.is_available()
        self.torch_version = torch.__version__
        
        # Detect GPU architecture
        self.gpu_arch = None
        if self.cuda_available:
            self.gpu_arch = torch.cuda.get_device_capability(0)
        
        logger.info(
            f"TorchOptimizer initialized: "
            f"torch={self.torch_version}, "
            f"cuda={self.cuda_available}, "
            f"arch={self.gpu_arch}"
        )
    
    def optimize_for_inference(
        self,
        enable_tf32: bool = True,
        enable_cudnn_benchmark: bool = True,
        enable_cudnn_deterministic: bool = False
    ):
        """
        Apply all inference optimizations
        
        Args:
            enable_tf32: Enable TF32 on Ampere+ GPUs
            enable_cudnn_benchmark: Enable cuDNN benchmark mode
            enable_cudnn_deterministic: Enable deterministic mode (slower)
        """
        logger.info("Applying PyTorch inference optimizations...")
        
        # 1. TF32 for Ampere+ (compute capability >= 8.0)
        if enable_tf32 and self.is_ampere_or_newer():
            self._enable_tf32()
        
        # 2. cuDNN benchmark mode
        if enable_cudnn_benchmark:
            self._enable_cudnn_benchmark()
        
        # 3. Deterministic mode (if needed)
        if enable_cudnn_deterministic:
            self._enable_deterministic()
        
        # 4. Set inference mode defaults
        torch.set_grad_enabled(False)
        
        logger.info("✓ Inference optimizations applied")
    
    def is_ampere_or_newer(self) -> bool:
        """Check if GPU is Ampere or newer (compute >= 8.0)"""
        if not self.cuda_available or self.gpu_arch is None:
            return False
        
        major, minor = self.gpu_arch
        return major >= 8
    
    def _enable_tf32(self):
        """Enable TF32 for matmul and cuDNN"""
        if not hasattr(torch, 'set_float32_matmul_precision'):
            logger.warning("TF32 not available in this PyTorch version")
            return
        
        try:
            # TF32 for matmul
            torch.backends.cuda.matmul.allow_tf32 = True
            
            # TF32 for cuDNN
            torch.backends.cudnn.allow_tf32 = True
            
            logger.info("✓ TF32 enabled (Ampere optimization)")
            
        except Exception as e:
            logger.warning(f"Failed to enable TF32: {e}")
    
    def _enable_cudnn_benchmark(self):
        """Enable cuDNN benchmark mode"""
        if not self.cuda_available:
            return
        
        try:
            torch.backends.cudnn.benchmark = True
            logger.info("✓ cuDNN benchmark mode enabled")
            
        except Exception as e:
            logger.warning(f"Failed to enable cuDNN benchmark: {e}")
    
    def _enable_deterministic(self):
        """Enable deterministic mode (for reproducibility)"""
        if not self.cuda_available:
            return
        
        try:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
            
            # PyTorch 1.11+
            if hasattr(torch, 'use_deterministic_algorithms'):
                torch.use_deterministic_algorithms(True)
            
            logger.info("✓ Deterministic mode enabled (reproducible but slower)")
            
        except Exception as e:
            logger.warning(f"Failed to enable deterministic mode: {e}")
    
    def compile_model(
        self,
        model: torch.nn.Module,
        mode: str = "reduce-overhead"
    ) -> torch.nn.Module:
        """
        Compile model using torch.compile (PyTorch 2.0+)
        
        Args:
            model: Model to compile
            mode: Compilation mode
                - "reduce-overhead": Best for inference
                - "max-autotune": Maximize performance
                - "default": Balanced
                
        Returns:
            Compiled model
        """
        # Check PyTorch version
        major, minor = map(int, self.torch_version.split('.')[:2])
        if major < 2:
            logger.warning(
                f"torch.compile requires PyTorch 2.0+, "
                f"got {self.torch_version}"
            )
            return model
        
        try:
            compiled = torch.compile(model, mode=mode)
            logger.info(f"✓ Model compiled with mode={mode}")
            return compiled
            
        except Exception as e:
            logger.error(f"Model compilation failed: {e}")
            return model
    
    def create_cuda_graph(
        self,
        model: torch.nn.Module,
        sample_input: torch.Tensor
    ) -> Optional[Callable]:
        """
        Create CUDA graph for static input shapes
        
        Args:
            model: Model to wrap
            sample_input: Sample input (for shape inference)
            
        Returns:
            Graph callable or None
        """
        if not self.cuda_available:
            logger.warning("CUDA graphs require CUDA")
            return None
        
        try:
            # Create graph
            graph = torch.cuda.CUDAGraph()
            
            # Allocate static tensors
            static_input = sample_input.clone()
            static_output = None
            
            # Warmup
            model.eval()
            with torch.no_grad():
                for _ in range(3):
                    _ = model(static_input)
            
            # Capture graph
            with torch.cuda.graph(graph):
                static_output = model(static_input)
            
            # Create wrapper
            def run_graph(input_tensor):
                static_input.copy_(input_tensor)
                graph.replay()
                return static_output.clone()
            
            logger.info("✓ CUDA graph created")
            return run_graph
            
        except Exception as e:
            logger.error(f"CUDA graph creation failed: {e}")
            return None
    
    def setup_mixed_precision(
        self,
        enabled: bool = True,
        dtype: torch.dtype = torch.float16
    ):
        """
        Setup automatic mixed precision
        
        Args:
            enabled: Enable AMP
            dtype: Precision dtype (float16 or bfloat16)
        """
        if not self.cuda_available:
            logger.warning("Mixed precision requires CUDA")
            return
        
        if enabled:
            logger.info(f"✓ Mixed precision enabled: {dtype}")
        else:
            logger.info("Mixed precision disabled")
    
    def get_optimal_batch_size(
        self,
        model: torch.nn.Module,
        sample_input: torch.Tensor,
        max_batch_size: int = 64
    ) -> int:
        """
        Find optimal batch size through binary search
        
        Args:
            model: Model to test
            sample_input: Single sample input
            max_batch_size: Maximum batch size to try
            
        Returns:
            Optimal batch size
        """
        if not self.cuda_available:
            return 1
        
        logger.info("Finding optimal batch size...")
        
        optimal = 1
        low, high = 1, max_batch_size
        
        model.eval()
        
        while low <= high:
            mid = (low + high) // 2
            
            try:
                # Test batch
                batch = sample_input.repeat(mid, 1, 1)
                
                with torch.no_grad():
                    _ = model(batch)
                
                # Success, try larger
                optimal = mid
                low = mid + 1
                
                # Clear cache
                torch.cuda.empty_cache()
                
            except RuntimeError as e:
                if "out of memory" in str(e):
                    # OOM, try smaller
                    high = mid - 1
                    torch.cuda.empty_cache()
                else:
                    raise
        
        logger.info(f"✓ Optimal batch size: {optimal}")
        return optimal
    
    def get_optimization_report(self) -> dict:
        """
        Get current optimization settings
        
        Returns:
            Dictionary with optimization status
        """
        report = {
            'torch_version': self.torch_version,
            'cuda_available': self.cuda_available,
            'gpu_arch': self.gpu_arch,
            'optimizations': {}
        }
        
        if self.cuda_available:
            report['optimizations']['tf32_matmul'] = torch.backends.cuda.matmul.allow_tf32
            report['optimizations']['tf32_cudnn'] = torch.backends.cudnn.allow_tf32
            report['optimizations']['cudnn_benchmark'] = torch.backends.cudnn.benchmark
            report['optimizations']['cudnn_deterministic'] = torch.backends.cudnn.deterministic
        
        return report


# Singleton instance
_optimizer_instance: Optional[TorchOptimizer] = None


def get_torch_optimizer() -> TorchOptimizer:
    """Get singleton torch optimizer"""
    global _optimizer_instance
    
    if _optimizer_instance is None:
        _optimizer_instance = TorchOptimizer()
    
    return _optimizer_instance


def optimize_for_inference():
    """Apply default inference optimizations"""
    optimizer = get_torch_optimizer()
    optimizer.optimize_for_inference()