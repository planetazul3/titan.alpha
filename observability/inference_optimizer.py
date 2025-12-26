"""
Production Optimization Module.

Implements inference optimization techniques for deployment:
- TorchScript compilation for faster inference
- Mixed precision (FP16) inference
- Feature caching for reduced computation
- Latency profiling and benchmarking

ARCHITECTURAL PRINCIPLE:
Production models must be fast AND reliable. This module
provides utilities to optimize models for deployment while
maintaining correctness guarantees.

Example:
    >>> from observability.inference_optimizer import InferenceOptimizer
    >>> optimizer = InferenceOptimizer()
    >>> optimized_model = optimizer.compile(model)
    >>> with optimizer.fp16_mode():
    ...     predictions = optimized_model(inputs)
"""

import logging
import time
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Callable

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


@dataclass
class LatencyStats:
    """
    Latency statistics from benchmarking.
    
    Attributes:
        mean_ms: Mean latency in milliseconds
        std_ms: Standard deviation
        min_ms: Minimum latency
        max_ms: Maximum latency
        p50_ms: 50th percentile (median)
        p95_ms: 95th percentile
        p99_ms: 99th percentile
        samples: Number of samples
    """
    mean_ms: float
    std_ms: float
    min_ms: float
    max_ms: float
    p50_ms: float
    p95_ms: float
    p99_ms: float
    samples: int
    
    def to_dict(self) -> dict[str, float]:
        """Serialize to dictionary."""
        return {
            "mean_ms": self.mean_ms,
            "std_ms": self.std_ms,
            "min_ms": self.min_ms,
            "max_ms": self.max_ms,
            "p50_ms": self.p50_ms,
            "p95_ms": self.p95_ms,
            "p99_ms": self.p99_ms,
            "samples": self.samples,
        }


class FeatureCache:
    """
    LRU cache for expensive feature computations.
    
    Caches feature tensors to avoid redundant computation
    when the same data is processed multiple times.
    """
    
    def __init__(self, max_size: int = 100):
        """
        Initialize cache.
        
        Args:
            max_size: Maximum number of cached entries
        """
        self.max_size = max_size
        self._cache: dict[str, torch.Tensor] = {}
        self._access_order: list[str] = []
        self._hits = 0
        self._misses = 0
    
    def _make_key(self, data: torch.Tensor) -> str:
        """Create cache key from tensor."""
        # Use shape and hash of first/last elements
        shape_str = str(tuple(data.shape))
        if data.numel() > 0:
            sample = f"{data.flatten()[0].item():.6f}_{data.flatten()[-1].item():.6f}"
        else:
            sample = "empty"
        return f"{shape_str}_{sample}"
    
    def get(self, data: torch.Tensor) -> torch.Tensor | None:
        """
        Get cached features if available.
        
        Args:
            data: Input data to look up
        
        Returns:
            Cached features or None
        """
        key = self._make_key(data)
        
        if key in self._cache:
            self._hits += 1
            # Move to end of access order
            self._access_order.remove(key)
            self._access_order.append(key)
            return self._cache[key]
        
        self._misses += 1
        return None
    
    def put(self, data: torch.Tensor, features: torch.Tensor) -> None:
        """
        Store features in cache.
        
        Args:
            data: Original data
            features: Computed features
        """
        key = self._make_key(data)
        
        # Evict if at capacity
        while len(self._cache) >= self.max_size:
            oldest = self._access_order.pop(0)
            del self._cache[oldest]
        
        self._cache[key] = features.detach().clone()
        self._access_order.append(key)
    
    def clear(self) -> None:
        """Clear the cache."""
        self._cache.clear()
        self._access_order.clear()
    
    def get_stats(self) -> dict[str, Any]:
        """Get cache statistics."""
        total = self._hits + self._misses
        return {
            "size": len(self._cache),
            "max_size": self.max_size,
            "hits": self._hits,
            "misses": self._misses,
            "hit_rate": self._hits / total if total > 0 else 0.0,
        }


class InferenceOptimizer:
    """
    Optimize models for production inference.
    
    Provides:
    - TorchScript compilation
    - Mixed precision inference
    - Latency benchmarking
    - Feature caching integration
    
    Example:
        >>> optimizer = InferenceOptimizer()
        >>> stats = optimizer.benchmark(model, sample_input)
        >>> if stats.mean_ms > 50:
        ...     model = optimizer.compile_torchscript(model, sample_input)
    """
    
    def __init__(
        self,
        enable_caching: bool = True,
        cache_size: int = 100,
    ):
        """
        Initialize optimizer.
        
        Args:
            enable_caching: Enable feature caching
            cache_size: Maximum cache size
        """
        self.enable_caching = enable_caching
        self.cache = FeatureCache(cache_size) if enable_caching else None
        
        # Check FP16 support
        self.fp16_available = torch.cuda.is_available()
        
        logger.info(
            f"InferenceOptimizer initialized: "
            f"caching={enable_caching}, fp16={self.fp16_available}"
        )
    
    def benchmark(
        self,
        model: nn.Module,
        sample_input: tuple[torch.Tensor, ...],
        num_warmup: int = 10,
        num_runs: int = 100,
    ) -> LatencyStats:
        """
        Benchmark model inference latency.
        
        Args:
            model: Model to benchmark
            sample_input: Sample input tensors
            num_warmup: Warmup iterations
            num_runs: Benchmark iterations
        
        Returns:
            LatencyStats with detailed metrics
        """
        import numpy as np
        
        model.eval()
        latencies_list: list[float] = []
        
        with torch.no_grad():
            # Warmup
            for _ in range(num_warmup):
                _ = model(*sample_input)
            
            # Synchronize if CUDA
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            
            # Benchmark
            for _ in range(num_runs):
                start = time.perf_counter()
                _ = model(*sample_input)
                
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                
                elapsed = (time.perf_counter() - start) * 1000
                latencies_list.append(elapsed)
        
        latencies = np.array(latencies_list)
        
        stats = LatencyStats(
            mean_ms=float(np.mean(latencies)),
            std_ms=float(np.std(latencies)),
            min_ms=float(np.min(latencies)),
            max_ms=float(np.max(latencies)),
            p50_ms=float(np.percentile(latencies, 50)),
            p95_ms=float(np.percentile(latencies, 95)),
            p99_ms=float(np.percentile(latencies, 99)),
            samples=num_runs,
        )
        
        logger.info(
            f"Benchmark complete: mean={stats.mean_ms:.2f}ms, "
            f"p95={stats.p95_ms:.2f}ms"
        )
        
        return stats
    
    def compile_torchscript(
        self,
        model: nn.Module,
        sample_input: tuple[torch.Tensor, ...],
        optimize: bool = True,
    ) -> torch.jit.ScriptModule:
        """
        Compile model to TorchScript for faster inference.
        
        Args:
            model: Model to compile
            sample_input: Sample input for tracing
            optimize: Apply optimizations
        
        Returns:
            Compiled TorchScript module
        """
        model.eval()
        
        try:
            with torch.no_grad():
                traced = torch.jit.trace(model, sample_input)
            
            if optimize:
                traced = torch.jit.optimize_for_inference(traced)
            
            logger.info("TorchScript compilation successful")
            return traced  # type: ignore[no-any-return]
            
        except Exception as e:
            logger.warning(f"TorchScript compilation failed: {e}")
            logger.info("Falling back to scripting...")
            
            try:
                scripted = torch.jit.script(model)
                logger.info("TorchScript scripting successful")
                return scripted
            except Exception as e2:
                logger.error(f"TorchScript scripting also failed: {e2}")
                raise
    
    @contextmanager
    def fp16_mode(self):
        """
        Context manager for FP16 inference.
        
        Usage:
            with optimizer.fp16_mode():
                output = model(input.half())
        """
        if self.fp16_available:
            with torch.cuda.amp.autocast():
                yield
        else:
            yield
    
    def optimize_for_inference(
        self,
        model: nn.Module,
        sample_input: tuple[torch.Tensor, ...],
    ) -> nn.Module:
        """
        Apply all available optimizations.
        
        Args:
            model: Model to optimize
            sample_input: Sample input tensors
        
        Returns:
            Optimized model
        """
        model.eval()
        
        # Fuse batch norm layers
        try:
            model = torch.quantization.fuse_modules(model, [])  # No-op if no fuseable layers
        except Exception:
            pass
        
        # Try TorchScript
        try:
            model = self.compile_torchscript(model, sample_input)
            logger.info("Applied TorchScript optimization")
        except Exception as e:
            logger.warning(f"Could not apply TorchScript: {e}")
        
        return model
    
    def create_cached_forward(
        self,
        model: nn.Module,
    ) -> Callable:
        """
        Create a cached forward function.
        
        Args:
            model: Model to wrap
        
        Returns:
            Cached forward function
        """
        if not self.enable_caching or self.cache is None:
            return model.forward
        
        cache = self.cache
        
        def cached_forward(*args):
            # Try to get from cache
            if len(args) > 0 and isinstance(args[0], torch.Tensor):
                cached = cache.get(args[0])
                if cached is not None:
                    return cached
            
            # Compute
            with torch.no_grad():
                result = model(*args)
            
            # Cache if tensor
            if len(args) > 0 and isinstance(args[0], torch.Tensor):
                if isinstance(result, torch.Tensor):
                    cache.put(args[0], result)
            
            return result
        
        return cached_forward
    
    def get_optimization_report(
        self,
        model: nn.Module,
        sample_input: tuple[torch.Tensor, ...],
    ) -> dict[str, Any]:
        """
        Generate optimization report for model.
        
        Args:
            model: Model to analyze
            sample_input: Sample input tensors
        
        Returns:
            Report with recommendations
        """
        report: dict[str, Any] = {
            "model_params": sum(p.numel() for p in model.parameters()),
            "model_size_mb": sum(p.numel() * p.element_size() for p in model.parameters()) / 1e6,
            "fp16_available": self.fp16_available,
            "recommendations": [],
        }
        recommendations: list[str] = []
        
        # Benchmark baseline
        baseline_stats = self.benchmark(model, sample_input, num_warmup=5, num_runs=50)
        report["baseline_latency"] = baseline_stats.to_dict()
        
        # Recommendations
        if baseline_stats.mean_ms > 50:
            recommendations.append("Consider TorchScript compilation")
        
        if report["model_size_mb"] > 100:
            recommendations.append("Consider model quantization")
        
        if self.fp16_available:
            recommendations.append("FP16 inference available for speedup")
        
        if baseline_stats.std_ms > baseline_stats.mean_ms * 0.2:
            recommendations.append("High latency variance - check for GC or memory issues")
        
        report["recommendations"] = recommendations
        
        return report
