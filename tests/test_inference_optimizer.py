"""
Unit tests for inference optimizer.
"""

import pytest
import torch
import torch.nn as nn

from observability.inference_optimizer import (
    FeatureCache,
    InferenceOptimizer,
    LatencyStats,
)


class SimpleModel(nn.Module):
    """Simple model for testing."""
    
    def __init__(self, input_dim: int = 10, output_dim: int = 5):
        super().__init__()
        self.fc = nn.Linear(input_dim, output_dim)
    
    def forward(self, x):
        return self.fc(x)


class TestLatencyStats:
    """Tests for LatencyStats."""

    def test_to_dict(self):
        """Test serialization."""
        stats = LatencyStats(
            mean_ms=10.0,
            std_ms=2.0,
            min_ms=5.0,
            max_ms=20.0,
            p50_ms=9.0,
            p95_ms=15.0,
            p99_ms=18.0,
            samples=100,
        )
        
        d = stats.to_dict()
        
        assert d["mean_ms"] == 10.0
        assert d["samples"] == 100


class TestFeatureCache:
    """Tests for FeatureCache."""

    def test_cache_miss(self):
        """Test cache miss."""
        cache = FeatureCache(max_size=10)
        data = torch.randn(5, 10)
        
        result = cache.get(data)
        
        assert result is None

    def test_cache_hit(self):
        """Test cache hit."""
        cache = FeatureCache(max_size=10)
        data = torch.randn(5, 10)
        features = torch.randn(5, 20)
        
        cache.put(data, features)
        result = cache.get(data)
        
        assert result is not None
        assert torch.allclose(result, features)

    def test_lru_eviction(self):
        """Test LRU eviction."""
        cache = FeatureCache(max_size=2)
        
        data1 = torch.randn(5, 10)
        data2 = torch.randn(5, 10)
        data3 = torch.randn(5, 10)
        
        cache.put(data1, torch.randn(5, 20))
        cache.put(data2, torch.randn(5, 20))
        cache.put(data3, torch.randn(5, 20))
        
        # data1 should be evicted
        assert cache.get(data1) is None
        assert cache.get(data2) is not None

    def test_get_stats(self):
        """Test statistics."""
        cache = FeatureCache(max_size=10)
        data = torch.randn(5, 10)
        
        cache.get(data)  # Miss
        cache.put(data, torch.randn(5, 20))
        cache.get(data)  # Hit
        
        stats = cache.get_stats()
        
        assert stats["hits"] == 1
        assert stats["misses"] == 1
        assert stats["hit_rate"] == 0.5

    def test_clear(self):
        """Test cache clearing."""
        cache = FeatureCache()
        cache.put(torch.randn(5, 10), torch.randn(5, 20))
        
        cache.clear()
        
        assert cache.get_stats()["size"] == 0


class TestInferenceOptimizer:
    """Tests for InferenceOptimizer."""

    def test_initialization(self):
        """Test initialization."""
        optimizer = InferenceOptimizer()
        
        assert optimizer.enable_caching is True
        assert optimizer.cache is not None

    def test_benchmark(self):
        """Test benchmarking."""
        optimizer = InferenceOptimizer()
        model = SimpleModel()
        sample_input = (torch.randn(1, 10),)
        
        stats = optimizer.benchmark(model, sample_input, num_warmup=2, num_runs=10)
        
        assert stats.samples == 10
        assert stats.mean_ms > 0

    def test_compile_torchscript(self):
        """Test TorchScript compilation."""
        optimizer = InferenceOptimizer()
        model = SimpleModel()
        sample_input = (torch.randn(1, 10),)
        
        compiled = optimizer.compile_torchscript(model, sample_input)
        
        assert isinstance(compiled, torch.jit.ScriptModule)
        
        # Test that compiled model works
        output = compiled(sample_input[0])
        assert output.shape == (1, 5)

    def test_create_cached_forward(self):
        """Test cached forward creation."""
        optimizer = InferenceOptimizer(enable_caching=True)
        model = SimpleModel()
        
        cached_forward = optimizer.create_cached_forward(model)
        
        input1 = torch.randn(1, 10)
        result1 = cached_forward(input1)
        result2 = cached_forward(input1)  # Should hit cache
        
        assert optimizer.cache.get_stats()["hits"] == 1

    def test_get_optimization_report(self):
        """Test optimization report."""
        optimizer = InferenceOptimizer()
        model = SimpleModel()
        sample_input = (torch.randn(1, 10),)
        
        report = optimizer.get_optimization_report(model, sample_input)
        
        assert "model_params" in report
        assert "baseline_latency" in report
        assert "recommendations" in report

    def test_fp16_mode(self):
        """Test FP16 mode context manager."""
        optimizer = InferenceOptimizer()
        
        # Should not raise
        with optimizer.fp16_mode():
            result = torch.randn(10) + 1
        
        assert result is not None
