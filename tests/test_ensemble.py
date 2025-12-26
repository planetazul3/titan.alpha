"""
Unit tests for ensemble module.
"""

import pytest
import torch
import torch.nn as nn

from models.ensemble import (
    EnsemblePredictor,
    MetaLearner,
    ModelMetadata,
    ModelRegistry,
    PerformanceTracker,
)


class MockModel(nn.Module):
    """Mock trading model for testing."""
    
    def __init__(self, base_prob: float = 0.5):
        super().__init__()
        self.base_prob = base_prob
        self.fc = nn.Linear(10, 1)  # Dummy parameter
    
    def predict_probs(self, ticks, candles, vol_metrics):
        batch_size = ticks.shape[0]
        return {
            "rise_fall_prob": torch.full((batch_size, 1), self.base_prob),
            "touch_prob": torch.full((batch_size, 1), self.base_prob),
        }
    
    def get_volatility_anomaly_score(self, vol_metrics):
        return torch.zeros(vol_metrics.shape[0])


class TestPerformanceTracker:
    """Tests for PerformanceTracker."""

    def test_initial_accuracy(self):
        """Test initial accuracy is 0.5."""
        tracker = PerformanceTracker()
        assert tracker.get_accuracy() == 0.5

    def test_record_updates_accuracy(self):
        """Test recording updates accuracy."""
        tracker = PerformanceTracker(window_size=10)
        
        # Record 8 correct predictions
        for _ in range(8):
            tracker.record(0.7, 1)  # High prob, win
        # Record 2 incorrect
        for _ in range(2):
            tracker.record(0.7, 0)  # High prob, loss
        
        assert tracker.get_accuracy() == 0.8

    def test_win_rate(self):
        """Test win rate calculation."""
        tracker = PerformanceTracker()
        
        for _ in range(6):
            tracker.record(0.6, 1)
        for _ in range(4):
            tracker.record(0.4, 0)
        
        assert tracker.get_win_rate() == 0.6

    def test_score_calculation(self):
        """Test composite score."""
        tracker = PerformanceTracker()
        
        # Perfect predictions
        for _ in range(10):
            tracker.record(0.7, 1)
        
        score = tracker.get_score()
        # Should be high (near 1.0)
        assert score > 0.8


class TestModelRegistry:
    """Tests for ModelRegistry."""

    def test_register_model(self):
        """Test model registration."""
        registry = ModelRegistry()
        model = MockModel()
        
        registry.register("v1", model, version="1.0.0")
        
        assert "v1" in registry.list_models()
        assert registry.get("v1") is model

    def test_primary_model(self):
        """Test primary model designation."""
        registry = ModelRegistry()
        model1 = MockModel()
        model2 = MockModel()
        
        registry.register("v1", model1)
        registry.register("v2", model2, make_primary=True)
        
        assert registry.get_primary() is model2
        assert registry.get_metadata("v2").is_primary

    def test_remove_model(self):
        """Test model removal."""
        registry = ModelRegistry()
        model1 = MockModel()
        model2 = MockModel()
        
        registry.register("v1", model1, make_primary=True)
        registry.register("v2", model2)
        
        registry.remove("v2")
        
        assert "v2" not in registry.list_models()

    def test_cannot_remove_primary(self):
        """Test cannot remove primary model."""
        registry = ModelRegistry()
        model = MockModel()
        
        registry.register("v1", model, make_primary=True)
        
        with pytest.raises(ValueError):
            registry.remove("v1")

    def test_record_prediction(self):
        """Test recording predictions."""
        registry = ModelRegistry()
        model = MockModel()
        registry.register("v1", model)
        
        registry.record_prediction("v1", 0.7, 1)
        
        tracker = registry.get_tracker("v1")
        assert tracker.get_win_rate() == 1.0


class TestMetaLearner:
    """Tests for MetaLearner."""

    def test_output_shape(self):
        """Test meta-learner output shape."""
        learner = MetaLearner(context_dim=10, num_models=3)
        context = torch.randn(4, 10)
        
        weights = learner(context)
        
        assert weights.shape == (4, 3)

    def test_weights_sum_to_one(self):
        """Test weights sum to 1."""
        learner = MetaLearner(context_dim=10, num_models=3)
        context = torch.randn(4, 10)
        
        weights = learner(context)
        
        sums = weights.sum(dim=-1)
        assert torch.allclose(sums, torch.ones(4), atol=1e-5)


class TestEnsemblePredictor:
    """Tests for EnsemblePredictor."""

    def test_equal_weights(self):
        """Test equal weighting strategy."""
        registry = ModelRegistry()
        registry.register("v1", MockModel(0.6))
        registry.register("v2", MockModel(0.8))
        
        ensemble = EnsemblePredictor(registry, strategy="equal")
        weights = ensemble.get_weights()
        
        assert weights["v1"] == 0.5
        assert weights["v2"] == 0.5

    def test_performance_weights(self):
        """Test performance-based weighting."""
        registry = ModelRegistry()
        registry.register("v1", MockModel())
        registry.register("v2", MockModel())
        
        # Make v2 perform better
        for _ in range(10):
            registry.record_prediction("v1", 0.6, 0)  # All wrong
            registry.record_prediction("v2", 0.6, 1)  # All right
        
        ensemble = EnsemblePredictor(registry, strategy="performance")
        weights = ensemble.get_weights()
        
        # v2 should have higher weight
        assert weights["v2"] > weights["v1"]

    def test_predict(self):
        """Test ensemble prediction."""
        registry = ModelRegistry()
        registry.register("v1", MockModel(0.6))
        registry.register("v2", MockModel(0.8))
        
        ensemble = EnsemblePredictor(registry, strategy="equal")
        
        ticks = torch.randn(2, 50)
        candles = torch.randn(2, 30, 10)
        vol = torch.randn(2, 4)
        
        probs = ensemble.predict(ticks, candles, vol)
        
        # Average of 0.6 and 0.8 = 0.7
        assert probs["rise_fall_prob"][0].item() == pytest.approx(0.7, rel=1e-5)

    def test_predict_best(self):
        """Test best model prediction."""
        registry = ModelRegistry()
        registry.register("v1", MockModel(0.5))
        registry.register("v2", MockModel(0.9))
        
        # Make v2 perform better
        for _ in range(10):
            registry.record_prediction("v2", 0.6, 1)
        
        ensemble = EnsemblePredictor(registry)
        
        ticks = torch.randn(2, 50)
        candles = torch.randn(2, 30, 10)
        vol = torch.randn(2, 4)
        
        probs, best_id = ensemble.predict_best(ticks, candles, vol)
        
        assert best_id == "v2"
        assert probs["rise_fall_prob"][0].item() == pytest.approx(0.9, rel=1e-5)

    def test_get_statistics(self):
        """Test statistics retrieval."""
        registry = ModelRegistry()
        registry.register("v1", MockModel(), make_primary=True)
        registry.register("v2", MockModel())
        
        ensemble = EnsemblePredictor(registry)
        stats = ensemble.get_statistics()
        
        assert stats["num_models"] == 2
        assert "v1" in stats["models"]
        assert stats["models"]["v1"]["is_primary"] is True


class TestModelMetadata:
    """Tests for ModelMetadata."""

    def test_to_dict(self):
        """Test serialization."""
        meta = ModelMetadata(
            model_id="test",
            version="1.0.0",
            is_primary=True,
            weight=0.8,
        )
        
        d = meta.to_dict()
        
        assert d["model_id"] == "test"
        assert d["version"] == "1.0.0"
        assert d["is_primary"] is True
