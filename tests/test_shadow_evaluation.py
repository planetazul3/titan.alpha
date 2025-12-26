"""
Unit tests for shadow evaluation module.
"""

import pytest
import torch
import torch.nn as nn

from training.shadow_evaluation import (
    ABTestResult,
    ModelPromoter,
    ShadowEvaluator,
    ShadowPrediction,
    ABTestStatus,
    ValidationPipeline,
)


class MockModel(nn.Module):
    """Mock model for testing."""
    
    def __init__(self, base_prob: float = 0.5):
        super().__init__()
        self.base_prob = base_prob
        self.fc = nn.Linear(10, 1)
    
    def predict_probs(self, ticks, candles, vol_metrics):
        batch_size = ticks.shape[0]
        return {
            "rise_fall_prob": torch.full((batch_size, 1), self.base_prob),
        }


class MockRegistry:
    """Mock model registry."""
    
    def __init__(self):
        self._models = {}
        self._primary = None
    
    def list_models(self):
        return list(self._models.keys())
    
    def get_metadata(self, mid):
        class Meta:
            is_primary = (mid == self._primary)
        return Meta() if mid in self._models else None
    
    def set_primary(self, mid):
        if mid not in self._models:
            raise ValueError(f"Model {mid} not found")
        self._primary = mid


class TestShadowPrediction:
    """Tests for ShadowPrediction."""

    def test_initial_state(self):
        """Test initial state has no outcome."""
        pred = ShadowPrediction(0.7, 0.8)
        assert pred.outcome is None
        assert pred.champion_correct() is None

    def test_set_outcome(self):
        """Test setting outcome."""
        pred = ShadowPrediction(0.7, 0.8)
        pred.set_outcome(1, 5.0)
        
        assert pred.outcome == 1
        assert pred.pnl == 5.0

    def test_correctness_check(self):
        """Test correctness evaluation."""
        pred = ShadowPrediction(0.7, 0.3)
        pred.set_outcome(1)
        
        assert pred.champion_correct() is True  # 0.7 >= 0.5 and outcome = 1
        assert pred.challenger_correct() is False  # 0.3 < 0.5 and outcome = 1


class TestShadowEvaluator:
    """Tests for ShadowEvaluator."""

    def test_record_predictions(self):
        """Test recording predictions."""
        champion = MockModel(0.6)
        challenger = MockModel(0.7)
        evaluator = ShadowEvaluator(champion, challenger)
        
        ticks = torch.randn(1, 50)
        candles = torch.randn(1, 30, 10)
        vol = torch.randn(1, 4)
        
        pred_id = evaluator.record_predictions(ticks, candles, vol)
        
        assert pred_id is not None
        assert len(evaluator._predictions) == 1

    def test_record_outcome(self):
        """Test recording outcomes."""
        champion = MockModel(0.6)
        challenger = MockModel(0.7)
        evaluator = ShadowEvaluator(champion, challenger)
        
        ticks = torch.randn(1, 50)
        candles = torch.randn(1, 30, 10)
        vol = torch.randn(1, 4)
        
        pred_id = evaluator.record_predictions(ticks, candles, vol)
        evaluator.record_outcome(pred_id, 1)
        
        resolved = evaluator.get_resolved_predictions()
        assert len(resolved) == 1

    def test_comparison_insufficient_samples(self):
        """Test comparison with insufficient samples."""
        champion = MockModel(0.6)
        challenger = MockModel(0.7)
        evaluator = ShadowEvaluator(champion, challenger)
        
        result = evaluator.get_comparison()
        
        assert result.status == ABTestStatus.PENDING

    def test_comparison_with_samples(self):
        """Test comparison with sufficient samples."""
        champion = MockModel(0.6)
        challenger = MockModel(0.7)
        evaluator = ShadowEvaluator(champion, challenger)
        
        ticks = torch.randn(1, 50)
        candles = torch.randn(1, 30, 10)
        vol = torch.randn(1, 4)
        
        # Record 50 predictions with outcomes
        for i in range(50):
            pred_id = evaluator.record_predictions(ticks, candles, vol)
            evaluator.record_outcome(pred_id, 1)
        
        result = evaluator.get_comparison()
        
        assert result.samples_collected == 50
        assert result.status in [ABTestStatus.RUNNING, ABTestStatus.SIGNIFICANT]

    def test_max_samples_limit(self):
        """Test max samples limit."""
        champion = MockModel(0.6)
        challenger = MockModel(0.7)
        evaluator = ShadowEvaluator(champion, challenger, max_samples=10)
        
        ticks = torch.randn(1, 50)
        candles = torch.randn(1, 30, 10)
        vol = torch.randn(1, 4)
        
        # Record more than max
        for _ in range(20):
            evaluator.record_predictions(ticks, candles, vol)
        
        assert len(evaluator._predictions) == 10


class TestValidationPipeline:
    """Tests for ValidationPipeline."""

    def test_validate_valid_model(self):
        """Test validation of valid model."""
        pipeline = ValidationPipeline()
        model = MockModel()
        
        ticks = torch.randn(1, 50)
        candles = torch.randn(1, 30, 10)
        vol = torch.randn(1, 4)
        
        passed, failures = pipeline.validate(model, (ticks, candles, vol))
        
        assert passed is True
        assert len(failures) == 0

    def test_custom_check(self):
        """Test custom validation check."""
        pipeline = ValidationPipeline()
        
        # Add check that always fails
        pipeline.add_check(lambda m: (False, "Custom check failed"))
        
        model = MockModel()
        passed, failures = pipeline.validate(model)
        
        assert passed is False
        assert "Custom check failed" in failures


class TestModelPromoter:
    """Tests for ModelPromoter."""

    def test_evaluate_insufficient_samples(self):
        """Test evaluation with insufficient samples."""
        registry = MockRegistry()
        registry._models = {"v1": None, "v2": None}
        registry._primary = "v1"
        
        promoter = ModelPromoter(registry, min_samples=100)
        
        champion = MockModel(0.6)
        challenger = MockModel(0.7)
        evaluator = ShadowEvaluator(champion, challenger)
        
        should_promote, reason = promoter.evaluate_promotion(evaluator, "v2")
        
        assert should_promote is False
        assert "Insufficient samples" in reason

    def test_promote(self):
        """Test model promotion."""
        registry = MockRegistry()
        registry._models = {"v1": None, "v2": None}
        registry._primary = "v1"
        
        promoter = ModelPromoter(registry)
        
        success = promoter.promote("v2", "Better accuracy")
        
        assert success is True
        assert registry._primary == "v2"

    def test_rollback(self):
        """Test rollback."""
        registry = MockRegistry()
        registry._models = {"v1": None, "v2": None}
        registry._primary = "v1"
        
        promoter = ModelPromoter(registry)
        promoter.promote("v2", "Testing")
        
        success = promoter.rollback()
        
        assert success is True
        assert registry._primary == "v1"

    def test_history(self):
        """Test promotion history."""
        registry = MockRegistry()
        registry._models = {"v1": None, "v2": None}
        registry._primary = "v1"
        
        promoter = ModelPromoter(registry)
        promoter.promote("v2", "Better")
        
        history = promoter.get_history()
        
        assert len(history) == 1
        assert history[0]["promoted"] == "v2"


class TestABTestResult:
    """Tests for ABTestResult."""

    def test_to_dict(self):
        """Test serialization."""
        result = ABTestResult(
            status=ABTestStatus.SIGNIFICANT,
            champion_metric=0.6,
            challenger_metric=0.65,
            p_value=0.03,
            samples_collected=200,
            improvement_pct=8.3,
            recommendation="Promote",
        )
        
        d = result.to_dict()
        
        assert d["status"] == "significant"
        assert d["improvement_pct"] == 8.3
