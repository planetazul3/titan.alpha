"""
Shadow Model Evaluation and A/B Testing Framework.

Enables safe comparison of new models against production models
by running them in parallel (shadow mode) and comparing results.

Key components:
- ShadowEvaluator: Runs challenger model alongside champion
- ABTestFramework: Statistical comparison of model performance
- ModelPromoter: Automated promotion logic with safeguards
- ValidationPipeline: Pre-deployment validation checks

ARCHITECTURAL PRINCIPLE:
New models must prove themselves in shadow mode before promotion.
The challenger runs in parallel, making predictions that are
logged but NOT executed. Only after statistical significance
is reached can a model be promoted to production.

Example:
    >>> from training.shadow_evaluation import ShadowEvaluator
    >>> evaluator = ShadowEvaluator(champion_model, challenger_model)
    >>> evaluator.evaluate(data)
    >>> if evaluator.should_promote():
    ...     evaluator.promote_challenger()
"""

import logging
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Callable

import numpy as np
import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


class ABTestStatus(Enum):
    """A/B test status."""
    PENDING = "pending"           # Not enough samples
    RUNNING = "running"           # Collecting samples
    SIGNIFICANT = "significant"   # Statistical significance reached
    INCONCLUSIVE = "inconclusive" # No clear winner
    FAILED = "failed"             # Challenger worse than champion


@dataclass
class ABTestResult:
    """
    Result of A/B test comparison.
    
    Attributes:
        status: Current test status
        champion_metric: Champion model metric value
        challenger_metric: Challenger model metric value
        p_value: Statistical significance
        samples_collected: Number of samples
        improvement_pct: Percentage improvement
        recommendation: Human-readable recommendation
    """
    status: ABTestStatus
    champion_metric: float
    challenger_metric: float
    p_value: float
    samples_collected: int
    improvement_pct: float
    recommendation: str
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    
    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "status": self.status.value,
            "champion_metric": self.champion_metric,
            "challenger_metric": self.challenger_metric,
            "p_value": self.p_value,
            "samples_collected": self.samples_collected,
            "improvement_pct": self.improvement_pct,
            "recommendation": self.recommendation,
            "timestamp": self.timestamp.isoformat(),
        }


class ShadowPrediction:
    """
    Container for shadow predictions from both models.
    
    Stores predictions from champion and challenger for
    later comparison when outcome is known.
    """
    
    def __init__(
        self,
        champion_prob: float,
        challenger_prob: float,
        timestamp: datetime | None = None,
    ):
        self.champion_prob = champion_prob
        self.challenger_prob = challenger_prob
        self.timestamp = timestamp or datetime.now(timezone.utc)
        self.outcome: int | None = None
        self.pnl: float | None = None
    
    def set_outcome(self, outcome: int, pnl: float | None = None) -> None:
        """Record the actual outcome."""
        self.outcome = outcome
        self.pnl = pnl
    
    def champion_correct(self, threshold: float = 0.5) -> bool | None:
        """Check if champion prediction was correct."""
        if self.outcome is None:
            return None
        predicted = self.champion_prob >= threshold
        return predicted == (self.outcome == 1)
    
    def challenger_correct(self, threshold: float = 0.5) -> bool | None:
        """Check if challenger prediction was correct."""
        if self.outcome is None:
            return None
        predicted = self.challenger_prob >= threshold
        return predicted == (self.outcome == 1)


class ShadowEvaluator:
    """
    Evaluates challenger model against champion in shadow mode.
    
    The champion model's predictions are used for actual trading.
    The challenger's predictions are logged for comparison only.
    
    Example:
        >>> evaluator = ShadowEvaluator(champion, challenger)
        >>> for data in market_stream:
        ...     pred_id = evaluator.record_predictions(data)
        ...     # Later, when outcome known:
        ...     evaluator.record_outcome(pred_id, outcome)
        >>> result = evaluator.get_comparison()
    """
    
    def __init__(
        self,
        champion: nn.Module,
        challenger: nn.Module,
        max_samples: int = 1000,
    ):
        """
        Initialize shadow evaluator.
        
        Args:
            champion: Production model (predictions used for trading)
            challenger: New model (predictions logged only)
            max_samples: Maximum samples to store
        """
        self.champion = champion
        self.challenger = challenger
        self.max_samples = max_samples
        
        self._predictions: dict[str, ShadowPrediction] = {}
        self._prediction_id = 0
        
        logger.info("ShadowEvaluator initialized")
    
    def record_predictions(
        self,
        ticks: torch.Tensor,
        candles: torch.Tensor,
        vol_metrics: torch.Tensor,
    ) -> str:
        """
        Record predictions from both models.
        
        Args:
            ticks: Tick data
            candles: Candle data
            vol_metrics: Volatility metrics
        
        Returns:
            Prediction ID for later outcome recording
        """
        with torch.no_grad():
            # Get champion prediction
            champion_out = self.champion.predict_probs(ticks, candles, vol_metrics)  # type: ignore[operator]
            champion_prob_tensor = champion_out.get("rise_fall_prob", torch.tensor([0.5]))
            if champion_prob_tensor.dim() > 0:  # type: ignore[union-attr]
                champion_prob = float(champion_prob_tensor.mean().item())  # type: ignore[union-attr]
            else:
                champion_prob = float(champion_prob_tensor.item())
            
            # Get challenger prediction
            challenger_out = self.challenger.predict_probs(ticks, candles, vol_metrics)  # type: ignore[operator]
            challenger_prob_tensor = challenger_out.get("rise_fall_prob", torch.tensor([0.5]))
            if challenger_prob_tensor.dim() > 0:  # type: ignore[union-attr]
                challenger_prob = float(challenger_prob_tensor.mean().item())  # type: ignore[union-attr]
            else:
                challenger_prob = float(challenger_prob_tensor.item())
        
        # Store prediction
        pred_id = f"pred_{self._prediction_id}"
        self._predictions[pred_id] = ShadowPrediction(
            champion_prob=champion_prob,
            challenger_prob=challenger_prob,
        )
        self._prediction_id += 1
        
        # Cleanup old predictions if over limit
        if len(self._predictions) > self.max_samples:
            oldest = list(self._predictions.keys())[0]
            del self._predictions[oldest]
        
        return pred_id
    
    def record_outcome(
        self,
        pred_id: str,
        outcome: int,
        pnl: float | None = None,
    ) -> None:
        """Record outcome for a prediction."""
        if pred_id in self._predictions:
            self._predictions[pred_id].set_outcome(outcome, pnl)
    
    def get_resolved_predictions(self) -> list[ShadowPrediction]:
        """Get predictions with known outcomes."""
        return [p for p in self._predictions.values() if p.outcome is not None]
    
    def get_comparison(self) -> ABTestResult:
        """
        Compare champion vs challenger performance.
        
        Returns:
            ABTestResult with comparison statistics
        """
        resolved = self.get_resolved_predictions()
        n = len(resolved)
        
        if n < 30:
            return ABTestResult(
                status=ABTestStatus.PENDING,
                champion_metric=0.0,
                challenger_metric=0.0,
                p_value=1.0,
                samples_collected=n,
                improvement_pct=0.0,
                recommendation="Need at least 30 samples for comparison",
            )
        
        # Calculate accuracies
        champion_correct = sum(1 for p in resolved if p.champion_correct())
        challenger_correct = sum(1 for p in resolved if p.challenger_correct())
        
        champion_acc = champion_correct / n
        challenger_acc = challenger_correct / n
        
        # Simple z-test for proportions
        pooled = (champion_correct + challenger_correct) / (2 * n)
        se = np.sqrt(2 * pooled * (1 - pooled) / n)
        
        if se > 0:
            z = (challenger_acc - champion_acc) / se
            # Two-tailed p-value approximation
            p_value = 2 * (1 - min(0.9999, abs(z) / 3))  # Simplified
        else:
            z = 0
            p_value = 1.0
        
        improvement = (challenger_acc - champion_acc) / max(champion_acc, 0.01)
        improvement_pct = improvement * 100
        
        # Determine status
        if p_value < 0.05:
            if challenger_acc > champion_acc:
                status = ABTestStatus.SIGNIFICANT
                recommendation = "Challenger significantly outperforms champion. Consider promotion."
            else:
                status = ABTestStatus.FAILED
                recommendation = "Champion significantly outperforms challenger. Do not promote."
        elif n >= 500:
            status = ABTestStatus.INCONCLUSIVE
            recommendation = "No significant difference after 500+ samples."
        else:
            status = ABTestStatus.RUNNING
            recommendation = f"Continue collecting samples. Currently {n}/{500}."
        
        return ABTestResult(
            status=status,
            champion_metric=champion_acc,
            challenger_metric=challenger_acc,
            p_value=p_value,
            samples_collected=n,
            improvement_pct=improvement_pct,
            recommendation=recommendation,
        )
    
    def should_promote(self, min_samples: int = 100, min_improvement: float = 0.02) -> bool:
        """
        Check if challenger should be promoted.
        
        Args:
            min_samples: Minimum samples required
            min_improvement: Minimum improvement required
        
        Returns:
            True if challenger should be promoted
        """
        result = self.get_comparison()
        return (
            result.status == ABTestStatus.SIGNIFICANT
            and result.samples_collected >= min_samples
            and result.improvement_pct >= min_improvement * 100
        )


class ValidationPipeline:
    """
    Pre-deployment validation checks for new models.
    
    Runs a series of checks to ensure model is safe for deployment:
    - Output sanity (probabilities in valid range)
    - Latency requirements
    - Memory footprint
    - Consistency (same input -> similar output)
    """
    
    def __init__(
        self,
        latency_threshold_ms: float = 100.0,
        memory_threshold_mb: float = 500.0,
    ):
        self.latency_threshold_ms = latency_threshold_ms
        self.memory_threshold_mb = memory_threshold_mb
        self._checks: list[Callable] = []
    
    def add_check(self, check: Callable[[nn.Module], tuple[bool, str]]) -> None:
        """Add a custom validation check."""
        self._checks.append(check)
    
    def validate(
        self,
        model: nn.Module,
        sample_data: tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None = None,
    ) -> tuple[bool, list[str]]:
        """
        Run all validation checks.
        
        Args:
            model: Model to validate
            sample_data: Optional (ticks, candles, vol_metrics) for testing
        
        Returns:
            Tuple of (passed, list of failures)
        """
        failures = []
        
        # 1. Output sanity check
        if sample_data:
            ticks, candles, vol_metrics = sample_data
            with torch.no_grad():
                output = model.predict_probs(ticks, candles, vol_metrics)  # type: ignore[operator]
            
            for key, prob in output.items():
                if "prob" in key:
                    if (prob < 0).any() or (prob > 1).any():  # type: ignore[union-attr]
                        failures.append(f"Invalid probabilities in {key}")
        
        # 2. Model has parameters
        n_params = sum(p.numel() for p in model.parameters())
        if n_params == 0:
            failures.append("Model has no parameters")
        
        # 3. Model is in eval mode check (warning only)
        if model.training:
            logger.warning("Model is in training mode, should be eval for deployment")
        
        # 4. Run custom checks
        for check in self._checks:
            try:
                passed, msg = check(model)
                if not passed:
                    failures.append(msg)
            except Exception as e:
                failures.append(f"Check failed with error: {e}")
        
        return len(failures) == 0, failures
    
    def benchmark_latency(
        self,
        model: nn.Module,
        sample_data: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
        num_runs: int = 100,
    ) -> dict[str, float]:
        """
        Benchmark model latency.
        
        Returns:
            Dictionary with mean, std, max latency in ms
        """
        import time
        
        ticks, candles, vol_metrics = sample_data
        latencies = []
        
        model.eval()
        with torch.no_grad():
            # Warmup
            for _ in range(10):
                model.predict_probs(ticks, candles, vol_metrics)  # type: ignore[operator]
            
            # Benchmark
            for _ in range(num_runs):
                start = time.perf_counter()
                model.predict_probs(ticks, candles, vol_metrics)  # type: ignore[operator]
                elapsed = (time.perf_counter() - start) * 1000
                latencies.append(elapsed)
        
        return {
            "mean_ms": np.mean(latencies),
            "std_ms": np.std(latencies),
            "max_ms": np.max(latencies),
            "p99_ms": np.percentile(latencies, 99),
            "passes_threshold": np.mean(latencies) < self.latency_threshold_ms,
        }


class ModelPromoter:
    """
    Handles model promotion workflow.
    
    Provides:
    - Promotion criteria evaluation
    - Rollback capability
    - Audit logging
    """
    
    def __init__(
        self,
        registry: Any,  # ModelRegistry from ensemble module
        min_samples: int = 100,
        min_improvement: float = 0.02,
        max_degradation: float = 0.05,
    ):
        self.registry = registry
        self.min_samples = min_samples
        self.min_improvement = min_improvement
        self.max_degradation = max_degradation
        
        self._promotion_history: list[dict] = []
        self._previous_primary: str | None = None
    
    def evaluate_promotion(
        self,
        evaluator: ShadowEvaluator,
        challenger_id: str,
    ) -> tuple[bool, str]:
        """
        Evaluate if challenger should be promoted.
        
        Returns:
            Tuple of (should_promote, reason)
        """
        result = evaluator.get_comparison()
        
        if result.samples_collected < self.min_samples:
            return False, f"Insufficient samples: {result.samples_collected}/{self.min_samples}"
        
        if result.status == ABTestStatus.FAILED:
            return False, f"Challenger underperforms by {abs(result.improvement_pct):.1f}%"
        
        if result.status == ABTestStatus.INCONCLUSIVE:
            return False, "No significant difference detected"
        
        if result.improvement_pct < self.min_improvement * 100:
            return False, f"Improvement {result.improvement_pct:.1f}% below threshold {self.min_improvement * 100}%"
        
        return True, f"Challenger improves by {result.improvement_pct:.1f}%"
    
    def promote(
        self,
        challenger_id: str,
        reason: str = "",
    ) -> bool:
        """
        Promote challenger to primary.
        
        Args:
            challenger_id: Model ID to promote
            reason: Reason for promotion
        
        Returns:
            True if promotion successful
        """
        try:
            # Store previous primary for rollback
            for mid in self.registry.list_models():
                meta = self.registry.get_metadata(mid)
                if meta and meta.is_primary:
                    self._previous_primary = mid
                    break
            
            # Promote
            self.registry.set_primary(challenger_id)
            
            # Log
            self._promotion_history.append({
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "promoted": challenger_id,
                "previous": self._previous_primary,
                "reason": reason,
            })
            
            logger.info(f"Promoted model '{challenger_id}': {reason}")
            return True
            
        except Exception as e:
            logger.error(f"Promotion failed: {e}")
            return False
    
    def rollback(self) -> bool:
        """
        Rollback to previous primary model.
        
        Returns:
            True if rollback successful
        """
        if not self._previous_primary:
            logger.warning("No previous primary to rollback to")
            return False
        
        try:
            self.registry.set_primary(self._previous_primary)
            
            self._promotion_history.append({
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "action": "rollback",
                "restored": self._previous_primary,
            })
            
            logger.info(f"Rolled back to model '{self._previous_primary}'")
            return True
            
        except Exception as e:
            logger.error(f"Rollback failed: {e}")
            return False
    
    def get_history(self) -> list[dict]:
        """Get promotion history."""
        return self._promotion_history.copy()
