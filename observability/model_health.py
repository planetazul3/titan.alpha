"""
Model Health Monitor for Production Trading Systems.

Tracks model performance in production and triggers alerts/retraining
when degradation is detected.

Key capabilities:
- Accuracy degradation detection (rolling accuracy vs baseline)
- Calibration drift (expected vs actual probabilities)
- Prediction distribution shift (KS test)
- Feature drift detection
- Auto-retraining recommendations

Reference: 
- "Hidden Technical Debt in Machine Learning Systems" (Sculley et al., 2015)
- "Monitoring Machine Learning Models in Production" (Huyen, 2022)

ARCHITECTURAL PRINCIPLE:
Model monitoring is PRODUCTION CRITICAL. The monitor runs independently
of the model and can recommend retraining without human intervention.
However, actual retraining requires explicit approval unless in full
autonomous mode.

Example:
    >>> from observability.model_health import ModelHealthMonitor
    >>> monitor = ModelHealthMonitor(baseline_accuracy=0.58)
    >>> monitor.record_prediction(probability=0.65, actual_outcome=1)
    >>> health = monitor.check_health()
    >>> if not health.is_healthy:
    ...     print(f"Issues: {health.issues}")
"""

import logging
import math
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, cast

import numpy as np

logger = logging.getLogger(__name__)


class HealthStatus(Enum):
    """Model health status levels."""
    HEALTHY = "healthy"           # All metrics within bounds
    DEGRADED = "degraded"         # Some metrics drifting
    CRITICAL = "critical"         # Requires immediate attention
    RETRAINING_RECOMMENDED = "retraining_recommended"


@dataclass
class HealthCheckResult:
    """
    Result of model health check.
    
    Attributes:
        status: Overall health status
        is_healthy: Boolean healthy flag
        issues: List of detected issues
        metrics: All computed health metrics
        recommendation: Action recommendation
        timestamp: When check was performed
    """
    status: HealthStatus
    is_healthy: bool
    issues: list[str]
    metrics: dict[str, float]
    recommendation: str
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    
    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "status": self.status.value,
            "is_healthy": self.is_healthy,
            "issues": self.issues,
            "metrics": self.metrics,
            "recommendation": self.recommendation,
            "timestamp": self.timestamp.isoformat(),
        }


class AccuracyTracker:
    """
    Track rolling accuracy with baseline comparison.
    
    Detects when accuracy drops below baseline or shows
    sustained degradation pattern.
    """
    
    def __init__(
        self,
        baseline_accuracy: float,
        window_size: int = 100,
        degradation_threshold: float = 0.05,
    ):
        """
        Initialize accuracy tracker.
        
        Args:
            baseline_accuracy: Expected accuracy from training
            window_size: Rolling window for accuracy calculation
            degradation_threshold: Allowed drop from baseline
        """
        self.baseline = baseline_accuracy
        self.window_size = window_size
        self.threshold = degradation_threshold
        self._predictions: deque = deque(maxlen=window_size)
    
    def record(self, predicted: float, actual: int, threshold: float = 0.5) -> None:
        """Record a prediction outcome."""
        correct = (predicted >= threshold) == (actual == 1)
        self._predictions.append(1 if correct else 0)
    
    def get_rolling_accuracy(self) -> float:
        """Get current rolling accuracy."""
        if not self._predictions:
            return self.baseline
        return cast(float, sum(self._predictions) / len(self._predictions))
    
    def check_degradation(self) -> tuple[bool, float]:
        """
        Check if accuracy has degraded.
        
        Returns:
            Tuple of (is_degraded, accuracy_delta)
        """
        if len(self._predictions) < self.window_size // 2:
            return False, 0.0
        
        current = self.get_rolling_accuracy()
        delta = current - self.baseline
        is_degraded = delta < -self.threshold
        
        return is_degraded, delta


class CalibrationTracker:
    """
    Track probability calibration.
    
    A well-calibrated model should have:
    - Predictions of 0.7 that are correct 70% of the time
    - Predictions of 0.9 that are correct 90% of the time
    
    Uses binned expected calibration error (ECE).
    """
    
    # Minimum samples needed for reliable ECE calculation
    MIN_SAMPLES_FOR_ECE = 50
    
    def __init__(self, num_bins: int = 10, max_samples: int = 1000):
        """
        Initialize calibration tracker.
        
        Args:
            num_bins: Number of probability bins
            max_samples: Maximum samples to track
        """
        self.num_bins = num_bins
        self.max_samples = max_samples
        self._samples: deque = deque(maxlen=max_samples)
    
    def record(self, probability: float, actual: int) -> None:
        """Record a prediction with outcome."""
        self._samples.append((probability, actual))
    
    def compute_ece(self) -> float:
        """
        Compute Expected Calibration Error.
        
        Returns:
            ECE value (0 = perfect calibration)
        """
        if len(self._samples) < self.MIN_SAMPLES_FOR_ECE:
            return 0.0
        
        probs = np.array([s[0] for s in self._samples])
        actuals = np.array([s[1] for s in self._samples])
        
        bin_edges = np.linspace(0, 1, self.num_bins + 1)
        ece = 0.0
        
        for i in range(self.num_bins):
            mask = (probs > bin_edges[i]) & (probs <= bin_edges[i + 1])
            n_bin = mask.sum()
            
            if n_bin > 0:
                avg_confidence = probs[mask].mean()
                avg_accuracy = actuals[mask].mean()
                ece += (n_bin / len(probs)) * abs(avg_accuracy - avg_confidence)
        
        return float(ece)
    
    def check_miscalibration(self, threshold: float = 0.1) -> tuple[bool, float]:
        """
        Check if model is miscalibrated.
        
        Returns:
            Tuple of (is_miscalibrated, ece_value)
        """
        ece = self.compute_ece()
        return ece > threshold, ece


class DistributionDriftTracker:
    """
    Track prediction distribution shift using statistical tests.
    
    Compares recent prediction distribution against baseline
    using Kolmogorov-Smirnov test.
    """
    
    def __init__(
        self,
        baseline_samples: int = 500,
        recent_samples: int = 100,
    ):
        """
        Initialize drift tracker.
        
        Args:
            baseline_samples: Number of samples for baseline distribution
            recent_samples: Number of recent samples to compare
        """
        self.baseline_window = baseline_samples
        self.recent_window = recent_samples
        self._all_predictions: deque = deque(maxlen=baseline_samples + recent_samples)
    
    def record(self, probability: float) -> None:
        """Record a prediction."""
        self._all_predictions.append(probability)
    
    def compute_ks_statistic(self) -> tuple[float, float]:
        """
        Compute KS statistic between baseline and recent distributions.
        
        Returns:
            Tuple of (ks_statistic, approximate_p_value)
        """
        if len(self._all_predictions) < self.baseline_window + self.recent_window // 2:
            return 0.0, 1.0
        
        predictions = list(self._all_predictions)
        
        # Split into baseline (older) and recent (newer)
        split_point = len(predictions) - self.recent_window
        baseline = np.array(predictions[:split_point])
        recent = np.array(predictions[split_point:])
        
        if len(baseline) < 20 or len(recent) < 20:
            return 0.0, 1.0
        
        # Two-sample KS test (simplified)
        combined = np.sort(np.concatenate([baseline, recent]))
        
        cdf_baseline = np.searchsorted(np.sort(baseline), combined, side='right') / len(baseline)
        cdf_recent = np.searchsorted(np.sort(recent), combined, side='right') / len(recent)
        
        ks_stat = np.max(np.abs(cdf_baseline - cdf_recent))
        
        # Approximate p-value using asymptotic formula
        n = len(baseline) * len(recent) / (len(baseline) + len(recent))
        p_value = np.exp(-2 * n * ks_stat ** 2)
        
        return float(ks_stat), float(p_value)
    
    def check_drift(self, significance: float = 0.05) -> tuple[bool, float]:
        """
        Check if distribution has drifted.
        
        Returns:
            Tuple of (is_drifted, ks_statistic)
        """
        ks_stat, p_value = self.compute_ks_statistic()
        return p_value < significance, ks_stat


class ModelHealthMonitor:
    """
    Comprehensive model health monitoring for production.
    
    Combines multiple health signals:
    - Rolling accuracy vs baseline
    - Calibration error
    - Prediction distribution drift
    - Trade win rate
    
    Provides unified health status and retraining recommendations.
    
    Example:
        >>> monitor = ModelHealthMonitor(baseline_accuracy=0.58)
        >>> # In production loop:
        >>> monitor.record_prediction(prob, outcome)
        >>> health = monitor.check_health()
        >>> if health.status == HealthStatus.RETRAINING_RECOMMENDED:
        ...     trigger_retraining()
    """
    
    def __init__(
        self,
        baseline_accuracy: float = 0.55,
        accuracy_window: int = 100,
        accuracy_threshold: float = 0.05,
        calibration_threshold: float = 0.15,
        drift_significance: float = 0.05,
    ):
        """
        Initialize model health monitor.
        
        Args:
            baseline_accuracy: Expected accuracy from training
            accuracy_window: Rolling window for accuracy
            accuracy_threshold: Allowed accuracy drop
            calibration_threshold: Max acceptable ECE
            drift_significance: p-value for drift detection
        """
        self.accuracy_tracker = AccuracyTracker(
            baseline_accuracy, accuracy_window, accuracy_threshold
        )
        self.calibration_tracker = CalibrationTracker()
        self.drift_tracker = DistributionDriftTracker()
        
        self.calibration_threshold = calibration_threshold
        self.drift_significance = drift_significance
        
        self._total_predictions = 0
        self._total_wins = 0
        self._recent_checks: deque = deque(maxlen=10)
        
        logger.info(
            f"ModelHealthMonitor initialized: "
            f"baseline={baseline_accuracy}, window={accuracy_window}"
        )
    
    def record_prediction(
        self,
        probability: float,
        actual_outcome: int,
        threshold: float = 0.5,
    ) -> None:
        """
        Record a prediction with its outcome.
        
        Args:
            probability: Model's predicted probability
            actual_outcome: Actual outcome (1=win, 0=loss)
            threshold: Classification threshold
        """
        self.accuracy_tracker.record(probability, actual_outcome, threshold)
        self.calibration_tracker.record(probability, actual_outcome)
        self.drift_tracker.record(probability)
        
        self._total_predictions += 1
        if actual_outcome == 1:
            self._total_wins += 1
    
    def check_health(self) -> HealthCheckResult:
        """
        Perform comprehensive health check.
        
        Returns:
            HealthCheckResult with status, issues, and recommendations
        """
        issues = []
        metrics = {}
        
        # 1. Accuracy check
        is_degraded, acc_delta = self.accuracy_tracker.check_degradation()
        metrics["rolling_accuracy"] = self.accuracy_tracker.get_rolling_accuracy()
        metrics["accuracy_delta"] = acc_delta
        
        if is_degraded:
            issues.append(f"Accuracy degraded by {abs(acc_delta):.2%}")
        
        # 2. Calibration check
        is_miscal, ece = self.calibration_tracker.check_miscalibration(
            self.calibration_threshold
        )
        metrics["ece"] = ece
        
        if is_miscal:
            issues.append(f"Model miscalibrated (ECE={ece:.3f})")
        
        # 3. Distribution drift check
        is_drifted, ks_stat = self.drift_tracker.check_drift(
            self.drift_significance
        )
        metrics["ks_statistic"] = ks_stat
        
        if is_drifted:
            issues.append(f"Prediction distribution drift (KS={ks_stat:.3f})")
        
        # 4. Win rate
        if self._total_predictions > 0:
            win_rate = self._total_wins / self._total_predictions
            metrics["win_rate"] = win_rate
            metrics["total_predictions"] = self._total_predictions
        
        # Determine overall status
        n_issues = len(issues)
        if n_issues == 0:
            status = HealthStatus.HEALTHY
            recommendation = "Model performing within expected parameters"
        elif n_issues == 1:
            status = HealthStatus.DEGRADED
            recommendation = "Monitor closely, consider retraining if persists"
        elif n_issues == 2:
            status = HealthStatus.CRITICAL
            recommendation = "Immediate investigation required"
        else:
            status = HealthStatus.RETRAINING_RECOMMENDED
            recommendation = "Retraining strongly recommended"
        
        result = HealthCheckResult(
            status=status,
            is_healthy=(status == HealthStatus.HEALTHY),
            issues=issues,
            metrics=metrics,
            recommendation=recommendation,
        )
        
        self._recent_checks.append(result)
        
        if issues:
            logger.warning(
                f"Model health check: {status.value}, "
                f"issues={issues}, recommendation={recommendation}"
            )
        else:
            logger.debug(f"Model health check: {status.value}")
        
        return result
    
    def should_retrain(self) -> bool:
        """
        Check if retraining is recommended.
        
        Uses sustained degradation pattern, not single check.
        
        Returns:
            True if retraining is recommended
        """
        if len(self._recent_checks) < 3:
            return False
        
        # Check if last 3 checks had issues
        recent = list(self._recent_checks)[-3:]
        all_degraded = all(
            check.status in [HealthStatus.CRITICAL, HealthStatus.RETRAINING_RECOMMENDED]
            for check in recent
        )
        
        return all_degraded
    
    def get_statistics(self) -> dict[str, Any]:
        """Get monitoring statistics."""
        return {
            "total_predictions": self._total_predictions,
            "total_wins": self._total_wins,
            "win_rate": self._total_wins / self._total_predictions if self._total_predictions > 0 else 0,
            "rolling_accuracy": self.accuracy_tracker.get_rolling_accuracy(),
            "ece": self.calibration_tracker.compute_ece(),
            "recent_checks": len(self._recent_checks),
            "should_retrain": self.should_retrain(),
        }
    
    def reset(self) -> None:
        """Reset all tracking (e.g., after retraining)."""
        self.accuracy_tracker._predictions.clear()
        self.calibration_tracker._samples.clear()
        self.drift_tracker._all_predictions.clear()
        self._total_predictions = 0
        self._total_wins = 0
        self._recent_checks.clear()
        logger.info("ModelHealthMonitor reset")
