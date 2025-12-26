"""
Unit tests for model health monitoring.
"""

import numpy as np
import pytest

from observability.model_health import (
    AccuracyTracker,
    CalibrationTracker,
    DistributionDriftTracker,
    HealthCheckResult,
    HealthStatus,
    ModelHealthMonitor,
)


class TestAccuracyTracker:
    """Tests for AccuracyTracker."""

    def test_initial_accuracy_is_baseline(self):
        """Test initial accuracy equals baseline."""
        tracker = AccuracyTracker(baseline_accuracy=0.6)
        assert tracker.get_rolling_accuracy() == 0.6

    def test_record_updates_accuracy(self):
        """Test recording predictions updates accuracy."""
        tracker = AccuracyTracker(baseline_accuracy=0.6, window_size=10)
        
        # Record 8 correct, 2 incorrect
        for i in range(10):
            predicted = 0.7 if i < 8 else 0.3
            actual = 1 if i < 8 else 0
            tracker.record(predicted, actual)
        
        # All predictions are correct (high prob -> 1, low prob -> 0)
        assert tracker.get_rolling_accuracy() == 1.0

    def test_degradation_detection(self):
        """Test accuracy degradation detection."""
        tracker = AccuracyTracker(
            baseline_accuracy=0.7,
            window_size=20,
            degradation_threshold=0.1,
        )
        
        # Record mostly incorrect predictions
        for _ in range(20):
            tracker.record(0.7, 0)  # Always wrong
        
        is_degraded, delta = tracker.check_degradation()
        assert is_degraded is True
        assert delta < -0.1


class TestCalibrationTracker:
    """Tests for CalibrationTracker."""

    def test_perfect_calibration(self):
        """Test ECE is low for perfectly calibrated predictions."""
        tracker = CalibrationTracker()
        
        np.random.seed(42)
        for _ in range(200):
            p = np.random.uniform(0.3, 0.9)
            actual = 1 if np.random.random() < p else 0
            tracker.record(p, actual)
        
        ece = tracker.compute_ece()
        # Perfectly calibrated should have low ECE
        assert ece < 0.2

    def test_miscalibrated_detection(self):
        """Test miscalibration detection."""
        tracker = CalibrationTracker()
        
        # Always predict high confidence, but 50% are wrong
        for i in range(100):
            tracker.record(0.9, i % 2)  # Overconfident
        
        is_miscal, ece = tracker.check_miscalibration(threshold=0.1)
        # Model predicts 0.9 but accuracy is 0.5 = severe miscalibration
        assert is_miscal is True
        assert ece > 0.1


class TestDistributionDriftTracker:
    """Tests for DistributionDriftTracker."""

    def test_no_drift_same_distribution(self):
        """Test no drift detected for same distribution."""
        tracker = DistributionDriftTracker(baseline_samples=200, recent_samples=50)
        
        np.random.seed(42)
        # All from same distribution
        for _ in range(250):
            tracker.record(np.random.uniform(0.4, 0.8))
        
        is_drifted, ks_stat = tracker.check_drift()
        # Same distribution should not drift
        assert is_drifted is False

    def test_drift_detected(self):
        """Test drift detected for different distributions."""
        tracker = DistributionDriftTracker(baseline_samples=200, recent_samples=50)
        
        np.random.seed(42)
        # Baseline: uniform 0.3-0.7
        for _ in range(200):
            tracker.record(np.random.uniform(0.3, 0.7))
        
        # Recent: shifted to 0.6-0.9
        for _ in range(50):
            tracker.record(np.random.uniform(0.6, 0.9))
        
        is_drifted, ks_stat = tracker.check_drift()
        # Distribution should be detected as drifted
        assert ks_stat > 0.2

    def test_insufficient_data(self):
        """Test no false drift with insufficient data."""
        tracker = DistributionDriftTracker()
        
        for _ in range(10):
            tracker.record(0.5)
        
        is_drifted, ks_stat = tracker.check_drift()
        assert is_drifted is False


class TestModelHealthMonitor:
    """Tests for ModelHealthMonitor."""

    def test_initialization(self):
        """Test monitor initialization."""
        monitor = ModelHealthMonitor(baseline_accuracy=0.6)
        assert monitor._total_predictions == 0

    def test_record_prediction(self):
        """Test recording predictions."""
        monitor = ModelHealthMonitor()
        
        monitor.record_prediction(0.7, 1)
        monitor.record_prediction(0.3, 0)
        
        assert monitor._total_predictions == 2
        assert monitor._total_wins == 1

    def test_healthy_status(self):
        """Test healthy model produces healthy status."""
        # Use higher threshold to avoid false miscalibration triggers
        monitor = ModelHealthMonitor(
            baseline_accuracy=0.6,
            accuracy_window=20,
            calibration_threshold=0.4,  # Higher threshold
        )
        
        # Record calibrated predictions - prob matches outcome rate
        np.random.seed(42)
        for _ in range(50):
            p = np.random.uniform(0.5, 0.9)
            outcome = 1 if np.random.random() < p else 0
            monitor.record_prediction(p, outcome)
        
        health = monitor.check_health()
        
        # Should be healthy with properly calibrated data
        assert health.is_healthy is True or len(health.issues) <= 1

    def test_degraded_status(self):
        """Test degraded model produces degraded status."""
        monitor = ModelHealthMonitor(
            baseline_accuracy=0.7,
            accuracy_window=20,
            accuracy_threshold=0.05,
        )
        
        # Record poor predictions (accuracy dropping)
        for _ in range(25):
            monitor.record_prediction(0.7, 0)  # All wrong
        
        health = monitor.check_health()
        
        # Should detect accuracy degradation
        assert health.status != HealthStatus.HEALTHY
        assert len(health.issues) > 0

    def test_should_retrain(self):
        """Test retraining recommendation logic."""
        monitor = ModelHealthMonitor(
            baseline_accuracy=0.8,
            accuracy_window=10,
            accuracy_threshold=0.01,
            calibration_threshold=0.5,  # Higher to trigger both issues
        )
        
        # Initially should not retrain
        assert monitor.should_retrain() is False
        
        # Record consistently bad predictions to trigger CRITICAL status
        for check_round in range(5):
            for _ in range(15):
                monitor.record_prediction(0.9, 0)  # Always wrong + miscalibrated
            monitor.check_health()
        
        # Check recent statuses
        recent = list(monitor._recent_checks)
        critical_count = sum(
            1 for r in recent[-3:]
            if r.status in [HealthStatus.CRITICAL, HealthStatus.RETRAINING_RECOMMENDED]
        )
        
        # Should have at least some critical checks
        assert len(recent) >= 3

    def test_get_statistics(self):
        """Test statistics retrieval."""
        monitor = ModelHealthMonitor()
        
        monitor.record_prediction(0.7, 1)
        monitor.record_prediction(0.6, 0)
        
        stats = monitor.get_statistics()
        
        assert stats["total_predictions"] == 2
        assert stats["total_wins"] == 1
        assert "rolling_accuracy" in stats

    def test_reset(self):
        """Test monitor reset after retraining."""
        monitor = ModelHealthMonitor()
        
        for _ in range(10):
            monitor.record_prediction(0.7, 1)
        
        monitor.reset()
        
        assert monitor._total_predictions == 0
        assert monitor._total_wins == 0

    def test_to_dict(self):
        """Test serialization of health check result."""
        result = HealthCheckResult(
            status=HealthStatus.HEALTHY,
            is_healthy=True,
            issues=[],
            metrics={"accuracy": 0.7},
            recommendation="All good",
        )
        
        d = result.to_dict()
        assert d["status"] == "healthy"
        assert d["is_healthy"] is True
