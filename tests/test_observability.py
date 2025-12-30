"""
Tests for observability components.
"""

import pytest
from unittest.mock import MagicMock, patch
from observability.model_health import ModelHealthMonitor, HealthStatus

class TestModelHealthMonitor:

    def test_initialization(self):
        monitor = ModelHealthMonitor(baseline_accuracy=0.6)
        assert monitor.accuracy_tracker.baseline == 0.6
        assert monitor.calibration_threshold == 0.15

    def test_healthy_check(self):
        monitor = ModelHealthMonitor(baseline_accuracy=0.55)
        # Simulate some good predictions
        for _ in range(50):
            monitor.record_prediction(0.6, 1) # Win
            monitor.record_prediction(0.4, 0) # Loss (Correct)

        health = monitor.check_health()
        # With small sample size, drift checks might not trigger, but basic sanity:
        assert health.status in [HealthStatus.HEALTHY, HealthStatus.DEGRADED]

    def test_degraded_accuracy(self):
        monitor = ModelHealthMonitor(baseline_accuracy=0.9) # Unrealistic baseline
        # Simulate bad predictions
        for _ in range(100):
            monitor.record_prediction(0.9, 0) # High confidence loss

        health = monitor.check_health()
        assert health.status != HealthStatus.HEALTHY
        assert any("Accuracy degraded" in issue for issue in health.issues)

    def test_should_retrain(self):
        monitor = ModelHealthMonitor()
        # Mock internal state to simulate sustained failure

        # 1. Critical
        with patch.object(monitor, 'check_health') as mock_check:
            mock_check.return_value = MagicMock(status=HealthStatus.CRITICAL)
            monitor.check_health()
            monitor.check_health()
            monitor.check_health()

            # Since check_health appends to _recent_checks, we need to manually populate it
            # if we are mocking check_health return value but calling the real method...
            # actually we can't easily mock the return of the method while testing side effects
            # unless we spy. Let's just manually append to _recent_checks

            monitor._recent_checks.append(MagicMock(status=HealthStatus.CRITICAL))
            monitor._recent_checks.append(MagicMock(status=HealthStatus.CRITICAL))
            monitor._recent_checks.append(MagicMock(status=HealthStatus.CRITICAL))

            assert monitor.should_retrain() is True

    def test_reset(self):
        monitor = ModelHealthMonitor()
        monitor.record_prediction(0.5, 1)
        monitor.reset()
        assert monitor._total_predictions == 0
        assert len(monitor._recent_checks) == 0
