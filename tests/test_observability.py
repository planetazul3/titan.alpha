"""
Unit tests for observability metrics module.

Tests cover:
- Counter increments and label tracking
- Histogram observations and bucketing
- Gauge set/inc/dec operations
- Alert callbacks
- Metric export
- Latency timer context manager
"""

import time
from unittest.mock import MagicMock

import pytest

from observability import (
    InternalCounter,
    InternalGauge,
    InternalHistogram,
    LatencyTimer,
    TradingMetrics,
    get_metrics,
)


class TestInternalCounter:
    """Tests for InternalCounter."""

    def test_simple_increment(self):
        """Should increment without labels."""
        counter = InternalCounter("test_counter")

        counter.inc()
        counter.inc(5)

        assert counter.get_all()["total"] == 6

    def test_labeled_increment(self):
        """Should track increments by label."""
        counter = InternalCounter("trades", ["outcome", "type"])

        counter.inc(outcome="success", type="CALL")
        counter.inc(outcome="success", type="CALL")
        counter.inc(outcome="failed", type="PUT")

        assert counter.get(outcome="success", type="CALL") == 2
        assert counter.get(outcome="failed", type="PUT") == 1
        assert counter.get(outcome="unknown", type="CALL") == 0

    def test_reset(self):
        """Reset should clear all values."""
        counter = InternalCounter("test", ["label"])
        counter.inc(label="a")
        counter.inc(label="b")

        counter.reset()

        assert counter.get(label="a") == 0
        assert counter.get(label="b") == 0


class TestInternalHistogram:
    """Tests for InternalHistogram."""

    def test_observation_counting(self):
        """Should count observations."""
        hist = InternalHistogram("latency")

        hist.observe(0.01)
        hist.observe(0.02)
        hist.observe(0.03)

        stats = hist.get_stats()
        assert stats["count"] == 3
        assert stats["sum"] == pytest.approx(0.06)
        assert stats["mean"] == pytest.approx(0.02)

    def test_bucket_assignment(self):
        """Should assign to correct buckets."""
        hist = InternalHistogram("test", buckets=[0.01, 0.1, 1.0])

        hist.observe(0.005)  # <= 0.01
        hist.observe(0.05)  # <= 0.1
        hist.observe(0.5)  # <= 1.0

        stats = hist.get_stats()
        assert stats["buckets"]["le_0.01"] == 1
        assert stats["buckets"]["le_0.1"] == 2  # Cumulative
        assert stats["buckets"]["le_1.0"] == 3  # Cumulative

    def test_reset(self):
        """Reset should clear histogram."""
        hist = InternalHistogram("test")
        hist.observe(0.1)
        hist.observe(0.2)

        hist.reset()

        stats = hist.get_stats()
        assert stats["count"] == 0
        assert stats["sum"] == 0.0


class TestInternalGauge:
    """Tests for InternalGauge."""

    def test_set(self):
        """Should set value."""
        gauge = InternalGauge("balance")

        gauge.set(100.0)
        assert gauge.get() == 100.0

        gauge.set(50.0)
        assert gauge.get() == 50.0

    def test_inc_dec(self):
        """Should increment and decrement."""
        gauge = InternalGauge("pnl")

        gauge.inc(10.0)
        assert gauge.get() == 10.0

        gauge.dec(3.0)
        assert gauge.get() == 7.0


class TestTradingMetrics:
    """Tests for TradingMetrics class."""

    @pytest.fixture
    def metrics(self):
        """Create fresh metrics instance."""
        m = TradingMetrics(enable_prometheus=False)
        m.reset()
        return m

    def test_record_trade_attempt(self, metrics):
        """Should record trade attempts."""
        metrics.record_trade_attempt("executed", "RISE_FALL")
        metrics.record_trade_attempt("executed", "RISE_FALL")
        metrics.record_trade_attempt("blocked_rate_limit", "TOUCH")

        trades = metrics.trades_total.get_all()
        assert "executed_RISE_FALL" in trades
        assert trades["executed_RISE_FALL"] == 2

    def test_record_regime_assessment(self, metrics):
        """Should record regime assessments."""
        metrics.record_regime_assessment("TRUSTED")
        metrics.record_regime_assessment("TRUSTED")
        metrics.record_regime_assessment("VETO")

        regime = metrics.regime_assessments.get_all()
        assert regime["TRUSTED"] == 2
        assert regime["VETO"] == 1

    def test_record_latencies(self, metrics):
        """Should record latency histograms."""
        metrics.record_inference_latency(0.01)
        metrics.record_inference_latency(0.02)
        metrics.record_execution_latency(0.05)

        inf_stats = metrics.inference_latency.get_stats()
        assert inf_stats["count"] == 2

        exec_stats = metrics.execution_latency.get_stats()
        assert exec_stats["count"] == 1

    def test_gauge_operations(self, metrics):
        """Should track gauge values."""
        metrics.set_daily_pnl(-25.0)
        metrics.set_reconstruction_error(0.15)
        metrics.set_account_balance(500.0)

        assert metrics.daily_pnl.get() == -25.0
        assert metrics.reconstruction_error.get() == 0.15
        assert metrics.account_balance.get() == 500.0

    def test_alert_callback(self, metrics):
        """Should trigger alert when reconstruction error exceeds threshold."""
        callback = MagicMock()
        metrics.register_alert_callback(callback)

        # Below threshold - no alert
        metrics.set_reconstruction_error(0.1)
        callback.assert_not_called()

        # Above threshold - alert triggered
        metrics.set_reconstruction_error(0.5)
        callback.assert_called_once()
        assert "reconstruction_error_high" in callback.call_args[0][0]

    def test_export(self, metrics):
        """Should export all metrics as dict."""
        metrics.record_trade_attempt("executed", "RISE_FALL")
        metrics.record_inference_latency(0.01)
        metrics.set_daily_pnl(10.0)

        export = metrics.export()

        assert "timestamp" in export
        assert "counters" in export
        assert "histograms" in export
        assert "gauges" in export
        assert export["gauges"]["daily_pnl"] == 10.0


class TestLatencyTimer:
    """Tests for LatencyTimer context manager."""

    def test_measures_elapsed_time(self):
        """Should measure elapsed time."""
        recorded_times = []

        with LatencyTimer(recorded_times.append):
            time.sleep(0.01)  # 10ms

        assert len(recorded_times) == 1
        assert recorded_times[0] >= 0.01

    def test_works_with_metrics(self):
        """Should work with metrics callbacks."""
        metrics = TradingMetrics(enable_prometheus=False)
        metrics.reset()

        with LatencyTimer(metrics.record_inference_latency):
            time.sleep(0.005)

        stats = metrics.inference_latency.get_stats()
        assert stats["count"] == 1
        assert stats["sum"] >= 0.005


class TestGlobalMetrics:
    """Tests for global metrics singleton."""

    def test_get_metrics_returns_same_instance(self):
        """get_metrics should return singleton."""
        m1 = get_metrics()
        m2 = get_metrics()

        assert m1 is m2

    def test_get_metrics_creates_trading_metrics(self):
        """Should return TradingMetrics instance."""
        m = get_metrics()
        assert isinstance(m, TradingMetrics)
