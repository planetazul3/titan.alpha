"""
Unit tests for adaptive risk module.
"""

import pytest
from datetime import datetime, timezone, timedelta

from execution.adaptive_risk import (
    AdaptiveRiskManager,
    PerformanceTracker,
    RiskLevel,
    RiskLimits,
)


class TestPerformanceTracker:
    """Tests for PerformanceTracker."""

    def test_initial_state(self):
        """Test initial state."""
        tracker = PerformanceTracker()
        assert tracker.get_win_rate() == 0.5
        assert tracker.get_sharpe_ratio() == 0.0

    def test_record_updates_win_rate(self):
        """Test recording updates win rate."""
        tracker = PerformanceTracker()
        
        for _ in range(8):
            tracker.record(5.0)  # Win
        for _ in range(2):
            tracker.record(-5.0)  # Loss
        
        assert tracker.get_win_rate() == 0.8

    def test_drawdown_tracking(self):
        """Test drawdown is tracked correctly."""
        tracker = PerformanceTracker()
        
        # Peak at 1000, drop to 900
        tracker.record(10.0, current_equity=1000.0)
        tracker.record(-10.0, current_equity=900.0)
        
        assert tracker.get_drawdown() == pytest.approx(0.1,rel=0.01)

    def test_profit_factor(self):
        """Test profit factor calculation."""
        tracker = PerformanceTracker()
        
        tracker.record(50.0)
        tracker.record(30.0)
        tracker.record(-20.0)
        
        # Profit factor = 80 / 20 = 4.0
        assert tracker.get_profit_factor() == 4.0


class TestRiskLimits:
    """Tests for RiskLimits."""

    def test_to_dict(self):
        """Test serialization."""
        limits = RiskLimits(
            daily_loss_limit=20.0,
            max_stake=5.0,
            max_trades_per_hour=10,
            max_drawdown=0.15,
            risk_level=RiskLevel.NORMAL,
        )
        
        d = limits.to_dict()
        
        assert d["daily_loss_limit"] == 20.0
        assert d["risk_level"] == "normal"


class TestAdaptiveRiskManager:
    """Tests for AdaptiveRiskManager."""

    def test_initialization(self):
        """Test initialization."""
        mgr = AdaptiveRiskManager(base_daily_limit=20.0, base_max_stake=5.0)
        
        assert mgr.base_daily_limit == 20.0
        assert mgr.base_max_stake == 5.0

    def test_record_trade(self):
        """Test recording trades."""
        mgr = AdaptiveRiskManager()
        
        mgr.record_trade(5.0)
        mgr.record_trade(-3.0)
        
        assert mgr._daily_pnl == 2.0

    def test_get_adjusted_limits_normal(self):
        """Test normal adjusted limits."""
        mgr = AdaptiveRiskManager(base_daily_limit=20.0, base_max_stake=5.0)
        
        limits = mgr.get_adjusted_limits(trust_score=0.8)
        
        assert limits.risk_level == RiskLevel.NORMAL
        assert limits.daily_loss_limit > 0

    def test_conservative_on_low_trust(self):
        """Test conservative mode on low trust."""
        mgr = AdaptiveRiskManager(base_daily_limit=20.0)
        
        limits = mgr.get_adjusted_limits(trust_score=0.3)
        
        assert limits.risk_level == RiskLevel.CONSERVATIVE
        assert limits.daily_loss_limit < 20.0

    def test_conservative_on_drawdown(self):
        """Test conservative mode during drawdown."""
        mgr = AdaptiveRiskManager(drawdown_threshold=0.1)
        
        # Simulate drawdown
        mgr.performance.record(100.0, current_equity=1000.0)
        mgr.performance.record(-200.0, current_equity=800.0)  # 20% drawdown
        
        limits = mgr.get_adjusted_limits(trust_score=0.8)
        
        assert limits.risk_level == RiskLevel.CONSERVATIVE

    def test_should_pause_daily_limit(self):
        """Test pause when daily limit hit."""
        mgr = AdaptiveRiskManager(base_daily_limit=20.0)
        
        # Hit daily limit
        mgr.record_trade(-25.0)
        
        assert mgr.should_pause() is True

    def test_reset_daily(self):
        """Test daily reset."""
        mgr = AdaptiveRiskManager()
        
        mgr.record_trade(-10.0)
        mgr.reset_daily()
        
        assert mgr._daily_pnl == 0.0

    def test_pause_until(self):
        """Test explicit pause."""
        mgr = AdaptiveRiskManager()
        
        future = datetime.now(timezone.utc) + timedelta(hours=1)
        mgr.pause_until(future)
        
        assert mgr.should_pause() is True

    def test_volatility_regime_adjustment(self):
        """Test volatility regime affects limits."""
        mgr = AdaptiveRiskManager(base_max_stake=10.0)
        
        limits_low = mgr.get_adjusted_limits(trust_score=0.8, volatility_regime="low")
        limits_high = mgr.get_adjusted_limits(trust_score=0.8, volatility_regime="high")
        
        # Low vol should have higher limits than high vol
        assert limits_low.max_stake > limits_high.max_stake

    def test_get_statistics(self):
        """Test statistics retrieval."""
        mgr = AdaptiveRiskManager()
        
        mgr.record_trade(5.0)
        
        stats = mgr.get_statistics()
        
        assert "daily_pnl" in stats
        assert "sharpe_ratio" in stats
        assert stats["daily_pnl"] == 5.0
