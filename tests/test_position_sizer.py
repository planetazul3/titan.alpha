"""
Unit tests for position_sizer module.

Tests for Kelly Criterion and Compounding position sizing strategies.
"""

import pytest

from execution.position_sizer import (
    CompoundingPositionSizer,
    FixedStakeSizer,
    KellyPositionSizer,
    MartingalePositionSizer,
    PositionSizeResult,
)


class TestCompoundingPositionSizer:
    """Tests for the CompoundingPositionSizer."""

    def test_init(self):
        """Test proper initialization."""
        sizer = CompoundingPositionSizer(base_stake=2.0, max_consecutive_wins=3)
        assert sizer.base_stake == 2.0
        assert sizer.max_consecutive_wins == 3
        assert sizer.get_current_streak() == 0

        # Invalid
        with pytest.raises(ValueError):
            CompoundingPositionSizer(base_stake=-1.0)

    def test_basic_progression_reinvest(self):
        """Test stake decreases after wins with profit reinvestment (default)."""
        sizer = CompoundingPositionSizer(base_stake=10.0, max_consecutive_wins=3)
        
        # 1. Trade 1: Base stake
        res = sizer.compute_stake(probability=0.8)
        assert res.stake == 10.0
        
        # Win with $9.5 profit (0.95 payout)
        sizer.record_outcome(pnl=9.5, won=True)
        assert sizer.get_current_streak() == 1
        
        # 2. Trade 2: Base + Profit
        res = sizer.compute_stake(probability=0.8)
        assert res.stake == 19.5
        
        # Win again ($18 profit)
        sizer.record_outcome(pnl=18.0, won=True)
        assert sizer.get_current_streak() == 2
        
        # 3. Trade 3
        res = sizer.compute_stake(probability=0.8)
        assert res.stake == 37.5  # 19.5 + 18

    def test_multiplier_logic(self):
        """Test specific 2x multiplier logic."""
        sizer = CompoundingPositionSizer(
            base_stake=10.0, 
            max_consecutive_wins=3,
            streak_multiplier=2.0
        )
        
        # Trade 1
        assert sizer.compute_stake(0.8).stake == 10.0
        
        # Win 1
        sizer.record_outcome(9.0, True)
        # Trade 2 should be previous * 2.0 = 20.0 (regardless of pnl)
        assert sizer.compute_stake(0.8).stake == 20.0
        
        # Win 2
        sizer.record_outcome(18.0, True)
        # Trade 3 should be 20.0 * 2.0 = 40.0
        assert sizer.compute_stake(0.8).stake == 40.0

    def test_reset_on_loss(self):
        """Test reset to base stake after a loss."""
        sizer = CompoundingPositionSizer(base_stake=10.0)
        
        # Win first
        sizer.record_outcome(9.5, True)
        assert sizer.compute_stake(0.8).stake == 19.5
        
        # Loss
        sizer.record_outcome(-19.5, False)
        
        assert sizer.get_current_streak() == 0
        assert sizer.compute_stake(0.8).stake == 10.0

    def test_reset_on_max_streak(self):
        """Test reset after hitting max streak cap."""
        sizer = CompoundingPositionSizer(base_stake=1.0, max_consecutive_wins=2)
        
        # Win 1
        sizer.record_outcome(0.9, True) 
        assert sizer.get_current_streak() == 1
        
        # Win 2 (Limit hit)
        sizer.record_outcome(1.5, True)
        # Should reset immediately after recording the Nth win
        assert sizer.get_current_streak() == 0
        assert sizer.compute_stake(0.8).stake == 1.0

    def test_low_confidence_reset(self):
        """Test that stake resets if confidence drops below threshold."""
        # Configured to require 0.75 confidence to compound
        sizer = CompoundingPositionSizer(
            base_stake=10.0, 
            min_confidence_to_compound=0.75
        )
        
        # Win 1 (Streak active)
        sizer.record_outcome(9.5, True)
        assert sizer.get_current_streak() == 1
        
        # Next trade only has 0.70 confidence (below threshold)
        # Should safeguard profit and use base stake
        res = sizer.compute_stake(probability=0.70)
        
        assert res.stake == 10.0
        assert sizer.get_current_streak() == 0  # Streak reset

    def test_max_stake_cap(self):
        """Test that absolute max cap is enforced."""
        sizer = CompoundingPositionSizer(
            base_stake=10.0, 
            max_stake_cap=20.0
        )
        
        # Win $50 profit
        sizer.record_outcome(50.0, True)
        # Next potential stake would be 60.0
        
        res = sizer.compute_stake(probability=0.8)
        assert res.stake == 20.0  # Capped
        assert "Capped" in res.reason


class TestMartingalePositionSizer:
    """Tests for Martingale strategy."""

    def test_loss_doubling(self):
        """Test stake doubling on loss."""
        sizer = MartingalePositionSizer(base_stake=10.0, multiplier=2.0)
        
        # Trade 1
        assert sizer.compute_stake().stake == 10.0
        
        # Loss 1
        sizer.record_outcome(-10.0, False)
        # Trade 2 -> 20.0
        assert sizer.compute_stake().stake == 20.0
        
        # Loss 2
        sizer.record_outcome(-20.0, False)
        # Trade 3 -> 40.0
        assert sizer.compute_stake().stake == 40.0
        
        # Win -> Reset
        sizer.record_outcome(36.0, True)
        assert sizer.compute_stake().stake == 10.0

    def test_max_streak_stop_loss(self):
        """Test reset after max loss streak."""
        sizer = MartingalePositionSizer(base_stake=10.0, max_streak=2)
        
        # T1
        assert sizer.compute_stake().stake == 10.0
        # Loss 1
        sizer.record_outcome(-10, False)
        
        # T2
        assert sizer.compute_stake().stake == 20.0
        # Loss 2 (Max hit)
        sizer.record_outcome(-20, False)
        
        # Should reset
        assert sizer.compute_stake().stake == 10.0


class TestKellyPositionSizer:
    """Tests for KellyPositionSizer class."""

    def test_init_valid_params(self):
        """Test valid initialization."""
        sizer = KellyPositionSizer(
            base_stake=1.0,
            safety_factor=0.5,
            max_stake=10.0,
            min_stake=0.35,
        )
        assert sizer.base_stake == 1.0
        
    def test_compute_stake_positive_edge(self):
        """Test stake computation with positive edge."""
        sizer = KellyPositionSizer(
            base_stake=10.0,
            safety_factor=0.5,
            min_stake=0.35,
            max_stake=10.0,
        )
        result = sizer.compute_stake(probability=0.65, payout_ratio=0.9)
        assert result.stake > 0
        assert "Kelly" in result.reason
