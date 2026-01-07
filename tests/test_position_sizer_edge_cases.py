"""
Edge case tests for position sizing (REC-004).

Tests threshold boundaries and edge cases for position sizers:
- Kelly fraction at exact breakeven threshold
- Probability boundaries (0, 0.5, 1)
- Stake clamping at min/max boundaries
- Zero/negative edge handling
"""

import pytest
from execution.position_sizer import (
    KellyPositionSizer,
    PositionSizeResult,
)


class TestKellyPositionSizerEdgeCases:
    """Edge case tests for Kelly position sizer thresholds."""

    def test_probability_at_breakeven_threshold(self):
        """
        REC-004: Test Kelly fraction at exact breakeven probability.
        
        For payout_ratio = 0.9, breakeven is p = 1 / (1 + 0.9) ≈ 0.5263
        Kelly should return 0 or very small positive at this threshold.
        """
        sizer = KellyPositionSizer(base_stake=1.0, safety_factor=0.5)
        
        # Breakeven probability for 90% payout: p = 1 / (1 + b) = 1 / 1.9 ≈ 0.5263
        breakeven_prob = 1 / (1 + 0.9)
        
        result = sizer.compute_kelly_fraction(breakeven_prob, payout_ratio=0.9)
        
        # At breakeven, Kelly fraction should be ~0
        assert abs(result) < 0.01, f"Expected ~0 at breakeven, got {result}"

    def test_probability_just_below_breakeven(self):
        """Kelly should be negative (no bet) just below breakeven."""
        sizer = KellyPositionSizer(base_stake=1.0, safety_factor=0.5)
        
        breakeven_prob = 1 / (1 + 0.9)
        below_breakeven = breakeven_prob - 0.05
        
        result = sizer.compute_kelly_fraction(below_breakeven, payout_ratio=0.9)
        
        # Below breakeven = negative edge = negative Kelly
        assert result < 0, f"Expected negative Kelly below breakeven, got {result}"

    def test_probability_just_above_breakeven(self):
        """Kelly should be positive (bet) just above breakeven."""
        sizer = KellyPositionSizer(base_stake=1.0, safety_factor=0.5)
        
        breakeven_prob = 1 / (1 + 0.9)
        above_breakeven = breakeven_prob + 0.05
        
        result = sizer.compute_kelly_fraction(above_breakeven, payout_ratio=0.9)
        
        # Above breakeven = positive edge = positive Kelly
        assert result > 0, f"Expected positive Kelly above breakeven, got {result}"

    def test_probability_boundary_zero(self):
        """Probability of 0% win should give max negative Kelly."""
        sizer = KellyPositionSizer(base_stake=1.0, safety_factor=0.5)
        
        result = sizer.compute_kelly_fraction(0.0, payout_ratio=0.9)
        
        # 0% win probability = guaranteed loss = negative fraction
        assert result < 0

    def test_probability_boundary_one(self):
        """Probability of 100% win should give max positive Kelly."""
        sizer = KellyPositionSizer(base_stake=1.0, safety_factor=0.5)
        
        result = sizer.compute_kelly_fraction(1.0, payout_ratio=0.9)
        
        # 100% win probability = bet everything = fraction = 1.0
        assert result == pytest.approx(1.0), f"100% prob should give Kelly=1, got {result}"

    def test_probability_exactly_fifty_percent(self):
        """At 50% with 0.9 payout, should have zero or negative edge."""
        sizer = KellyPositionSizer(base_stake=1.0, safety_factor=0.5)
        
        # 50% win with 90% payout = expected value = 0.5 * 0.9 - 0.5 * 1.0 = -0.05 (losing)
        result = sizer.compute_kelly_fraction(0.5, payout_ratio=0.9)
        
        assert result < 0, f"50% with 0.9 payout should be negative edge, got {result}"

    def test_stake_clamped_to_max(self):
        """High probability should clamp stake to max_stake."""
        sizer = KellyPositionSizer(
            base_stake=100.0, 
            safety_factor=1.0,  # Full Kelly
            max_stake=10.0,
            min_stake=0.35
        )
        
        result = sizer.compute_stake(probability=0.95, payout_ratio=0.9)
        
        assert result.stake <= 10.0, f"Stake should be clamped to max, got {result.stake}"
        assert result.stake == 10.0, "With very high prob, should hit max"

    def test_stake_clamped_to_min_when_positive_edge(self):
        """Low but positive edge should clamp stake to min_stake."""
        sizer = KellyPositionSizer(
            base_stake=1.0, 
            safety_factor=0.5,
            max_stake=100.0,
            min_stake=0.35
        )
        
        # Just above breakeven - small positive edge
        breakeven_prob = 1 / (1 + 0.9)
        result = sizer.compute_stake(probability=breakeven_prob + 0.01, payout_ratio=0.9)
        
        # Should be clamped to min_stake due to small edge
        assert result.stake >= 0.35, "Should not go below min_stake"

    def test_zero_stake_returned_for_negative_edge(self):
        """Negative edge (below breakeven) should return zero stake."""
        sizer = KellyPositionSizer(base_stake=10.0, safety_factor=0.5)
        
        # 40% probability with 0.9 payout = negative edge
        result = sizer.compute_stake(probability=0.40, payout_ratio=0.9)
        
        assert result.stake == 0.0, f"Negative edge should give zero stake, got {result.stake}"
        assert "negative" in result.reason.lower() or "no bet" in result.reason.lower() or "zero" in result.reason.lower()

    def test_initialization_rejects_invalid_safety_factor(self):
        """Safety factor > 1 should raise ValueError."""
        with pytest.raises(ValueError, match="safety_factor"):
            KellyPositionSizer(base_stake=1.0, safety_factor=1.5)

    def test_initialization_rejects_zero_base_stake(self):
        """Zero base stake should raise ValueError."""
        with pytest.raises(ValueError, match="base_stake"):
            KellyPositionSizer(base_stake=0.0, safety_factor=0.5)

    def test_initialization_rejects_max_less_than_min(self):
        """max_stake < min_stake should raise ValueError."""
        with pytest.raises(ValueError, match="max_stake"):
            KellyPositionSizer(base_stake=1.0, max_stake=0.1, min_stake=1.0)
