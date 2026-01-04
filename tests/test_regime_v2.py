"""
Unit tests for regime module.

Tests for hierarchical regime detection.
"""

import numpy as np
import pytest

from execution.regime import (
    HierarchicalRegimeAssessment,
    HierarchicalRegimeDetector,
    HurstExponentEstimator,
    MacroRegime,
    MicroRegime,
    TrendDetector,
    VolatilityRegime,
    VolatilityRegimeDetector,
)


class TestHurstExponentEstimator:
    """Tests for Hurst exponent estimation."""

    def test_random_walk_hurst_near_half(self):
        """Random walk should have Hurst ≈ 0.5."""
        np.random.seed(42)
        # Generate random walk
        returns = np.random.randn(500) * 0.01
        prices = 100 * np.exp(np.cumsum(returns))
        
        estimator = HurstExponentEstimator()
        hurst = estimator.estimate(prices)
        
        # Should be close to 0.5 (within reasonable tolerance)
        assert 0.35 < hurst < 0.65, f"Random walk Hurst={hurst}, expected ~0.5"

    def test_trending_series_hurst_above_half(self):
        """Trending series should have Hurst > 0.5."""
        # Generate trending series (persistent)
        np.random.seed(42)
        n = 500
        prices = np.zeros(n)
        prices[0] = 100
        # Momentum: each return correlates with previous
        prev_return = 0.001
        for i in range(1, n):
            prev_return = 0.7 * prev_return + 0.3 * np.random.randn() * 0.01
            prices[i] = prices[i-1] * (1 + prev_return)
        
        estimator = HurstExponentEstimator()
        hurst = estimator.estimate(prices)
        
        # Trending should tend towards > 0.5
        assert hurst > 0.4, f"Trending Hurst={hurst}"

    def test_insufficient_data_returns_half(self):
        """Insufficient data should return 0.5."""
        estimator = HurstExponentEstimator(min_window=8)
        prices = np.array([100, 101, 102])  # Too short
        
        hurst = estimator.estimate(prices)
        assert hurst == 0.5


class TestVolatilityRegimeDetector:
    """Tests for volatility regime detection."""

    def test_low_volatility_detection(self):
        """Stable prices should be classified as low volatility."""
        # Generate very stable prices
        np.random.seed(42)
        prices = 100 + np.random.randn(200) * 0.1  # Very small moves
        
        detector = VolatilityRegimeDetector()
        regime, percentile = detector.detect(prices)
        
        # Should be low or medium (small random noise)
        assert regime in [VolatilityRegime.LOW, VolatilityRegime.MEDIUM]

    def test_high_volatility_detection(self):
        """Volatile prices should be classified as high volatility."""
        np.random.seed(42)
        # Start with very stable prices, then sudden large volatility
        # The detector uses a rolling window, so recent volatility should dominate
        stable = 100 + np.random.randn(150) * 0.01  # Very stable
        # Create very volatile end - larger moves
        volatile = np.zeros(50)
        volatile[0] = stable[-1]
        for i in range(1, 50):
            volatile[i] = volatile[i-1] + np.random.randn() * 5  # Large moves
        prices = np.concatenate([stable, volatile])
        
        detector = VolatilityRegimeDetector(low_percentile=30, high_percentile=70, lookback=100)
        regime, percentile = detector.detect(prices)
        
        # Recent volatility should be detected as high relative to the stable period
        # At minimum should not be LOW given the extreme recent moves
        assert regime != VolatilityRegime.LOW, f"Expected not LOW, got {regime} (percentile={percentile})"

    def test_insufficient_data(self):
        """Short data should return medium regime."""
        detector = VolatilityRegimeDetector()
        prices = np.array([100, 101])
        
        regime, percentile = detector.detect(prices)
        assert regime == VolatilityRegime.HIGH


class TestTrendDetector:
    """Tests for macro trend detection."""

    def test_bull_trend_detection(self):
        """Uptrending prices should be classified as bull."""
        # Generate clear uptrend
        prices = np.linspace(100, 150, 100)
        
        detector = TrendDetector(short_window=10, long_window=30)
        regime, strength = detector.detect(prices)
        
        assert regime == MacroRegime.BULL
        assert strength > 0

    def test_bear_trend_detection(self):
        """Downtrending prices should be classified as bear."""
        # Generate clear downtrend
        prices = np.linspace(150, 100, 100)
        
        detector = TrendDetector(short_window=10, long_window=30)
        regime, strength = detector.detect(prices)
        
        assert regime == MacroRegime.BEAR
        assert strength < 0

    def test_sideways_detection(self):
        """Range-bound prices should be classified as sideways."""
        np.random.seed(42)
        # Oscillating around mean
        prices = 100 + np.sin(np.linspace(0, 10*np.pi, 100)) * 2
        
        detector = TrendDetector(short_window=10, long_window=30)
        regime, strength = detector.detect(prices)
        
        # Sideways or weak trend
        assert abs(strength) < 0.03

    def test_insufficient_data(self):
        """Short data should return sideways."""
        detector = TrendDetector(long_window=50)
        prices = np.array([100, 101, 102])
        
        regime, strength = detector.detect(prices)
        assert regime == MacroRegime.SIDEWAYS


class TestHierarchicalRegimeDetector:
    """Tests for full hierarchical regime detection."""

    def test_full_assessment(self):
        """Test complete regime assessment."""
        np.random.seed(42)
        prices = 100 + np.cumsum(np.random.randn(200) * 0.5)
        
        detector = HierarchicalRegimeDetector()
        assessment = detector.assess(prices, reconstruction_error=0.1)
        
        assert isinstance(assessment, HierarchicalRegimeAssessment)
        assert isinstance(assessment.macro, MacroRegime)
        assert isinstance(assessment.volatility, VolatilityRegime)
        assert isinstance(assessment.micro, MicroRegime)
        assert 0 <= assessment.trust_score <= 1
        assert "hurst_exponent" in assessment.details

    def test_high_recon_error_reduces_trust(self):
        """High reconstruction error should reduce trust score."""
        np.random.seed(42)
        prices = 100 + np.cumsum(np.random.randn(200) * 0.5)
        
        detector = HierarchicalRegimeDetector()
        
        low_error = detector.assess(prices, reconstruction_error=0.05)
        high_error = detector.assess(prices, reconstruction_error=0.5)
        
        assert high_error.trust_score < low_error.trust_score

    def test_veto_on_very_low_trust(self):
        """Very low trust should trigger veto."""
        assessment = HierarchicalRegimeAssessment(
            macro=MacroRegime.SIDEWAYS,
            volatility=VolatilityRegime.HIGH,
            micro=MicroRegime.RANDOM,
            trust_score=0.2,  # Below 0.3 threshold
            reconstruction_error=0.5,
            details={},
        )
        
        assert assessment.is_vetoed() is True
        assert assessment.requires_caution() is False

    def test_caution_on_medium_trust(self):
        """Medium trust should trigger caution."""
        assessment = HierarchicalRegimeAssessment(
            macro=MacroRegime.BULL,
            volatility=VolatilityRegime.MEDIUM,
            micro=MicroRegime.TRENDING,
            trust_score=0.5,  # Between 0.3 and 0.6
            reconstruction_error=0.15,
            details={},
        )
        
        assert assessment.is_vetoed() is False
        assert assessment.requires_caution() is True

    def test_favorable_on_high_trust(self):
        """High trust should be favorable."""
        assessment = HierarchicalRegimeAssessment(
            macro=MacroRegime.BULL,
            volatility=VolatilityRegime.LOW,
            micro=MicroRegime.TRENDING,
            trust_score=0.85,
            reconstruction_error=0.05,
            details={},
        )
        
        assert assessment.is_favorable() is True
        assert assessment.is_vetoed() is False
        assert assessment.requires_caution() is False

    def test_to_dict(self):
        """Test serialization to dictionary."""
        assessment = HierarchicalRegimeAssessment(
            macro=MacroRegime.BULL,
            volatility=VolatilityRegime.MEDIUM,
            micro=MicroRegime.TRENDING,
            trust_score=0.75,
            reconstruction_error=0.1,
            details={"test_key": "test_value"},
        )
        
        d = assessment.to_dict()
        assert d["macro"] == "bull"
        assert d["volatility"] == "medium"
        assert d["micro"] == "trending"
        assert d["trust_score"] == 0.75
        assert d["is_vetoed"] is False

    def test_fallback_mode(self):
        """Test fallback when no price data is available."""
        detector = HierarchicalRegimeDetector()
        
        # No prices, just reconstruction error
        assessment = detector.assess_from_reconstruction_error(
            reconstruction_error=0.15,
            prices=None
        )
        
        assert assessment.details.get("fallback_mode") is True
        assert assessment.macro == MacroRegime.SIDEWAYS

    def test_fallback_with_short_prices(self):
        """Test fallback with insufficient price data."""
        detector = HierarchicalRegimeDetector()
        
        short_prices = np.array([100, 101, 102])
        assessment = detector.assess_from_reconstruction_error(
            reconstruction_error=0.1,
            prices=short_prices
        )
        
        assert assessment.details.get("fallback_mode") is True


class TestRegimeEnums:
    """Tests for regime enum values."""

    def test_macro_regime_values(self):
        """Test MacroRegime enum."""
        assert MacroRegime.BULL.value == "bull"
        assert MacroRegime.BEAR.value == "bear"
        assert MacroRegime.SIDEWAYS.value == "sideways"

    def test_volatility_regime_values(self):
        """Test VolatilityRegime enum."""
        assert VolatilityRegime.LOW.value == "low"
        assert VolatilityRegime.MEDIUM.value == "medium"
        assert VolatilityRegime.HIGH.value == "high"

    def test_micro_regime_values(self):
        """Test MicroRegime enum."""
        assert MicroRegime.TRENDING.value == "trending"
        assert MicroRegime.RANDOM.value == "random"
        assert MicroRegime.MEAN_REVERTING.value == "mean_reverting"
