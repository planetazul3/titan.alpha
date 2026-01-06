"""
Unit tests for execution/integration.py module.
"""

from datetime import datetime, timezone

import numpy as np
import pytest

from execution.integration import (
    EnhancedRegimeVeto,
    RegimeAwarePositionSizer,
)
from execution.regime import (
    HierarchicalRegimeAssessment,
    HierarchicalRegimeDetector,
    MacroRegime,
    MicroRegime,
    VolatilityRegime,
)
from execution.signals import TradeSignal


def _make_signal(probability: float = 0.65) -> TradeSignal:
    """Helper to create TradeSignal with required fields."""
    from execution.signals import SIGNAL_TYPES
    return TradeSignal(
        signal_type=SIGNAL_TYPES.REAL_TRADE,
        contract_type="rise_fall",
        direction="CALL",
        probability=probability,
        timestamp=datetime.now(timezone.utc),
    )


class TestRegimeAwarePositionSizer:
    """Tests for RegimeAwarePositionSizer."""

    def test_initialization(self):
        """Test initialization."""
        sizer = RegimeAwarePositionSizer(base_stake=2.0, safety_factor=0.5)
        assert sizer.kelly_sizer.base_stake == 2.0
        assert sizer.kelly_sizer.safety_factor == 0.5

    def test_compute_stake_without_regime(self):
        """Test stake computation without regime assessment."""
        sizer = RegimeAwarePositionSizer(base_stake=10.0)
        
        signal = _make_signal(probability=0.65)
        
        result = sizer.compute_stake_for_signal(signal)
        assert result.stake > 0

    def test_compute_stake_with_favorable_regime(self):
        """Test stake with favorable regime assessment."""
        sizer = RegimeAwarePositionSizer(base_stake=10.0, max_stake=50.0)
        
        signal = _make_signal(probability=0.65)
        
        favorable_regime = HierarchicalRegimeAssessment(
            macro=MacroRegime.BULL,
            volatility=VolatilityRegime.LOW,
            micro=MicroRegime.TRENDING,
            trust_score=0.9,
            reconstruction_error=0.05,
            details={},
        )
        
        result = sizer.compute_stake_for_signal(signal, regime_assessment=favorable_regime)
        assert result.stake > 0
        assert result.confidence_multiplier == 0.9

    def test_compute_stake_with_risky_regime(self):
        """Test stake with risky regime assessment reduces size."""
        sizer = RegimeAwarePositionSizer(base_stake=10.0, max_stake=50.0)
        
        signal = _make_signal(probability=0.65)
        
        risky_regime = HierarchicalRegimeAssessment(
            macro=MacroRegime.SIDEWAYS,
            volatility=VolatilityRegime.HIGH,
            micro=MicroRegime.RANDOM,
            trust_score=0.4,
            reconstruction_error=0.25,
            details={},
        )
        
        favorable_regime = HierarchicalRegimeAssessment(
            macro=MacroRegime.BULL,
            volatility=VolatilityRegime.LOW,
            micro=MicroRegime.TRENDING,
            trust_score=0.9,
            reconstruction_error=0.05,
            details={},
        )
        
        result_risky = sizer.compute_stake_for_signal(signal, regime_assessment=risky_regime)
        result_favorable = sizer.compute_stake_for_signal(signal, regime_assessment=favorable_regime)
        
        # Risky regime should produce smaller stake
        assert result_risky.stake <= result_favorable.stake

    def test_create_stake_resolver(self):
        """Test stake resolver creation."""
        sizer = RegimeAwarePositionSizer(base_stake=5.0)
        
        def get_regime():
            return HierarchicalRegimeAssessment(
                macro=MacroRegime.BULL,
                volatility=VolatilityRegime.MEDIUM,
                micro=MicroRegime.TRENDING,
                trust_score=0.7,
                reconstruction_error=0.1,
                details={},
            )
        
        def get_drawdown():
            return 0.05
        
        resolver = sizer.create_stake_resolver(
            get_regime_assessment=get_regime,
            get_drawdown=get_drawdown,
        )
        
        signal = _make_signal(probability=0.65)
        
        stake = resolver(signal)
        assert isinstance(stake, float)
        assert stake >= 0


class TestEnhancedRegimeVeto:
    """Tests for EnhancedRegimeVeto adapter."""

    def test_initialization(self):
        """Test initialization with defaults."""
        veto = EnhancedRegimeVeto()
        # In unified model, these attributes are on the base class now
        assert veto.hierarchical_detector is not None
        assert veto.use_hierarchical is True

    def test_assess_with_price_cache(self):
        """Test assessment with cached prices."""
        veto = EnhancedRegimeVeto()
        
        # Cache some prices
        np.random.seed(42)
        prices = 100 + np.cumsum(np.random.randn(100) * 0.5)
        veto.update_prices(prices)
        
        assessment = veto.assess(reconstruction_error=0.1)
        
        assert isinstance(assessment, HierarchicalRegimeAssessment)
        assert 0 <= assessment.trust_score <= 1

    def test_assess_without_price_cache(self):
        """Test fallback when no prices cached."""
        veto = EnhancedRegimeVeto()
        
        # No prices cached
        assessment = veto.assess(reconstruction_error=0.15)
        
        assert isinstance(assessment, HierarchicalRegimeAssessment)
        # Fallback mode uses only reconstruction error
        assert assessment.details.get("fallback_mode") is True

    def test_assess_with_high_recon_error(self):
        """Test high reconstruction error produces low trust."""
        veto = EnhancedRegimeVeto()
        
        assessment = veto.assess(reconstruction_error=0.5)
        
        # High error should mean low trust
        assert assessment.trust_score < 0.5

    def test_threshold_compatibility(self):
        """Test that thresholds are exposed for compatibility."""
        veto = EnhancedRegimeVeto()
        
        # Should have threshold attributes for compatibility
        assert hasattr(veto, "threshold_caution")
        assert hasattr(veto, "threshold_veto")

    def test_get_hierarchical_assessment(self):
        """Test direct hierarchical assessment via hierarchical_detector."""
        veto = EnhancedRegimeVeto()
        
        np.random.seed(42)
        prices = 100 + np.cumsum(np.random.randn(100) * 0.5)
        
        # Use hierarchical_detector directly as get_hierarchical_assessment was removed
        assessment = veto.hierarchical_detector.assess(
            prices=prices,
            reconstruction_error=0.1,
        )
        
        assert isinstance(assessment, HierarchicalRegimeAssessment)
        assert "hurst_exponent" in assessment.details


class TestIntegrationFlow:
    """Integration tests for combined components."""

    def test_full_flow(self):
        """Test full flow from signal to stake."""
        # Create components
        regime_veto = EnhancedRegimeVeto()
        position_sizer = RegimeAwarePositionSizer(base_stake=10.0)
        
        # Simulate market data
        np.random.seed(42)
        prices = 100 + np.cumsum(np.random.randn(100) * 0.5)
        regime_veto.update_prices(prices)
        
        # Get regime assessment
        assessment = regime_veto.assess(reconstruction_error=0.1)
        
        # Create signal
        signal = _make_signal(probability=0.65)
        
        # Compute stake
        result = position_sizer.compute_stake_for_signal(
            signal=signal,
            regime_assessment=assessment,
            current_drawdown=0.05,
        )
        
        assert result.stake > 0
        assert result.confidence_multiplier == assessment.trust_score
