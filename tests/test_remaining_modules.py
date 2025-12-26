"""
Unit tests for temporal_v2, decision_v2, and rl_integration.
"""

import numpy as np
import pytest
import torch

from models.tft import MultiHorizonHead
from execution.rl_integration import RLDecision, RLTradingIntegration


class TestMultiHorizonHead:
    """Tests for MultiHorizonHead."""

    def test_output_shape(self):
        """Test multi-horizon output shapes."""
        head = MultiHorizonHead(
            hidden_size=64,
            horizons=[1, 5, 10],
            output_size=1,
        )
        # Sequence input [batch, seq, hidden]
        hidden = torch.randn(4, 20, 64)
        
        predictions = head(hidden)
        
        assert 1 in predictions
        assert 5 in predictions
        assert 10 in predictions
        # Should preserve sequence dimension
        assert predictions[1].shape == (4, 20, 1, 1)  # batch, seq, output, quantiles

    def test_quantile_output(self):
        """Test quantile output shape."""
        head = MultiHorizonHead(
            hidden_size=32,
            horizons=[1],
            quantiles=[0.1, 0.5, 0.9],
        )
        hidden = torch.randn(2, 10, 32)
        
        predictions = head(hidden)
        
        assert predictions[1].shape == (2, 10, 1, 3)  # 3 quantiles

    def test_point_predictions(self):
        """Test getting point predictions."""
        head = MultiHorizonHead(hidden_size=32, horizons=[1, 5])
        hidden = torch.randn(2, 10, 32)
        
        point_preds = head.get_point_predictions(hidden)
        
        assert 1 in point_preds
        assert point_preds[1].shape == (2, 10, 1)




class TestRLDecision:
    """Tests for RLDecision."""

    def test_creation(self):
        """Test decision creation."""
        decision = RLDecision(
            recommended_stake=5.0,
            confidence=0.8,
            action_raw=0.5,
            state_embedding=[0.1] * 11,
            use_rl_sizing=True,
        )
        
        assert decision.recommended_stake == 5.0
        assert decision.use_rl_sizing is True


class TestRLTradingIntegration:
    """Tests for RLTradingIntegration."""

    @pytest.fixture
    def rl_integration(self):
        """Create RL integration for tests."""
        from models.policy import TradingActor
        actor = TradingActor(state_dim=11, action_dim=1)
        return RLTradingIntegration(actor)

    def test_initialization(self, rl_integration):
        """Test initialization."""
        assert rl_integration.max_stake == 10.0
        assert rl_integration.min_confidence_for_rl == 0.6

    def test_record_outcome(self, rl_integration):
        """Test recording outcomes."""
        decision = RLDecision(
            recommended_stake=5.0,
            confidence=0.8,
            action_raw=0.5,
            state_embedding=[0.1] * 11,
            use_rl_sizing=True,
        )
        
        rl_integration.record_outcome(decision, pnl=5.0, won=True)
        
        experiences = rl_integration.get_experiences()
        assert len(experiences) == 1

    def test_statistics(self, rl_integration):
        """Test statistics."""
        stats = rl_integration.get_statistics()
        
        assert "experiences" in stats

    def test_get_experiences(self, rl_integration):
        """Test getting experiences with limit."""
        for i in range(5):
            decision = RLDecision(
                recommended_stake=float(i),
                confidence=0.5,
                action_raw=0.0,
                state_embedding=[0.1] * 11,
                use_rl_sizing=False,
            )
            rl_integration.record_outcome(decision, pnl=float(i), won=i % 2 == 0)
        
        all_exp = rl_integration.get_experiences()
        limited = rl_integration.get_experiences(limit=3)
        
        assert len(all_exp) == 5
        assert len(limited) == 3
