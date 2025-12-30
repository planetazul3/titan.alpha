"""
Unit tests for execution module.

Tests decision engine, signals, and filters.
"""

from datetime import datetime, timezone
from unittest.mock import MagicMock

import pytest

from config.constants import CONTRACT_TYPES, SIGNAL_TYPES


class TestTradeSignal:
    """Tests for TradeSignal dataclass."""

    def test_signal_creation(self):
        """Basic signal creation should work."""
        from execution.signals import TradeSignal

        signal = TradeSignal(
            signal_type=SIGNAL_TYPES.REAL_TRADE,
            contract_type=CONTRACT_TYPES.RISE_FALL,
            direction="CALL",
            probability=0.85,
            timestamp=datetime.now(timezone.utc),
        )

        assert signal.signal_type == SIGNAL_TYPES.REAL_TRADE
        assert signal.probability == 0.85

    def test_signal_to_dict(self):
        """Signal serialization should work."""
        from execution.signals import TradeSignal

        now = datetime.now(timezone.utc)
        signal = TradeSignal(
            signal_type=SIGNAL_TYPES.REAL_TRADE,
            contract_type=CONTRACT_TYPES.RISE_FALL,
            direction="CALL",
            probability=0.85,
            timestamp=now,
        )

        d = signal.to_dict()

        assert isinstance(d["timestamp"], str)
        assert d["probability"] == 0.85

    def test_signal_from_dict(self):
        """Signal deserialization should work."""
        from execution.signals import TradeSignal

        d = {
            "signal_type": SIGNAL_TYPES.REAL_TRADE,
            "contract_type": CONTRACT_TYPES.RISE_FALL,
            "direction": "CALL",
            "probability": 0.85,
            "timestamp": "2024-01-01T12:00:00+00:00",
            "metadata": {},
        }

        signal = TradeSignal.from_dict(d)

        assert signal.probability == 0.85
        assert isinstance(signal.timestamp, datetime)


class TestShadowTrade:
    """Tests for ShadowTrade dataclass."""

    def test_shadow_trade_creation(self):
        """Shadow trade creation with auto-generated ID."""
        from execution.signals import ShadowTrade

        trade = ShadowTrade(
            signal_type=SIGNAL_TYPES.SHADOW_TRADE,
            contract_type=CONTRACT_TYPES.RISE_FALL,
            direction="CALL",
            probability=0.65,
            timestamp=datetime.now(timezone.utc),
        )

        assert trade.trade_id is not None
        assert len(trade.trade_id) > 0

    def test_shadow_trade_update_outcome_win(self):
        """Update outcome for winning trade."""
        from execution.signals import ShadowTrade

        trade = ShadowTrade(
            signal_type=SIGNAL_TYPES.SHADOW_TRADE,
            contract_type=CONTRACT_TYPES.RISE_FALL,
            direction="CALL",
            probability=0.65,
            timestamp=datetime.now(timezone.utc),
        )

        trade = trade.with_outcome(outcome=True, exit_price=1.05, stake=10.0, payout=0.95)

        assert trade.outcome is True
        assert trade.pnl == 9.5  # 10 * 0.95

    def test_shadow_trade_update_outcome_loss(self):
        """Update outcome for losing trade."""
        from execution.signals import ShadowTrade

        trade = ShadowTrade(
            signal_type=SIGNAL_TYPES.SHADOW_TRADE,
            contract_type=CONTRACT_TYPES.RISE_FALL,
            direction="CALL",
            probability=0.65,
            timestamp=datetime.now(timezone.utc),
        )

        trade = trade.with_outcome(outcome=False, exit_price=0.95, stake=10.0)

        assert trade.outcome is False
        assert trade.pnl == -10.0


class TestDecisionEngine:
    """Tests for DecisionEngine."""

    @pytest.fixture
    def mock_settings(self):
        """Create real settings for decision engine."""
        from config.settings import Settings, Thresholds, ModelHyperparams, Trading
        thresholds = Thresholds.model_construct(
            confidence_threshold_high=0.75,
            learning_threshold_min=0.40,
            learning_threshold_max=0.60
        )
        hyperparams = ModelHyperparams.model_construct(
            regime_caution_threshold=0.1,
            regime_veto_threshold=0.3
        )
        trading = Trading.model_construct(symbol="R_100", stake_amount=10.0)
        return Settings.model_construct(
            thresholds=thresholds,
            hyperparams=hyperparams,
            trading=trading,
            environment="development"
        )

    def test_engine_initialization(self, mock_settings):
        """Engine should initialize with zero stats."""
        from execution.decision import DecisionEngine

        engine = DecisionEngine(mock_settings)
        stats = engine.get_statistics()

        assert stats["processed"] == 0
        assert stats["real"] == 0
        assert stats["shadow"] == 0




class TestFilters:
    """Tests for signal filtering."""

    @pytest.fixture
    def mock_settings(self):
        """Create real settings for filters."""
        from config.settings import Settings, Thresholds, Trading
        thresholds = Thresholds.model_construct(
            confidence_threshold_high=0.75,
            learning_threshold_min=0.40,
            learning_threshold_max=0.60
        )
        trading = Trading.model_construct(symbol="R_100", stake_amount=10.0)
        return Settings.model_construct(
            thresholds=thresholds,
            trading=trading,
            environment="development"
        )

    def test_high_confidence_becomes_real_trade(self, mock_settings):
        """High confidence signals should be REAL_TRADE."""
        from execution.filters import filter_signals

        probs = {"rise_fall_prob": 0.85, "touch_prob": 0.50, "range_prob": 0.50}
        timestamp = datetime.now(timezone.utc)

        signals = filter_signals(probs, mock_settings, timestamp)

        real_trades = [s for s in signals if s.signal_type == SIGNAL_TYPES.REAL_TRADE]
        assert len(real_trades) >= 1

    def test_mid_confidence_becomes_shadow(self, mock_settings):
        """Mid confidence signals should be SHADOW_TRADE."""
        from execution.filters import filter_signals

        probs = {
            "rise_fall_prob": 0.50,  # In shadow range
            "touch_prob": 0.30,
            "range_prob": 0.30,
        }
        timestamp = datetime.now(timezone.utc)

        signals = filter_signals(probs, mock_settings, timestamp)

        shadow_trades = [s for s in signals if s.signal_type == SIGNAL_TYPES.SHADOW_TRADE]
        assert len(shadow_trades) >= 1

    def test_low_confidence_ignored(self, mock_settings):
        """Low confidence signals should be classified as IGNORE."""
        from execution.filters import filter_signals, get_actionable_signals

        # Use 0.68: above shadow range (0.60) but below real trade (0.75)
        # CALL: 0.68 -> IGNORE (not in 0.40-0.60 shadow range, not >= 0.75 real)
        # PUT: 0.32 -> IGNORE (not in 0.40-0.60 shadow range, not >= 0.75 real)
        probs = {
            "rise_fall_prob": 0.68,
            "touch_prob": 0.68,
            "range_prob": 0.68,
        }
        timestamp = datetime.now(timezone.utc)

        signals = filter_signals(probs, mock_settings, timestamp)

        # All signals should be classified as IGNORE
        ignored = [s for s in signals if s.signal_type == SIGNAL_TYPES.IGNORE]
        assert len(ignored) > 0, "Mid-gap confidence should produce IGNORE signals"

        # get_actionable_signals should filter out IGNORE signals
        real_trades, shadow_trades = get_actionable_signals(signals)
        assert len(real_trades) == 0, "No real trades in ignore zone"
        assert len(shadow_trades) == 0, "No shadow trades in ignore zone"


class TestRegimeVeto:
    """Tests for RegimeVeto authority."""

    def test_regime_veto_initialization(self):
        """RegimeVeto should initialize with valid thresholds."""
        from execution.regime import RegimeVeto

        veto = RegimeVeto(threshold_caution=0.1, threshold_veto=0.3)

        assert veto.threshold_caution == 0.1
        assert veto.threshold_veto == 0.3

    def test_regime_veto_invalid_thresholds_raises(self):
        """Should raise if caution >= veto threshold."""
        from execution.regime import RegimeVeto

        with pytest.raises(ValueError, match="must be <"):
            RegimeVeto(threshold_caution=0.5, threshold_veto=0.3)

    def test_regime_trusted_state(self):
        """Low reconstruction error should return TRUSTED state."""
        import torch

        from execution.regime import RegimeVeto
        from execution.common.types import TrustState

        veto = RegimeVeto(threshold_caution=0.1, threshold_veto=0.3)
        assessment = veto.assess(torch.tensor(0.05))

        assert assessment.trust_state == TrustState.TRUSTED
        assert not assessment.is_vetoed()
        assert not assessment.requires_caution()

    def test_regime_caution_state(self):
        """Medium reconstruction error should return CAUTION state."""
        import torch

        from execution.regime import RegimeVeto
        from execution.common.types import TrustState

        veto = RegimeVeto(threshold_caution=0.1, threshold_veto=0.3)
        assessment = veto.assess(torch.tensor(0.15))

        assert assessment.trust_state == TrustState.CAUTION
        assert not assessment.is_vetoed()
        assert assessment.requires_caution()

    def test_regime_veto_state(self):
        """High reconstruction error should return VETO state."""
        import torch

        from execution.regime import RegimeVeto
        from execution.common.types import TrustState

        veto = RegimeVeto(threshold_caution=0.1, threshold_veto=0.3)
        assessment = veto.assess(torch.tensor(0.5))

        assert assessment.trust_state == TrustState.VETO
        assert assessment.is_vetoed()
        assert not assessment.requires_caution()

    @pytest.mark.asyncio
    async def test_regime_veto_blocks_high_confidence_trades(self):
        """
        CRITICAL TEST: Regime veto MUST block trades even with 99% confidence.

        This is the absolute authority test - no probability threshold
        can override the regime veto.
        """
        from execution.decision import DecisionEngine
        from execution.regime import RegimeVeto
        from config.settings import Settings, Thresholds, ModelHyperparams, Trading
        thresholds = Thresholds.model_construct(
            confidence_threshold_high=0.75,
            learning_threshold_min=0.40,
            learning_threshold_max=0.60
        )
        hyperparams = ModelHyperparams.model_construct(
            regime_caution_threshold=0.1,
            regime_veto_threshold=0.3
        )
        trading = Trading.model_construct(symbol="R_100", stake_amount=10.0)
        settings = Settings.model_construct(
            thresholds=thresholds,
            hyperparams=hyperparams,
            trading=trading,
            environment="development"
        )

        regime_veto = RegimeVeto(threshold_caution=0.1, threshold_veto=0.3)
        engine = DecisionEngine(settings, regime_veto=regime_veto)

        # 99% confidence - would normally be a real trade
        probs = {"rise_fall_prob": 0.99, "touch_prob": 0.99, "range_prob": 0.99}

        # But reconstruction error is above veto threshold
        reconstruction_error = 0.5  # Way above 0.3 threshold

        trades = await engine.process_model_output(probs, reconstruction_error)

        # ABSOLUTE: No trades allowed, even with 99% confidence
        assert len(trades) == 0
        assert engine.get_statistics()["regime_vetoed"] == 1


class TestTradeExecutor:
    """Tests for TradeExecutor abstraction."""

    def test_mock_executor_records_signals(self):
        """MockTradeExecutor should record signals without executing."""
        import asyncio

        from execution.executor import MockTradeExecutor
        from execution.signals import TradeSignal

        executor = MockTradeExecutor()

        signal = TradeSignal(
            signal_type=SIGNAL_TYPES.REAL_TRADE,
            contract_type=CONTRACT_TYPES.RISE_FALL,
            direction="CALL",
            probability=0.85,
            timestamp=datetime.now(timezone.utc),
        )

        result = asyncio.run(executor.execute(signal))

        assert result.success is True
        assert result.contract_id.startswith("MOCK_")
        assert len(executor.get_signals()) == 1

    def test_trade_result_defaults(self):
        """TradeResult should have sensible defaults."""
        from execution.executor import TradeResult

        result = TradeResult(success=True)

        assert result.success is True
        assert result.contract_id is None
        assert result.error is None
        assert result.timestamp is not None
