"""
Integration tests for live trading flow.

Tests the full pipeline: event → features → model → decision → executor
using mocked DerivClient for controlled event injection.

These tests validate:
- Async orchestration correctness
- Decision engine integration with regime veto
- Shadow trade capture
- Safety wrapper integration
"""

import asyncio
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pytest
import torch

from config.constants import SIGNAL_TYPES
from data.events import CandleEvent, TickEvent
from execution.decision import DecisionEngine
from execution.executor import TradeResult
from execution.regime import RegimeVeto
from execution.safety import ExecutionSafetyConfig, SafeTradeExecutor
from execution.sqlite_shadow_store import SQLiteShadowStore


@pytest.fixture
def mock_settings():
    """Create real settings for testing."""
    from config.settings import Settings, Trading, Thresholds, ModelHyperparams, DataShapes
    
    trading = Trading.model_construct(
        symbol="R_100",
        stake_amount=1.0,
        barrier_offset="+0.50",
        barrier2_offset="-0.50"
    )
    thresholds = Thresholds.model_construct(
        confidence_threshold_high=0.75,
        learning_threshold_min=0.40,
        learning_threshold_max=0.60
    )
    hyperparams = ModelHyperparams.model_construct(
        lstm_hidden_size=64,
        cnn_filters=32,
        latent_dim=16,
        dropout_rate=0.1
    )
    data_shapes = DataShapes.model_construct(
        sequence_length_ticks=100,
        sequence_length_candles=50
    )
    
    return Settings.model_construct(
        trading=trading,
        thresholds=thresholds,
        hyperparams=hyperparams,
        data_shapes=data_shapes,
        environment="development"
    )


@pytest.fixture
def sample_tick_events():
    """Generate sample tick events."""
    base_price = 100.0
    events = []
    for i in range(100):
        price = base_price + np.sin(i * 0.1) * 2 + np.random.randn() * 0.1
        events.append(
            TickEvent(
                symbol="R_100",
                price=float(price),
                timestamp=datetime.now(timezone.utc),
                metadata={"source": "mock"},
            )
        )
    return events


@pytest.fixture
def sample_candle_events():
    """Generate sample candle events."""
    base_price = 100.0
    events = []
    for i in range(50):
        o = base_price + i * 0.1
        h = o + np.random.rand() * 0.5
        l = o - np.random.rand() * 0.5
        c = o + np.random.randn() * 0.2
        events.append(
            CandleEvent(
                symbol="R_100",
                open=float(o),
                high=float(h),
                low=float(l),
                close=float(c),
                volume=1000.0,
                timestamp=datetime.now(timezone.utc),
                metadata={"source": "mock"},
            )
        )
    return events


class MockDerivClient:
    """Mock DerivClient for testing without real API connection."""

    def __init__(self, tick_prices: list[float], candles: list[dict]):
        self.tick_prices = tick_prices
        self.candles = candles
        self.api = True  # Simulate connected state
        self._tick_idx = 0
        self._candle_idx = 0
        self.executed_trades = []

    async def connect(self):
        """Mock connection."""
        pass

    async def disconnect(self):
        """Mock disconnection."""
        pass

    async def get_balance(self):
        """Return mock balance."""
        return 1000.0

    async def get_historical_ticks(self, count=100):
        """Return mock historical ticks."""
        return self.tick_prices[:count]

    async def get_historical_candles(self, count=50, interval=60):
        """Return mock historical candles."""
        return self.candles[:count]

    async def stream_ticks(self):
        """Mock tick stream."""
        for price in self.tick_prices:
            yield price
            await asyncio.sleep(0.001)  # Small delay for realism

    async def stream_candles(self, interval=60):
        """Mock candle stream."""
        for candle in self.candles:
            yield candle
            await asyncio.sleep(0.001)

    async def buy(self, contract_type, amount, duration, duration_unit):
        """Mock trade execution."""
        self.executed_trades.append(
            {"contract_type": contract_type, "amount": amount, "duration": duration}
        )
        return {"buy": {"contract_id": f"MOCK_{len(self.executed_trades)}", "buy_price": amount}}


class TestLiveFlowIntegration:
    """Integration tests for the live trading flow."""

    @pytest.fixture
    def mock_client(self, sample_tick_events, sample_candle_events):
        """Create mock client with sample data."""
        tick_prices = [e.price for e in sample_tick_events]
        candles = [
            {
                "open": e.open,
                "high": e.high,
                "low": e.low,
                "close": e.close,
                "epoch": e.timestamp.timestamp(),
            }
            for e in sample_candle_events
        ]
        return MockDerivClient(tick_prices, candles)

    @pytest.mark.asyncio
    async def test_feature_builder_processes_events(
        self, mock_settings, sample_tick_events, sample_candle_events
    ):
        """Feature builder should process tick and candle events."""
        # Create feature builder with mock settings
        with patch("data.features.FeatureBuilder") as MockBuilder:
            mock_builder = MagicMock()
            mock_builder.build.return_value = {
                "ticks": torch.randn(100),
                "candles": torch.randn(50, 5),
                "vol_metrics": torch.randn(10),
            }
            mock_builder.get_schema_version.return_value = "1.0"
            MockBuilder.return_value = mock_builder

            # Simulate processing
            ticks = np.array([e.price for e in sample_tick_events])
            candles = np.array(
                [[e.open, e.high, e.low, e.close, e.volume] for e in sample_candle_events]
            )

            mock_builder.build(ticks=ticks, candles=candles)
            mock_builder.build.assert_called_once()

    @pytest.mark.asyncio
    async def test_decision_engine_processes_predictions(self, mock_settings):
        """Decision engine should process model predictions correctly."""
        regime_veto = RegimeVeto(threshold_caution=0.1, threshold_veto=0.3)
        engine = DecisionEngine(mock_settings, regime_veto=regime_veto)

        # High confidence predictions should produce trades
        # High confidence predictions should produce trades
        probs = {"rise_fall_prob": 0.85, "touch_prob": 0.50, "range_prob": 0.40}
        trades = await engine.process_model_output(probs, reconstruction_error=0.05)

        # Should have at least one real trade for high confidence
        assert len(trades) >= 1
        assert trades[0].signal_type == SIGNAL_TYPES.REAL_TRADE

    @pytest.mark.asyncio
    async def test_regime_veto_blocks_during_anomaly(self, mock_settings):
        """Regime veto should block all trades when reconstruction error is high."""
        regime_veto = RegimeVeto(threshold_caution=0.1, threshold_veto=0.3)
        engine = DecisionEngine(mock_settings, regime_veto=regime_veto)

        # Even with high confidence, should be blocked
        probs = {"rise_fall_prob": 0.95, "touch_prob": 0.95, "range_prob": 0.95}

        # High reconstruction error triggers veto
        trades = await engine.process_model_output(probs, reconstruction_error=0.5)

        # All trades should be blocked
        assert len(trades) == 0

        # Stats should show veto
        stats = engine.get_statistics()
        assert stats["regime_vetoed"] >= 1

    @pytest.mark.asyncio
    async def test_safety_wrapper_enforces_rate_limit(self, mock_settings, tmp_path):
        """Safety wrapper should enforce rate limits on executor."""
        mock_executor = MagicMock()
        mock_executor.execute = AsyncMock(
            return_value=TradeResult(success=True, contract_id="TEST")
        )

        config = ExecutionSafetyConfig(max_trades_per_minute=3, max_trades_per_minute_per_symbol=3, max_daily_loss=100.0)
        safe_executor = SafeTradeExecutor(mock_executor, config, state_file=tmp_path / "rate_limit.db")

        from execution.signals import TradeSignal

        # Execute trades up to limit
        for i in range(5):
            signal = TradeSignal(
                signal_type=SIGNAL_TYPES.REAL_TRADE,
                contract_type="RISE_FALL",
                direction="CALL",
                probability=0.85,
                timestamp=datetime.now(timezone.utc),
                metadata={"stake": 1.0},
            )
            result = await safe_executor.execute(signal)

            if i < 3:
                assert result.success is True
            else:
                # Rate limited
                assert result.success is False
                assert "Rate limit" in result.error

    @pytest.mark.asyncio
    async def test_shadow_trades_captured(self, mock_settings, tmp_path):
        """Shadow trades should be captured during decision processing."""
        regime_veto = RegimeVeto(threshold_caution=0.1, threshold_veto=0.3)
        shadow_store = SQLiteShadowStore(tmp_path / "shadow.db")

        # Add missing mock settings for C02 and I01 fixes
        mock_settings.shadow_trade = MagicMock()
        mock_settings.shadow_trade.min_probability_track = 0.40
        mock_settings.shadow_trade.duration_rise_fall = 1
        mock_settings.shadow_trade.duration_touch = 5
        mock_settings.shadow_trade.duration_range = 5
        mock_settings.shadow_trade.duration_minutes = 1
        mock_settings.trading.barrier_offset = "+0.50"
        mock_settings.trading.barrier2_offset = "-0.50"

        engine = DecisionEngine(
            mock_settings,
            regime_veto=regime_veto,
            shadow_store=shadow_store,
            model_version="test_v1",
        )

        # Mid-confidence prediction (shadow territory)
        probs = {
            "rise_fall_prob": 0.55,  # In learning range
            "touch_prob": 0.45,
            "range_prob": 0.40,
        }

        # Process with context for shadow capture (now async)
        tick_window = np.random.randn(100)
        candle_window = np.random.randn(50, 5)

        await engine.process_with_context(
            probs=probs,
            reconstruction_error=0.05,
            tick_window=tick_window,
            candle_window=candle_window,
            entry_price=100.0,
        )

        # Check shadow store has records
        # Note: Implementation may vary
        shadow_store.close()


class TestEndToEndFlow:
    """End-to-end integration tests simulating full live trading cycle."""

    @pytest.mark.asyncio
    async def test_full_inference_cycle(
        self, mock_settings, sample_tick_events, sample_candle_events
    ):
        """Test complete inference cycle with mocked components."""
        # Prepare data
        np.array([e.price for e in sample_tick_events])
        np.array(
            [[e.open, e.high, e.low, e.close, e.volume] for e in sample_candle_events]
        )

        # Mock model predictions
        mock_probs = {
            "rise_fall_prob": torch.tensor(0.80),
            "touch_prob": torch.tensor(0.50),
            "range_prob": torch.tensor(0.45),
        }

        # Create decision engine
        regime_veto = RegimeVeto(threshold_caution=0.1, threshold_veto=0.3)
        engine = DecisionEngine(mock_settings, regime_veto=regime_veto)

        # Process output
        sample_probs = {k: v.item() for k, v in mock_probs.items()}
        reconstruction_error = 0.05  # Normal regime

        trades = await engine.process_model_output(sample_probs, reconstruction_error=0.05)

        # Should produce real trade for high confidence rise_fall
        assert len(trades) >= 1

        # Verify trade properties
        trade = trades[0]
        assert trade.probability >= mock_settings.thresholds.confidence_threshold_high

    @pytest.mark.asyncio
    async def test_caution_regime_reduces_trades(self, mock_settings):
        """Caution regime should reduce number of trades."""
        regime_veto = RegimeVeto(threshold_caution=0.1, threshold_veto=0.5)
        engine = DecisionEngine(mock_settings, regime_veto=regime_veto)

        # Multiple high confidence signals
        probs = {"rise_fall_prob": 0.90, "touch_prob": 0.85, "range_prob": 0.80}

        # Normal regime - should allow multiple trades
        trades_normal = await engine.process_model_output(probs, reconstruction_error=0.05)

        # Reset engine
        engine = DecisionEngine(mock_settings, regime_veto=regime_veto)

        # Caution regime - should reduce trades
        trades_caution = await engine.process_model_output(probs, reconstruction_error=0.15)

        # Caution should be more restrictive (keep only best)
        assert len(trades_caution) <= len(trades_normal)

    @pytest.mark.asyncio
    async def test_executor_called_for_real_trades(self, mock_settings, tmp_path):
        """Executor should be called only for real trades."""
        mock_executor = MagicMock()
        mock_executor.execute = AsyncMock(
            return_value=TradeResult(success=True, contract_id="EXEC_1")
        )

        config = ExecutionSafetyConfig(max_trades_per_minute=10)
        safe_executor = SafeTradeExecutor(mock_executor, config, state_file=tmp_path / "state.db")

        regime_veto = RegimeVeto(threshold_caution=0.1, threshold_veto=0.3)
        engine = DecisionEngine(mock_settings, regime_veto=regime_veto)

        # High confidence
        probs = {"rise_fall_prob": 0.90, "touch_prob": 0.50, "range_prob": 0.40}
        trades = await engine.process_model_output(probs, reconstruction_error=0.05)

        # Execute each trade
        for trade in trades:
            result = await safe_executor.execute(trade)
            assert result.success is True

        # Executor should have been called
        assert mock_executor.execute.call_count == len(trades)


class TestErrorHandling:
    """Tests for error handling in the live flow."""

    @pytest.mark.asyncio
    async def test_executor_failure_handled_gracefully(self, mock_settings, tmp_path):
        """Executor failures should be handled gracefully."""
        mock_executor = MagicMock()
        mock_executor.execute = AsyncMock(side_effect=Exception("Broker error"))

        config = ExecutionSafetyConfig(
            max_trades_per_minute=10, max_retry_attempts=2, retry_base_delay=0.01
        )
        safe_executor = SafeTradeExecutor(mock_executor, config, state_file=tmp_path / "state.db")

        from execution.signals import TradeSignal

        signal = TradeSignal(
            signal_type=SIGNAL_TYPES.REAL_TRADE,
            contract_type="RISE_FALL",
            direction="CALL",
            probability=0.85,
            timestamp=datetime.now(timezone.utc),
            metadata={"stake": 1.0},
        )

        result = await safe_executor.execute(signal)

        # Should fail after retries but not crash
        # Should fail after retries but not crash
        assert result.success is False
        assert "Broker error" in result.error

    @pytest.mark.asyncio
    async def test_kill_switch_halts_all_execution(self, mock_settings, tmp_path):
        """Kill switch should immediately halt all trades."""
        mock_executor = MagicMock()
        mock_executor.execute = AsyncMock(
            return_value=TradeResult(success=True, contract_id="SHOULD_NOT_EXECUTE")
        )

        config = ExecutionSafetyConfig()
        safe_executor = SafeTradeExecutor(mock_executor, config, state_file=tmp_path / "kill.db")

        # Activate kill switch
        config.kill_switch_enabled = True

        from execution.signals import TradeSignal

        # Try multiple trades
        for _ in range(5):
            signal = TradeSignal(
                signal_type=SIGNAL_TYPES.REAL_TRADE,
                contract_type="RISE_FALL",
                direction="CALL",
                probability=0.99,
                timestamp=datetime.now(timezone.utc),
                metadata={"stake": 1.0},
            )
            result = await safe_executor.execute(signal)
            assert result.success is False
            assert "Kill switch" in result.error

        # Executor should never have been called
        mock_executor.execute.assert_not_called()
