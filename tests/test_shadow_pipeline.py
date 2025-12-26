"""
Tests for shadow trade data pipeline.

Tests ShadowTradeStore, OutcomeResolver, and ShadowTradeDataset integration.
"""

import tempfile
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest


class TestShadowTradeStore:
    """Tests for ShadowTradeStore."""

    def test_shadow_trade_record_create(self):
        """ShadowTradeRecord.create should generate UUID and timestamp."""
        from execution.shadow_store import ShadowTradeRecord

        record = ShadowTradeRecord.create(
            contract_type="RISE_FALL",
            direction="CALL",
            probability=0.85,
            entry_price=100.0,
            reconstruction_error=0.15,
            regime_state="CAUTION",
            tick_window=np.array([99.0, 100.0, 100.5]),
            candle_window=np.array([[100, 101, 99, 100.5, 1000, 123]]),
        )

        assert record.trade_id is not None
        assert record.timestamp is not None
        assert record.contract_type == "RISE_FALL"
        assert record.probability == 0.85
        assert not record.is_resolved()

    def test_shadow_trade_record_with_outcome(self):
        """with_outcome should return a new resolved record."""
        from execution.shadow_store import ShadowTradeRecord

        record = ShadowTradeRecord.create(
            contract_type="RISE_FALL",
            direction="CALL",
            probability=0.85,
            entry_price=100.0,
            reconstruction_error=0.15,
            regime_state="TRUSTED",
            tick_window=np.array([100.0]),
            candle_window=np.array([[100, 101, 99, 100, 0, 0]]),
        )

        resolved = record.with_outcome(outcome=True, exit_price=101.0)

        # Original should be unchanged
        assert not record.is_resolved()

        # New record should be resolved
        assert resolved.is_resolved()
        assert resolved.outcome is True
        assert resolved.exit_price == 101.0
        assert resolved.trade_id == record.trade_id  # Same ID

    def test_shadow_trade_store_append_query(self):
        """Store should support append and query roundtrip."""
        from execution.shadow_store import ShadowTradeRecord, ShadowTradeStore

        with tempfile.TemporaryDirectory() as tmpdir:
            store_path = Path(tmpdir) / "shadow_trades.ndjson"
            store = ShadowTradeStore(store_path)

            # Append records
            for i in range(3):
                record = ShadowTradeRecord.create(
                    contract_type="RISE_FALL",
                    direction="CALL",
                    probability=0.8 + i * 0.05,
                    entry_price=100.0 + i,
                    reconstruction_error=0.1,
                    regime_state="TRUSTED",
                    tick_window=np.array([100.0]),
                    candle_window=np.array([[100, 101, 99, 100, 0, 0]]),
                )
                store.append(record)

            # Query
            records = store.query()

            assert len(records) == 3
            assert records[0].probability == 0.8
            assert records[2].probability == 0.9

    def test_shadow_trade_store_query_filters(self):
        """Store query should filter by resolution status."""
        from execution.shadow_store import ShadowTradeRecord, ShadowTradeStore

        with tempfile.TemporaryDirectory() as tmpdir:
            store_path = Path(tmpdir) / "shadow_trades.ndjson"
            store = ShadowTradeStore(store_path)

            # Append unresolved
            record = ShadowTradeRecord.create(
                contract_type="RISE_FALL",
                direction="CALL",
                probability=0.8,
                entry_price=100.0,
                reconstruction_error=0.1,
                regime_state="TRUSTED",
                tick_window=np.array([100.0]),
                candle_window=np.array([[100, 101, 99, 100, 0, 0]]),
            )
            store.append(record)

            # Append resolved
            resolved = record.with_outcome(True, 101.0)
            store.append(resolved)

            # Query filters
            all_records = store.query()
            assert len(all_records) == 2

            resolved_only = store.query(resolved_only=True)
            assert len(resolved_only) == 1

            unresolved_only = store.query(unresolved_only=True)
            assert len(unresolved_only) == 1


class TestOutcomeResolver:
    """Tests for OutcomeResolver."""

    def test_resolver_rise_fall_call_win(self):
        """CALL should win if price rises."""
        from execution.outcome_resolver import OutcomeResolver
        from execution.shadow_store import ShadowTradeRecord

        resolver = OutcomeResolver()

        record = ShadowTradeRecord.create(
            contract_type="RISE_FALL",
            direction="CALL",
            probability=0.8,
            entry_price=100.0,
            reconstruction_error=0.1,
            regime_state="TRUSTED",
            tick_window=np.array([100.0]),
            candle_window=np.array([[100, 101, 99, 100, 0, 0]]),
        )

        # Simulated tick data: price rises to 102
        tick_data = np.array([100.0, 100.5, 101.0, 101.5, 102.0])
        tick_timestamps = np.array(
            [
                record.timestamp.timestamp(),
                record.timestamp.timestamp() + 10,
                record.timestamp.timestamp() + 20,
                record.timestamp.timestamp() + 30,
                record.timestamp.timestamp() + 50,
            ]
        )

        resolved = resolver.resolve_from_cache([record], tick_data, tick_timestamps)

        assert len(resolved) == 1
        assert resolved[0].is_resolved()
        assert resolved[0].outcome is True  # CALL won, price rose
        assert resolved[0].exit_price == 102.0

    def test_resolver_rise_fall_put_win(self):
        """PUT should win if price falls."""
        from execution.outcome_resolver import OutcomeResolver
        from execution.shadow_store import ShadowTradeRecord

        resolver = OutcomeResolver()

        record = ShadowTradeRecord.create(
            contract_type="RISE_FALL",
            direction="PUT",
            probability=0.8,
            entry_price=100.0,
            reconstruction_error=0.1,
            regime_state="TRUSTED",
            tick_window=np.array([100.0]),
            candle_window=np.array([[100, 101, 99, 100, 0, 0]]),
        )

        # Simulated tick data: price falls to 98
        tick_data = np.array([100.0, 99.5, 99.0, 98.5, 98.0])
        tick_timestamps = np.array(
            [
                record.timestamp.timestamp(),
                record.timestamp.timestamp() + 10,
                record.timestamp.timestamp() + 20,
                record.timestamp.timestamp() + 30,
                record.timestamp.timestamp() + 50,
            ]
        )

        resolved = resolver.resolve_from_cache([record], tick_data, tick_timestamps)

        assert len(resolved) == 1
        assert resolved[0].outcome is True  # PUT won, price fell


class TestPipelineIntegration:
    """Integration tests for the full pipeline."""

    def test_full_pipeline_flow(self):
        """Test complete flow: capture → resolve → dataset."""
        from execution.outcome_resolver import OutcomeResolver
        from execution.shadow_store import ShadowTradeRecord, ShadowTradeStore

        with tempfile.TemporaryDirectory() as tmpdir:
            store_path = Path(tmpdir) / "shadow_trades.ndjson"

            # Step 1: Capture shadow trade
            store = ShadowTradeStore(store_path)
            record = ShadowTradeRecord.create(
                contract_type="RISE_FALL",
                direction="CALL",
                probability=0.85,
                entry_price=100.0,
                reconstruction_error=0.15,
                regime_state="CAUTION",
                tick_window=np.random.rand(100).astype(np.float32) * 100 + 1,
                candle_window=np.random.rand(50, 6).astype(np.float32) * 100 + 1,
            )
            store.append(record)

            # Step 2: Resolve outcome
            resolver = OutcomeResolver()
            unresolved = store.query(unresolved_only=True)

            # Create tick data showing price rise (CALL wins)
            tick_data = np.array([100.0, 100.5, 101.0, 101.5, 102.0])
            tick_timestamps = np.array(
                [
                    record.timestamp.timestamp(),
                    record.timestamp.timestamp() + 10,
                    record.timestamp.timestamp() + 20,
                    record.timestamp.timestamp() + 30,
                    record.timestamp.timestamp() + 50,
                ]
            )

            resolved = resolver.resolve_from_cache(unresolved, tick_data, tick_timestamps)

            # Step 3: Store resolved
            store.append_resolved(resolved)

            # Verify
            assert len(resolved) == 1
            assert resolved[0].is_resolved()
            assert resolved[0].outcome is True

            # Verify statistics
            stats = store.get_statistics()
            assert stats["total_records"] == 1

    @pytest.mark.asyncio
    async def test_decision_engine_with_shadow_store(self):
        """Test DecisionEngine captures shadow trades with context."""
        from execution.decision import DecisionEngine
        from execution.regime import RegimeVeto
        from execution.shadow_store import ShadowTradeStore

        with tempfile.TemporaryDirectory() as tmpdir:
            store_path = Path(tmpdir) / "shadow_trades.ndjson"
            store = ShadowTradeStore(store_path)

            settings = MagicMock()
            settings.thresholds.confidence_threshold_high = 0.75
            settings.thresholds.learning_threshold_min = 0.40
            settings.thresholds.learning_threshold_max = 0.60
            # Add missing shadow_trade settings for I01 and C02 fixes
            settings.shadow_trade.min_probability_track = 0.40
            settings.shadow_trade.duration_rise_fall = 1
            settings.shadow_trade.duration_touch = 5
            settings.shadow_trade.duration_range = 5
            settings.shadow_trade.duration_minutes = 1
            # Add trading settings for filter_signals
            settings.trading.symbol = "R_100"
            settings.trading.barrier_offset = "+0.50"
            settings.trading.barrier2_offset = "-0.50"

            regime_veto = RegimeVeto(threshold_caution=0.1, threshold_veto=0.3)

            engine = DecisionEngine(
                settings, regime_veto=regime_veto, shadow_store=store, model_version="test_v1.0"
            )

            probs = {"rise_fall_prob": 0.85, "touch_prob": 0.25, "range_prob": 0.25}

            tick_window = np.random.rand(100).astype(np.float32) * 100 + 1
            candle_window = np.random.rand(50, 6).astype(np.float32) * 100 + 1

            # Process with context - now async
            trades = await engine.process_with_context(
                probs=probs,
                reconstruction_error=0.05,  # Below caution threshold
                tick_window=tick_window,
                candle_window=candle_window,
                entry_price=100.0,
            )

            # Should have real trades (not vetoed)
            assert len(trades) > 0

            # Should have stored shadow trades
            stored = store.query()
            assert len(stored) > 0

            # Check stored record has context
            record = stored[0]
            assert record.model_version == "test_v1.0"
            assert len(record.tick_window) == 100
            assert len(record.candle_window) == 50
