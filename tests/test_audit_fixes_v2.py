import pytest
from unittest.mock import MagicMock, AsyncMock
from datetime import datetime, timezone
from execution.executor import DerivTradeExecutor, TradeResult
from execution.common.types import ExecutionRequest
from execution.decision import DecisionEngine
from execution.policy import ExecutionPolicy, SafetyProfile
from config.settings import Settings, Trading, Thresholds, ExecutionSafety, ModelHyperparams, DataShapes
from config.constants import SIGNAL_TYPES, CONTRACT_TYPES
from execution.shadow_store import ShadowTradeRecord, ShadowTradeStore
from pathlib import Path
import asyncio

def create_real_settings():
    # Instantiate real Pydantic models to avoid Mock formatting/attribute issues
    trading = Trading(symbol="R_100", stake_amount=10.0)
    thresholds = Thresholds(
        confidence_threshold_high=0.7,
        learning_threshold_min=0.4,
        learning_threshold_max=0.6
    )
    hyperparams = ModelHyperparams(
        learning_rate=0.001,
        batch_size=32,
        lstm_hidden_size=64,
        cnn_filters=32,
        latent_dim=16
    )
    data_shapes = DataShapes(
        sequence_length_ticks=20,
        sequence_length_candles=20,
        warmup_steps=0
    )
    safety = ExecutionSafety()
    
    settings = Settings.model_construct(
        trading=trading,
        thresholds=thresholds,
        hyperparams=hyperparams,
        data_shapes=data_shapes,
        execution_safety=safety,
        environment="development"
    )
    return settings

@pytest.mark.asyncio
async def test_idempotency_failure_blocks_execution():
    """Test that idempotency check failure blocks execution (ID-001)."""
    client = MagicMock()
    client.get_balance = AsyncMock(return_value=1000.0)
    
    # Mocking idempotency store failure
    store = MagicMock()
    # Fix C-001: Use AsyncMock for async methods
    store.get_contract_id_async = AsyncMock(side_effect=Exception("Idempotency check failure"))
    store.check_and_reserve_async = AsyncMock(side_effect=Exception("Idempotency check failure"))
    
    settings = create_real_settings()
    executor = DerivTradeExecutor(client, settings, idempotency_store=store)
    
    signal = ExecutionRequest(
        signal_id="SIG_IDEM_TEST",
        symbol="R_100",
        contract_type="RISE_FALL",
        stake=10.0,
        duration=1,
        duration_unit="m"
    )
    
    result = await executor.execute(signal)
    
    assert result.success is False
    assert "Idempotency check failure" in result.error
    client.buy.assert_not_called()

# Note: test_circuit_breaker_triggers_on_repeated_failures was removed.
# The executor's internal circuit breaker was refactored to delegate to DerivClient.
# The delegated behavior is now tested in tests/test_executor_circuit_breaker.py.

@pytest.mark.asyncio
async def test_policy_integration_blocks_decision():
    """Test that policy vetoes block decision making in DecisionEngine."""
    settings = create_real_settings()
    settings.execution_safety.kill_switch_enabled = True
    
    # Mock other deps
    regime_veto = MagicMock()
    regime_veto.threshold_caution = 0.2
    regime_veto.threshold_veto = 0.5
    # assess returns a mocked assessment
    assessment = MagicMock()
    assessment.is_vetoed.return_value = False
    regime_veto.assess.return_value = assessment
    
    shadow_store = MagicMock(spec=ShadowTradeStore)
    shadow_store._store_path = Path("test_shadow.db") # Fix: must be Path object
    
    engine = DecisionEngine(settings, regime_veto=regime_veto, shadow_store=shadow_store)
    
    # Fake model output (flat format as expected by filter_signals)
    probs = {"rise_fall_prob": 0.9}
    
    # Must apply SafetyProfile to register the kill switch veto
    SafetyProfile.apply(engine.policy, settings)
    
    # process_model_output is now async
    signals = await engine.process_model_output(probs, reconstruction_error=0.1)
    
    assert len(signals) == 0
    # Check stats for ignored (DecisionEngine uses "ignored" for policy vetoes)
    assert engine.get_statistics()["ignored"] >= 1

@pytest.mark.asyncio
async def test_shadow_store_append_async_standardization():
    """Test that append_async exists and works on base ShadowTradeStore."""
    store_path = Path("test_shadow_verify.jsonl")
    store = ShadowTradeStore(store_path)
    
    record = ShadowTradeRecord(
        trade_id="test",
        timestamp=datetime.now(timezone.utc),
        contract_type="RISE_FALL",
        direction="CALL",
        probability=0.9,
        entry_price=100.0,
        reconstruction_error=0.1,
        regime_state="TRUSTED",
        metadata={}
    )
    
    try:
        await store.append_async(record)
        assert store_path.exists()
        content = store_path.read_text()
        assert "test" in content
    finally:
        if store_path.exists():
            store_path.unlink()
