import unittest
import pytest
from unittest.mock import MagicMock, AsyncMock
from execution.decision.core import DecisionEngine
from config.settings import Settings, DataShapes, Trading, Thresholds, ModelHyperparams, ExecutionSafety

def create_mock_settings():
    return Settings.model_construct(
        data_shapes=DataShapes(
            warmup_steps=5, 
            sequence_length_candles=20, # Must be >= 16
            sequence_length_ticks=16    # Must be >= 16
        ),
        trading=Trading(symbol="R_100", stake_amount=10),
        thresholds=Thresholds(
            confidence_threshold_high=0.8,
            learning_threshold_min=0.4,
            learning_threshold_max=0.6
        ),
        hyperparams=ModelHyperparams(
            learning_rate=0.001,
            batch_size=32,
            lstm_hidden_size=64,
            cnn_filters=32,
            latent_dim=16
        ),
        execution_safety=ExecutionSafety(),
        environment="test"
    )

@pytest.mark.asyncio
async def test_decision_engine_warmup_veto():
    """Test that DecisionEngine blocks signals during warmup period (C-003)."""
    settings = create_mock_settings()
    # Mock dependencies
    shadow = MagicMock()
    shadow.append_async = AsyncMock()
    safety = MagicMock()
    policy = MagicMock()
    policy.async_check_vetoes = AsyncMock(return_value=None)
    
    engine = DecisionEngine(settings, shadow_store=shadow, safety_store=safety, policy=policy)
    
    # Mock internal components to avoid complexity
    engine.safety_sync = MagicMock()
    engine.safety_sync.sync = AsyncMock()
    
    # Create a proper TradeSignal-like object that passes filters
    mock_signal = MagicMock()
    mock_signal.contract_type = "RISE_FALL"
    mock_signal.direction = "CALL" 
    mock_signal.probability = 0.9
    mock_signal.signal_id = "test_signal"
    mock_signal.metadata = {}
    
    # process_signals_batch expects signals to have certain attributes
    # But since we are mocking process_signals_batch (simulating it), we just need to ensure
    # the engine passes it through.
    
    # Wait, we are NOT mocking process_signals_batch in the previous run.
    # Let's mock it now to isolate DecisionEngine logic.
    with unittest.mock.patch("execution.decision.core.process_signals_batch") as mock_psb:
        mock_psb.return_value = ([mock_signal], []) # real, shadow
        
        engine.processor = MagicMock()
        engine.processor.process.return_value = [mock_signal]
    
        # 1. During warmup (steps 1-4)
        for i in range(1, 5):
            signals = await engine.process_model_output({}, 0.1)
            assert len(signals) == 0, f"Signals should be blocked at step {i}"
            assert engine._warmup_complete is False
            
        # 2. Completion (step 5)
        signals = await engine.process_model_output({}, 0.1)
        # Depending on logic, step 5 might be the first accepted or 6.
        # Logic: increment, then check < threshold. 
        # Step 4: obs=4 < 5 -> Blocked.
        # Step 5: obs=5 < 5 -> False (5 is not < 5) -> unblocked.
        assert engine._warmup_complete is True
        # If unblocked, it should return mock signals
        assert len(signals) == 1
        assert signals[0] == mock_signal

