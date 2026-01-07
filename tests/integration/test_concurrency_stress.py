import pytest
import asyncio
from unittest.mock import MagicMock, AsyncMock
from execution.executor import DerivTradeExecutor
from execution.common.types import ExecutionRequest
from execution.policy import ExecutionPolicy
from config.settings import Settings, Trading, Thresholds, ExecutionSafety, ModelHyperparams
from execution.idempotency_store import SQLiteIdempotencyStore

def create_test_settings():
    return Settings.model_construct(
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
        execution_safety=ExecutionSafety(
            max_trades_per_minute=60,
            max_daily_loss=1000,
            max_stake_per_trade=50
        ),
        environment="test"
    )

@pytest.mark.asyncio
async def test_concurrent_idempotency_under_load(tmp_path):
    """Verify idempotency holds up under high concurrent load."""
    
    # 1. Setup
    client = MagicMock()
    # Simulate API delay
    async def delayed_buy(**kwargs):
        await asyncio.sleep(0.01)
        return {"buy": {"contract_id": "12345", "buy_price": 10.0}}
    client.buy = AsyncMock(side_effect=delayed_buy)
    client.get_balance = AsyncMock(return_value=1000.0)

    db_path = tmp_path / "idempotency.db"
    store = SQLiteIdempotencyStore(db_path)
    # Store auto-initializes in __init__
    
    settings = create_test_settings()
    executor = DerivTradeExecutor(client, settings, idempotency_store=store)

    # 2. Fire 50 identical requests concurrently
    req = ExecutionRequest(
        signal_id="CONCURRENT_SIG_1",
        symbol="R_100",
        contract_type="RISE_FALL",
        stake=10.0,
        duration=1,
        duration_unit="m"
    )
    
    tasks = [executor.execute(req) for _ in range(50)]
    results = await asyncio.gather(*tasks)
    
    # 3. Verify ONLY ONE succeeded with execution
    successful_executions = 0
    idempotent_skips = 0
    
    for r in results:
        # TradeResult doesn't strictly have a "skipped" flag, but we check if client.buy was called
        # Wait, client.buy is called only if check_and_reserve returns True
        if r.success and r.contract_id == "12345":
             # We need to distinguish between "Executed" and "Returned Existing ID"
             # But implementation returns success=True for both.
             pass
            
    # Check physical calls to client
    assert client.buy.call_count == 1
    
    await executor.shutdown()


# Note: test_circuit_breaker_concurrency was removed.
# The executor's internal circuit breaker was refactored to delegate to DerivClient.
# The delegated behavior is tested in tests/test_executor_circuit_breaker.py.
