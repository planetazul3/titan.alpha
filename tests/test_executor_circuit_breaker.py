import pytest
import time
import asyncio
from unittest.mock import MagicMock, AsyncMock
from execution.executor import DerivTradeExecutor, TradeResult, CIRCUIT_BREAKER_WINDOW_SECONDS, CIRCUIT_BREAKER_FAILURE_THRESHOLD
from execution.common.types import ExecutionRequest

@pytest.fixture
def mock_client():
    client = MagicMock()
    # Mock buy to return a success response by default
    client.buy = AsyncMock(return_value={"buy": {"contract_id": "123456", "buy_price": 10.0}})
    return client

@pytest.fixture
def mock_settings():
    settings = MagicMock()
    settings.trading.stake_amount = 10.0
    return settings

@pytest.fixture
def executor(mock_client, mock_settings):
    return DerivTradeExecutor(client=mock_client, settings=mock_settings)

@pytest.mark.asyncio
async def test_circuit_breaker_does_not_trigger_under_threshold(executor):
    # Simulate N-1 failures
    for i in range(CIRCUIT_BREAKER_FAILURE_THRESHOLD - 1):
        executor._record_failure(f"Error {i}")
        
    assert not executor._is_circuit_breaker_active()
    
    # Execute should proceed
    req = ExecutionRequest(signal_id="test", symbol="R_100", contract_type="CALL", stake=10.0, duration=1, duration_unit="m")
    result = await executor.execute(req)
    assert result.success

@pytest.mark.asyncio
async def test_circuit_breaker_triggers_at_threshold(executor, mock_client):
    # Simulate N failures
    for i in range(CIRCUIT_BREAKER_FAILURE_THRESHOLD):
        executor._record_failure(f"Error {i}")
        
    assert executor._is_circuit_breaker_active()
    
    # Execute should fail immediately without calling client
    mock_client.buy.reset_mock()
    req = ExecutionRequest(signal_id="test_blocked", symbol="R_100", contract_type="CALL", stake=10.0, duration=1, duration_unit="m")
    
    result = await executor.execute(req)
    assert not result.success
    assert "Circuit breaker active" in result.error
    
    # Client buy should NOT have been called
    mock_client.buy.assert_not_called()

@pytest.mark.asyncio
async def test_circuit_breaker_resets_manually(executor, mock_client):
    # Trigger breaker
    for i in range(CIRCUIT_BREAKER_FAILURE_THRESHOLD):
        executor._record_failure(f"Error {i}")
    
    assert executor._is_circuit_breaker_active()
    
    # Reset
    executor.reset_circuit_breaker()
    assert not executor._is_circuit_breaker_active()
    assert len(executor._failure_timestamps) == 0
    
    # Should work now
    req = ExecutionRequest(signal_id="test_reset", symbol="R_100", contract_type="CALL", stake=10.0, duration=1, duration_unit="m")
    result = await executor.execute(req)
    assert result.success

@pytest.mark.asyncio
async def test_circuit_breaker_auto_prunes_old_failures(executor):
    # We need to simulate time passing. Since _record_failure uses time.monotonic(),
    # and _is_circuit_breaker_active uses it too.
    # We can mock time.monotonic or manually inject timestamps.
    
    # Inject old timestamps
    old_time = time.monotonic() - (CIRCUIT_BREAKER_WINDOW_SECONDS + 10)
    for _ in range(CIRCUIT_BREAKER_FAILURE_THRESHOLD):
        executor._failure_timestamps.append(old_time)
        
    # Check status - should auto-prune
    assert not executor._is_circuit_breaker_active()
    assert len(executor._failure_timestamps) == 0

