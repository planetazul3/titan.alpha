import pytest
from unittest.mock import MagicMock, AsyncMock, PropertyMock
from execution.executor import DerivTradeExecutor, TradeResult
from execution.common.types import ExecutionRequest
from data.ingestion.client import CircuitState

@pytest.fixture
def mock_client():
    client = MagicMock()
    # Mock buy to return a success response by default
    client.buy = AsyncMock(return_value={"buy": {"contract_id": "123456", "buy_price": 10.0}})
    # Default circuit state: CLOSED
    type(client).circuit_state = PropertyMock(return_value=CircuitState.CLOSED)
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
async def test_delegated_circuit_breaker_open(executor, mock_client):
    """Executor should reject trade if client circuit breaker is OPEN."""
    # Set client state to OPEN
    type(mock_client).circuit_state = PropertyMock(return_value=CircuitState.OPEN)
    
    req = ExecutionRequest(signal_id="test_blocked", symbol="R_100", contract_type="CALL", stake=10.0, duration=1, duration_unit="m")
    
    result = await executor.execute(req)
    
    assert not result.success
    assert "Circuit breaker OPEN" in result.error
    assert "(delegated to client)" in result.error
    
    # Client buy should NOT have been called
    mock_client.buy.assert_not_called()

@pytest.mark.asyncio
async def test_delegated_circuit_breaker_closed(executor, mock_client):
    """Executor should allow trade if client circuit breaker is CLOSED."""
    # Set client state to CLOSED
    type(mock_client).circuit_state = PropertyMock(return_value=CircuitState.CLOSED)
    
    req = ExecutionRequest(signal_id="test_allowed", symbol="R_100", contract_type="CALL", stake=10.0, duration=1, duration_unit="m")
    
    result = await executor.execute(req)
    
    assert result.success
    mock_client.buy.assert_called_once()


# ========== C1-FIX: Rolling Window Circuit Breaker Tests ==========

@pytest.mark.asyncio
async def test_rolling_window_circuit_breaker_trips(mock_client, mock_settings):
    """Rolling window should trip after 5 failures within 600 seconds."""
    from config.constants import CIRCUIT_BREAKER_FAILURE_THRESHOLD
    
    executor = DerivTradeExecutor(client=mock_client, settings=mock_settings)
    
    # Mock buy to fail
    mock_client.buy = AsyncMock(return_value={"error": {"message": "Insufficient balance"}})
    
    # First 5 failures should be tracked but still attempt execution
    for i in range(CIRCUIT_BREAKER_FAILURE_THRESHOLD):
        req = ExecutionRequest(signal_id=f"fail_{i}", symbol="R_100", contract_type="CALL", stake=10.0, duration=1, duration_unit="m")
        result = await executor.execute(req)
        assert not result.success
        assert "Insufficient balance" in result.error
    
    # Verify we have 5 failures tracked
    assert len(executor._failure_timestamps) == CIRCUIT_BREAKER_FAILURE_THRESHOLD
    
    # 6th call should be rejected by circuit breaker without calling buy
    mock_client.buy.reset_mock()
    req = ExecutionRequest(signal_id="blocked", symbol="R_100", contract_type="CALL", stake=10.0, duration=1, duration_unit="m")
    result = await executor.execute(req)
    
    assert not result.success
    assert "executor rolling window" in result.error
    mock_client.buy.assert_not_called()


@pytest.mark.asyncio
async def test_rolling_window_expiry(mock_client, mock_settings, monkeypatch):
    """Failures older than 600 seconds should be removed from rolling window."""
    import time
    from config.constants import CIRCUIT_BREAKER_WINDOW_SECONDS
    
    executor = DerivTradeExecutor(client=mock_client, settings=mock_settings)
    
    # Manually inject old failure timestamps (older than window)
    old_time = time.time() - CIRCUIT_BREAKER_WINDOW_SECONDS - 10
    executor._failure_timestamps = [old_time] * 10  # 10 old failures
    
    # Reset buy to succeed
    mock_client.buy = AsyncMock(return_value={"buy": {"contract_id": "999", "buy_price": 10.0}})
    
    req = ExecutionRequest(signal_id="should_work", symbol="R_100", contract_type="CALL", stake=10.0, duration=1, duration_unit="m")
    result = await executor.execute(req)
    
    # Should succeed because old failures are filtered out
    assert result.success
    assert executor._failure_timestamps == []  # All old entries removed


@pytest.mark.asyncio
async def test_statistics_includes_circuit_breaker_info(executor):
    """get_statistics() should include rolling window info."""
    stats = executor.get_statistics()
    
    assert "rolling_window_failures" in stats
    assert "circuit_breaker_threshold" in stats
    assert stats["rolling_window_failures"] == 0
    assert stats["circuit_breaker_threshold"] == 5
