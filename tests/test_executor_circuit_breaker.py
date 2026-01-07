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


