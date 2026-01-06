import pytest
from unittest.mock import MagicMock, AsyncMock
from execution.signal_adapter_service import SignalAdapterService
from execution.signals import TradeSignal
from execution.common.types import ExecutionRequest
from execution.contract_params import ContractParameterService

@pytest.fixture
def mock_settings():
    settings = MagicMock()
    settings.trading.stake_amount = 10.0
    return settings

@pytest.fixture
def mock_sizer():
    sizer = MagicMock()
    # Mock suggest_stake_for_signal
    # Protocol defines suggest_stake_for_signal(signal) -> float
    sizer.suggest_stake_for_signal.return_value = 15.0
    return sizer

@pytest.fixture
def service(mock_settings, mock_sizer):
    return SignalAdapterService(mock_settings, mock_sizer)

@pytest.mark.asyncio
async def test_adapt_signal_success(service):
    signal = TradeSignal(
        signal_id="test_sig",
        symbol="R_100",
        signal_type="ML_MODEL",
        contract_type="RISE_FALL",
        direction="CALL",
        probability=0.8,
        timestamp=1234567890.0
    )
    
    # Mock internal adapter's to_execution_request since it depends on other things
    # Or rely on real one if simple. 
    # Real one uses param_service which might need mocking if it checks internal things.
    # ContractParameterService usually just looks up constants.
    
    # We'll just run it. The duration/barrier resolution might fail if not mocked or config valid.
    # Mock internal adapter to isolate service logic
    service._internal_adapter = MagicMock()
    service._internal_adapter.to_execution_request = AsyncMock(return_value=ExecutionRequest(
        signal_id="test_sig",
        symbol="R_100", 
        contract_type="CALL",
        stake=15.0,
        duration=5,
        duration_unit="m",
        barrier=None,
        barrier2=None
    ))
    
    req = await service.adapt(signal)
    
    assert req.signal_id == "test_sig"
    assert req.stake == 15.0

@pytest.mark.asyncio
async def test_adapt_signal_invalid_stake_warning(service):
    signal = TradeSignal("id", "sym", "type", "contract", "dir", 0.5, 0.0)
    
    service._internal_adapter = MagicMock()
    # Return negative stake
    service._internal_adapter.to_execution_request = AsyncMock(return_value=ExecutionRequest(
        signal_id="id", symbol="sym", contract_type="CALL", stake=-5.0, duration=1, duration_unit="m"
    ))
    
    # Should log warning but return request
    req = await service.adapt(signal)
    assert req.stake == -5.0

@pytest.mark.asyncio
async def test_adapt_signal_failure(service):
    signal = TradeSignal("id", "sym", "type", "contract", "dir", 0.5, 0.0)
    service._internal_adapter.to_execution_request = AsyncMock(side_effect=ValueError("Adapt failed"))
    
    with pytest.raises(ValueError, match="Adapt failed"):
        await service.adapt(signal)
