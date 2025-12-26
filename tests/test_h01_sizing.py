import pytest
from unittest.mock import AsyncMock, Mock, patch
from execution.executor import DerivTradeExecutor, TradeSignal
from config.settings import Settings
from datetime import datetime
from config.constants import CONTRACT_TYPES, SIGNAL_TYPES
from execution.position_sizer import KellyPositionSizer

@pytest.fixture
def mock_client():
    client = AsyncMock()
    # Mock buy response
    client.buy.return_value = {
        "buy": {
            "contract_id": "12345",
            "buy_price": 10.0,
            "balance_after": 990.0
        }
    }
    return client

@pytest.fixture
def settings():
    s = Settings()
    s.trading.stake_amount = 10.0
    return s

@pytest.mark.asyncio
async def test_executor_uses_balance_for_sizing(mock_client, settings):
    """Verify that executor fetches balance and passes it to sizer."""
    
    # Setup Kelly sizer which uses balance
    # Base stake 0.1 means 10% of balance (roughly, depends on adjustments)
    # Safety factor 1.0 for simple math
    sizer = KellyPositionSizer(base_stake=0.1, safety_factor=1.0, max_stake=1000.0)
    
    executor = DerivTradeExecutor(mock_client, settings, position_sizer=sizer)
    
    # Case 1: Balance = 1000
    mock_client.get_balance.return_value = 1000.0
    
    signal = TradeSignal(
        signal_type=SIGNAL_TYPES.REAL_TRADE, # Valid type from constants
        contract_type=CONTRACT_TYPES.RISE_FALL,
        direction="CALL",
        probability=0.7, # High prob to ensure positive Kelly
        timestamp=datetime.now()
        # payout_ratio=0.9 
    )
    signal.payout_ratio = 0.9 # Manually attach for test context
    
    # Kelly calculation:
    # b = 0.9, p = 0.7, q = 0.3
    # f = (0.7 * 0.9 - 0.3) / 0.9 = (0.63 - 0.3) / 0.9 = 0.33 / 0.9 = 0.3667
    # Adjusted (safety=1.0) = 0.3667
    # Stake = 0.3667 * 1000 = 366.67
    
    await executor.execute(signal)
    
    mock_client.get_balance.assert_called()
    
    # Verify buy was called with dynamic amount
    # We don't check exact float due to potential minor logic diffs in sizer
    # But it should be ~366, definitely not the default 10.0
    call_args = mock_client.buy.call_args
    assert call_args is not None
    amount_arg = call_args[0][1] # amount is 2nd arg
    
    assert amount_arg > 100.0 
    assert abs(amount_arg - 366.67) < 50.0  # Allow some buffer for confidence scaling etc

    print(f"Executed with stake: {amount_arg} for balance 1000")

@pytest.mark.asyncio
async def test_executor_fallback_on_balance_error(mock_client, settings):
    """Verify that executor proceeds safely if get_balance fails."""
    
    sizer = KellyPositionSizer(base_stake=0.1) # Using fractional sizing
    executor = DerivTradeExecutor(mock_client, settings, position_sizer=sizer)
    
    # Simulate API error
    mock_client.get_balance.side_effect = Exception("API Error")
    
    signal = TradeSignal(
        signal_type=SIGNAL_TYPES.REAL_TRADE,
        contract_type=CONTRACT_TYPES.RISE_FALL,
        direction="CALL",
        probability=0.7,
        timestamp=datetime.now()
        # payout_ratio=0.9  <-- Removed, not in dataclass
    )
    # inject attribute manually if needed for sizer, or rely on default
    signal.payout_ratio = 0.9
    
    await executor.execute(signal)
    
    # Should still execute using base logic (no balance passed)
    # KellyPositionSizer defaults to base_stake * 10 scaler if balance is None
    # effectively acting as fixed stake-ish or just small stake
    
    call_args = mock_client.buy.call_args
    assert call_args is not None
    amount_arg = call_args[0][1]
    
    # Just assert it didn't crash and produced specific positive number
    assert amount_arg > 0
    print(f"Executed with fallback stake: {amount_arg}")
