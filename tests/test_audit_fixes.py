
import asyncio
import numpy as np
import pytest
from unittest.mock import MagicMock, AsyncMock, patch
from data.buffer import MarketDataBuffer
from execution.real_trade_tracker import RealTradeTracker, TradeIntent

@pytest.mark.asyncio
async def test_buffer_snapshot():
    print("\n--- Testing MarketDataBuffer.get_snapshot ---")
    buffer = MarketDataBuffer(tick_length=10, candle_length=5)
    
    # Fill buffer
    for i in range(5):
        buffer.append_tick(float(i))
    
    # Add candles
    for i in range(6): # 6 candles (last one is forming)
        c = [i, i+1, i-0.5, i, 100, 1000+i]
        buffer.update_candle_from_array(c)
        
    print(f"Buffer state: {buffer}")
    
    # Take snapshot
    snap = buffer.get_snapshot()
    print("Snapshot taken.")
    
    # Verify keys
    assert "ticks" in snap
    assert "candles" in snap
    
    # Verify lengths
    assert len(snap["ticks"]) == 5
    # Should exclude the 6th forming candle if length > candle_length (5)
    # Wait, update_candle appends. If we added 6, and maxlen is 5+1=6.
    # get_snapshot logic: if len > 5, exclude last.
    assert len(snap["candles"]) == 5
    
    # Verify deep copy
    snap["ticks"][0] = 999.0
    assert buffer._ticks[0] == 0.0
    print("Deep copy verified.")

@pytest.mark.asyncio
async def test_tracker_intent():
    print("\n--- Testing RealTradeTracker.intent ---")
    
    # Mock Tracker
    tracker = RealTradeTracker("test_token", "test_account")
    tracker._store = MagicMock()
    
    # Test Context Manager Success
    async with tracker.intent("CALL", 100.0, 10.0, 0.8) as intent_id:
        print(f"Intent created: {intent_id}")
        assert intent_id.startswith("intent_")
        tracker._store.prepare_trade_intent.assert_called_once()
        
        # Simulate Success
        tracker.confirm_intent(intent_id, "contract_123")
        tracker._store.confirm_trade.assert_called_with(intent_id, "contract_123")
        
    # Test Context Manager Failure (Exception)
    try:
        async with tracker.intent("PUT", 100.0, 10.0, 0.8) as intent_id:
            raise ValueError("Execution Failed")
    except ValueError:
        pass
        
    # Verify cleanup was called
    tracker._store.remove_trade.assert_called()
    print("Intent cleanup verified.")

if __name__ == "__main__":
    import sys
    sys.exit(pytest.main(["-v", __file__]))
