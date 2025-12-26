
import asyncio
import pytest
from datetime import datetime, timedelta, timezone
from unittest.mock import MagicMock
import numpy as np

from execution.shadow_store import ShadowTradeRecord
from execution.sqlite_shadow_store import SQLiteShadowStore
from execution.shadow_resolution import ShadowTradeResolver
from config.settings import ShadowTradeConfig
from config.constants import CONTRACT_TYPES

@pytest.fixture
def mock_store(tmp_path):
    return SQLiteShadowStore(tmp_path / "c01_verification.db")

@pytest.fixture
def mock_resolver(mock_store):
    return ShadowTradeResolver(mock_store, duration_minutes=1) # Default

@pytest.mark.asyncio
async def test_c01_touch_trade_resolution_after_entry(mock_store, mock_resolver):
    """
    Verify C01 fix: Ensure TOUCH trade outcome is determined by post-entry candles (resolution_context),
    not pre-entry features (candle_window).
    """
    # 1. Create a TOUCH trade
    now = datetime.now(timezone.utc)
    entry_price = 100.0
    barrier_offset = 1.0 # Upper barrier at 101.0
    duration_minutes = 2
    
    trade = ShadowTradeRecord.create(
        contract_type=CONTRACT_TYPES.TOUCH_NO_TOUCH,
        direction="TOUCH", # Predicts touch
        probability=0.8,
        entry_price=entry_price,
        reconstruction_error=0.01,
        regime_state="normal",
        tick_window=np.zeros(10), # Dummy
        candle_window=np.zeros((10, 5)), # Dummy pre-entry data
        barrier_level=barrier_offset,
        duration_minutes=duration_minutes
    )
    # Ensure no resolution context initially
    assert len(trade.resolution_context) == 0
    await mock_store.append_async(trade)
    
    # 2. Simulate Minute 1: Price goes up but DOES NOT hit barrier
    # Candle: Open=100, High=100.8, Low=99.8, Close=100.5
    # Should NOT satisfy barrier (101.0)
    current_time_m1 = now + timedelta(minutes=1)
    
    # We call resolve_trades. It should:
    # a) Update resolution_context with the candle
    # b) NOT resolve the trade yet because it's not expired (duration=2m)
    # However, for testing, we can simulate the update flow.
    # The real loop calls resolve_trades every minute.
    
    # Pass Minute 1 candle
    resolved_count = await mock_resolver.resolve_trades(
        current_price=100.5,
        current_time=current_time_m1,
        high_price=100.8,
        low_price=99.8
    )
    
    assert resolved_count == 0 # Not expired yet
    
    # Verify resolution_context was updated
    updated_trade = mock_store.get_by_id(trade.trade_id)
    assert len(updated_trade.resolution_context) == 1
    assert updated_trade.resolution_context[0] == [100.8, 99.8, 100.5] # High, Low, Close
    
    # 3. Simulate Minute 2: Price hits barrier!
    # Candle: Open=100.5, High=101.5, Low=100.0, Close=101.0
    # Should satisfy barrier (101.5 > 101.0)
    current_time_m2 = now + timedelta(minutes=2, seconds=1) # Expired now
    
    resolved_count = await mock_resolver.resolve_trades(
        current_price=101.0,
        current_time=current_time_m2,
        high_price=101.5,
        low_price=100.0
    )
    
    # Now it should be resolved
    assert resolved_count == 1
    
    # Verify outcome
    final_trade = mock_store.get_by_id(trade.trade_id)
    assert final_trade.is_resolved()
    assert final_trade.outcome is True # WIN - touched barrier
    
    # Verify final resolution context
    # Note: The final candle is used for resolution (combined in memory) but typically 
    # not written to resolution_context in DB to avoid double writes at resolution time.
    # Since outcome is True, we know the final candle was used (first candle didn't breach).
    assert len(final_trade.resolution_context) == 1
    # assert final_trade.resolution_context[1] == [101.5, 100.0, 101.0]

@pytest.mark.asyncio
async def test_c01_stays_between_resolution_breach(mock_store, mock_resolver):
    """
    Verify C01 fix: Ensure STAYS_BETWEEN trade fails if barrier breached in post-entry candles.
    """
    now = datetime.now(timezone.utc)
    entry_price = 100.0
    barrier_offset = 1.0 
    barrier2_offset = -1.0
    duration_minutes = 2
    
    trade = ShadowTradeRecord.create(
        contract_type=CONTRACT_TYPES.STAYS_BETWEEN,
        direction="STAYS_BETWEEN",
        probability=0.8,
        entry_price=entry_price,
        reconstruction_error=0.01,
        regime_state="normal",
        tick_window=np.zeros(10),
        candle_window=np.zeros((10, 5)),
        barrier_level=barrier_offset,
        barrier2_level=barrier2_offset,
        duration_minutes=duration_minutes
    )
    await mock_store.append_async(trade)
    
    # Minute 1: Safe
    await mock_resolver.resolve_trades(
        current_price=100.0,
        current_time=now + timedelta(minutes=1),
        high_price=100.5,
        low_price=99.5
    )
    
    # Minute 2: BREACH upper barrier (101.2 > 101.0)
    await mock_resolver.resolve_trades(
        current_price=100.0,
        current_time=now + timedelta(minutes=2, seconds=1),
        high_price=101.2,
        low_price=99.5
    )
    
    final_trade = mock_store.get_by_id(trade.trade_id)
    assert final_trade.is_resolved()
    assert final_trade.outcome is False # LOSS - breached
