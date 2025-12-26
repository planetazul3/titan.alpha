import pytest
import os
import shutil
from pathlib import Path
from unittest.mock import Mock, AsyncMock
import asyncio

from execution.safety_store import SQLiteSafetyStateStore
from execution.adaptive_risk import AdaptiveRiskManager
from execution.safety import SafeTradeExecutor, ExecutionSafetyConfig
from execution.executor import TradeResult, TradeSignal
from config.constants import CONTRACT_TYPES, SIGNAL_TYPES
from datetime import datetime

TEST_DB_PATH = Path("test_data/safety_test.db")

@pytest.fixture
def clean_db():
    if TEST_DB_PATH.exists():
        os.remove(TEST_DB_PATH)
    TEST_DB_PATH.parent.mkdir(exist_ok=True, parents=True)
    yield TEST_DB_PATH
    if TEST_DB_PATH.exists():
        os.remove(TEST_DB_PATH)

def test_store_persistence(clean_db):
    """Verify raw store persistence."""
    store1 = SQLiteSafetyStateStore(clean_db)
    store1.set_value("foo", "bar")
    store1.increment_daily_trade_count()
    store1.update_daily_pnl(-10.0)
    
    # Re-open
    store2 = SQLiteSafetyStateStore(clean_db)
    assert store2.get_value("foo") == "bar"
    
    count, pnl = store2.get_daily_stats()
    assert count == 1
    assert pnl == -10.0

def test_risk_manager_persistence(clean_db):
    """Verify AdaptiveRiskManager restores state."""
    store = SQLiteSafetyStateStore(clean_db)
    
    # Init manager and simulate trading activity
    risk_mgr = AdaptiveRiskManager(state_store=store)
    
    # Record a loss to update drawdown
    # Peak equity default is 0.0 in PerformanceTracker, so we need to set positive equity first
    risk_mgr.record_trade(0.0, current_equity=1000.0) # Set peak to 1000
    risk_mgr.record_trade(-100.0, current_equity=900.0) # Drawdown 10%
    
    # Verify in-memory state
    assert risk_mgr.performance.get_drawdown() == 0.1
    
    # Re-init manager with same store (simulate restart)
    # Note: store instance is "fresh" view on same DB
    risk_mgr2 = AdaptiveRiskManager(state_store=store) # Should auto-load
    
    assert risk_mgr2.performance.get_drawdown() == 0.1
    assert risk_mgr2.performance._peak_equity == 1000.0

@pytest.mark.asyncio
async def test_safe_executor_daily_limit(clean_db):
    """Verify SafeTradeExecutor respects persisted daily limits."""
    
    # Config: Max 2 trades/day for testing (via pnl limit or trade count?)
    # Config has max_daily_loss. Let's use that.
    config = ExecutionSafetyConfig(max_daily_loss=50.0)
    
    mock_inner = AsyncMock()
    mock_inner.execute.return_value = TradeResult(success=True)
    
    executor = SafeTradeExecutor(
        inner_executor=mock_inner,
        config=config,
        state_file=clean_db
    )
    
    # 1. Simulate previous losses stored in DB
    executor.store.update_daily_pnl(-60.0) # Exceeds limit of 50
    
    signal = TradeSignal(
        signal_type=SIGNAL_TYPES.REAL_TRADE,
        contract_type=CONTRACT_TYPES.RISE_FALL,
        direction="CALL",
        probability=0.8,
        timestamp=datetime.now(),
        metadata={"symbol": "R_100"}
    )
    
    # 2. Attempt trade - should be rejected
    result = await executor.execute(signal)
    
    assert not result.success
    assert "Daily limits exceeded" in result.error
    assert mock_inner.execute.call_count == 0

@pytest.mark.asyncio
async def test_safe_executor_rate_limit(clean_db):
    """Verify rate limits."""
    config = ExecutionSafetyConfig(max_trades_per_minute=2)
    mock_inner = AsyncMock()
    mock_inner.execute.return_value = TradeResult(success=True)
    
    executor = SafeTradeExecutor(
        inner_executor=mock_inner,
        config=config,
        state_file=clean_db
    )
    
    signal = TradeSignal(
        signal_type=SIGNAL_TYPES.REAL_TRADE,
        contract_type=CONTRACT_TYPES.RISE_FALL,
        direction="CALL",
        probability=0.8,
        timestamp=datetime.now(),
        metadata={"symbol": "R_100"}
    )
    
    # Execute 2 trades (allowed)
    r1 = await executor.execute(signal)
    r2 = await executor.execute(signal)
    assert r1.success
    assert r2.success
    
    # Execute 3rd trade (should fail rate limit)
    r3 = await executor.execute(signal)
    assert not r3.success
    assert "Rate limit exceeded" in r3.error
