import pytest
import asyncio
import time
from unittest.mock import MagicMock, AsyncMock
from execution.executor import DerivTradeExecutor, TradeResult
from execution.common.types import ExecutionRequest
from execution.signals import TradeSignal
from execution.idempotency_store import SQLiteIdempotencyStore

@pytest.fixture
def store(tmp_path):
    return SQLiteIdempotencyStore(tmp_path / "stress.db")

@pytest.fixture
def mock_client():
    client = MagicMock()
    # Simulate slight latency
    async def fast_buy(**kwargs):
        await asyncio.sleep(0.01)
        return {"buy": {"contract_id": f"CID_{int(time.time()*1000)}", "buy_price": 10.0}}
    client.buy = AsyncMock(side_effect=fast_buy)
    return client

@pytest.fixture
def mock_settings():
    settings = MagicMock()
    settings.trading.stake_amount = 10.0
    return settings

@pytest.fixture
def executor(mock_client, mock_settings, store):
    return DerivTradeExecutor(
        client=mock_client, 
        settings=mock_settings, 
        idempotency_store=store
    )

@pytest.mark.asyncio
async def test_concurrent_execution_idempotency(executor):
    """Verify that multiple concurrent calls for SAME signal result in only ONE execution."""
    signal_id = "concurrent_sig_1"
    
    # Create 20 concurrent requests for same signal
    req = ExecutionRequest(
        signal_id=signal_id,
        symbol="R_100", 
        contract_type="CALL",
        stake=10.0,
        duration=1,
        duration_unit="m"
    )
    
    tasks = [executor.execute(req) for _ in range(20)]
    results = await asyncio.gather(*tasks)
    
    # Analyze results
    success_count = sum(1 for r in results if r.success)
    assert success_count == 20 # All should "succeed" from caller perspective
    
    # But only one should have actually called the API (contract_id is unique per call in mock)
    # Actually, executor returns existing contract_id if found.
    # In atomic check, everyone gets success=True. The winner gets is_new=True. The losers get is_new=False.
    # Executor logic: if not is_new, return TradeResult(success=True, contract_id=cached_id)
    # So we need to check how many unique contract_ids we got? 
    # Or check executor._executed_count if meaningful?
    
    # If API called once, mock_client.buy.call_count should be 1
    assert executor.client.buy.call_count == 1
    
    # Check that all results have same contract_id eventually?
    # actually, the first one might still be in progress while others hit "PENDING"
    # Executor "PENDING" handling returns success=True, contract_id="PENDING" or None?
    # Let's check logic:
    # is_new, cached_id = check_and_reserve
    # if not is_new: return cached_id
    # If it's PENDING, it returns "PENDING".
    
    contract_ids = [r.contract_id for r in results]
    assert "PENDING" in contract_ids or executor.client.buy.call_count == 1
    # It's possible the winner finishes before some losers check, so they might see the real ID.
    # But definitely call count should be 1.

@pytest.mark.asyncio
async def test_high_concurrency_stress(executor):
    """Verify system stability under high load of DIFFERENT signals."""
    count = 50
    tasks = []
    
    for i in range(count):
        req = ExecutionRequest(
            signal_id=f"stress_sig_{i}",
            symbol="R_100", 
            contract_type="CALL",
            stake=10.0,
            duration=1,
            duration_unit="m"
        )
        tasks.append(executor.execute(req))
        
    start = time.time()
    results = await asyncio.gather(*tasks)
    end = time.time()
    
    assert len(results) == count
    assert executor.client.buy.call_count == count
    
    print(f"Executed {count} trades in {end-start:.2f}s")
