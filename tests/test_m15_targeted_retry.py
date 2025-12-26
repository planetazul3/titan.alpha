
import pytest
import asyncio
from datetime import datetime
from unittest.mock import MagicMock, AsyncMock
from execution.safety import SafeTradeExecutor, ExecutionSafetyConfig, TradeSignal
from execution.executor import TradeExecutor, TradeResult
from config.constants import SIGNAL_TYPES, CONTRACT_TYPES

class TestTargetedRetry:
    
    @pytest.mark.asyncio
    async def test_transient_error_retry(self, tmp_path):
        """Verify that transient errors trigger retries."""
        # Setup
        mock_inner = MagicMock(spec=TradeExecutor)
        # Fail twice with ConnectionError, then succeed
        mock_inner.execute = AsyncMock(side_effect=[
            ConnectionError("Network Blip"),
            ConnectionError("Still Down"),
            TradeResult(success=True, contract_id="123")
        ])
        
        config = ExecutionSafetyConfig(max_retry_attempts=3, retry_base_delay=0.01)
        # Dummy DB path
        db_path = tmp_path / "safety.db"
        
        executor = SafeTradeExecutor(mock_inner, config, db_path)
        
        # Test
        signal = TradeSignal(
            signal_type=SIGNAL_TYPES.REAL_TRADE,
            contract_type=CONTRACT_TYPES.RISE_FALL,
            direction="CALL",
            probability=0.8,
            timestamp=datetime.now(),
            metadata={"symbol": "test"}
        )
        async with asyncio.timeout(2): # fail-safe
            result = await executor._execute_with_retry(signal)
            
        # Verify
        assert result.success
        assert mock_inner.execute.call_count == 3

    @pytest.mark.asyncio
    async def test_permanent_error_fail_fast(self, tmp_path):
        """Verify that permanent errors fail immediately without retry."""
        # Setup
        mock_inner = MagicMock(spec=TradeExecutor)
        # Fail with ValueError (logic error)
        mock_inner.execute = AsyncMock(side_effect=ValueError("Invalid Parameter"))
        
        config = ExecutionSafetyConfig(max_retry_attempts=3, retry_base_delay=0.1)
        db_path = tmp_path / "safety2.db"
        
        executor = SafeTradeExecutor(mock_inner, config, db_path)
        
        # Test
        signal = TradeSignal(
            signal_type=SIGNAL_TYPES.REAL_TRADE,
            contract_type=CONTRACT_TYPES.RISE_FALL,
            direction="CALL",
            probability=0.8,
            timestamp=datetime.now(),
            metadata={"symbol": "test"}
        )
        
        # Should return error result immediately (call count 1)
        # Note: function catches Exception and returns failed TradeResult, so it won't raise.
        result = await executor._execute_with_retry(signal)
        
        # Verify
        assert not result.success
        assert "Permanent Error" in result.error
        assert mock_inner.execute.call_count == 1 # Did NOT retry

if __name__ == "__main__":
    pytest.main([__file__])
