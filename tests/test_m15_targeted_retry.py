
import pytest
import asyncio
from datetime import datetime
from unittest.mock import MagicMock, AsyncMock
from execution.safety import SafeTradeExecutor, ExecutionSafetyConfig, TradeSignal
from execution.executor import TradeExecutor, TradeResult
from config.constants import SIGNAL_TYPES, CONTRACT_TYPES

class TestTargetedRetry:
    
    @pytest.mark.asyncio
    async def test_transport_error_fails_fast(self, tmp_path):
        """Verify that transport errors fail fast (handled by DerivClient).
        
        Transport errors (ConnectionError, TimeoutError) are NO LONGER retried
        at the SafeTradeExecutor level because DerivClient._reconnect already
        handles them with its own exponential backoff.
        """
        # Setup
        mock_inner = MagicMock(spec=TradeExecutor)
        mock_inner.execute = AsyncMock(side_effect=ConnectionError("Network Blip"))
        
        config = ExecutionSafetyConfig(max_retry_attempts=3, retry_base_delay=0.01)
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
        async with asyncio.timeout(2):  # fail-safe
            result = await executor._execute_with_retry(signal)
            
        # Verify: fails fast without retries (count = 1)
        assert not result.success
        assert "Transport Error" in result.error
        assert mock_inner.execute.call_count == 1

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
