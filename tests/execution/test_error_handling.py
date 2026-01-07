
import unittest
import asyncio
from unittest.mock import MagicMock, AsyncMock
from execution.executor import DerivTradeExecutor, TradeResult
from execution.common.types import ExecutionRequest
from deriv_api import APIError
from config.settings import Settings

from data.ingestion.client import CircuitState

class TestExecutorErrorHandling(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.client = AsyncMock()
        # Mock the circuit state to CLOSED to allow execution
        self.client.circuit_state = CircuitState.CLOSED
        
        self.settings = MagicMock(spec=Settings)
        self.settings.trading = MagicMock()
        self.settings.trading.stake_amount = 10.0
        self.settings.contracts = MagicMock() 
        self.settings.contracts.duration_rise_fall = 3
        self.settings.contracts.duration_touch = 3
        self.settings.contracts.duration_range = 3
        
        # Mock duration resolver defaults too just in case
        self.settings.trading.default_duration = 5
        self.settings.trading.default_duration_unit = "m"
        self.settings.trading.barrier_offset = 0.5
        
        self.executor = DerivTradeExecutor(self.client, self.settings)
        
        # Dummy signal as ExecutionRequest
        self.request = ExecutionRequest(
            signal_id="test_sig_1",
            symbol="R_100",
            contract_type="CALL",
            stake=10.0,
            duration=1,
            duration_unit="m"
        )

    async def test_api_error_handling(self):
        """Test handling of Deriv APIError."""
        # Mock client.buy to raise APIError
        error = APIError("RateLimit triggered")
        error.code = "RateLimit"
        self.client.buy.side_effect = error
        
        result = await self.executor.execute(self.request)
        
        self.assertFalse(result.success)
        self.assertIn("APIError", result.error)
        self.assertIn("RateLimit", result.error)
        # Check that it recorded failure
        self.assertEqual(self.executor._failed_count, 1)

    async def test_connection_error_handling(self):
        """Test handling of ConnectionError."""
        self.client.buy.side_effect = ConnectionError("Network down")
        
        result = await self.executor.execute(self.request)
        
        self.assertFalse(result.success)
        self.assertEqual(result.error, "ConnectionError")
        self.assertEqual(self.executor._failed_count, 1) # Should increment failure count

    async def test_shutdown_idempotency(self):
        """Test shutdown logic."""
        # Mock idempotency store
        mock_store = AsyncMock()
        self.executor.idempotency_store = mock_store
        
        await self.executor.shutdown()
        
        mock_store.close.assert_called_once()
        
    async def test_shutdown_no_store(self):
        """Test shutdown without store."""
        self.executor.idempotency_store = None
        # Should not raise
        await self.executor.shutdown()

if __name__ == "__main__":
    unittest.main()
