
import unittest
import asyncio
from unittest.mock import MagicMock, AsyncMock
from execution.executor import DerivTradeExecutor, TradeResult
from execution.signals import TradeSignal
from deriv_api import APIError
from config.settings import Settings

class TestExecutorErrorHandling(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.client = AsyncMock()
        self.settings = MagicMock(spec=Settings)
        self.settings.trading = MagicMock()
        self.settings.trading.stake_amount = 10.0
        self.settings.contracts = MagicMock() # Needed for DurationResolver
        # Mock duration resolver to return fixed values
        self.settings.trading.default_duration = 5
        self.settings.trading.default_duration_unit = "m"
        
        self.executor = DerivTradeExecutor(self.client, self.settings)
        
        # Dummy signal
        self.signal = TradeSignal(
            signal_id="test_sig_1",
            signal_type="ML_MODEL",
            timestamp=1000.0,
            direction="CALL",
            contract_type="RISE_FALL",
            probability=0.8,
            metadata={"symbol": "R_100"}
        )

    async def test_api_error_handling(self):
        """Test handling of Deriv APIError."""
        # Mock client.buy to raise APIError
        error = APIError("RateLimit triggered")
        error.code = "RateLimit"
        self.client.buy.side_effect = error
        
        result = await self.executor.execute(self.signal)
        
        self.assertFalse(result.success)
        self.assertIn("APIError", result.error)
        self.assertIn("RateLimit", result.error)
        # Check that it recorded failure
        self.assertEqual(self.executor._failed_count, 1)

    async def test_connection_error_handling(self):
        """Test handling of ConnectionError."""
        self.client.buy.side_effect = ConnectionError("Network down")
        
        result = await self.executor.execute(self.signal)
        
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
