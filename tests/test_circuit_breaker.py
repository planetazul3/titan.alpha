
import unittest
from unittest.mock import MagicMock, patch, AsyncMock
import asyncio
from data.ingestion.client import DerivClient, CircuitState
from config.settings import Settings

class TestCircuitBreaker(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.mock_settings = MagicMock()
        self.mock_settings.trading.symbol = "R_100"
        self.mock_settings.deriv_app_id = 12345
        self.mock_settings.deriv_api_token.get_secret_value.return_value = "token"
        
        self.client = DerivClient(self.mock_settings)
        self.client.api = AsyncMock()

    async def test_buy_checks_circuit_breaker(self):
        """Test that buy() checks circuit breaker state."""
        # Force circuit open and ensure it stays open
        self.client._circuit_breaker._state = CircuitState.OPEN
        self.client._circuit_breaker._current_cooldown = 100000 # Long cooldown
        self.client._circuit_breaker._last_failure_time = asyncio.get_running_loop().time() + 1000 # Future? No, monotonic.
        # Just use time.monotonic()
        import time
        self.client._circuit_breaker._last_failure_time = time.monotonic()
        
        with self.assertRaises(RuntimeError) as cm:
            await self.client.buy("CALL", 10.0, 1)
            
        self.assertIn("Circuit breaker open", str(cm.exception))
        
        # Verify API was NOT called
        self.client.api.proposal.assert_not_called()

    async def test_buy_records_success(self):
        """Test that successful buy records success."""
        # Setup successful response
        self.client.api.proposal.return_value = {"proposal": {"id": "123"}}
        self.client.api.buy.return_value = {"buy": {"contract_id": "456"}}
        
        # Pre-fail to set count > 0
        self.client._circuit_breaker._failure_count = 3
        
        await self.client.buy("CALL", 10.0, 1)
        
        # Should reset failure count
        self.assertEqual(self.client._circuit_breaker._failure_count, 0)
        self.assertEqual(self.client._circuit_breaker.state, CircuitState.CLOSED)

    async def test_buy_records_failure(self):
        """Test that failed buy records failure."""
        # Setup failure
        self.client.api.proposal.side_effect = Exception("API Error")
        
        with self.assertRaises(Exception):
            await self.client.buy("CALL", 10.0, 1)
            
        self.assertEqual(self.client._circuit_breaker._failure_count, 1)
        
        # Trigger threshold
        self.client._circuit_breaker.failure_threshold = 2
        with self.assertRaises(Exception):
            await self.client.buy("CALL", 10.0, 1)
            
        self.assertEqual(self.client._circuit_breaker.state, CircuitState.OPEN)

    async def test_get_open_contracts_circuit_breaker(self):
        """Test get_open_contracts uses circuit breaker."""
        import time
        self.client._circuit_breaker._state = CircuitState.OPEN
        self.client._circuit_breaker._current_cooldown = 100000
        self.client._circuit_breaker._last_failure_time = time.monotonic()
        
        with self.assertRaises(RuntimeError):
            await self.client.get_open_contracts()
            
    async def test_subscribe_contract_circuit_breaker(self):
        """Test subscribe_contract uses circuit breaker."""
        import time
        self.client._circuit_breaker._state = CircuitState.OPEN
        self.client._circuit_breaker._current_cooldown = 100000
        self.client._circuit_breaker._last_failure_time = time.monotonic()
        
        result = await self.client.subscribe_contract("123")
        self.assertFalse(result)

if __name__ == '__main__':
    unittest.main()
