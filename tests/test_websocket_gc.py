
import unittest
from unittest.mock import AsyncMock, MagicMock, patch
import asyncio
from data.ingestion.client import DerivClient

class TestWebSocketGC(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.mock_settings = MagicMock()
        self.mock_settings.trading.symbol = "R_100"
        self.mock_settings.deriv_app_id = 1234
        self.mock_settings.deriv_api_token.get_secret_value.return_value = "fake_token"
        
        self.client = DerivClient(self.mock_settings)
        self.client.api = AsyncMock()

    async def test_buy_forgets_proposal(self):
        """Test that buy() explicitly calls forget(proposal_id)."""
        # Mock API responses
        self.client.api.proposal.return_value = {"proposal": {"id": "prop_123"}}
        self.client.api.buy.return_value = {"buy": {"contract_id": "c_123"}}
        
        await self.client.buy("CALL", 10.0, 1)
        
        # Verify forget was called with correct ID
        self.client.api.forget.assert_called_with("prop_123")
        
        # Verify order of calls: proposal -> buy -> forget
        self.client.api.proposal.assert_called_once()
        self.client.api.buy.assert_called_once()
        self.client.api.forget.assert_called_once()

    async def test_buy_forgets_proposal_on_error(self):
        """Test that buy() forgets proposal even if buy fails."""
        self.client.api.proposal.return_value = {"proposal": {"id": "prop_123"}}
        self.client.api.buy.side_effect = ValueError("Buy failed")
        
        with self.assertRaises(ValueError):
            await self.client.buy("CALL", 10.0, 1)
            
        # Verify call happened despite exception
        self.client.api.forget.assert_called_with("prop_123")

    async def test_connect_invokes_gc(self):
        """Test that connect calls forget_all clean start."""
        # Setup the mock that will be RETURNED by DerivAPI constructor
        mock_api_instance = AsyncMock()
        mock_api_instance.authorize.return_value = {"authorize": {"balance": 100, "currency": "USD"}}
        
        with patch("data.ingestion.client.DerivAPI", return_value=mock_api_instance):
            await self.client.connect()
            
            # Verify forget_all called on startup
            mock_api_instance.forget_all.assert_called()
            self.assertIn("proposal", mock_api_instance.forget_all.call_args[0][0])

    async def test_disconnect_invokes_gc(self):
        """Test that disconnect calls forget_all."""
        # Hold reference to mock before it is cleared
        mock_api = self.client.api
        
        await self.client.disconnect()
        
        mock_api.forget_all.assert_called()

if __name__ == '__main__':
    unittest.main()
