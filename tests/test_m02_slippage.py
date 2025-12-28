import pytest
import asyncio
from unittest.mock import AsyncMock, MagicMock
from pydantic import SecretStr
from data.ingestion.client import DerivClient
from config.settings import Settings, Trading

class TestSlippageProtection:
    @pytest.mark.asyncio
    async def test_buy_slippage_limit(self):
        """Verify that buy() calls API with price limit exactly equal to stake amount."""
        
        # Setup using model_construct for frozen settings
        trading = Trading.model_construct(
            symbol="R_100",
            stake_amount=10.0,
            barrier_offset=0.5,
            barrier2_offset=-0.5
        )
        settings = Settings.model_construct(
            trading=trading,
            deriv_api_token=SecretStr("fake_token"),
            deriv_app_id=1089,
        )
        
        client = DerivClient(settings)
        client.api = AsyncMock()
        
        # Mock responses
        client.api.proposal.return_value = {
            "proposal": {"id": "prop_123", "ask_price": 10.0}
        }
        client.api.buy.return_value = {
            "buy": {"contract_id": "cont_456"}
        }
        
        # Execute
        amount = 10.0
        await client.buy("CALL", amount, 1, "m")
        
        # Verify
        # Check that api.buy was called with price=10.0, NOT 110.0
        client.api.buy.assert_called_once()
        call_args = client.api.buy.call_args[0][0]
        
        assert call_args["buy"] == "prop_123"
        assert call_args["price"] == amount
        assert call_args["price"] == 10.0

if __name__ == "__main__":
    asyncio.run(TestSlippageProtection().test_buy_slippage_limit())
