import pytest
import asyncio
from unittest.mock import MagicMock, patch, AsyncMock
from data.ingestion.client import DerivClient

class TestClientResilience:
    @pytest.mark.asyncio
    async def test_jitter_backoff(self):
        """Verify that backoff duration includes jitter."""
        settings = MagicMock()
        settings.trading.symbol = "R_100"
        client = DerivClient(settings)
        
        # Mock APIError to trigger retry logic
        from deriv_api import APIError
        
        # We need to mock DerivAPI to raise APIError on init/authorize
        # But DerivAPI is instantiated inside connect()
        
        with patch("data.ingestion.client.DerivAPI") as mock_api_cls:
            mock_api = MagicMock()
            mock_api_cls.return_value = mock_api
            # Make ping raise error first few times
            mock_api.ping = AsyncMock(side_effect=[APIError("Fail1"), APIError("Fail2"), {"ping": 1}])
            mock_api.authorize = AsyncMock(return_value={"authorize": {"balance": 100, "currency": "USD"}})
            
            # Mock asyncio.sleep to capture duration
            with patch("asyncio.sleep", new_callable=AsyncMock) as mock_sleep:
                await client.connect(max_retries=3)
                
                # Should have slept 2 times
                assert mock_sleep.call_count == 2
                
                # Check durations - shouldn't be exactly 1.0 and 2.0 due to jitter
                delays = [call.args[0] for call in mock_sleep.call_args_list]
                
                # Attempt 0: 2^0 = 1. Range 0.5 - 1.5
                assert 0.5 <= delays[0] <= 1.5
                assert delays[0] != 1.0 # Unlikely to be exactly 1.0 with float random
                
                # Attempt 1: 2^1 = 2. Range 1.0 - 3.0
                assert 1.0 <= delays[1] <= 3.0

    @pytest.mark.asyncio
    async def test_is_connected(self):
        """Verify is_connected property."""
        settings = MagicMock()
        client = DerivClient(settings)
        
        assert not client.is_connected
        
        # Manually pretend we are connected
        client.api = MagicMock()
        client._keep_alive_task = asyncio.create_task(asyncio.sleep(0.1))
        
        assert client.is_connected
        
        # Cancel task
        client._keep_alive_task.cancel()
        try:
            await client._keep_alive_task
        except asyncio.CancelledError:
            pass
            
        assert not client.is_connected

if __name__ == "__main__":
    pytest.main([__file__])
