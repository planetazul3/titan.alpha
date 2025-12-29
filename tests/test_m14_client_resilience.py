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


class TestCircuitBreakerProbeLock:
    """Test HALF_OPEN probe lock-out behavior."""

    def test_half_open_allows_only_one_probe(self):
        """Verify only ONE request passes in HALF_OPEN state."""
        from data.ingestion.client import CircuitBreaker, CircuitState
        
        cb = CircuitBreaker(failure_threshold=1, initial_cooldown=0.01)
        
        # Trigger OPEN state
        cb.record_failure()
        assert cb.state == CircuitState.OPEN
        
        # Wait for cooldown to transition to HALF_OPEN
        import time
        time.sleep(0.02)
        
        # First request should be allowed (probe)
        assert cb.should_allow_request() is True
        assert cb.state == CircuitState.HALF_OPEN
        
        # Second concurrent request should be blocked
        assert cb.should_allow_request() is False
        assert cb.should_allow_request() is False  # Still blocked

    def test_half_open_probe_reset_on_success(self):
        """Verify probe lock resets after success."""
        from data.ingestion.client import CircuitBreaker, CircuitState
        
        cb = CircuitBreaker(failure_threshold=1, initial_cooldown=0.01)
        
        cb.record_failure()
        import time
        time.sleep(0.02)
        
        # First probe allowed
        assert cb.should_allow_request() is True
        assert cb._probing is True
        
        # Success resets everything
        cb.record_success()
        assert cb.state == CircuitState.CLOSED
        assert cb._probing is False
        
        # Now all requests allowed
        assert cb.should_allow_request() is True
        assert cb.should_allow_request() is True

    def test_half_open_probe_reset_on_failure(self):
        """Verify probe lock resets when going back to OPEN."""
        from data.ingestion.client import CircuitBreaker, CircuitState
        
        cb = CircuitBreaker(failure_threshold=1, initial_cooldown=0.01)
        
        cb.record_failure()
        import time
        time.sleep(0.02)
        
        # First probe allowed
        assert cb.should_allow_request() is True
        assert cb._probing is True
        
        # Failure during probe goes back to OPEN
        cb.record_failure()
        assert cb.state == CircuitState.OPEN
        assert cb._probing is False  # Reset when leaving HALF_OPEN

    def test_state_change_tracks_duration(self, caplog):
        """Verify state transitions log duration in previous state."""
        import logging
        from data.ingestion.client import CircuitBreaker, CircuitState
        
        with caplog.at_level(logging.INFO):
            cb = CircuitBreaker(failure_threshold=1, initial_cooldown=0.01)
            
            import time
            time.sleep(0.05)
            
            # Trigger state change
            cb.record_failure()
        
        # Check log contains duration info
        assert any("was in closed for" in record.message.lower() for record in caplog.records)


if __name__ == "__main__":
    pytest.main([__file__])
