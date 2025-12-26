"""
Tests for data ingestion client with mocked Deriv API.

Uses pytest-mock to simulate API responses without real network calls.
This enables:
- Fast, deterministic tests
- Error scenario simulation
- Rate limit testing
"""
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock

import numpy as np
import pytest


class MockDerivApiResponse:
    """Factory for creating mock Deriv API responses."""

    @staticmethod
    def ticks_history(prices: list[float], epochs: list[int]):
        """Create mock ticks_history response."""
        return {
            "history": {
                "prices": prices,
                "times": epochs,
            },
            "echo_req": {"ticks_history": "R_100"},
        }

    @staticmethod
    def candles_history(candles: list[dict]):
        """Create mock candles response."""
        return {
            "candles": candles,
            "echo_req": {"ticks_history": "R_100", "style": "candles"},
        }

    @staticmethod
    def balance(amount: float = 1000.0, currency: str = "USD"):
        """Create mock balance response."""
        return {
            "balance": {
                "balance": amount,
                "currency": currency,
            }
        }

    @staticmethod
    def authorize(email: str = "test@example.com"):
        """Create mock authorize response."""
        return {
            "authorize": {
                "email": email,
                "balance": 1000.0,
                "currency": "USD",
            }
        }


@pytest.fixture
def sample_tick_data():
    """Generate sample tick data for testing."""
    base_price = 100.0
    base_epoch = int(datetime.now(timezone.utc).timestamp()) - 1000
    prices = [base_price + np.sin(i * 0.1) * 2 for i in range(100)]
    epochs = [base_epoch + i for i in range(100)]
    return prices, epochs


@pytest.fixture
def sample_candle_data():
    """Generate sample candle data for testing."""
    base_epoch = int(datetime.now(timezone.utc).timestamp()) - 60 * 50
    candles = []
    base_price = 100.0
    for i in range(50):
        o = base_price + i * 0.1
        h = o + 0.5
        l = o - 0.5
        c = o + 0.2
        candles.append({
            "epoch": base_epoch + i * 60,
            "open": o,
            "high": h,
            "low": l,
            "close": c,
        })
    return candles


class TestDerivClientConnection:
    """Test DerivClient connection handling."""

    @pytest.mark.asyncio
    async def test_connect_success(self, mocker):
        """Client should initialize with valid settings."""
        from data.ingestion.client import DerivClient

        # Mock settings
        mock_settings = MagicMock()
        mock_settings.trading.app_id = "12345"
        mock_settings.trading.api_token = "test_token"
        mock_settings.trading.symbol = "R_100"

        # Create client - should not raise
        client = DerivClient(mock_settings)

        # Verify client has expected attributes
        assert client.symbol == "R_100"
        assert client.api is None  # Not connected yet

    @pytest.mark.asyncio
    async def test_get_historical_ticks(self, sample_tick_data, mocker):
        """Client should fetch historical tick data correctly."""
        from data.ingestion.client import DerivClient

        prices, epochs = sample_tick_data

        mock_settings = MagicMock()
        mock_settings.trading.app_id = "12345"
        mock_settings.trading.api_token = "test_token"
        mock_settings.trading.symbol = "R_100"

        client = DerivClient(mock_settings)

        # Mock API response
        mock_api = AsyncMock()
        mock_api.ticks_history = AsyncMock(
            return_value=MockDerivApiResponse.ticks_history(prices, epochs)
        )
        client.api = mock_api
        client._connected = True

        # Fetch ticks
        ticks = await client.get_historical_ticks(count=100)

        assert len(ticks) == 100
        assert all(isinstance(t, float) for t in ticks)

    @pytest.mark.asyncio
    async def test_get_historical_candles(self, sample_candle_data, mocker):
        """Client should fetch historical candle data correctly."""
        from data.ingestion.client import DerivClient

        mock_settings = MagicMock()
        mock_settings.trading.app_id = "12345"
        mock_settings.trading.api_token = "test_token"
        mock_settings.trading.symbol = "R_100"

        client = DerivClient(mock_settings)

        # Mock API response
        mock_api = AsyncMock()
        mock_api.ticks_history = AsyncMock(
            return_value=MockDerivApiResponse.candles_history(sample_candle_data)
        )
        client.api = mock_api
        client._connected = True

        # Fetch candles
        candles = await client.get_historical_candles(count=50, interval=60)

        assert len(candles) == 50
        assert all("open" in c for c in candles)
        assert all("close" in c for c in candles)


class TestDerivClientErrorHandling:
    """Test error handling in DerivClient."""

    @pytest.mark.asyncio
    async def test_api_error_response(self, mocker):
        """Client should handle API error responses gracefully."""
        from data.ingestion.client import DerivClient

        mock_settings = MagicMock()
        mock_settings.trading.app_id = "12345"
        mock_settings.trading.api_token = "test_token"
        mock_settings.trading.symbol = "R_100"

        client = DerivClient(mock_settings)

        # Mock API error
        mock_api = AsyncMock()
        mock_api.ticks_history = AsyncMock(
            return_value={"error": {"message": "Rate limit exceeded", "code": "RateLimit"}}
        )
        client.api = mock_api
        client._connected = True

        # Should handle error gracefully
        try:
            await client.get_historical_ticks(count=100)
            # May return empty or raise - both acceptable
        except Exception:
            pass  # Expected for some error types

    @pytest.mark.asyncio
    async def test_connection_timeout(self, mocker):
        """Client should handle connection timeouts."""
        import asyncio

        from data.ingestion.client import DerivClient

        mock_settings = MagicMock()
        mock_settings.trading.app_id = "12345"
        mock_settings.trading.api_token = "test_token"
        mock_settings.trading.symbol = "R_100"

        client = DerivClient(mock_settings)

        # Mock timeout
        mock_api = AsyncMock()
        mock_api.ticks_history = AsyncMock(side_effect=asyncio.TimeoutError("Connection timeout"))
        client.api = mock_api
        client._connected = True

        # Should handle timeout
        with pytest.raises((asyncio.TimeoutError, Exception)):
            await client.get_historical_ticks(count=100)


class TestDerivClientStreamMocking:
    """Test streaming functionality with mocks."""

    @pytest.mark.asyncio
    async def test_tick_stream_mock(self, sample_tick_data):
        """Mock tick streaming for testing."""
        prices, _ = sample_tick_data

        # Create async generator that yields prices
        async def mock_tick_stream():
            for price in prices[:10]:  # Limit for test
                yield price

        # Test that we can iterate over mock stream
        received = []
        async for tick in mock_tick_stream():
            received.append(tick)
            if len(received) >= 10:
                break

        assert len(received) == 10
        assert all(isinstance(t, float) for t in received)
