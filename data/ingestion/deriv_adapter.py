"""
Deriv API adapter for MarketEventBus.

This adapter translates Deriv-specific API messages into normalized
MarketEventBus events. Strategy code never sees Deriv API details.

This isolation enables:
- Testing with mock event buses
- Swapping to different brokers
- Replay of historical events
"""

import logging
from collections.abc import AsyncGenerator
from datetime import datetime, timezone

from data.events import CandleEvent, MarketEventBus, TickEvent
from data.ingestion.client import DerivClient

logger = logging.getLogger(__name__)


class DerivEventAdapter(MarketEventBus):
    """
    Adapter that translates Deriv API to normalized events.

    This is the ONLY place in the codebase that knows about Deriv API details.
    Strategy code depends on MarketEventBus interface, not this adapter.

    Attributes:
        client: Connected Deriv# client instance

    Example:
        >>> from config.settings import load_settings
        >>> settings = load_settings()
        >>> client = DerivClient(settings)
        >>> await client.connect()
        >>>
        >>> # Strategy code uses normalized interface
        >>> event_bus = DerivEventAdapter(client)
        >>> async for tick in event_bus.subscribe_ticks(settings.trading.symbol):
        ...     process_tick(tick)  # Receives TickEvent, not Deriv message
    """

    def __init__(self, client: DerivClient):
        """
        Initialize Deriv adapter.

        Args:
            client: Connected DerivClient instance
        """
        if not client.api:
            raise ValueError("DerivClient must be connected before creating adapter")

        self.client = client
        logger.info("DerivEventAdapter initialized")

    async def subscribe_ticks(self, symbol: str) -> AsyncGenerator[TickEvent, None]:
        """
        Subscribe to Deriv tick stream, yield normalized TickEvents.

        Translates Deriv API tick messages into broker-agnostic TickEvent objects.

        Args:
            symbol: Deriv symbol (e.g., 'R_100')

        Yields:
            TickEvent instances

        Raises:
            ConnectionError: If Deriv API connection fails
        """
        logger.info(f"Subscribing to Deriv ticks for {symbol}")

        try:
            async for price in self.client.stream_ticks():
                # Translate Deriv price to TickEvent
                yield TickEvent(
                    symbol=symbol,
                    price=float(price),
                    timestamp=datetime.now(timezone.utc),
                    metadata={"source": "deriv"},
                )
        except Exception as e:
            logger.error(f"Deriv tick stream error: {e}")
            raise ConnectionError(f"Failed to stream ticks from Deriv: {e}")

    async def subscribe_candles(
        self, symbol: str, interval: int = 60
    ) -> AsyncGenerator[CandleEvent, None]:
        """
        Subscribe to Deriv candle stream, yield normalized CandleEvents.

        Translates Deriv API OHLC messages into broker-agnostic CandleEvent objects.

        Args:
            symbol: Deriv symbol (e.g., 'R_100')
            interval: Candle interval in seconds (default: 60 for 1m)

        Yields:
            CandleEvent instances

        Raises:
            ConnectionError: If Deriv API connection fails
        """
        logger.info(f"Subscribing to Deriv candles for {symbol} ({interval}s)")

        try:
            async for candle_dict in self.client.stream_candles(interval=interval):
                # Translate Deriv candle dict to CandleEvent
                # Deriv candle format: {'open': ..., 'high': ..., 'low': ..., 'close': ..., 'epoch': ...}
                yield CandleEvent(
                    symbol=symbol,
                    open=float(candle_dict["open"]),
                    high=float(candle_dict["high"]),
                    low=float(candle_dict["low"]),
                    close=float(candle_dict["close"]),
                    volume=0.0,  # Deriv doesn't provide volume for synthetic indices
                    timestamp=datetime.fromtimestamp(candle_dict["epoch"], tz=timezone.utc),
                    metadata={"source": "deriv", "epoch": candle_dict["epoch"]},
                )
        except Exception as e:
            logger.error(f"Deriv candle stream error: {e}")
            raise ConnectionError(f"Failed to stream candles from Deriv: {e}")

    async def get_balance(self) -> float:
        """
        Get account balance (broker-specific extension).

        Note: This is NOT part of MarketEventBus interface.
        Only use when broker-specific operations are required.
        """
        return await self.client.get_balance()
