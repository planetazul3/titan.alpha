"""
Normalized market event interfaces.

This module defines broker-agnostic event types for market data.
Strategy code should only depend on these normalized events, never
on broker-specific message formats or connection details.

This isolation allows:
- Swapping brokers without touching strategy code
- Testing with mocked event streams
- Replay of historical events for debugging
"""

from abc import abstractmethod
from collections.abc import AsyncGenerator
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Protocol


@dataclass(frozen=True)
class TickEvent:
    """
    Normalized tick (price update) event.

    Represents a single price update for an instrument.
    Broker-agnostic representation.

    Attributes:
        symbol: Instrument symbol
        price: Current price (quote)
        timestamp: Event timestamp (UTC)
        metadata: Optional broker-specific metadata
    """

    symbol: str
    price: float
    timestamp: datetime
    metadata: dict = field(default_factory=dict)


@dataclass(frozen=True)
class CandleEvent:
    """
    Normalized candlestick (OHLCV) event.

    Represents a completed candlestick for an instrument.
    Broker-agnostic representation.

    Attributes:
        symbol: Instrument symbol
        open: Opening price
        high: Highest price
        low: Lowest price
        close: Closing price
        volume: Trading volume (may be 0 if unavailable)
        timestamp: Candle close timestamp (UTC)
        metadata: Optional broker-specific metadata
    """

    symbol: str
    open: float
    high: float
    low: float
    close: float
    volume: float
    timestamp: datetime
    metadata: dict = field(default_factory=dict)


class MarketEventBus(Protocol):
    """
    Protocol for normalized market event streams.

    Strategy code depends on this interface, NOT on specific broker APIs.
    This enables:
    - Broker swapping without strategy changes
    - Mock event streams for testing
    - Historical replay for debugging

    Implementations:
    - DerivEventAdapter: Adapter for Deriv.com API
    - MockEventBus: In-memory stub for testing
    - ReplayEventBus: Historical data replay

    Example:
        >>> async def trading_strategy(event_bus: MarketEventBus):
        ...     async for tick in event_bus.subscribe_ticks("R_100"):
        ...         print(f"Price: {tick.price}")
        ...
        >>> # Works with any implementation
        >>> event_bus = DerivEventAdapter(deriv_client)
        >>> # OR
        >>> event_bus = MockEventBus()
    """

    @abstractmethod
    async def subscribe_ticks(self, symbol: str) -> AsyncGenerator[TickEvent, None]:
        """
        Subscribe to tick events for a symbol.

        Args:
            symbol: Instrument symbol to subscribe to

        Yields:
            TickEvent instances as they occur

        Raises:
            ConnectionError: If unable to establish subscription
        """

        if False:
             yield TickEvent("", 0.0, datetime.now())

    @abstractmethod
    async def subscribe_candles(
        self, symbol: str, interval: int = 60
    ) -> AsyncGenerator[CandleEvent, None]:
        """
        Subscribe to candle events for a symbol.

        Args:
            symbol: Instrument symbol to subscribe to
            interval: Candle interval in seconds (default: 60 for 1m)

        Yields:
            CandleEvent instances as candles complete

        Raises:
            ConnectionError: If unable to establish subscription
        """
        if False:
             yield CandleEvent("", 0.0, 0.0, 0.0, 0.0, 0.0, datetime.now())


class MockEventBus(MarketEventBus):
    """
    Mock event bus for testing.

    Generates synthetic events for unit testing without broker connection.

    Example:
        >>> mock_bus = MockEventBus()
        >>> tick_count = 0
        >>> async for tick in mock_bus.subscribe_ticks("TEST"):
        ...     tick_count += 1
        ...     if tick_count >= 10:
        ...         break
    """

    async def subscribe_ticks(self, symbol: str) -> AsyncGenerator[TickEvent, None]:
        """Generate mock tick events."""
        import asyncio
        import random

        base_price = 100.0
        while True:
            await asyncio.sleep(0.1)  # 10 ticks/sec
            price = base_price + random.uniform(-0.5, 0.5)
            yield TickEvent(symbol=symbol, price=price, timestamp=datetime.now(timezone.utc))

    async def subscribe_candles(
        self, symbol: str, interval: int = 60
    ) -> AsyncGenerator[CandleEvent, None]:
        """Generate mock candle events."""
        import asyncio
        import random

        base_price = 100.0
        while True:
            await asyncio.sleep(interval)
            open_price = base_price + random.uniform(-1.0, 1.0)
            close_price = open_price + random.uniform(-0.5, 0.5)
            high_price = max(open_price, close_price) + random.uniform(0, 0.3)
            low_price = min(open_price, close_price) - random.uniform(0, 0.3)

            yield CandleEvent(
                symbol=symbol,
                open=open_price,
                high=high_price,
                low=low_price,
                close=close_price,
                volume=random.uniform(100, 1000),
                timestamp=datetime.now(timezone.utc),
            )


class ReplayEventBus(MarketEventBus):
    """
    Replay event bus for historical data playback.

    Replays pre-recorded tick and candle data as if they were live events.
    Essential for:
    - Backtesting strategies
    - Debugging with deterministic data
    - Offline development

    Example:
        >>> import numpy as np
        >>> ticks = np.array([100.0, 100.5, 101.0, 100.8])
        >>> timestamps = [datetime(2024, 1, 1, 0, 0, i) for i in range(4)]
        >>>
        >>> replay_bus = ReplayEventBus(tick_prices=ticks, tick_timestamps=timestamps)
        >>> async for tick in replay_bus.subscribe_ticks("R_100"):
        ...     print(f"{tick.timestamp}: {tick.price}")
    """

    def __init__(
        self,
        tick_prices: list | None = None,
        tick_timestamps: list | None = None,
        candles: list | None = None,  # List of (open, high, low, close, volume, timestamp)
        replay_speed: float = 1.0,  # 1.0 = realtime, 0 = instant
    ):
        """
        Initialize replay event bus.

        Args:
            tick_prices: List of tick prices
            tick_timestamps: List of datetime objects for each tick
            candles: List of candle tuples (open, high, low, close, volume, timestamp)
            replay_speed: Speed multiplier (1.0 = realtime, 0 = instant)
        """
        self._tick_prices = tick_prices or []
        self._tick_timestamps = tick_timestamps or []
        self._candles = candles or []
        self._replay_speed = replay_speed

    async def subscribe_ticks(self, symbol: str) -> AsyncGenerator[TickEvent, None]:
        """Replay historical tick data."""
        import asyncio

        for i, price in enumerate(self._tick_prices):
            # Calculate delay if we have timestamps
            if self._replay_speed > 0 and i > 0 and self._tick_timestamps:
                time_diff = (
                    self._tick_timestamps[i] - self._tick_timestamps[i - 1]
                ).total_seconds()
                await asyncio.sleep(time_diff / self._replay_speed)

            timestamp = (
                self._tick_timestamps[i] if i < len(self._tick_timestamps) else datetime.now(timezone.utc)
            )

            yield TickEvent(
                symbol=symbol,
                price=float(price),
                timestamp=timestamp,
                metadata={"source": "replay", "index": i},
            )

    async def subscribe_candles(
        self, symbol: str, interval: int = 60
    ) -> AsyncGenerator[CandleEvent, None]:
        """Replay historical candle data."""
        import asyncio

        for i, candle in enumerate(self._candles):
            # Calculate delay if we have timestamps
            if self._replay_speed > 0 and i > 0:
                prev_ts = self._candles[i - 1][5]
                curr_ts = candle[5]
                if isinstance(prev_ts, datetime) and isinstance(curr_ts, datetime):
                    time_diff = (curr_ts - prev_ts).total_seconds()
                    await asyncio.sleep(time_diff / self._replay_speed)

            timestamp = candle[5] if isinstance(candle[5], datetime) else datetime.now(timezone.utc)

            yield CandleEvent(
                symbol=symbol,
                open=float(candle[0]),
                high=float(candle[1]),
                low=float(candle[2]),
                close=float(candle[3]),
                volume=float(candle[4]) if len(candle) > 4 else 0.0,
                timestamp=timestamp,
                metadata={"source": "replay", "index": i},
            )
