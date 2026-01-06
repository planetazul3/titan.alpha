"""
Market Data Buffer.

This module provides the MarketDataBuffer class which encapsulates
buffering logic for tick and candle data streams.

The buffer handles:
1. Fixed-size sliding windows for ticks and candles
2. Candle close detection (new period vs in-place update)
3. Ready state checking for minimum data requirements

This abstraction was extracted from scripts/live.py per architectural
audit recommendations to improve testability and separation of concerns.

Example:
    >>> buffer = MarketDataBuffer(tick_length=128, candle_length=64)
    >>> buffer.append_tick(1.2345)
    >>> is_new = buffer.update_candle(candle_event)
    >>> if is_new and buffer.is_ready():
    ...     run_inference(buffer.get_ticks(), buffer.get_candles())
"""

import collections
import logging
from dataclasses import dataclass
from datetime import datetime
from typing import Any

import numpy as np


from typing import TypedDict, Any

class MarketSnapshot(TypedDict):
    ticks: list[float]
    candles: list[list[float]] # or np.ndarray if we convert
    tick_count: int
    candle_count: int
    timestamp: float

logger = logging.getLogger(__name__)


@dataclass
class CandleData:
    """Normalized candle representation."""

    open: float
    high: float
    low: float
    close: float
    volume: float
    timestamp: float  # Unix timestamp

    def to_array(self) -> list[float]:
        """Convert to array format [open, high, low, close, volume, timestamp]."""
        return [self.open, self.high, self.low, self.close, self.volume, self.timestamp]

    @classmethod
    def from_array(cls, arr: list[float]) -> "CandleData":
        """Create from array format."""
        return cls(
            open=arr[0], high=arr[1], low=arr[2], close=arr[3], volume=arr[4], timestamp=arr[5]
        )


class MarketDataBuffer:
    """
    Manages tick and candle buffers with candle-close detection.

    This class encapsulates the buffering logic that was previously
    inline in scripts/live.py. Key responsibilities:

    1. **Fixed-size windows**: Uses deques with maxlen for O(1) append
    2. **Candle close detection**: Distinguishes between in-place updates
       (same candle period) and new candles (previous period closed)
    3. **Ready state**: Checks if minimum data is available for inference

    The candle detection logic handles Deriv API behavior where streaming
    updates include both in-progress and completed candles.

    Attributes:
        tick_length: Maximum number of ticks to buffer
        candle_length: Maximum number of candles to buffer

    Example:
        >>> buffer = MarketDataBuffer(tick_length=128, candle_length=64)
        >>>
        >>> # Append ticks as they arrive
        >>> buffer.append_tick(1.2345)
        >>>
        >>> # Update candle buffer, check if new candle
        >>> from data.events import CandleEvent
        >>> is_new = buffer.update_candle(candle_event)
        >>>
        >>> if is_new and buffer.is_ready():
        ...     ticks = buffer.get_ticks_array()
        ...     candles = buffer.get_candles_array()
        ...     await run_inference(ticks, candles)
    """

    def __init__(self, tick_length: int, candle_length: int):
        """
        Initialize market data buffer.

        Args:
            tick_length: Maximum number of ticks to keep (sequence length)
            candle_length: Maximum number of candles to keep (sequence length)
        """
        self.tick_length = tick_length
        self.candle_length = candle_length

        self._ticks: collections.deque[float] = collections.deque(maxlen=tick_length)
        # +1 capacity allows holding N closed candles + 1 forming candle
        # This prevents data skew during live inference where the forming candle
        # would otherwise push out the oldest closed candle
        self._candles: collections.deque[list[float]] = collections.deque(maxlen=candle_length + 1)

        # Track last candle timestamp for close detection
        self._last_candle_ts: float | None = None

        logger.debug(
            f"MarketDataBuffer initialized: "
            f"tick_length={tick_length}, candle_length={candle_length} (+1 for forming)"
        )

    def append_tick(self, price: float) -> None:
        """
        Append a tick price to the buffer.

        Args:
            price: Tick price value
        """
        self._ticks.append(price)

    def append_ticks(self, prices: list[float]) -> None:
        """
        Append multiple tick prices to the buffer.

        Args:
            prices: List of tick price values
        """
        for price in prices:
            self._ticks.append(price)

    def update_candle(self, candle_event: Any) -> bool:
        """
        Update candle buffer with incoming event.

        This method handles the Deriv API streaming behavior where:
        - Updates for the SAME candle period should update in-place
        - Updates for a NEW candle period indicate the previous candle closed

        The detection uses timestamp comparison with 1-second tolerance
        to handle clock drift and API latency.

        Args:
            candle_event: Candle event with open, high, low, close, volume, timestamp
                         Can be a CandleEvent object or any object with these attributes.

        Returns:
            True if this was a NEW candle (previous period closed),
            False if this was an UPDATE to the current candle.
        """
        # Extract timestamp - handle both datetime and float
        if hasattr(candle_event, "timestamp"):
            ts = candle_event.timestamp
            if isinstance(ts, datetime):
                incoming_ts = ts.timestamp()
            else:
                incoming_ts = float(ts)
        else:
            raise ValueError("candle_event must have timestamp attribute")

        # Build candle array
        new_candle = [
            float(candle_event.open),
            float(candle_event.high),
            float(candle_event.low),
            float(candle_event.close),
            float(getattr(candle_event, "volume", 0.0)),
            incoming_ts,
        ]

        is_new_candle = False

        if len(self._candles) == 0:
            # First candle
            self._candles.append(new_candle)
            self._last_candle_ts = incoming_ts
            is_new_candle = True
            logger.info(f"First candle: ts={incoming_ts:.0f}")
        else:
            # Compare with last candle timestamp
            last_candle_ts = self._candles[-1][5]

            if abs(incoming_ts - last_candle_ts) < 1.0:
                # Same candle period - UPDATE in place
                # Merge High/Low/Volume with existing candle
                current_candle = self._candles[-1]
                merged_candle = [
                    current_candle[0],  # Open stays same
                    max(current_candle[1], new_candle[1]),  # High
                    min(current_candle[2], new_candle[2]),  # Low
                    new_candle[3],  # Close updates
                    current_candle[4] + new_candle[4],  # Volume accumulates
                    current_candle[5],  # Timestamp stays same
                ]
                self._candles[-1] = merged_candle
                logger.debug(
                    f"Candle updated: ts={incoming_ts:.0f}, close={candle_event.close:.5f}"
                )
            else:
                # New candle period - previous candle CLOSED
                is_new_candle = True
                self._candles.append(new_candle)
                self._last_candle_ts = incoming_ts
                logger.info(
                    f"Candle closed: new candle ts={incoming_ts:.0f}, "
                    f"buffer_size={len(self._candles)}"
                )

        return is_new_candle

    def update_candle_from_array(self, candle_array: list[float]) -> bool:
        """
        Update candle buffer from array format.

        Args:
            candle_array: [open, high, low, close, volume, timestamp]

        Returns:
            True if new candle, False if update.
        """
        candle = CandleData.from_array(candle_array)
        return self.update_candle(candle)

    def preload_candles(self, candles: list[list[float]]) -> None:
        """
        Preload historical candles into buffer.

        Args:
            candles: List of candle arrays [open, high, low, close, volume, timestamp]
        """
        for candle in candles:
            self._candles.append(candle)

        if self._candles:
            self._last_candle_ts = self._candles[-1][5]

        logger.info(f"Pre-loaded {len(self._candles)} candles")

    def is_ready(self) -> bool:
        """
        Check if buffers have enough data for inference.

        Returns:
            True if both tick and candle buffers have minimum required data.
            Note: Candle buffer may have candle_length+1 if forming candle present.
        """
        return len(self._ticks) >= self.tick_length and len(self._candles) >= self.candle_length

    def get_ticks(self) -> collections.deque[float]:
        """Get raw tick buffer."""
        return self._ticks

    def get_candles(self) -> collections.deque[list[float]]:
        """Get raw candle buffer."""
        return self._candles

    def get_ticks_array(self) -> np.ndarray:
        """Get ticks as numpy array."""
        return np.array(list(self._ticks))

    def get_candles_array(self, include_forming: bool = False) -> np.ndarray:
        """
        Get candles as numpy array.

        Args:
            include_forming: If False (default), return only closed candles.
                            This ensures inference uses the same data format as training.
                            If True, include the forming (current) candle.

        Returns:
            Numpy array of candle data, shape (N, 6) where N is candle_length
            (or candle_length+1 if include_forming=True and forming candle exists).
        """
        candles_list = list(self._candles)
        
        # If buffer has more than candle_length entries, the last one is forming
        if not include_forming and len(candles_list) > self.candle_length:
            # Exclude the forming candle (most recent)
            candles_list = candles_list[:-1]
        
        return np.array(candles_list)

    def get_snapshot(self) -> dict[str, np.ndarray]:
        """
        Get an atomic snapshot of current buffer state.
        
        Returns a dictionary with deep copies of ticks and candles to ensure
        data consistency during inference, preventing race conditions where
        background tasks might update the buffer while inference is running.
        
        Returns:
            Dict with 'ticks' and 'candles' numpy arrays.
        """
        # Create copies of data
        ticks_snap = list(self._ticks)
        
        # Snapshot candles (excluding forming one for consistency with training)
        candles_list = list(self._candles)
        if len(candles_list) > self.candle_length:
            candles_list = candles_list[:-1]
        candles_snap = [c.to_array() for c in candles_list]
        
        return {
            "ticks": ticks_snap, 
            "candles": candles_snap,
            "tick_count": len(ticks_snap),
            "candle_count": len(candles_snap)
        }

    def tick_count(self) -> int:
        """Get current number of ticks in buffer."""
        return len(self._ticks)

    def candle_count(self) -> int:
        """Get current number of candles in buffer."""
        return len(self._candles)

    def get_last_price(self) -> float:
        """Get the most recent tick price.
        
        Returns:
            Last tick price, or 0.0 if buffer is empty
        """
        if self._ticks:
            return self._ticks[-1]
        return 0.0

    def clear(self) -> None:
        """Clear all buffered data."""
        self._ticks.clear()
        self._candles.clear()
        self._last_candle_ts = None
        logger.debug("MarketDataBuffer cleared")

    def __repr__(self) -> str:
        return (
            f"MarketDataBuffer("
            f"ticks={len(self._ticks)}/{self.tick_length}, "
            f"candles={len(self._candles)}/{self.candle_length}, "
            f"ready={self.is_ready()})"
        )
