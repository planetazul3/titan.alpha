"""
Outcome Resolver - Join shadow trades with market outcomes.

This module resolves shadow trade outcomes by analyzing market data
after the trade signal was generated. It determines whether the trade
would have been a win or loss.

Usage:
    >>> resolver = OutcomeResolver()
    >>> resolved = resolver.resolve_from_cache(unresolved_trades, historical_df)
    >>> for trade in resolved:
    ...     print(f"{trade.trade_id}: {'WIN' if trade.outcome else 'LOSS'}")
"""

import logging
from dataclasses import dataclass

import numpy as np
from typing import Any

from execution.shadow_store import ShadowTradeRecord

try:
    import pandas as pd
except ImportError:
    pd = Any  # type: ignore

logger = logging.getLogger(__name__)


@dataclass
class ResolutionConfig:
    """
    Configuration for outcome resolution.

    Attributes:
        rise_fall_window_seconds: How long to track rise/fall outcome
        touch_barrier_percent: Percentage for touch barrier detection
        range_band_percent: Percentage for range bands
    """

    rise_fall_window_seconds: int = 60  # 1 minute default
    touch_barrier_percent: float = 0.005  # 0.5%
    range_band_percent: float = 0.003  # 0.3%


class OutcomeResolver:
    """
    Resolves shadow trade outcomes from market data.

    This takes shadow trades (with entry price and direction) and determines
    whether they would have been wins or losses based on subsequent market
    price movement.

    Supports multiple contract types:
    - RISE_FALL: Did price rise or fall after window?
    - TOUCH_NO_TOUCH: Did price touch a barrier?
    - STAYS_BETWEEN: Did price stay within range?
    """

    def __init__(self, config: ResolutionConfig | None = None):
        """
        Initialize outcome resolver.

        Args:
            config: Resolution configuration (uses defaults if None)
        """
        self.config = config or ResolutionConfig()
        logger.info(f"OutcomeResolver initialized: window={self.config.rise_fall_window_seconds}s")

    def resolve_from_cache(
        self, trades: list[ShadowTradeRecord], tick_data: np.ndarray, tick_timestamps: np.ndarray
    ) -> list[ShadowTradeRecord]:
        """
        Resolve trade outcomes using cached historical tick data.

        This is the primary resolution method for offline batch processing.

        Args:
            trades: List of unresolved shadow trade records
            tick_data: Historical tick prices, shape (N,)
            tick_timestamps: Corresponding Unix timestamps, shape (N,)

        Returns:
            List of resolved shadow trade records (new objects, immutable)
        """
        resolved = []

        for trade in trades:
            if trade.is_resolved():
                resolved.append(trade)
                continue

            try:
                outcome, exit_price = self._resolve_single(trade, tick_data, tick_timestamps)
                resolved.append(trade.with_outcome(outcome, exit_price))
            except Exception as e:
                logger.warning(f"Could not resolve {trade.trade_id}: {e}")
                resolved.append(trade)  # Keep unresolved

        resolved_count = sum(1 for t in resolved if t.is_resolved())
        logger.info(f"Resolved {resolved_count}/{len(trades)} trades")

        return resolved

    def _resolve_single(
        self, trade: ShadowTradeRecord, tick_data: np.ndarray, tick_timestamps: np.ndarray
    ) -> tuple[bool, float]:
        """
        Resolve a single trade outcome.

        Args:
            trade: Shadow trade to resolve
            tick_data: Historical tick prices
            tick_timestamps: Corresponding timestamps

        Returns:
            Tuple of (outcome, exit_price)

        Raises:
            ValueError: If insufficient data for resolution
        """
        # Find trade time in tick data
        trade_ts = trade.timestamp.timestamp()
        window_end_ts = trade_ts + self.config.rise_fall_window_seconds

        # Get tick indices within window
        start_idx = np.searchsorted(tick_timestamps, trade_ts)
        end_idx = np.searchsorted(tick_timestamps, window_end_ts)

        if end_idx <= start_idx:
            raise ValueError(f"No tick data in window [{trade_ts}, {window_end_ts}]")

        window_ticks = tick_data[start_idx:end_idx]
        exit_price = float(window_ticks[-1])

        # Resolve based on contract type
        if trade.contract_type == "RISE_FALL":
            outcome = self._resolve_rise_fall(trade.entry_price, exit_price, trade.direction)
        elif trade.contract_type == "TOUCH_NO_TOUCH":
            outcome = self._resolve_touch(trade.entry_price, window_ticks, trade.direction)
        elif trade.contract_type == "STAYS_BETWEEN":
            outcome = self._resolve_range(trade.entry_price, window_ticks)
        else:
            logger.warning(f"Unknown contract type: {trade.contract_type}")
            outcome = False

        return outcome, exit_price

    def _resolve_rise_fall(self, entry_price: float, exit_price: float, direction: str) -> bool:
        """
        Resolve RISE_FALL contract outcome.

        - CALL: Win if exit_price > entry_price
        - PUT: Win if exit_price < entry_price
        """
        price_rose = exit_price > entry_price

        if direction == "CALL":
            return price_rose
        elif direction == "PUT":
            return not price_rose
        else:
            logger.warning(f"Unknown direction: {direction}")
            return False

    def _resolve_touch(self, entry_price: float, window_ticks: np.ndarray, direction: str) -> bool:
        """
        Resolve TOUCH_NO_TOUCH contract outcome.

        - TOUCH: Win if price touched barrier (moved > threshold)
        - NO_TOUCH: Win if price stayed within barrier
        """
        barrier = entry_price * self.config.touch_barrier_percent

        max_deviation = np.max(np.abs(window_ticks - entry_price))
        touched = max_deviation > barrier

        if direction == "TOUCH":
            return bool(touched)
        elif direction == "NO_TOUCH":
            return bool(not touched)
        else:
            logger.warning(f"Unknown direction: {direction}")
            return False

    def _resolve_range(self, entry_price: float, window_ticks: np.ndarray) -> bool:
        """
        Resolve STAYS_BETWEEN contract outcome.

        Win if price stayed within ±range_band_percent of entry_price.
        """
        band = entry_price * self.config.range_band_percent

        max_price = np.max(window_ticks)
        min_price = np.min(window_ticks)

        stayed_in_range = max_price <= entry_price + band and min_price >= entry_price - band

        return bool(stayed_in_range)

    def resolve_from_dataframe(
        self, trades: list[ShadowTradeRecord], df: "pd.DataFrame"
    ) -> list[ShadowTradeRecord]:
        """
        Resolve trades from a pandas DataFrame.

        Convenience method that extracts tick data from DataFrame.

        Args:
            trades: List of unresolved shadow trades
            df: DataFrame with 'quote' and 'epoch' columns

        Returns:
            List of resolved trades
        """

        if "quote" not in df.columns or "epoch" not in df.columns:
            raise ValueError("DataFrame must have 'quote' and 'epoch' columns")

        tick_data = df["quote"].values.astype(np.float64)
        tick_timestamps = df["epoch"].values.astype(np.float64)

        return self.resolve_from_cache(trades, tick_data, tick_timestamps)

    def get_resolution_window(self) -> int:
        """Get the resolution window in seconds."""
        return self.config.rise_fall_window_seconds


def batch_resolve(store_path: str, tick_data_path: str, output_path: str | None = None) -> int:
    """
    CLI utility to batch resolve shadow trades.

    Args:
        store_path: Path to shadow trade store
        tick_data_path: Path to tick data Parquet file
        output_path: Optional path for resolved output

    Returns:
        Number of trades resolved
    """
    from pathlib import Path

    import pandas as pd

    from execution.shadow_store import ShadowTradeStore

    store = ShadowTradeStore(Path(store_path))
    unresolved = store.query(unresolved_only=True)

    if not unresolved:
        logger.info("No unresolved trades to process")
        return 0

    tick_df = pd.read_parquet(tick_data_path)
    resolver = OutcomeResolver()
    resolved = resolver.resolve_from_dataframe(unresolved, tick_df)

    # Append resolved to store
    resolved_only = [r for r in resolved if r.is_resolved()]
    if resolved_only:
        store.append_resolved(resolved_only)

    return len(resolved_only)
