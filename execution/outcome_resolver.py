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
from typing import Any, TYPE_CHECKING

if TYPE_CHECKING:
    from execution.sqlite_shadow_store import SQLiteShadowStore

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
        prefer_tick_data: If True, prefer tick data over OHLC for accuracy
    """

    rise_fall_window_seconds: int = 60  # 1 minute default
    touch_barrier_percent: float = 0.005  # 0.5%
    range_band_percent: float = 0.003  # 0.3%
    prefer_tick_data: bool = True  # PERF: Tick data preferred for tick-perfect resolution


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
            outcome = self._resolve_touch(
                trade.entry_price, window_ticks, trade.direction, getattr(trade, "barrier_level", None)
            )
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

    def _resolve_touch(
        self, 
        entry_price: float, 
        window_ticks: np.ndarray, 
        direction: str, 
        barrier_level: float | None = None,
        ohlc_data: tuple[np.ndarray, np.ndarray] | None = None,
    ) -> bool:
        """
        Resolve TOUCH_NO_TOUCH contract outcome.

        - TOUCH: Win if price touched barrier (moved > threshold)
        - NO_TOUCH: Win if price stayed within barrier
        
        Resolution Strategy (aligned with Deriv specifications):
        1. PREFER tick data for tick-perfect resolution when available
        2. FALLBACK to OHLC (High/Low) during data gaps with warning
        
        Args:
            entry_price: Trade entry price
            window_ticks: Tick prices in the resolution window
            direction: "TOUCH" or "NO_TOUCH"
            barrier_level: Absolute barrier level (or None for percentage-based)
            ohlc_data: Optional tuple of (high_prices, low_prices) for OHLC fallback
        """
        if barrier_level is not None:
            # Absolute barrier checks
            # Standard Deriv 'barrier' is usually an offset (+1.23) or absolute.
            # Treat barrier_level as the absolute delta threshold from entry.
            barrier = barrier_level
        else:
            barrier = entry_price * self.config.touch_barrier_percent

        # Determine tick data availability
        has_tick_data = len(window_ticks) > 0 and not np.all(np.isnan(window_ticks))
        
        if has_tick_data and self.config.prefer_tick_data:
            # PREFERRED: Tick-perfect resolution
            max_deviation = np.max(np.abs(window_ticks - entry_price))
        elif ohlc_data is not None:
            # FALLBACK: Use OHLC High/Low for barrier check (per Deriv specs)
            high_prices, low_prices = ohlc_data
            if len(high_prices) > 0 and len(low_prices) > 0:
                max_above = np.max(high_prices) - entry_price  # How far above entry
                max_below = entry_price - np.min(low_prices)   # How far below entry
                max_deviation = max(abs(max_above), abs(max_below))
                logger.warning(
                    f"Touch resolution using OHLC fallback (tick data unavailable). "
                    f"Accuracy may be reduced. max_deviation={max_deviation:.5f}"
                )
            else:
                logger.warning("Touch resolution: No OHLC data available for fallback")
                max_deviation = 0.0
        else:
            # Last resort: use ticks even if sparse
            if len(window_ticks) > 0:
                max_deviation = np.max(np.abs(window_ticks - entry_price))
            else:
                logger.warning("Touch resolution: No price data available")
                max_deviation = 0.0
        
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


def resolve_trade_transactionally(
    store: "SQLiteShadowStore", 
    trade_id: str, 
    resolver: OutcomeResolver, 
    tick_data: np.ndarray, 
    tick_timestamps: np.ndarray,
    max_retries: int = 3
) -> bool:
    """
    Resolve a trade with Optimistic Concurrency Control (OCC) retry loop.
    
    1. Fetch latest trade state
    2. detailed resolution logic
    3. Try atomic update (update_outcome)
    4. If OptimisticLockError, retry
    
    Args:
        store: SQLiteShadowStore instance
        trade_id: ID of trade to resolve
        resolver: OutcomeResolver instance
        tick_data: Tick prices
        tick_timestamps: Tick timestamps
        max_retries: Number of retries on version conflict
        
    Returns:
        True if successfully resolved/updated, False otherwise
    """
    from execution.sqlite_shadow_store import OptimisticLockError, SQLiteShadowStore
    import time
    
    # Ensure strict typing for the store to access get_by_id and update_outcome
    if not isinstance(store, SQLiteShadowStore):
        logger.warning(f"Store {type(store)} does not support transactional resolution")
        return False

    for attempt in range(max_retries):
        try:
            # 1. Fetch latest state
            trade = store.get_by_id(trade_id)
            if not trade:
                logger.warning(f"Trade {trade_id} not found in store")
                return False
                
            # 2. Check if already resolved
            if trade.is_resolved():
                logger.debug(f"Trade {trade_id} already resolved (outcome={trade.outcome})")
                return True
                
            # 3. Resolve
            outcome, exit_price = resolver._resolve_single(trade, tick_data, tick_timestamps)
            
            # 4. Try atomic update
            # This will raise OptimisticLockError if version changed since step 1
            store.update_outcome(trade, outcome, exit_price)
            return True
            
        except OptimisticLockError:
            logger.info(f"OCC conflict for trade {trade_id} (attempt {attempt+1}/{max_retries}) - Retrying...")
            time.sleep(0.05 * (attempt + 1)) # Backoff
            continue
            
        except ValueError as e:
            # Data not available yet?
            logger.debug(f"Skipping {trade_id}: {e}")
            return False
            
        except Exception as e:
            logger.error(f"Error resolving {trade_id}: {e}")
            return False
            
    logger.error(f"Failed to resolve {trade_id} after {max_retries} attempts due to concurrency")
    return False



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


def batch_resolve_sqlite(store_path: str, tick_data_path: str) -> int:
    """
    Batch resolve trades using SQLite store and OCC.
    """
    from concurrent.futures import ThreadPoolExecutor
    from pathlib import Path
    
    from execution.sqlite_shadow_store import SQLiteShadowStore
    
    if not Path(store_path).exists():
        logger.error(f"Store not found: {store_path}")
        return 0
        
    store = SQLiteShadowStore(Path(store_path))
    unresolved = store.query(unresolved_only=True)
    
    if not unresolved:
        logger.info("No unresolved trades to process")
        return 0
        
    logger.info(f"Processing {len(unresolved)} unresolved trades...")
    
    # Load market data ONCE
    # Note: For massive datasets, we might need chunking, but for batch jobs this is likely fine
    try:
        tick_df = pd.read_parquet(tick_data_path)
        if "quote" not in tick_df.columns or "epoch" not in tick_df.columns:
            raise ValueError("DataFrame missing quote/epoch columns")
        
        tick_data = tick_df["quote"].values.astype(np.float64)
        tick_timestamps = tick_df["epoch"].values.astype(np.float64)
    except Exception as e:
        logger.error(f"Failed to load tick data: {e}")
        return 0

    resolver = OutcomeResolver()
    resolved_count = 0
    
    # Process serially or in parallel?
    # Parallel is safe due to per-row OCC!
    # But SQLite writes are serialized by the driver/WAL.
    # Let's simple serial loop for safety first.
    
    for trade in unresolved:
        if resolve_trade_transactionally(store, trade.trade_id, resolver, tick_data, tick_timestamps):
            resolved_count += 1
            
    return resolved_count
