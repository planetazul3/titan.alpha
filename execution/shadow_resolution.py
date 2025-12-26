
import logging
import asyncio
from datetime import datetime, timedelta, timezone
from typing import List, Optional, Any

from config.constants import CONTRACT_TYPES
from execution.shadow_store import ShadowTradeRecord, ShadowTradeStore


# Candle array column indices (must match data/buffer.py CandleData.to_array)
CANDLE_COL_OPEN = 0
CANDLE_COL_HIGH = 1
CANDLE_COL_LOW = 2
CANDLE_COL_CLOSE = 3
CANDLE_COL_VOLUME = 4
CANDLE_COL_TIMESTAMP = 5

logger = logging.getLogger(__name__)


class ShadowTradeResolver:
    """
    Resolves outcomes for shadow trades.

    Monitors unresolved trades in the store and determines if they won or lost
    based on subsequent market data.

    Assumption: Trades are 1-candle duration (expiration is 1 minute for '1m' timeframe).
    """

    # Maximum time past expiration where we trust current_price as valid.
    # Trades expiring longer than this ago are flagged as unresolvable without history.
    STALENESS_THRESHOLD = timedelta(minutes=5)

    def __init__(
        self,
        shadow_store: ShadowTradeStore,
        duration_minutes: int = 1,
        client: Any = None,
        default_touch_barrier_pct: float = 0.005,
        default_range_barrier_pct: float = 0.003,
    ):
        """
        Initialize the resolver.

        Args:
            shadow_store: Store containing shadow trades.
            duration_minutes: Trade duration in minutes (default 1).
            client: DerivClient instance for fetching historical data (optional).
            default_touch_barrier_pct: Default barrier percentage for TOUCH/NO_TOUCH if not in trade (0.005 = 0.5%).
            default_range_barrier_pct: Default barrier percentage for STAYS_BETWEEN if not in trade (0.003 = 0.3%).
        """
        self.store = shadow_store
        self.duration = timedelta(minutes=duration_minutes)
        self.client = client
        self.default_touch_barrier_pct = default_touch_barrier_pct
        self.default_range_barrier_pct = default_range_barrier_pct
        self.logger = logging.getLogger(__name__)

    async def resolve_trades(self, current_price: float, current_time: datetime) -> int:
        """
        Check unresolved trades and update outcomes if expired.

        Args:
            current_price: Current market price (e.g. candle close).
            current_time: Current timestamp (aware).

        Returns:
            Number of trades resolved in this pass.
        """
        # Ensure current_time is aware
        if current_time.tzinfo is None:
            current_time = current_time.replace(tzinfo=timezone.utc)

        # Get unresolved trades
        # Queries are still sync for now unless we change query() too, but query is usually fast or less frequent.
        # However, plan said "Queries generally happen in background...".
        # Let's keep query sync for now or upgrade it too?
        # The hot path is update_outcome in loop.
        unresolved = self.store.query(unresolved_only=True)
        
        resolved_count = 0
        wins = 0
        losses = 0
        
        stale_trades: List[ShadowTradeRecord] = []
        normal_resolutions: List[tuple[ShadowTradeRecord, float]] = []

        for trade in unresolved:
            # Check if trade has expired
            expiration_time = trade.timestamp + self.duration
            
            if current_time >= expiration_time:
                staleness = current_time - expiration_time
                if staleness > self.STALENESS_THRESHOLD:
                    # Collect as potentially stale
                    stale_trades.append(trade)
                else:
                    # Normal resolution using current price
                    normal_resolutions.append((trade, current_price))

        # 1. Resolve normal trades
        for trade, exit_price in normal_resolutions:
            await self._apply_resolution_async(trade, exit_price, current_price)
            resolved_count += 1
            if trade.outcome is not None:
                if trade.outcome > 0: wins += 1
                else: losses += 1

        # 2. Handle stale trades (fetch history if client available)
        if stale_trades and self.client:
            try:
                # Calculate range
                timestamps = [t.timestamp.timestamp() for t in stale_trades]
                min_ts = int(min(timestamps))
                max_ts = int(max(timestamps)) + int(self.duration.total_seconds()) + 60
                
                self.logger.info(f"Fetching history for {len(stale_trades)} stale trades ({min_ts} to {max_ts})")
                
                # Fetch history
                candles = await self.client.get_historical_candles_by_range(
                    start_time=min_ts, 
                    end_time=max_ts, 
                    interval=60 # Assuming 1m candles
                )
                
                # Map candles by epoch
                candles_map = {c['epoch']: c for c in candles}
                
                for trade in stale_trades:
                    # Match trade timestamp (entry time) to candle epoch
                    # The candle covering the trade duration starts at trade.timestamp
                    matching_candle = candles_map.get(int(trade.timestamp.timestamp()))
                    
                    if matching_candle:
                        exit_price = float(matching_candle['close'])
                        await self._apply_resolution_async(trade, exit_price, current_price) # Pass current_price mainly for logging context if needed, but resolved price is exit_price
                        resolved_count += 1
                        if trade.outcome is not None:
                            if trade.outcome > 0: wins += 1
                            else: losses += 1
                    else:
                        self.logger.warning(f"Could not find historical candle for trade {trade.trade_id} @ {trade.timestamp}")
                        await self._mark_as_stale_async(trade, current_price)

            except Exception as e:
                self.logger.error(f"Error fetching history for stale trades: {e}")
                # Fallback to marking all stale
                for trade in stale_trades:
                    await self._mark_as_stale_async(trade, current_price)
        
        elif stale_trades:
            # No client provided, must mark stale
            for trade in stale_trades:
                await self._mark_as_stale_async(trade, current_price)

        # Log summary
        if resolved_count > 0:
            win_rate = (wins / resolved_count * 100) if resolved_count > 0 else 0
            self.logger.info(
                f"🎯 Resolved {resolved_count} shadow trades: "
                f"{wins} wins, {losses} losses "
                f"(win rate: {win_rate:.1f}%)"
            )

        return resolved_count

    async def _apply_resolution_async(self, trade: ShadowTradeRecord, exit_price: float, current_price_ref: float):
        """Apply resolution logic to a trade asynchronously."""
        outcome = self._determine_outcome(trade, exit_price)
        
        if outcome is not None:
            # H08: Async DB update
            await self.store.update_outcome_async(
                trade=trade,
                outcome=outcome,
                exit_price=exit_price
            )
            
            # Update trade object for local counting
            trade.outcome = outcome
            
            price_change_pct = ((exit_price - trade.entry_price) / trade.entry_price) * 100
            simulated_pnl = 0.95 if outcome else -1.0
            outcome_str = "✅ WIN" if outcome else "❌ LOSS"
            
            self.logger.info(
                f"{outcome_str} | {trade.direction} @ {trade.probability:.1%} confidence | "
                f"{trade.entry_price:.2f} → {exit_price:.2f} ({price_change_pct:+.2f}%) | "
                f"P&L: ${simulated_pnl:+.2f} | Regime: {trade.regime_state}"
            )

    async def _mark_as_stale_async(self, trade: ShadowTradeRecord, current_price: float):
        """Mark a trade as stale asynchronously."""
        expiration_time = trade.timestamp + self.duration
        staleness = datetime.now(timezone.utc) - expiration_time
        
        self.logger.warning(
            f"⚠️ Trade {trade.trade_id[:8]} expired {staleness.total_seconds()/60:.1f} min ago - "
            f"marking as STALE (excluded from training)"
        )
        # H08: Async DB update
        await self.store.mark_stale_async(
            trade_id=trade.trade_id,
            exit_price=current_price # Use current price as fallback exit for db rec
        )

    def _determine_outcome(self, trade: ShadowTradeRecord, exit_price: float) -> Optional[bool]:
        """
        Determine win/loss based on contract type and direction.

        For RISE_FALL: Uses exit_price vs entry_price
        For TOUCH/NO_TOUCH: Uses High/Low from candle_window to check barrier hits
        For STAYS_BETWEEN: Uses High/Low from candle_window to check range containment

        Returns:
            True (Win), False (Loss), or None (Unknown/Error)
        """
        import numpy as np

        try:
            if trade.contract_type == CONTRACT_TYPES.RISE_FALL:
                if trade.direction == "CALL":
                    return exit_price > trade.entry_price
                elif trade.direction == "PUT":
                    return exit_price < trade.entry_price

            elif trade.contract_type == CONTRACT_TYPES.TOUCH_NO_TOUCH:
                # TOUCH/NO_TOUCH requires checking High/Low from candle period
                if trade.candle_window and len(trade.candle_window) > 0:
                    candles = np.array(trade.candle_window)

                    # Validate shape (must have at least columns 0-3 for OHLC)
                    if candles.ndim < 2 or candles.shape[1] < 4:
                        self.logger.warning(
                            f"Invalid candle_window shape {candles.shape} for trade {trade.trade_id}. "
                            f"Expected at least 4 columns (OHLC)."
                        )
                        return None

                    high_prices = candles[:, CANDLE_COL_HIGH]
                    low_prices = candles[:, CANDLE_COL_LOW]

                    # C02: Use stored barrier levels if available, else default
                    if trade.barrier_level is not None:
                        # barrier_level is usually high barrier offset, barrier2_level is low barrier offset
                        # Assuming symmetric around entry if only one is provided, or barrier_level is the 'touch' distance
                        # The report says: "Calculate upper_barrier as entry_price plus barrier_level. For lower_barrier, use the negative of the absolute barrier distance."
                        offset = abs(trade.barrier_level)
                        upper_barrier = trade.entry_price + offset
                        lower_barrier = trade.entry_price - offset
                    else:
                        # Fallback to percentage
                        self.logger.debug(
                            f"Using default barrier pct {self.default_touch_barrier_pct} for trade {trade.trade_id}"
                        )
                        upper_barrier = trade.entry_price * (1 + self.default_touch_barrier_pct)
                        lower_barrier = trade.entry_price * (1 - self.default_touch_barrier_pct)

                    # Check if any high/low touched the barriers
                    touched = bool(
                        np.any(high_prices >= upper_barrier) or np.any(low_prices <= lower_barrier)
                    )

                    if trade.direction == "TOUCH":
                        return touched
                    elif trade.direction == "NO_TOUCH":
                        return not touched
                else:
                    self.logger.warning(
                        f"TOUCH trade {trade.trade_id[:8]} missing candle_window"
                    )
                    return None

            elif trade.contract_type == CONTRACT_TYPES.STAYS_BETWEEN:
                # STAYS_BETWEEN: price must stay within ±band of entry
                if trade.candle_window and len(trade.candle_window) > 0:
                    candles = np.array(trade.candle_window)

                    # Validate shape
                    if candles.ndim < 2 or candles.shape[1] < 4:
                        self.logger.warning(
                            f"Invalid candle_window shape {candles.shape} for trade {trade.trade_id}. "
                            f"Expected at least 4 columns (OHLC)."
                        )
                        return None

                    high_prices = candles[:, CANDLE_COL_HIGH]
                    low_prices = candles[:, CANDLE_COL_LOW]

                    # C02: Use stored barrier levels
                    if trade.barrier_level is not None and trade.barrier2_level is not None:
                        # barrier_level is usually high barrier offset, barrier2_level is low barrier offset
                        upper_band = trade.entry_price + trade.barrier_level
                        lower_band = trade.entry_price + trade.barrier2_level
                    else:
                        # Fallback
                        self.logger.debug(
                            f"Using default range pct {self.default_range_barrier_pct} for trade {trade.trade_id}"
                        )
                        upper_band = trade.entry_price * (1 + self.default_range_barrier_pct)
                        lower_band = trade.entry_price * (1 - self.default_range_barrier_pct)

                    # Price stayed in range if ALL highs below upper and ALL lows above lower
                    stayed_in = bool(
                        np.all(high_prices <= upper_band) and np.all(low_prices >= lower_band)
                    )
                    return stayed_in
                else:
                    self.logger.warning(
                        f"RANGE trade {trade.trade_id[:8]} missing candle_window"
                    )
                    return None

            # Unsupported contract type
            self.logger.warning(f"Unsupported contract type: {trade.contract_type}")
            return None

        except Exception as e:
            self.logger.error(f"Error determining outcome for {trade.trade_id}: {e}")
            return None
