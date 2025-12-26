
import logging
import asyncio
from datetime import datetime, timedelta, timezone
from typing import List, Optional, Any

from config.constants import CONTRACT_TYPES
from execution.shadow_store import ShadowTradeRecord, ShadowTradeStore
from execution.barriers import BarrierCalculator
from observability.shadow_logging import shadow_trade_logger

try:
    from opentelemetry import trace
    TRACING_ENABLED = True
except ImportError:
    TRACING_ENABLED = False


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

    @staticmethod
    def _parse_timeframe_to_seconds(timeframe: str) -> int:
        """
        I03 Fix: Parse timeframe string to seconds.
        
        Args:
            timeframe: Timeframe string like "1m", "5m", "15m", "1h"
            
        Returns:
            Interval in seconds
        """
        timeframe = timeframe.lower().strip()
        
        multipliers = {
            's': 1,      # seconds
            'm': 60,     # minutes
            'h': 3600,   # hours
            'd': 86400,  # days
        }
        
        # Extract number and unit
        if len(timeframe) < 2:
            return 60  # Default to 1 minute
            
        try:
            number = int(timeframe[:-1])
            unit = timeframe[-1]
            return number * multipliers.get(unit, 60)
        except (ValueError, KeyError):
            return 60  # Default to 1 minute

    def __init__(
        self,
        shadow_store: ShadowTradeStore,
        duration_minutes: int = 1,
        client: Any = None,
        default_touch_barrier_pct: float = 0.005,
        default_range_barrier_pct: float = 0.003,
        staleness_threshold_minutes: int = 5,
        timeframe: str = "1m",  # I03 Fix: Configurable timeframe
    ):
        """
        Initialize the resolver.

        Args:
            shadow_store: Store containing shadow trades.
            duration_minutes: Trade duration in minutes (default 1).
            client: DerivClient instance for fetching historical data (optional).
            default_touch_barrier_pct: Default barrier percentage for TOUCH/NO_TOUCH if not in trade (0.005 = 0.5%).
            default_range_barrier_pct: Default barrier percentage for STAYS_BETWEEN if not in trade (0.003 = 0.3%).
            staleness_threshold_minutes: Max time past expiration before trade is flagged stale (default 5).
            timeframe: Candle timeframe for historical fetch (e.g., "1m", "5m", "1h").
        """
        self.store = shadow_store
        self.duration = timedelta(minutes=duration_minutes)
        self.client = client
        self.default_touch_barrier_pct = default_touch_barrier_pct
        self.default_range_barrier_pct = default_range_barrier_pct
        self.staleness_threshold = timedelta(minutes=staleness_threshold_minutes)
        self.barrier_calculator = BarrierCalculator()
        self.logger = logging.getLogger(__name__)
        
        # I03 Fix: Parse and store interval
        self.candle_interval_seconds = self._parse_timeframe_to_seconds(timeframe)
        
        if TRACING_ENABLED:
            self.tracer = trace.get_tracer(__name__)
        else:
            self.tracer = None
            
        self.logger.info(f"ShadowTradeResolver initialized with {timeframe} ({self.candle_interval_seconds}s) interval")

    async def resolve_trades(
        self,
        current_price: float,
        current_time: datetime,
        high_price: float | None = None,
        low_price: float | None = None,
    ) -> int:
        """
        Check unresolved trades and update outcomes if expired.

        C01 Fix: For path-dependent contracts (TOUCH, RANGE), accumulate candle
        OHLC data in resolution_context AFTER trade entry, then use that data
        for barrier checking at expiration.

        Args:
            current_price: Current market price (e.g. candle close).
            current_time: Current timestamp (aware).
            high_price: High price of current candle (for barrier checking).
            low_price: Low price of current candle (for barrier checking).

        Returns:
            Number of trades resolved in this pass.
        """
        from config.constants import CONTRACT_TYPES
        
        if self.tracer:
            with self.tracer.start_as_current_span("shadow_resolver.resolve_trades") as span:
                res = await self._resolve_trades_internal(current_price, current_time, high_price, low_price)
                span.set_attribute("current_price", current_price)
                span.set_attribute("resolved_count", res)
                return res
        else:
            return await self._resolve_trades_internal(current_price, current_time, high_price, low_price)

    async def _resolve_trades_internal(
        self,
        current_price: float,
        current_time: datetime,
        high_price: float | None = None,
        low_price: float | None = None,
    ) -> int:
        """Internal resolution logic moved from resolve_trades for tracing wrapper."""
        from config.constants import CONTRACT_TYPES
        
        # Ensure current_time is aware
        if current_time.tzinfo is None:
            current_time = current_time.replace(tzinfo=timezone.utc)

        # Get unresolved trades
        unresolved = self.store.query(unresolved_only=True)
        
        resolved_count = 0
        wins = 0
        losses = 0
        
        stale_trades: List[ShadowTradeRecord] = []
        normal_resolutions: List[tuple[ShadowTradeRecord, float]] = []
        active_trades: List[ShadowTradeRecord] = []  # C01: Trades still within duration

        for trade in unresolved:
            # C02 Fix: Use per-trade duration instead of global default
            trade_duration = timedelta(minutes=getattr(trade, 'duration_minutes', self.duration.total_seconds() / 60))
            expiration_time = trade.timestamp + trade_duration
            
            if current_time >= expiration_time:
                staleness = current_time - expiration_time
                if staleness > self.staleness_threshold:
                    stale_trades.append(trade)
                else:
                    normal_resolutions.append((trade, current_price))
            else:
                # C01 Fix: Trade is still active (within duration window)
                active_trades.append(trade)

        # C01 Fix: Accumulate resolution context for active path-dependent trades
        if high_price is not None and low_price is not None:
            for trade in active_trades:
                # Only accumulate for path-dependent contracts
                if trade.contract_type in (CONTRACT_TYPES.TOUCH_NO_TOUCH, CONTRACT_TYPES.STAYS_BETWEEN,
                                           "TOUCH_NO_TOUCH", "STAYS_BETWEEN"):
                    if hasattr(self.store, 'update_resolution_context_async'):
                        await self.store.update_resolution_context_async(
                            trade.trade_id, high_price, low_price, current_price
                        )
                    elif hasattr(self.store, 'update_resolution_context'):
                        self.store.update_resolution_context(
                            trade.trade_id, high_price, low_price, current_price
                        )

        # 1. Resolve normal trades
        for trade, exit_price in normal_resolutions:
            await self._apply_resolution_async(trade, exit_price, current_price, high_price, low_price)
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
                    interval=self.candle_interval_seconds  # I03 Fix: Use configurable interval
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

    async def _apply_resolution_async(
        self,
        trade: ShadowTradeRecord,
        exit_price: float,
        current_price_ref: float,
        high_price: float | None = None,
        low_price: float | None = None,
    ):
        """Apply resolution logic to a trade asynchronously."""
        outcome = self._determine_outcome(trade, exit_price, high_price, low_price)
        
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
            
            shadow_trade_logger.log_resolved(
                trade_id=trade.trade_id,
                outcome=outcome,
                exit_price=exit_price,
            )
            
            # Legacy log for console/debug
            self.logger.info(
                f"{outcome_str} | {trade.direction} @ {trade.probability:.1%} confidence | "
                f"{trade.entry_price:.2f} → {exit_price:.2f} ({price_change_pct:+.2f}%) | "
                f"P&L: ${simulated_pnl:+.2f} | Regime: {trade.regime_state}"
            )

    async def _mark_as_stale_async(self, trade: ShadowTradeRecord, current_price: float):
        """Mark a trade as stale asynchronously."""
        expiration_time = trade.timestamp + self.duration
        staleness = datetime.now(timezone.utc) - expiration_time
        
        # Structured log
        shadow_trade_logger.log_stale(trade.trade_id, reason="expired_unresolved")
        # H08: Async DB update
        await self.store.mark_stale_async(
            trade_id=trade.trade_id,
            exit_price=current_price # Use current price as fallback exit for db rec
        )

    def _determine_outcome(
        self,
        trade: ShadowTradeRecord,
        exit_price: float,
        high_price: float | None = None,
        low_price: float | None = None,
    ) -> Optional[bool]:
        """
        Determine win/loss based on contract type and direction.

        For RISE_FALL: Uses exit_price vs entry_price
        For TOUCH/NO_TOUCH: Uses passed high_price/low_price or candle_window to check barrier hits
        For STAYS_BETWEEN: Uses passed high_price/low_price or candle_window to check range containment

        Args:
            trade: The shadow trade record
            exit_price: Exit price (candle close)
            high_price: High price of resolution candle (for barrier checking)
            low_price: Low price of resolution candle (for barrier checking)

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
                # C01 Fix: Use resolution_context (post-entry candles) instead of candle_window (pre-entry)
                # Priority: 1) resolution_context, 2) passed high/low, 3) candle_window (legacy fallback)
                
                # C01 Fix: Use resolution_context (post-entry candles) if available, and append current candle
                all_highs = []
                all_lows = []

                if trade.resolution_context and len(trade.resolution_context) > 0:
                     context = np.array(trade.resolution_context)
                     all_highs.extend(context[:, 0].tolist())
                     all_lows.extend(context[:, 1].tolist())

                # Always include the current resolution candle if provided (it's the final candle)
                if high_price is not None and low_price is not None:
                     all_highs.append(high_price)
                     all_lows.append(low_price)

                if all_highs:
                    high_prices = np.array(all_highs)
                    low_prices = np.array(all_lows)
                    self.logger.debug(
                        f"TOUCH resolution using {len(high_prices)} candles (context + current)"
                    )
                elif trade.candle_window and len(trade.candle_window) > 0:
                    # Legacy fallback to candle_window (pre-entry data - less accurate)
                    candles = np.array(trade.candle_window)

                    if candles.ndim < 2 or candles.shape[1] < 4:
                        self.logger.warning(
                            f"Invalid candle_window shape {candles.shape} for trade {trade.trade_id}. "
                            f"Expected at least 4 columns (OHLC)."
                        )
                        return None

                    high_prices = candles[:, CANDLE_COL_HIGH]
                    low_prices = candles[:, CANDLE_COL_LOW]
                    self.logger.warning(
                        f"TOUCH trade {trade.trade_id[:8]} using legacy candle_window (pre-entry data)"
                    )
                else:
                    self.logger.warning(
                        f"TOUCH trade {trade.trade_id[:8]} missing resolution_context and candle_window"
                    )
                    return None

                # C02 & R05: Use BarrierCalculator
                if trade.barrier_level is not None:
                    levels = self.barrier_calculator.calculate(
                        entry_price=trade.entry_price,
                        contract_type=CONTRACT_TYPES.TOUCH_NO_TOUCH,
                        barrier_offset=trade.barrier_level
                    )
                else:
                    # Fallback to percentage
                    self.logger.debug(
                        f"Using default barrier pct {self.default_touch_barrier_pct} for trade {trade.trade_id}"
                    )
                    levels = self.barrier_calculator.calculate_from_percentage(
                        entry_price=trade.entry_price,
                        contract_type=CONTRACT_TYPES.TOUCH_NO_TOUCH,
                        barrier_pct=self.default_touch_barrier_pct
                    )
                
                upper_barrier = levels.upper
                lower_barrier = levels.lower

                # I04 Fix: Directional validation for barrier touches
                # For TOUCH trades, we need to check if barrier was touched in the correct direction
                barrier_direction = trade.metadata.get("barrier_direction") if trade.metadata else None
                
                if barrier_direction == "UP":
                    # Only upper barrier matters for UP direction
                    touched = bool(np.any(high_prices >= upper_barrier))
                elif barrier_direction == "DOWN":
                    # Only lower barrier matters for DOWN direction
                    touched = bool(np.any(low_prices <= lower_barrier))
                else:
                    # Default: either barrier touch counts (symmetric)
                    touched = bool(
                        np.any(high_prices >= upper_barrier) or np.any(low_prices <= lower_barrier)
                    )

                # Log when barrier hit detected
                if touched and high_price is not None:
                    self.logger.debug(
                        f"🎯 TOUCH barrier hit: high={high_price:.5f}, low={low_price:.5f}, "
                        f"upper={upper_barrier:.5f}, lower={lower_barrier:.5f}, dir={barrier_direction}"
                    )

                if trade.direction == "TOUCH":
                    return touched
                elif trade.direction == "NO_TOUCH":
                    return not touched

            elif trade.contract_type == CONTRACT_TYPES.STAYS_BETWEEN:
                # STAYS_BETWEEN: price must stay within ±band of entry
                # C01 Fix: Use resolution_context (post-entry candles) instead of candle_window (pre-entry)
                # Priority: 1) resolution_context, 2) passed high/low, 3) candle_window (legacy fallback)
                
                # C01 Fix: Use resolution_context (post-entry candles) if available, and append current candle
                all_highs = []
                all_lows = []

                if trade.resolution_context and len(trade.resolution_context) > 0:
                     context = np.array(trade.resolution_context)
                     all_highs.extend(context[:, 0].tolist())
                     all_lows.extend(context[:, 1].tolist())

                # Always include the current resolution candle if provided (it's the final candle)
                if high_price is not None and low_price is not None:
                     all_highs.append(high_price)
                     all_lows.append(low_price)

                if all_highs:
                    high_prices = np.array(all_highs)
                    low_prices = np.array(all_lows)
                    self.logger.debug(
                        f"RANGE resolution using {len(high_prices)} candles (context + current)"
                    )
                elif trade.candle_window and len(trade.candle_window) > 0:
                    # Legacy fallback to candle_window (pre-entry data - less accurate)
                    candles = np.array(trade.candle_window)

                    if candles.ndim < 2 or candles.shape[1] < 4:
                        self.logger.warning(
                            f"Invalid candle_window shape {candles.shape} for trade {trade.trade_id}. "
                            f"Expected at least 4 columns (OHLC)."
                        )
                        return None

                    high_prices = candles[:, CANDLE_COL_HIGH]
                    low_prices = candles[:, CANDLE_COL_LOW]
                    self.logger.warning(
                        f"RANGE trade {trade.trade_id[:8]} using legacy candle_window (pre-entry data)"
                    )
                else:
                    self.logger.warning(
                        f"RANGE trade {trade.trade_id[:8]} missing resolution_context and candle_window"
                    )
                    return None

                # C02 & R05: Use BarrierCalculator
                if trade.barrier_level is not None and trade.barrier2_level is not None:
                    levels = self.barrier_calculator.calculate(
                        entry_price=trade.entry_price,
                        contract_type=CONTRACT_TYPES.STAYS_BETWEEN,
                        barrier_offset=trade.barrier_level,
                        barrier2_offset=trade.barrier2_level
                    )
                else:
                    # Fallback to percentage
                    self.logger.debug(
                        f"Using default range pct {self.default_range_barrier_pct} for trade {trade.trade_id}"
                    )
                    levels = self.barrier_calculator.calculate_from_percentage(
                        entry_price=trade.entry_price,
                        contract_type=CONTRACT_TYPES.STAYS_BETWEEN,
                        barrier_pct=self.default_range_barrier_pct
                        # Note: calculate_from_percentage uses barrier_pct for range logic too if barrier2 not distinguished
                        # or we should update calculate_from_percentage to accept separate range pct
                    )
                
                upper_band = levels.upper
                lower_band = levels.lower

                # Price stayed in range if ALL highs below upper and ALL lows above lower
                stayed_in = bool(
                    np.all(high_prices <= upper_band) and np.all(low_prices >= lower_band)
                )
                
                # Log when range breach detected
                if not stayed_in and high_price is not None:
                    self.logger.debug(
                        f"🎯 RANGE breach: high={high_price:.5f}, low={low_price:.5f}, "
                        f"upper={upper_band:.5f}, lower={lower_band:.5f}"
                    )
                
                return stayed_in

            # Unsupported contract type
            self.logger.warning(f"Unsupported contract type: {trade.contract_type}")
            return None

        except Exception as e:
            self.logger.error(f"Error determining outcome for {trade.trade_id}: {e}")
            return None
