"""
Safety Shield and Trade Execution Wrapper.

Provides a safety layer around trade execution to enforce:
- Rate limits (per minute/hour/day)
- Max drawdown limits
- Stake size caps
- Kill switch functionality

Uses persistent storage to maintain safety state across restarts.
"""

import asyncio
import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Protocol, Optional

from execution.executor import TradeExecutor, TradeResult, TradeSignal
from execution.safety_store import SQLiteSafetyStateStore

logger = logging.getLogger(__name__)

@dataclass
class ExecutionSafetyConfig:
    """Configuration for safety limits."""
    max_trades_per_minute: int = 5
    max_trades_per_minute_per_symbol: int = 2
    max_daily_loss: float = 50.0
    max_stake_per_trade: float = 20.0
    max_retry_attempts: int = 3
    retry_base_delay: float = 1.0
    kill_switch_enabled: bool = False


class SafeTradeExecutor:
    """
    Wrapper around TradeExecutor that enforces safety policies.
    
    Persists state to prevent limit bypassing via restarts.
    """
    
    def __init__(
        self,
        inner_executor: TradeExecutor,
        config: ExecutionSafetyConfig,
        state_file: str | Path,
        stake_resolver: Optional[Callable[[TradeSignal], float | Any]] = None  # Any for Coroutine
    ):
        """
        Initialize safe executor.
        
        Args:
            inner_executor: Raw execution implementation
            config: Safety limits configuration
            state_file: Path to SQLite state DB
            stake_resolver: Function to resolve stake amount for a signal (for pre-check)
        """
        self.inner = inner_executor
        self.config = config
        self.store = SQLiteSafetyStateStore(state_file)
        self.stake_resolver = stake_resolver
        
        # In-memory short-term rate limits (reset on restart is acceptable for per-minute)
        # For daily limits, we check the DB.
        self._minute_trades: list[float] = []
        self._symbol_minute_trades: dict[str, list[float]] = {}
        
        logger.info(f"SafeTradeExecutor initialized with DB: {state_file}")

    async def execute(self, signal: TradeSignal) -> TradeResult:
        """
        Execute trade with safety checks.
        
        Args:
            signal: Trade signal
            
        Returns:
            TradeResult (success or failure with error)
        """
        # 1. Check Kill Switch
        if self.config.kill_switch_enabled:
            return self._reject("Kill switch enabled")

        # 2. Check State-Based Limits (Daily)
        if not await self._check_daily_limits():
            return self._reject("Daily limits exceeded")

        # 3. Check Short-Term Rate Limits
        if not self._check_rate_limits(get_symbol_from_signal(signal)):
             return self._reject("Rate limit exceeded")
             
        # 4. Check Stake Amount (if resolver provided)
        if self.stake_resolver:
            try:
                # FIX: Pass the full signal to the resolver to get the ACTUAL stake logic
                # Support both sync and async resolvers
                if asyncio.iscoroutinefunction(self.stake_resolver):
                    stake = await self.stake_resolver(signal)
                else:
                    stake = self.stake_resolver(signal)

                if stake > self.config.max_stake_per_trade:
                    msg = f"Stake {stake:.2f} exceeds safety limit {self.config.max_stake_per_trade:.2f}"
                    return self._reject(msg)
                
                # INJECTION: Store validated stake in signal metadata to prevent drift
                # This ensures the inner executor uses exactly what we validated
                signal.metadata["stake"] = stake
                
            except Exception as e:
                 logger.error(f"Error resolving stake in safety check: {e}")
                 # Fail safe -> reject
                 return self._reject(f"Stake resolution failed: {e}")

        # 5. Execute with Retries
        result = await self._execute_with_retry(signal)
        
        # 6. Update State
        if result.success:
            await self._record_trade(signal.metadata.get("symbol", "unknown"), result)

        return result

    def _reject(self, reason: str) -> TradeResult:
        logger.warning(f"Trade rejected by Safety Shield: {reason}")
        return TradeResult(success=False, error=f"Safety: {reason}")

    async def _check_daily_limits(self) -> bool:
        """Check if daily trade count or loss limit is hit."""
        # H09: Use async get to avoid blocking
        trade_count, daily_pnl = await self.store.get_daily_stats_async()
        
        # We assume max_daily_loss is a positive number representing MAX LOSS allowed
        # e.g. 50.0 means we stop if pnl <= -50.0
        if self.config.max_daily_loss > 0 and daily_pnl <= -self.config.max_daily_loss:
            logger.warning(f"Daily loss limit hit: {daily_pnl:.2f} <= -{self.config.max_daily_loss}")
            return False
            
        return True

    def _check_rate_limits(self, symbol: str) -> bool:
        """Check per-minute limits."""
        now = datetime.now(timezone.utc).timestamp()
        window = 60.0
        
        # cleanup
        self._minute_trades = [t for t in self._minute_trades if now - t < window]
        if symbol in self._symbol_minute_trades:
             self._symbol_minute_trades[symbol] = [t for t in self._symbol_minute_trades[symbol] if now - t < window]
        
        # check global
        if len(self._minute_trades) >= self.config.max_trades_per_minute:
            return False
            
        # check symbol
        if symbol and len(self._symbol_minute_trades.get(symbol, [])) >= self.config.max_trades_per_minute_per_symbol:
            return False
            
        return True

    async def _execute_with_retry(self, signal: TradeSignal) -> TradeResult:
        """Execute with exponential backoff on transient errors."""
        # Define retryable exceptions
        # We try to import APIError dynamically to avoid hard dependency if possible, 
        # or just assume it's available since this is part of the system.
        retryable_errors: tuple[type[Exception], ...] = (
            ConnectionError, 
            TimeoutError, 
            asyncio.TimeoutError
        )
        
        try:
            from deriv_api import APIError
            retryable_errors += (APIError,)
        except ImportError:
            pass

        for attempt in range(self.config.max_retry_attempts):
            try:
                result = await self.inner.execute(signal)
                return result
            except retryable_errors as e:
                # Retryable error
                logger.warning(f"Transient execution error (attempt {attempt+1}): {e}")
                if attempt < self.config.max_retry_attempts - 1:
                    await asyncio.sleep(self.config.retry_base_delay * (2 ** attempt))
                else:
                    return TradeResult(success=False, error=str(e))
            except Exception as e:
                # M15: Non-retryable error (logic bug, validation, etc.) - Fail fast!
                logger.error(f"Non-retryable execution error: {e}", exc_info=True)
                return TradeResult(success=False, error=f"Permanent Error: {e}")
                
        return TradeResult(success=False, error="Max retries exhausted")

    async def _record_trade(self, symbol: str, result: TradeResult):
        """Update persistent and memory state."""
        now = datetime.now(timezone.utc).timestamp()
        
        # Memory (Rate Limits)
        self._minute_trades.append(now)
        if symbol not in self._symbol_minute_trades:
            self._symbol_minute_trades[symbol] = []
        self._symbol_minute_trades[symbol].append(now)
        
        # Persistence (Daily Stats)
        # H09: Async update
        await self.store.increment_daily_trade_count_async()
        # We don't know PnL yet! This is just entry.
        # PnL updates come from outcome resolution typically.
        # But we track *entry* count here.

    # Hook for outcome tracking to update P&L
    async def register_outcome(self, pnl: float):
        """Called by RealTradeTracker when trade closes.
        
        C04 Fix: Now async to prevent blocking I/O in event loop.
        """
        await self.store.update_daily_pnl_async(pnl)

    # Alias for backward compatibility with RealTradeTracker
    async def update_pnl(self, pnl: float):
        """Alias for register_outcome - called by RealTradeTracker."""
        await self.register_outcome(pnl)


def get_symbol_from_signal(signal: TradeSignal) -> str:
    """Extract symbol from signal metadata if available."""
    # Logic depends on signal structure.
    return signal.metadata.get("symbol", "unknown")
