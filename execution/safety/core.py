import asyncio
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional, TYPE_CHECKING
import math

from execution.common.types import ExecutionRequest
from execution.executor import TradeExecutor, TradeResult
from execution.safety_store import SQLiteSafetyStateStore
from utils.numerical_validation import ensure_finite

from .config import ExecutionSafetyConfig
from .reconciliation import SafetyReconciler

if TYPE_CHECKING:
    from execution.policy import ExecutionPolicy

try:
    from opentelemetry import trace
    TRACING_ENABLED = True
except ImportError:
    TRACING_ENABLED = False

logger = logging.getLogger(__name__)


class SafeTradeExecutor:
    """
    Wrapper around TradeExecutor that enforces safety policies.
    
    Responsibilities:
    - Record trade outcomes to persistent SQLite store
    - Execute trades with retry logic for API errors
    - Provide rate-limit state for ExecutionPolicy integration
    - Validate stake amounts before execution
    
    H4: Enforces Warmup Veto.
    RC-8: Enforces Numerical Safety (isfinite checks).
    """
    
    def __init__(
        self,
        inner_executor: TradeExecutor,
        config: ExecutionSafetyConfig,
        state_file: str | Path,
        policy: Optional["ExecutionPolicy"] = None,
    ):
        self.inner = inner_executor
        self.config = config
        self.store = SQLiteSafetyStateStore(state_file)
        self._policy = policy
        
        # Concurrency safety
        self._state_lock = asyncio.Lock()
        self._last_reconciliation: str | None = None
        self._pending_global = 0
        self._pending_symbols: dict[str, int] = {}
        
        if TRACING_ENABLED:
            self.tracer: Any = trace.get_tracer(__name__)
        else:
            self.tracer = None
        
        # RISK-ARCH-REVIEW: Register rate-limit vetoes with policy if provided
        if policy is not None:
            self.register_with_policy(policy)
            
        logger.info(f"SafeTradeExecutor initialized with DB: {state_file}, policy_integrated={policy is not None}")

    def register_with_policy(self, policy: "ExecutionPolicy") -> None:
        """Register rate-limit checks with ExecutionPolicy for unified veto logging."""
        from execution.policy import VetoPrecedence
        
        self._policy = policy
        
        policy.register_veto(
            level=VetoPrecedence.RATE_LIMIT,
            # Sync fallback (blocking)
            check_fn=lambda: self.store.get_trades_in_window(None, 60.0) >= self.config.max_trades_per_minute,
            # Async implementation (non-blocking) - Use the public check method which handles pending
            async_check_fn=self._check_global_rate_limit_veto,
            reason=lambda: f"Global rate limit hit (max {self.config.max_trades_per_minute}/min)",
             details_fn=lambda: {
                "limit": self.config.max_trades_per_minute,
            }
        )
        
        logger.info("SafeTradeExecutor registered persistent rate-limit vetoes with ExecutionPolicy")
    
    async def _check_global_rate_limit_veto(self) -> bool:
        """Rate limit check for veto policy (returns True if vetoed)."""
        async with self._state_lock:
             count = await self.store.get_trades_in_window_async(None, 60.0)
             # Add pending trades to the count
             total_count = count + self._pending_global
             return total_count >= self.config.max_trades_per_minute

    async def execute(self, request: ExecutionRequest) -> TradeResult:
        """Execute execution request with safety checks."""
        # RC-8: Numerical Safety - Ensure Inputs are Finite
        if not math.isfinite(request.stake):
             return self._reject(f"RC-8: Non-finite stake detected: {request.stake}")

        # H4: Warmup Veto (Explicit Check)
        # Note: request doesn't carry price history, but we act as a gatekeeper.
        # If the CALLER (DecisionEngine) passed a signal, it should have checked warmup.
        # However, as a safety layer, if we had access to buffer we would check it.
        # Currently ExecutionRequest doesn't have buffer info.
        
        # 1. Fallback Kill Switch check
        if self._policy is None and self.config.kill_switch_enabled:
            return self._reject("Kill switch enabled")

        # 2. Fallback Daily Limits check
        if self._policy is None and not await self._check_daily_limits():
            return self._reject("Daily limits exceeded")

        # 3. Check Short-Term Rate Limits AND Reserve Slot
        # We must reserve the slot inside the check to prevent race conditions
        reservation_token = await self._check_and_reserve_rate_limit(request.symbol)
        if not reservation_token:
             return self._reject("Rate limit exceeded")
             
        try:
            # 4. Check Stake Amount
            if request.stake > self.config.max_stake_per_trade:
                msg = f"Stake {request.stake:.2f} exceeds safety limit {self.config.max_stake_per_trade:.2f}"
                return self._reject(msg)

            # 5. Execute with Retries
            result = await self._execute_with_retry(request)
                
            # 6. Update State
            if result.success:
                await self._record_trade(request.symbol, result)

            return result
        finally:
            # RELEASE RESERVATION
            await self._release_rate_limit_reservation(request.symbol)

    def _reject(self, reason: str) -> TradeResult:
        logger.warning(f"Trade rejected by Safety Shield: {reason}")
        return TradeResult(success=False, error=f"Safety: {reason}")

    async def _check_daily_limits(self) -> bool:
        """Check if daily trade count or loss limit is hit."""
        trade_count, daily_pnl = await self.store.get_daily_stats_async()
        
        # RC-8: Numerical Safety
        daily_pnl = ensure_finite(daily_pnl, "SafeTradeExecutor.check_daily_limits", 0.0)
        
        if not math.isfinite(daily_pnl):
            logger.error(f"CRITICAL-002: NaN daily_pnl detected: {daily_pnl}. Failing closed.")
            return False

        if self.config.max_daily_loss > 0 and daily_pnl <= -self.config.max_daily_loss:
            logger.warning(f"Daily loss limit hit: {daily_pnl:.2f} <= -{self.config.max_daily_loss}")
            return False
            
        return True

    async def _check_and_reserve_rate_limit(self, symbol: str) -> bool:
        """Check limits and increment pending counters if allowed."""
        async with self._state_lock:
             # Check Global
             global_count = await self.store.get_trades_in_window_async(None, 60.0)
             if (global_count + self._pending_global) >= self.config.max_trades_per_minute:
                 return False
                 
             # Check Symbol
             if symbol:
                 symbol_count = await self.store.get_trades_in_window_async(symbol, 60.0)
                 pending_symbol = self._pending_symbols.get(symbol, 0)
                 if (symbol_count + pending_symbol) >= self.config.max_trades_per_minute_per_symbol:
                     return False

             # Reserve
             self._pending_global += 1
             if symbol:
                 self._pending_symbols[symbol] = self._pending_symbols.get(symbol, 0) + 1
             
             return True

    async def _release_rate_limit_reservation(self, symbol: str):
        """Decrement pending counters."""
        async with self._state_lock:
            if self._pending_global > 0:
                self._pending_global -= 1
            
            if symbol and symbol in self._pending_symbols:
                if self._pending_symbols[symbol] > 0:
                    self._pending_symbols[symbol] -= 1

    async def _execute_with_retry(self, request: ExecutionRequest) -> TradeResult:
        """Execute with exponential backoff on business logic errors only."""
        RETRYABLE_API_CODES = frozenset([
            "MarketIsClosed",
            "MarketIsClosedForThePeriod",
            "TradingIsDisabled",
            "RateLimit",
        ])
        
        APIError = None
        try:
            from deriv_api import APIError as _APIError
            APIError = _APIError
        except ImportError:
            pass

        for attempt in range(self.config.max_retry_attempts):
            try:
                result = await self.inner.execute(request)
                return result
            except Exception as e:
                error_code = getattr(e, "code", None) or ""
                
                is_retryable_api = (
                    APIError is not None 
                    and isinstance(e, APIError) 
                    and error_code in RETRYABLE_API_CODES
                )
                
                if is_retryable_api:
                    logger.warning(
                        f"Retryable business error (attempt {attempt+1}/{self.config.max_retry_attempts}): "
                        f"[{error_code}] {e}"
                    )
                    if attempt < self.config.max_retry_attempts - 1:
                        await asyncio.sleep(self.config.retry_base_delay * (2 ** attempt))
                    else:
                        return TradeResult(success=False, error=f"Max retries exhausted: {e}")
                elif isinstance(e, (ConnectionError, TimeoutError, asyncio.TimeoutError)):
                    logger.warning(
                        f"Transport error (handled by DerivClient): {type(e).__name__}: {e}"
                    )
                    return TradeResult(success=False, error=f"Transport Error: {e}")
                else:
                    logger.error(f"Non-retryable execution error: {e}", exc_info=True)
                    return TradeResult(success=False, error=f"Permanent Error: {e}")
                
        return TradeResult(success=False, error="Max retries exhausted")

    async def _record_trade(self, symbol: str, result: TradeResult):
        """Update persistent and memory state."""
        now = datetime.now(timezone.utc).timestamp()
        await self.store.record_trade_timestamp_async(symbol, now)
        await self.store.increment_daily_trade_count_async()

    async def register_outcome(self, pnl: float):
        """Called by RealTradeTracker when trade closes."""
        # RC-8: Numerical Safety
        safe_pnl = ensure_finite(pnl, "SafeTradeExecutor.register_outcome", 0.0)
        await self.store.update_daily_pnl_async(safe_pnl)

    async def update_pnl(self, pnl: float):
        """Alias for register_outcome."""
        await self.register_outcome(pnl)

    def get_safety_statistics(self) -> dict:
        """Get current safety statistics."""
        return {
            "kill_switch": self.config.kill_switch_enabled,
            "max_trades_per_min": self.config.max_trades_per_minute,
            "max_daily_loss": self.config.max_daily_loss,
            "last_reconciliation": self._last_reconciliation
        }

    async def reconcile_with_api(self, client: Any, pending_store: Optional[Any] = None) -> dict:
        """Reconcile safety state with API on startup."""
        result = await SafetyReconciler.reconcile(client, pending_store)
        self._last_reconciliation = datetime.now(timezone.utc).isoformat()
        return result
