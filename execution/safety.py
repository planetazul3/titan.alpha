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

try:
    from opentelemetry import trace
    TRACING_ENABLED = True
except ImportError:
    TRACING_ENABLED = False

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
            executor: Raw execution implementation
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
        
        if TRACING_ENABLED:
            self.tracer = trace.get_tracer(__name__)
        else:
            self.tracer = None
            
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
        safe_signal = signal # Initialize safe_signal with original signal
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
                
                # CRITICAL-003: Use immutable pattern for stake injection
                # Instead of mutating the shared signal, we create a copy with the validated stake
                safe_signal = signal.with_metadata(stake=stake)
                
            except Exception as e:
                 logger.error(f"Error resolving stake in safety check: {e}")
                 # Fail safe -> reject
                 return self._reject(f"Stake resolution failed: {e}")

        # 5. Execute with Retries
        if self.tracer:
            with self.tracer.start_as_current_span("safety_executor.execute") as span:
                span.set_attribute("symbol", get_symbol_from_signal(signal))
                span.set_attribute("contract_type", str(signal.contract_type))
                span.set_attribute("stake", signal.metadata.get("stake", 0.0))
                
                result = await self._execute_with_retry(safe_signal)
        else:
            result = await self._execute_with_retry(safe_signal)
            
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
        """Execute with exponential backoff on business logic errors only.
        
        IMPORTANT: Transport-level errors (ConnectionError, TimeoutError) are 
        intentionally NOT retried here because DerivClient._reconnect already 
        handles them with its own exponential backoff (up to 5 attempts).
        
        This layer only retries on business logic errors from the API:
        - MarketIsClosed
        - InvalidContractParameters  
        - ContractBuyValidationError (retryable subset)
        
        This prevents "nested retry storms" where both layers back off
        independently, causing unpredictable multi-minute delays.
        """
        # Business logic error codes that are worth retrying at safety level
        # These are temporary conditions that may resolve shortly
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
                result = await self.inner.execute(signal)
                return result
            except Exception as e:
                error_code = getattr(e, "code", None) or ""
                
                # Check if this is a retryable business logic error
                is_retryable_api = (
                    APIError is not None 
                    and isinstance(e, APIError) 
                    and error_code in RETRYABLE_API_CODES
                )
                
                if is_retryable_api:
                    # Business logic error - retry with backoff
                    logger.warning(
                        f"Retryable business error (attempt {attempt+1}/{self.config.max_retry_attempts}): "
                        f"[{error_code}] {e}"
                    )
                    if attempt < self.config.max_retry_attempts - 1:
                        await asyncio.sleep(self.config.retry_base_delay * (2 ** attempt))
                    else:
                        return TradeResult(success=False, error=f"Max retries exhausted: {e}")
                elif isinstance(e, (ConnectionError, TimeoutError, asyncio.TimeoutError)):
                    # Transport error - DerivClient handles reconnection
                    # Fail fast here to let caller decide
                    logger.warning(
                        f"Transport error (handled by DerivClient): {type(e).__name__}: {e}"
                    )
                    return TradeResult(success=False, error=f"Transport Error: {e}")
                else:
                    # Non-retryable error (logic bug, validation, etc.) - Fail fast!
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

    def get_safety_statistics(self) -> dict:
        """Get current safety statistics."""
        # Note: Some stats require async store access, so we return what's easily available
        # or just placeholders for the keys live.py expects.
        return {
            "kill_switch": self.config.kill_switch_enabled,
            "max_trades_per_min": self.config.max_trades_per_minute,
            "max_daily_loss": self.config.max_daily_loss,
            "last_reconciliation": getattr(self, "_last_reconciliation", None)
        }

    async def reconcile_with_api(
        self, 
        client: Any, 
        pending_store: Optional[Any] = None
    ) -> dict:
        """
        Reconcile safety state with API on startup.
        
        Queries open contracts from the Deriv API and compares with the
        pending trade store to detect discrepancies:
        - api_only: Contracts on API not in store (potential missed tracking)
        - store_only: Contracts in store not on API (already settled)
        
        This is a DIAGNOSTIC tool - it logs discrepancies but does not
        auto-resolve them (safety-first approach).
        
        Args:
            client: DerivClient instance (must be connected)
            pending_store: Optional PendingTradeStore instance
            
        Returns:
            Reconciliation results dict with:
            - api_open_count: Number of open contracts from API
            - store_pending_count: Number of pending trades in store
            - matched: List of matched contract IDs
            - api_only: Contracts on API but not in store
            - store_only: Contracts in store but not on API
        """
        result = {
            "api_open_count": 0,
            "store_pending_count": 0,
            "matched": [],
            "api_only": [],
            "store_only": [],
            "error": None
        }
        
        try:
            # Get open contracts from API
            api_response = await client.get_open_contracts()
            
            # Parse contract IDs from API response
            api_contracts = set()
            if isinstance(api_response, dict):
                # proposal_open_contract returns a dict with contract details
                poc = api_response.get("proposal_open_contract", {})
                if poc and isinstance(poc, dict):
                    cid = poc.get("contract_id")
                    if cid:
                        api_contracts.add(str(cid))
            elif isinstance(api_response, list):
                for item in api_response:
                    if isinstance(item, dict):
                        cid = item.get("contract_id") or item.get("id")
                        if cid:
                            api_contracts.add(str(cid))
                            
            result["api_open_count"] = len(api_contracts)
            
            # Get pending trades from store
            store_contracts = set()
            if pending_store is not None:
                try:
                    pending = pending_store.get_all_pending()
                    for trade in pending:
                        cid = trade.get("contract_id")
                        if cid:
                            store_contracts.add(str(cid))
                except Exception as e:
                    logger.warning(f"Could not read pending store: {e}")
                    
            result["store_pending_count"] = len(store_contracts)
            
            # Compare
            result["matched"] = list(api_contracts & store_contracts)
            result["api_only"] = list(api_contracts - store_contracts)
            result["store_only"] = list(store_contracts - api_contracts)
            
            # Log reconciliation results
            if result["api_only"]:
                logger.warning(
                    f"⚠️ Reconciliation found {len(result['api_only'])} open contracts "
                    f"NOT in pending store: {result['api_only']}"
                )
            if result["store_only"]:
                logger.info(
                    f"Reconciliation: {len(result['store_only'])} stored trades "
                    f"already settled: {result['store_only']}"
                )
            
            logger.info(
                f"Reconciliation complete: {result['api_open_count']} API contracts, "
                f"{result['store_pending_count']} pending in store, "
                f"{len(result['matched'])} matched"
            )
            
            # Track last reconciliation time
            self._last_reconciliation = datetime.now(timezone.utc).isoformat()
            
        except Exception as e:
            logger.error(f"Reconciliation failed: {e}")
            result["error"] = str(e)
            
        return result


def get_symbol_from_signal(signal: TradeSignal) -> str:
    """Extract symbol from signal metadata if available."""
    # Logic depends on signal structure.
    return signal.metadata.get("symbol", "unknown")
