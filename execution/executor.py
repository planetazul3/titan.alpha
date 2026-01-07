"""
Trade execution abstraction.

Integrated with SQLiteIdempotencyStore (CRITICAL-002) and ExecutionLogger (REC-001).

Performance/Safety Improvements:
- Multi-failure circuit breaker: Triggers on any 5 consecutive API failures within 10 minutes
  (not just idempotency errors). Protects against API outages, insufficient balance, etc.
"""

import logging
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Protocol, runtime_checkable, Optional, Union

from execution.common.types import ExecutionRequest
from execution.contract_params import ContractParameterService
from execution.idempotency_store import SQLiteIdempotencyStore
from observability.execution_logging import execution_logger
from deriv_api import APIError

logger = logging.getLogger(__name__)

# Circuit breaker configuration
from config.constants import (
    CIRCUIT_BREAKER_FAILURE_THRESHOLD,
    CIRCUIT_BREAKER_WINDOW_SECONDS,
)

@dataclass
class TradeResult:
    success: bool
    contract_id: str | None = None
    entry_price: float | None = None
    error: str | None = None
    timestamp: datetime | None = None

    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now(timezone.utc)

@runtime_checkable
class TradeExecutor(Protocol):
    async def execute(self, request: ExecutionRequest) -> TradeResult:
        ...

class DerivTradeExecutor:
    """
    Deriv-specific trade executor with idempotency and structured logging.
    
    Safety Features:
    - Multi-failure circuit breaker: Triggers on 5 consecutive API failures within 10 minutes
    - Idempotency guard: Prevents duplicate executions via SQLiteIdempotencyStore
    - Structured logging: All attempts/results logged via ExecutionLogger
    """

    def __init__(
        self, 
        client, 
        settings, 
        position_sizer=None, 
        policy=None,
        idempotency_store: Optional[SQLiteIdempotencyStore] = None
    ):
        from execution.position_sizer import FixedStakeSizer
        self.client = client
        self.settings = settings
        self.position_sizer = position_sizer or FixedStakeSizer(stake=settings.trading.stake_amount)
        self.policy = policy
        self.idempotency_store = idempotency_store
        self.param_service = ContractParameterService(settings)
        
        self._executed_count = 0

        self._failed_count = 0
        
        # Multi-failure circuit breaker: track failure timestamps for rolling window
        self._failure_timestamps: list[float] = []
        
        logger.info(f"DerivTradeExecutor initialized with idempotency={idempotency_store is not None}")

    async def execute(self, request: ExecutionRequest, check_only: bool = False) -> TradeResult:
        """
        Execute validated request on Deriv platform.
        Strictly requires ExecutionRequest.
        """


        # CRITICAL-FIX (C-001): Check circuit breaker status
        if self._is_circuit_breaker_active():
            error_msg = f"Circuit breaker active: {len(self._failure_timestamps)} failures in last {CIRCUIT_BREAKER_WINDOW_SECONDS}s"
            logger.warning(error_msg)
            return TradeResult(success=False, error=error_msg)

        try:
            # 5. Idempotency Check (CRITICAL-002)
            # 5. Idempotency Check (CRITICAL-002)
            if self.idempotency_store:
                # Atomic check-and-reserve
                is_new, cached_id = await self.idempotency_store.check_and_reserve_async(request.signal_id, request.symbol)
                if not is_new:
                    logger.warning(f"Idempotency Guard: Signal {request.signal_id} already executed or pending. ID: {cached_id}")
                    return TradeResult(success=True, contract_id=cached_id, entry_price=request.stake)

            if check_only:
                # If check_only, we should release the reservation if we made one? 
                # Ideally check_only shouldn't reserve. But atomic reservation is safer.
                # We'll just release it.
                if self.idempotency_store:
                    await self.idempotency_store.delete_record_async(request.signal_id)
                return TradeResult(success=True)

            # 6. Structured Log Attempt (REC-001)
            execution_logger.log_trade_attempt(
                request.signal_id, 
                request.contract_type, 
                request.contract_type, # Direction is implicit in contract_type now (e.g. CALL)
                request.stake
            )

            # 7. Execute via Deriv API
            result = await self.client.buy(
                contract_type=request.contract_type,
                amount=request.stake,
                duration=request.duration,
                duration_unit=request.duration_unit,
                barrier=request.barrier,
                barrier2=request.barrier2
            )

            if result.get("buy"):
                contract_id = str(result["buy"]["contract_id"])
                buy_price = float(result["buy"]["buy_price"])
                
                execution_logger.log_trade_success(request.signal_id, contract_id, buy_price)
                
                # Record in idempotency store
                # Record in idempotency store (Update the reservation)
                if self.idempotency_store:
                    await self.idempotency_store.update_contract_id_async(request.signal_id, contract_id)
                
                self._executed_count += 1
                self._clear_failure_window()  # Reset on success
                return TradeResult(success=True, contract_id=contract_id, entry_price=buy_price)
            else:
                error_msg = result.get("error", {}).get("message", "Unknown execution error")
                execution_logger.log_trade_failure(request.signal_id, error_msg, details=result)
                self._failed_count += 1
                self._record_failure(error_msg)  # Multi-failure tracking
                return TradeResult(success=False, error=error_msg)

        except APIError as e:
            # Handle Deriv-specific API errors
            # APIError usually has a 'message' and 'code' if it comes from the API response
            error_code = getattr(e, "code", "UnknownAPIError")
            error_msg = str(e)
            
            logger.error(f"API Error during execution ({error_code}): {error_msg}")
            execution_logger.log_trade_failure(request.signal_id, error_msg, details={"code": error_code})
            
            self._failed_count += 1
            
            # Rate limits are critical but temporary
            if "RateLimit" in str(error_code) or "RateLimit" in error_msg:
                 logger.warning("Rate limit hit! Circuit breaker will likely trigger soon.")
            
            self._record_failure(f"{error_code}: {error_msg}")
            
            # Clean up reservation on definite API failure
            if self.idempotency_store:
                await self.idempotency_store.delete_record_async(request.signal_id)
                
            return TradeResult(success=False, error=f"APIError: {error_msg}")

        except ConnectionError as e:
            logger.error(f"Connection lost during execution: {e}")
            execution_logger.log_trade_failure(request.signal_id, "ConnectionError")
            self._failed_count += 1
            self._record_failure("ConnectionError")
            return TradeResult(success=False, error="ConnectionError")

        except Exception as e:
            logger.exception(f"Unexpected error during execution: {e}")
            execution_logger.log_trade_failure(request.signal_id, str(e))
            self._failed_count += 1
            self._record_failure(str(e))  # Multi-failure tracking for all exceptions
            return TradeResult(success=False, error=str(e))
            
    async def shutdown(self) -> None:
        """Gracefully close resources."""
        if self.idempotency_store:
            try:
                await self.idempotency_store.close()
                logger.info("Executor: Idempotency store closed.")
            except Exception as e:
                logger.error(f"Executor: Failed to close idempotency store: {e}")
    
    def _record_failure(self, error_msg: str) -> None:
        """
        Record a failure and check if circuit breaker should trigger.
        
        Triggers circuit breaker on any N consecutive failures within the rolling window,
        regardless of error type (Idempotency, InternalServerError, InsufficientBalance, etc.).
        """
        now = time.monotonic()
        self._failure_timestamps.append(now)
        
        # Prune failures outside the rolling window
        cutoff = now - CIRCUIT_BREAKER_WINDOW_SECONDS
        self._failure_timestamps = [t for t in self._failure_timestamps if t > cutoff]
        
        # Check if we've hit the threshold
        if len(self._failure_timestamps) >= CIRCUIT_BREAKER_FAILURE_THRESHOLD and self.policy:
            self.policy.trigger_circuit_breaker(
                f"Multi-failure circuit breaker: {len(self._failure_timestamps)} failures in {CIRCUIT_BREAKER_WINDOW_SECONDS}s. Last: {error_msg[:50]}"
            )

            
    def _is_circuit_breaker_active(self) -> bool:
        """Check if circuit breaker is currently active."""
        if len(self._failure_timestamps) >= CIRCUIT_BREAKER_FAILURE_THRESHOLD:
            # Check if we are still within the window relative to the last failure
            # Actually, simply having N failures in the list (which is pruned on append) 
            # implies they are within the window.
            # But we should double check time just to be safe if _record_failure hasn't been called recently.
            
            # Prune first
            now = time.monotonic()
            cutoff = now - CIRCUIT_BREAKER_WINDOW_SECONDS
            active_failures = [t for t in self._failure_timestamps if t > cutoff]
            
            if len(active_failures) < len(self._failure_timestamps):
                 # Lazy prune
                 self._failure_timestamps = active_failures
                 
            return len(self._failure_timestamps) >= CIRCUIT_BREAKER_FAILURE_THRESHOLD
            
        return False

    def reset_circuit_breaker(self) -> None:
        """Manually reset the circuit breaker."""
        logger.info(f"Manual circuit breaker reset. clearing {len(self._failure_timestamps)} failures.")
        self._failure_timestamps.clear()
        self._failed_count = 0
    
    def _clear_failure_window(self) -> None:
        """Clear failure window on successful execution."""
        self._failure_timestamps.clear()

    def get_statistics(self) -> dict:
        total = self._executed_count + self._failed_count
        return {
            "executed": self._executed_count,
            "failed": self._failed_count,
            "success_rate": self._executed_count / total if total > 0 else 0.0,
        }

class MockTradeExecutor:
    """
    Mock executor that records signals without executing them.
    Used for testing and system verification.
    """
    def __init__(self):
        self._signals = []
        
    async def execute(self, request: ExecutionRequest) -> TradeResult:
        self._signals.append(request)
        return TradeResult(success=True, contract_id=f"MOCK_{request.signal_id}")
    
    async def shutdown(self) -> None:
        pass
        
    def get_signals(self) -> list[ExecutionRequest]:
        return self._signals # type: ignore[no-any-return]
