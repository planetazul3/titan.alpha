"""
Trade execution abstraction.

Integrated with SQLiteIdempotencyStore (CRITICAL-002) and ExecutionLogger (REC-001).
"""

import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Protocol, runtime_checkable, Optional

from config.constants import CONTRACT_TYPES
from execution.contract_params import ContractDurationResolver
from execution.idempotency_store import SQLiteIdempotencyStore
from execution.signals import TradeSignal
from observability.execution_logging import execution_logger

logger = logging.getLogger(__name__)

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
    async def execute(self, signal: TradeSignal) -> TradeResult:
        ...

class DerivTradeExecutor:
    """
    Deriv-specific trade executor with idempotency and structured logging.
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
        self.duration_resolver = ContractDurationResolver(settings)
        self._executed_count = 0
        self._failed_count = 0
        
        logger.info(f"DerivTradeExecutor initialized with idempotency={idempotency_store is not None}")

    async def execute(self, signal: TradeSignal, check_only: bool = False) -> TradeResult:
        """
        Execute trade on Deriv platform with safety guards.
        """
        try:
            # 1. Fetch balance for sizing
            try:
                balance = await self.client.get_balance()
            except Exception:
                balance = None

            # 2. Determine stake
            amount = signal.metadata.get("stake")
            if amount is None:
                amount = self.position_sizer.suggest_stake_for_signal(signal, account_balance=balance)
            
            if amount <= 0:
                return TradeResult(success=False, error="Zero stake amount")

            # 3. Resolve Durations (IMPORTANT-001)
            duration, duration_unit = self.duration_resolver.resolve_duration(signal.contract_type)
            
            # 4. Map Contract Types
            if signal.contract_type == CONTRACT_TYPES.RISE_FALL:
                contract = "CALL" if signal.direction == "CALL" else "PUT"
            elif signal.contract_type == CONTRACT_TYPES.TOUCH_NO_TOUCH:
                contract = "ONETOUCH" if signal.direction == "TOUCH" else "NOTOUCH"
            elif signal.contract_type == CONTRACT_TYPES.STAYS_BETWEEN:
                contract = "RANGE" if signal.direction == "IN" else "UPORDOWN"
            else:
                return TradeResult(success=False, error=f"Unsupported contract type: {signal.contract_type}")

            # 5. Idempotency Check (CRITICAL-002)
            if self.idempotency_store:
                cached_id = await self.idempotency_store.get_contract_id_async(signal.signal_id)
                if cached_id:
                    logger.warning(f"Idempotency Guard: Signal {signal.signal_id} already executed.")
                    return TradeResult(success=True, contract_id=cached_id, entry_price=amount)

            if check_only:
                return TradeResult(success=True)

            # 6. Structured Log Attempt (REC-001)
            execution_logger.log_trade_attempt(signal.signal_id, signal.contract_type, signal.direction, amount)

            # 7. Execute via Deriv API
            # Fetch barriers from metadata
            barrier = signal.metadata.get("barrier")
            barrier2 = signal.metadata.get("barrier2")

            result = await self.client.buy(
                contract_type=contract,
                amount=amount,
                duration=duration,
                duration_unit=duration_unit,
                barrier=barrier,
                barrier2=barrier2
            )

            if result.get("buy"):
                contract_id = str(result["buy"]["contract_id"])
                buy_price = float(result["buy"]["buy_price"])
                
                execution_logger.log_trade_success(signal.signal_id, contract_id, buy_price)
                
                # Record in idempotency store
                if self.idempotency_store:
                    await self.idempotency_store.record_execution_async(signal.signal_id, contract_id)
                
                self._executed_count += 1
                return TradeResult(success=True, contract_id=contract_id, entry_price=buy_price)
            else:
                error_msg = result.get("error", {}).get("message", "Unknown execution error")
                execution_logger.log_trade_failure(signal.signal_id, error_msg, details=result)
                self._failed_count += 1
                return TradeResult(success=False, error=error_msg)

        except Exception as e:
            execution_logger.log_trade_failure(signal.signal_id, str(e))
            self._failed_count += 1
            return TradeResult(success=False, error=str(e))

    def get_statistics(self) -> dict:
        total = self._executed_count + self._failed_count
        return {
            "executed": self._executed_count,
            "failed": self._failed_count,
            "success_rate": self._executed_count / total if total > 0 else 0.0,
        }
