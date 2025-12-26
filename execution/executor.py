"""
Trade execution abstraction.

This module provides a broker-agnostic interface for trade execution.
Strategy code calls this interface, not broker APIs directly.

This isolation ensures:
- Broker failures don't cascade into model logic
- Trade execution can be mocked for testing
- Clean separation between decision and execution
"""

import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Protocol, runtime_checkable

from config.constants import CONTRACT_TYPES
from config.settings import Settings
from execution.signals import TradeSignal

logger = logging.getLogger(__name__)


@dataclass
class TradeResult:
    """
    Result of a trade execution attempt.

    Attributes:
        success: Whether the trade was executed successfully
        contract_id: Broker-assigned contract ID (if successful)
        entry_price: Price at which trade was entered
        error: Error message (if failed)
        timestamp: When the execution occurred
    """

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
    """
    Protocol for trade execution.

    Strategy code depends on this interface, not broker-specific implementations.
    This enables:
    - Testing with mock executors
    - Broker swapping without strategy changes
    - Shadow/paper trading modes

    Example:
        >>> async def execute_signals(executor: TradeExecutor, signals: List[TradeSignal]):
        ...     for signal in signals:
        ...         result = await executor.execute(signal)
        ...         if not result.success:
        ...             logger.error(f"Trade failed: {result.error}")
    """

    async def execute(self, signal: TradeSignal) -> TradeResult:
        """
        Execute a single trade signal.

        Args:
            signal: Trade signal with contract type, direction, probability

        Returns:
            TradeResult with success status and details
        """
        ...


class DerivTradeExecutor:
    """
    Deriv-specific trade executor.

    Translates TradeSignal objects into Deriv API calls.
    This is the ONLY place that knows Deriv trade execution details.

    Attributes:
        client: Connected DerivClient instance
        settings: Trading configuration

    Example:
        >>> client = DerivClient(settings)
        >>> await client.connect()
        >>> executor = DerivTradeExecutor(client, settings)
        >>> result = await executor.execute(signal)
    """

    def __init__(self, client, settings: Settings, position_sizer=None):
        """
        Initialize Deriv executor.

        Args:
            client: Connected DerivClient instance
            settings: Trading configuration with stake amounts
            position_sizer: Optional PositionSizer instance. If None, uses FixedStakeSizer 
                            with settings.trading.stake_amount.
        """
        from execution.position_sizer import FixedStakeSizer  # Local import to avoid circular dep

        self.client = client
        self.settings = settings
        self.position_sizer = position_sizer or FixedStakeSizer(stake=settings.trading.stake_amount)
        self._executed_count = 0
        self._failed_count = 0
        
        name = self.position_sizer.__class__.__name__
        logger.info(f"DerivTradeExecutor initialized with sizer: {name}")

    async def execute(self, signal: TradeSignal) -> TradeResult:
        """
        Execute trade on Deriv platform.

        Translates signal to Deriv-specific contract parameters and executes.

        Args:
            signal: Trade signal from decision engine

        Returns:
            TradeResult with execution outcome
        """
        try:
            # 1. Fetch current account balance for dynamic sizing
            try:
                balance = await self.client.get_balance()
            except Exception as e:
                logger.warning(f"Failed to fetch balance for sizing: {e}. Using static/fallback sizing.")
                balance = None

            # 2. Determine stake size using the injected position sizer
            # pass context (drawdown, balance, etc) if available in the future
            amount = self.position_sizer.suggest_stake_for_signal(
                signal, 
                account_balance=balance
            )
            
            # 3. Safety check: ensure amount > 0
            if amount <= 0:
                logger.warning(f"Position sizer returned ${amount:.2f} (<=0). Skipping trade.")
                return TradeResult(success=False, error="Zero stake amount")

            if signal.contract_type == CONTRACT_TYPES.RISE_FALL:
                duration = 1
                duration_unit = "m"
                contract = "CALL" if signal.direction == "CALL" else "PUT"
            elif signal.contract_type == CONTRACT_TYPES.TOUCH_NO_TOUCH:
                duration = 5
                duration_unit = "m"
                contract = "ONETOUCH" if signal.direction == "TOUCH" else "NOTOUCH"
            elif signal.contract_type == CONTRACT_TYPES.STAYS_BETWEEN:
                duration = 5
                duration_unit = "m"
                contract = "RANGE" if signal.direction == "IN" else "UPORDOWN"
            else:
                return TradeResult(
                    success=False, error=f"Unsupported contract type: {signal.contract_type}"
                )

            logger.info(
                f"Executing {contract} trade: amount=${amount:.2f}, duration={duration}{duration_unit}"
            )

            # D04: Idempotency Check
            # Before executing, check if we verify a similar trade exists to prevent double execution on retries
            try:
                open_contracts = await self.client.get_open_contracts()
                signal_ts = signal.timestamp.timestamp() if signal.timestamp else datetime.now(timezone.utc).timestamp()
                
                for c in open_contracts:
                    # Check matching parameters
                    if (c.get("underlying") == self.client.symbol and
                        c.get("contract_type") == contract and
                        not c.get("is_expired") and 
                        not c.get("is_sold")):
                        
                        # Check timing (purchase time close to signal time)
                        # We assume execution happens within 90s of signal generation
                        purchase_time = c.get("purchase_time")
                        if purchase_time and abs(purchase_time - signal_ts) < 90:
                            logger.warning(f"Idempotency check: Found existing contract {c.get('contract_id')} matching signal.")
                            return TradeResult(
                                success=True,
                                contract_id=str(c.get("contract_id")),
                                entry_price=float(c.get("buy_price", amount)),
                                timestamp=datetime.fromtimestamp(purchase_time, tz=timezone.utc)
                            )
            except Exception as e:
                logger.warning(f"Idempotency check failed (proceeding with execution): {e}")

            # Extract barrier from signal if present (required for Touch/No Touch and Range)
            barrier = signal.metadata.get("barrier")

            # Execute via Deriv API
            result = await self.client.buy(contract, amount, duration, duration_unit, barrier=barrier)

            self._executed_count += 1

            # Extract contract details from response
            # NOTE: client.buy() returns result["buy"] directly, so access fields directly
            contract_id = result.get("contract_id")
            entry_price = result.get("buy_price")

            logger.info(f"Trade executed successfully: contract_id={contract_id}")

            return TradeResult(
                success=True,
                contract_id=str(contract_id) if contract_id else None,
                entry_price=float(entry_price) if entry_price else None,
            )

        except Exception as e:
            self._failed_count += 1
            error_msg = str(e)
            logger.error(f"Trade execution failed: {error_msg}")

            return TradeResult(success=False, error=error_msg)

    def get_statistics(self) -> dict:
        """Get execution statistics."""
        total = self._executed_count + self._failed_count
        return {
            "executed": self._executed_count,
            "failed": self._failed_count,
            "success_rate": self._executed_count / total if total > 0 else 0.0,
        }


class MockTradeExecutor:
    """
    Mock executor for testing and paper trading.

    Records trade signals without actual execution.
    Useful for backtesting and shadow trading validation.
    """

    def __init__(self):
        self.executed_signals: list = []
        logger.info("MockTradeExecutor initialized (paper trading mode)")

    async def execute(self, signal: TradeSignal) -> TradeResult:
        """Record signal without executing."""
        self.executed_signals.append(signal)

        logger.info(f"[MOCK] Would execute: {signal}")

        return TradeResult(
            success=True,
            contract_id=f"MOCK_{len(self.executed_signals)}",
            entry_price=100.0,  # Placeholder
        )

    def get_signals(self) -> list:
        """Get all recorded signals."""
        return self.executed_signals.copy()
