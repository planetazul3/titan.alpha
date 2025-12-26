"""
Real Trade Outcome Tracking via Deriv API.

This module provides API-based outcome tracking for executed trades.
Uses Deriv's proposal_open_contract subscription to get real-time
updates and actual P&L when contracts settle.

Usage:
    >>> tracker = RealTradeTracker(client, sizer, executor)
    >>> await tracker.register_trade(signal, contract_id, entry_price, stake)
    >>> # Tracker automatically subscribes and updates sizer/executor on settlement
"""

import asyncio
import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path
from typing import Any

from execution.pending_trade_store import PendingTradeStore

logger = logging.getLogger(__name__)


@dataclass
class PendingTrade:
    """A trade awaiting outcome resolution."""
    contract_id: str
    direction: str  # "CALL" or "PUT"
    entry_price: float
    stake: float
    probability: float
    executed_at: datetime
    contract_type: str = "RISE_FALL"


class RealTradeTracker:
    """
    Tracks real trades using Deriv API for outcome resolution.
    
    Uses proposal_open_contract subscription to receive real-time
    updates and actual P&L directly from Deriv.
    
    On settlement:
    1. Updates position sizer (for compounding)
    2. Updates executor P&L (for daily limits)
    
    Attributes:
        client: DerivClient for API subscriptions
        sizer: Position sizer with record_outcome() method
        executor: SafeTradeExecutor with update_pnl() method
    """
    
    def __init__(
        self,
        client: Any = None,
        sizer: Any = None,
        executor: Any = None,

        model_monitor: Any = None,
        persistence_path: Path | None = None,
    ):
        """
        Initialize trade tracker.
        
        Args:
            client: DerivClient instance for API calls
            sizer: Position sizer (CompoundingPositionSizer, etc.)
            executor: SafeTradeExecutor for P&L updates
            persistence_path: Path for SQLite persistence (enables crash recovery)
        """
        self.client = client
        self.sizer = sizer
        self.executor = executor
        self.model_monitor = model_monitor
        
        # Persistence for crash recovery
        if persistence_path:
            self._store = PendingTradeStore(persistence_path)
        else:
            self._store = PendingTradeStore(Path("data_cache/pending_trades.db"))
        
        self._pending_trades: dict[str, PendingTrade] = {}
        self._active_tasks: set[asyncio.Task] = set()  # Track background tasks
        self._resolved_count = 0
        self._wins = 0
        self._losses = 0
        self._total_pnl = Decimal("0.0")
        
        logger.info(
            f"RealTradeTracker initialized (API mode): "
            f"sizer={type(sizer).__name__ if sizer else 'None'}, "
            f"persistence={self._store._db_path}"
        )
    
    async def register_trade(
        self,
        contract_id: str,
        direction: str,
        entry_price: float,
        stake: float,
        probability: float,
        contract_type: str = "RISE_FALL",
    ) -> None:
        """
        Register a newly executed trade for outcome tracking.
        
        Subscribes to Deriv API for real-time contract updates.
        
        Args:
            contract_id: Deriv contract ID
            direction: "CALL" or "PUT"
            entry_price: Price at trade entry
            stake: Stake amount
            probability: Model's predicted probability
            contract_type: Type of contract
        """
        trade = PendingTrade(
            contract_id=contract_id,
            direction=direction,
            entry_price=entry_price,
            stake=stake,
            probability=probability,
            executed_at=datetime.now(timezone.utc),
            contract_type=contract_type,
        )
        
        self._pending_trades[contract_id] = trade
        
        # Persist for crash recovery
        self._store.add_trade(
            contract_id=contract_id,
            direction=direction,
            entry_price=entry_price,
            stake=stake,
            probability=probability,
            contract_type=contract_type,
        )
        
        logger.info(
            f"[TRACKER] Registered trade {contract_id}: "
            f"{direction} @ {entry_price:.2f}, stake=${stake:.2f} (persisted)"
        )
        
        # Subscribe to contract outcome via API with task tracking
        if self.client:
            task = asyncio.create_task(self._watch_contract(contract_id, trade))
            self._active_tasks.add(task)
            task.add_done_callback(self._active_tasks.discard)
        else:
            logger.warning("[TRACKER] No client - cannot subscribe to contract")
    
    async def _watch_contract(self, contract_id: str, trade: PendingTrade) -> None:
        """Subscribe to contract updates and handle settlement with retry."""
        max_retries = 3
        
        def on_settled(profit: float, won: bool):
            """Called when contract settles."""
            self._handle_outcome(contract_id, profit, won)
        
        for attempt in range(max_retries):
            try:
                result = await self.client.subscribe_contract(
                    contract_id=contract_id,
                    on_update=None,  # Don't need intermediate updates
                    on_settled=on_settled,
                )
                
                if not result:
                     raise TimeoutError("Subscription timed out (180s)")
                     
                return  # Success - exit retry loop
            except Exception as e:
                if attempt < max_retries - 1:
                    wait_time = min(2 ** attempt, 10)  # Exponential backoff: 1, 2, 4s max 10s
                    logger.warning(
                        f"[TRACKER] Retry {attempt + 1}/{max_retries} for contract {contract_id}: {e}. "
                        f"Waiting {wait_time}s..."
                    )
                    await asyncio.sleep(wait_time)
                else:
                    logger.error(f"[TRACKER] Failed to watch contract {contract_id} after {max_retries} retries: {e}")
                    # Remove from pending after all retries exhausted
                    self._pending_trades.pop(contract_id, None)
    
    def _handle_outcome(self, contract_id: str, profit: float, won: bool) -> None:
        """Process trade outcome - update sizer and P&L."""
        
        # Remove from pending (memory and persistent store)
        trade = self._pending_trades.pop(contract_id, None)
        self._store.remove_trade(contract_id)  # Remove from SQLite
        
        if not trade:
            logger.warning(f"[TRACKER] Trade {contract_id} not found in pending")
            return
        
        # Update stats
        self._resolved_count += 1
        # M16: Use Decimal for precision
        self._total_pnl += Decimal(str(profit))
        if won:
            self._wins += 1
        else:
            self._losses += 1
        
        logger.info(
            f"[TRACKER] Trade {contract_id} resolved: "
            f"{'WIN' if won else 'LOSS'}, P&L=${profit:+.2f}, "
            f"Total: {self._wins}W/{self._losses}L"
        )
        
        # Update Model Health Monitor (Calibration/Drift tracking)
        if self.model_monitor and trade.probability:
            try:
                # Actual outcome: 1.0 (Win) or 0.0 (Loss)
                actual = 1.0 if won else 0.0
                self.model_monitor.record_prediction(trade.probability, actual)
                logger.debug(f"[TRACKER] Model monitor updated: prob={trade.probability:.3f} -> {actual}")
            except Exception as e:
                logger.error(f"[TRACKER] Failed to update model monitor: {e}")
        
        # Update position sizer (for compounding)
        if self.sizer and hasattr(self.sizer, 'record_outcome'):
            try:
                self.sizer.record_outcome(pnl=profit, won=won)
                logger.debug(f"[TRACKER] Sizer updated: won={won}")
            except Exception as e:
                logger.error(f"[TRACKER] Failed to update sizer: {e}")
        
        # Update P&L tracking (for daily limits) - dispatch as background task
        # update_pnl is async, but we're in a sync callback, so use create_task
        if self.executor and hasattr(self.executor, 'update_pnl'):
            try:
                # Non-blocking dispatch to prevent event loop jitter
                asyncio.create_task(self.executor.update_pnl(profit))
                logger.debug(f"[TRACKER] P&L update dispatched: ${profit:+.2f}")
            except Exception as e:
                logger.error(f"[TRACKER] Failed to dispatch P&L update: {e}")
    
    def get_pending_count(self) -> int:
        """Get number of pending trades awaiting resolution."""
        return len(self._pending_trades)
    
    def get_statistics(self) -> dict[str, Any]:
        """Get tracker statistics."""
        return {
            "pending_trades": len(self._pending_trades),
            "persisted_trades": self._store.get_count(),
            "resolved_trades": self._resolved_count,
            "wins": self._wins,
            "losses": self._losses,
            "win_rate": self._wins / self._resolved_count if self._resolved_count > 0 else 0,
            "total_pnl": float(self._total_pnl),
        }

    async def recover_pending_trades(self) -> int:
        """
        Recover pending trades from persistent storage after restart.
        
        Loads all persisted trades and re-subscribes to their outcomes.
        Should be called once during initialization when client is ready.
        
        Returns:
            Number of trades recovered
        """
        if not self.client:
            logger.warning("[TRACKER] No client available for recovery")
            return 0
        
        pending = self._store.get_all_pending()
        if not pending:
            logger.info("[TRACKER] No pending trades to recover")
            return 0
        
        logger.info(f"[TRACKER] Recovering {len(pending)} pending trades from storage...")
        
        recovered = 0
        for trade_data in pending:
            try:
                # Reconstruct PendingTrade object
                trade = PendingTrade(
                    contract_id=trade_data["contract_id"],
                    direction=trade_data["direction"],
                    entry_price=trade_data["entry_price"],
                    stake=trade_data["stake"],
                    probability=trade_data["probability"],
                    executed_at=trade_data["executed_at"],
                    contract_type=trade_data["contract_type"],
                )
                
                # Add to in-memory tracking
                self._pending_trades[trade.contract_id] = trade
                
                # Re-subscribe to contract outcome
                task = asyncio.create_task(self._watch_contract(trade.contract_id, trade))
                self._active_tasks.add(task)
                task.add_done_callback(self._active_tasks.discard)
                
                recovered += 1
                logger.info(f"[TRACKER] Recovered trade {trade.contract_id}: {trade.direction}")
                
            except Exception as e:
                logger.error(f"[TRACKER] Failed to recover trade: {e}")
                # Remove invalid trade from store
                self._store.remove_trade(trade_data["contract_id"])
        
        logger.info(f"[TRACKER] Recovery complete: {recovered} trades re-subscribed")
        return recovered
