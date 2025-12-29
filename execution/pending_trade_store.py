"""
Pending Trade Persistence Store.

SQLite-backed storage for pending (in-flight) real trades.
Enables recovery after restart by persisting contract IDs and trade metadata.

Zombie Trade Prevention (TRACKER-STORE-AUDIT):
- retry_count column tracks failed re-subscription attempts
- Trades exceeding MAX_RECOVERY_RETRIES are marked FAILED_TO_TRACK
- get_zombie_trades() returns trades requiring operator attention

Usage:
    >>> store = PendingTradeStore(Path("data_cache/pending_trades.db"))
    >>> store.add_trade(contract_id="12345", direction="CALL", ...)
    >>> pending = store.get_all_pending()  # Load on startup
    >>> store.remove_trade("12345")  # After settlement
"""

import logging
import sqlite3
import threading
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, cast

logger = logging.getLogger(__name__)

# TRACKER-STORE-AUDIT: Max retries before marking trade as zombie
MAX_RECOVERY_RETRIES = 5


class PendingTradeStore:
    """
    SQLite-backed store for pending trades awaiting settlement.
    
    Provides crash recovery for RealTradeTracker by persisting:
    - contract_id: Unique identifier for re-subscription
    - direction: CALL/PUT for outcome calculation
    - entry_price: Price at trade entry
    - stake: Stake amount for P&L calculation
    - probability: Model confidence at trade time
    - executed_at: Timestamp of trade execution
    - retry_count: Number of failed re-subscription attempts (zombie prevention)
    
    Thread-safe with WAL mode for concurrent access.
    """

    CREATE_TABLE_SQL = """
    CREATE TABLE IF NOT EXISTS pending_trades (
        contract_id TEXT PRIMARY KEY,
        direction TEXT NOT NULL,
        entry_price REAL NOT NULL,
        stake REAL NOT NULL,
        probability REAL NOT NULL,
        executed_at TEXT NOT NULL,
        contract_type TEXT DEFAULT 'RISE_FALL',
        status TEXT DEFAULT 'CONFIRMED',
        retry_count INTEGER DEFAULT 0,
        metadata TEXT
    )
    """

    def __init__(self, db_path: Path):
        """
        Initialize pending trade store.
        
        Args:
            db_path: Path to SQLite database file
        """
        self._db_path = Path(db_path)
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        
        self._local = threading.local()
        self._write_lock = threading.Lock()
        
        self._init_schema()
        
        logger.info(f"PendingTradeStore initialized: {db_path}")

    def _get_connection(self) -> sqlite3.Connection:
        """Get thread-local database connection."""
        if not hasattr(self._local, "conn") or self._local.conn is None:
            conn = sqlite3.connect(str(self._db_path), check_same_thread=False, timeout=30.0)
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA synchronous=NORMAL")
            conn.row_factory = sqlite3.Row
            self._local.conn = conn
        return cast(sqlite3.Connection, self._local.conn)

    @contextmanager
    def _transaction(self):
        """Context manager for atomic transactions."""
        conn = self._get_connection()
        with self._write_lock:
            try:
                yield conn
                conn.commit()
            except Exception:
                conn.rollback()
                raise

    def _init_schema(self) -> None:
        """Initialize database schema with migrations."""
        with self._transaction() as conn:
            conn.execute(self.CREATE_TABLE_SQL)
            
            # Migration: Add status column if missing
            try:
                conn.execute("ALTER TABLE pending_trades ADD COLUMN status TEXT DEFAULT 'CONFIRMED'")
                logger.info("Migrated pending_trades: added status column")
            except sqlite3.OperationalError:
                pass  # Column already exists
            
            # TRACKER-STORE-AUDIT: Migration for retry_count column (zombie prevention)
            try:
                conn.execute("ALTER TABLE pending_trades ADD COLUMN retry_count INTEGER DEFAULT 0")
                logger.info("Migrated pending_trades: added retry_count column")
            except sqlite3.OperationalError:
                pass  # Column already exists

    def prepare_trade_intent(
        self,
        intent_id: str,
        direction: str,
        entry_price: float,
        stake: float,
        probability: float,
        contract_type: str = "RISE_FALL",
    ) -> None:
        """
        Record trade intent BEFORE execution (crash safety).
        
        If the app crashes after this but before confirm_trade,
        we know a trade was attempted and can investigate.
        
        Args:
            intent_id: Temporary ID (use signal hash or UUID)
            direction: CALL or PUT
            entry_price: Current price at trade time
            stake: Stake amount
            probability: Model probability
            contract_type: Type of contract
        """
        executed_at = datetime.now(timezone.utc).isoformat()
        
        with self._transaction() as conn:
            conn.execute(
                """
                INSERT INTO pending_trades 
                (contract_id, direction, entry_price, stake, probability, executed_at, contract_type, status)
                VALUES (?, ?, ?, ?, ?, ?, ?, 'ATTEMPTING')
                """,
                (intent_id, direction, entry_price, stake, probability, executed_at, contract_type),
            )
        
        logger.info(f"Recorded trade intent: {intent_id[:12]}... (status=ATTEMPTING)")

    def confirm_trade(self, intent_id: str, contract_id: str) -> bool:
        """
        Confirm trade execution with actual contract ID.
        
        Updates the ATTEMPTING record with the real contract_id.
        
        Args:
            intent_id: Original intent ID
            contract_id: Actual contract ID from broker
            
        Returns:
            True if intent was found and updated
        """
        with self._transaction() as conn:
            # Update intent_id to actual contract_id and set CONFIRMED
            cursor = conn.execute(
                """
                UPDATE pending_trades 
                SET contract_id = ?, status = 'CONFIRMED'
                WHERE contract_id = ? AND status = 'ATTEMPTING'
                """,
                (contract_id, intent_id),
            )
            
            if cursor.rowcount > 0:
                logger.info(f"Confirmed trade: {intent_id[:12]}... -> {contract_id}")
                return True
            else:
                logger.warning(f"Trade intent not found for confirmation: {intent_id}")
                return False

    def get_attempting_trades(self) -> list[dict]:
        """Get trades that were started but never confirmed (potential phantoms)."""
        import json
        
        conn = self._get_connection()
        cursor = conn.execute("SELECT * FROM pending_trades WHERE status = 'ATTEMPTING'")
        
        trades = []
        for row in cursor:
            trades.append({
                "intent_id": row["contract_id"],
                "direction": row["direction"],
                "entry_price": row["entry_price"],
                "stake": row["stake"],
                "executed_at": row["executed_at"],
            })
        return trades

    def add_trade(
        self,
        contract_id: str,
        direction: str,
        entry_price: float,
        stake: float,
        probability: float,
        contract_type: str = "RISE_FALL",
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """
        Add a pending trade to the store.
        
        Args:
            contract_id: Deriv contract ID
            direction: CALL or PUT
            entry_price: Price at trade entry
            stake: Stake amount
            probability: Model probability
            contract_type: Type of contract
            metadata: Optional additional metadata
        """
        import json
        
        executed_at = datetime.now(timezone.utc).isoformat()
        
        with self._transaction() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO pending_trades 
                (contract_id, direction, entry_price, stake, probability, executed_at, contract_type, metadata)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    contract_id,
                    direction,
                    entry_price,
                    stake,
                    probability,
                    executed_at,
                    contract_type,
                    json.dumps(metadata) if metadata else None,
                ),
            )
        
        logger.debug(f"Added pending trade: {contract_id}")

    def remove_trade(self, contract_id: str) -> bool:
        """
        Remove a trade after settlement.
        
        Args:
            contract_id: Contract ID to remove
            
        Returns:
            True if trade was found and removed
        """
        with self._transaction() as conn:
            cursor = conn.execute(
                "DELETE FROM pending_trades WHERE contract_id = ?",
                (contract_id,),
            )
            removed = cursor.rowcount > 0
        
        if removed:
            logger.debug(f"Removed pending trade: {contract_id}")
        return cast(bool, removed)
    
    def increment_retry_count(self, contract_id: str) -> int:
        """
        Increment retry count for a trade and return new count.
        
        TRACKER-STORE-AUDIT: Tracks re-subscription attempts for zombie prevention.
        
        Args:
            contract_id: Contract ID to update
            
        Returns:
            New retry count (or -1 if trade not found)
        """
        with self._transaction() as conn:
            cursor = conn.execute(
                "UPDATE pending_trades SET retry_count = retry_count + 1 WHERE contract_id = ?",
                (contract_id,),
            )
            if cursor.rowcount == 0:
                return -1
            
            # Get new count
            result = conn.execute(
                "SELECT retry_count FROM pending_trades WHERE contract_id = ?",
                (contract_id,),
            ).fetchone()
            return cast(int, result[0]) if result else -1
    
    def mark_failed_to_track(self, contract_id: str) -> bool:
        """
        Mark a trade as FAILED_TO_TRACK (zombie - requires operator attention).
        
        TRACKER-STORE-AUDIT: Called when MAX_RECOVERY_RETRIES exceeded.
        
        Args:
            contract_id: Contract ID to mark
            
        Returns:
            True if trade was found and updated
        """
        with self._transaction() as conn:
            cursor = conn.execute(
                "UPDATE pending_trades SET status = 'FAILED_TO_TRACK' WHERE contract_id = ?",
                (contract_id,),
            )
            updated = cursor.rowcount > 0
        
        if updated:
            logger.error(f"⚠️ Trade marked as FAILED_TO_TRACK (zombie): {contract_id}")
        return cast(bool, updated)
    
    def get_zombie_trades(self) -> list[dict[str, Any]]:
        """
        Get trades that failed to track (zombies requiring operator attention).
        
        TRACKER-STORE-AUDIT: Returns trades with status=FAILED_TO_TRACK.
        
        Returns:
            List of zombie trade dictionaries
        """
        import json
        
        conn = self._get_connection()
        cursor = conn.execute("SELECT * FROM pending_trades WHERE status = 'FAILED_TO_TRACK'")
        
        trades = []
        for row in cursor:
            trades.append({
                "contract_id": row["contract_id"],
                "direction": row["direction"],
                "entry_price": row["entry_price"],
                "stake": row["stake"],
                "retry_count": row["retry_count"] if "retry_count" in row.keys() else 0,
                "executed_at": row["executed_at"],
            })
        return trades

    def get_all_pending(self) -> list[dict[str, Any]]:
        """
        Get all pending trades for recovery.
        
        Returns:
            List of pending trade dictionaries
        """
        import json
        
        conn = self._get_connection()
        cursor = conn.execute("SELECT * FROM pending_trades")
        
        trades = []
        for row in cursor:
            trade = {
                "contract_id": row["contract_id"],
                "direction": row["direction"],
                "entry_price": row["entry_price"],
                "stake": row["stake"],
                "probability": row["probability"],
                "executed_at": datetime.fromisoformat(row["executed_at"]),
                "contract_type": row["contract_type"],
                "metadata": json.loads(row["metadata"]) if row["metadata"] else {},
            }
            trades.append(trade)
        
        return trades

    def get_count(self) -> int:
        """Get number of pending trades."""
        conn = self._get_connection()
        return cast(int, conn.execute("SELECT COUNT(*) FROM pending_trades").fetchone()[0])

    def close(self) -> None:
        """Close database connection."""
        if hasattr(self._local, "conn") and self._local.conn:
            self._local.conn.close()
            self._local.conn = None
