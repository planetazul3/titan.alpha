"""
Idempotency Store - Persistent tracking of executed signals.

CRITICAL-002: Prevents double execution of trades by tracking 
deterministic signal IDs.
"""

import sqlite3
import logging
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

class SQLiteIdempotencyStore:
    """
    SQLite-backed store for tracking executed signal IDs.
    """
    
    def __init__(self, db_path: Path):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()
        
    def _init_db(self):
        """Initialize database schema."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS executed_signals (
                    signal_id TEXT PRIMARY KEY,
                    executed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    contract_id TEXT,
                    symbol TEXT
                )
            """)
            # Create index for cleanup
            conn.execute("CREATE INDEX IF NOT EXISTS idx_executed_at ON executed_signals(executed_at)")
            
    def exists(self, signal_id: str) -> bool:
        """Check if a signal ID has already been executed."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute(
                "SELECT 1 FROM executed_signals WHERE signal_id = ?", 
                (signal_id,)
            )
            return cursor.fetchone() is not None
            
    def get_contract_id(self, signal_id: str) -> Optional[str]:
        """Get the contract ID associated with a previously executed signal."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute(
                "SELECT contract_id FROM executed_signals WHERE signal_id = ?", 
                (signal_id,)
            )
            row = cursor.fetchone()
            return row[0] if row else None

            return row[0] if row else None

    # CRITICAL-002: Atomic checks
    def check_and_reserve(self, signal_id: str, symbol: str) -> tuple[bool, Optional[str]]:
        """
        Atomically check if signal exists and reserve if not.
        Returns: (is_new: bool, existing_contract_id: str|None)
        """
        with sqlite3.connect(self.db_path) as conn:
            # Try to reserve
            try:
                # Use INSERT OR IGNORE to atomically reserve
                # We use 'PENDING' as temporary contract_id
                cursor = conn.execute(
                    "INSERT INTO executed_signals (signal_id, contract_id, symbol) VALUES (?, 'PENDING', ?)",
                    (signal_id, symbol)
                )
                # If we get here without integrity error, we reserved it
                logger.debug(f"Reserved execution for signal {signal_id}")
                return True, None
            except sqlite3.IntegrityError:
                # Already exists, fetch current state
                cursor = conn.execute("SELECT contract_id FROM executed_signals WHERE signal_id = ?", (signal_id,))
                row = cursor.fetchone()
                existing_id = row[0] if row else None
                logger.debug(f"Signal {signal_id} already exists (contract: {existing_id})")
                return False, existing_id

    def update_contract_id(self, signal_id: str, contract_id: str):
        """Update a reserved execution with actual contract ID."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                "UPDATE executed_signals SET contract_id = ? WHERE signal_id = ?",
                (contract_id, signal_id)
            )
        logger.debug(f"Updated reservation for {signal_id} with contract {contract_id}")

    def delete_record(self, signal_id: str):
        """Remove a record (e.g. after failed execution of reserved signal)."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("DELETE FROM executed_signals WHERE signal_id = ?", (signal_id,))
        logger.debug(f"Deleted record for signal {signal_id}")

    def record_execution(self, signal_id: str, contract_id: str, symbol: str):
        """Record a successful execution. Updated to be an upsert for safety."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                "INSERT OR REPLACE INTO executed_signals (signal_id, contract_id, symbol) VALUES (?, ?, ?)",
                (signal_id, contract_id, symbol)
            )
        logger.debug(f"Recorded execution for signal {signal_id} (contract: {contract_id})")

    def cleanup(self, days: int = 7):
        """Clean up old records to prevent database bloat."""
        cutoff = (datetime.now(timezone.utc) - timedelta(days=days)).isoformat()
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("DELETE FROM executed_signals WHERE executed_at < ?", (cutoff,))
            logger.info(f"Cleaned up {cursor.rowcount} old idempotency records")

    # Async wrappers for use in async context (executor.py)
    async def get_contract_id_async(self, signal_id: str) -> Optional[str]:
        """Async wrapper for get_contract_id."""
        import asyncio
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, self.get_contract_id, signal_id)

    async def record_execution_async(self, signal_id: str, contract_id: str, symbol: str = "unknown"):
        """Async wrapper for record_execution."""
        import asyncio
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(None, lambda: self.record_execution(signal_id, contract_id, symbol))

    async def check_and_reserve_async(self, signal_id: str, symbol: str) -> tuple[bool, Optional[str]]:
        """Async wrapper for check_and_reserve."""
        import asyncio
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, self.check_and_reserve, signal_id, symbol)

    async def update_contract_id_async(self, signal_id: str, contract_id: str):
        """Async wrapper for update_contract_id."""
        import asyncio
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(None, self.update_contract_id, signal_id, contract_id)

    async def delete_record_async(self, signal_id: str):
        """Async wrapper for delete_record."""
        import asyncio
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(None, self.delete_record, signal_id)

    async def close(self):
        """Close resources (no-op for SQLite as we open per query)."""
        pass
