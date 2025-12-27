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

    def record_execution(self, signal_id: str, contract_id: str, symbol: str):
        """Record a successful execution."""
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
