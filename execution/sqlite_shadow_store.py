"""
SQLite-backed Shadow Trade Store.

This module provides a transactional, SQLite-backed storage for shadow trades.
Replaces the NDJSON update-by-rewrite approach with atomic operations.

Key improvements over NDJSON:
- WAL mode for concurrent read/write access
- Atomic outcome updates (no file rewrites)
- Indexed queries for efficient time-range lookups
- Automatic schema migrations

Usage:
    >>> store = SQLiteShadowStore(Path("data_cache/shadow_trades.db"))
    >>> store.append(ShadowTradeRecord(...))
    >>> store.update_outcome("trade-id", outcome=True, exit_price=1.05)
    >>> trades = store.query(start=yesterday, end=today, resolved_only=True)
"""

import json
import logging
import sqlite3
import threading
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, cast

from execution.shadow_store import SHADOW_STORE_SCHEMA_VERSION, ShadowTradeRecord

logger = logging.getLogger(__name__)

# SQLite database schema version
SQLITE_SCHEMA_VERSION = 2


class SQLiteShadowStore:
    """
    SQLite-backed shadow trade store with atomic operations.

    Design principles:
    - WAL MODE: Enables concurrent reads during writes
    - ATOMIC UPDATES: Outcome resolution is a single UPDATE, not file rewrite
    - INDEXED: time-based queries are fast via timestamp index
    - THREAD-SAFE: Uses per-thread connections with locking

    Migration from NDJSON:
    - Use SQLiteShadowStore.from_ndjson() to import existing data
    - Maintains full API compatibility with ShadowTradeStore
    """

    # SQL statements for schema creation
    CREATE_TABLE_SQL = """
    CREATE TABLE IF NOT EXISTS shadow_trades (
        trade_id TEXT PRIMARY KEY,
        timestamp TEXT NOT NULL,
        contract_type TEXT NOT NULL,
        direction TEXT NOT NULL,
        probability REAL NOT NULL,
        entry_price REAL NOT NULL,
        reconstruction_error REAL NOT NULL,
        regime_state TEXT NOT NULL,
        model_version TEXT DEFAULT 'unknown',
        feature_schema_version TEXT DEFAULT '1.0',
        tick_window TEXT,
        candle_window TEXT,
        outcome INTEGER,
        exit_price REAL,
        resolved_at TEXT,
        metadata TEXT,
        schema_version TEXT NOT NULL,
        created_at TEXT NOT NULL,
        barrier_level REAL,
        barrier2_level REAL
    )
    """

    CREATE_INDEXES_SQL = [
        "CREATE INDEX IF NOT EXISTS idx_timestamp ON shadow_trades(timestamp)",
        "CREATE INDEX IF NOT EXISTS idx_outcome ON shadow_trades(outcome)",
        "CREATE INDEX IF NOT EXISTS idx_regime_state ON shadow_trades(regime_state)",
    ]

    SCHEMA_VERSION_TABLE_SQL = """
    CREATE TABLE IF NOT EXISTS schema_meta (
        key TEXT PRIMARY KEY,
        value TEXT
    )
    """

    def __init__(self, db_path: Path):
        """
        Initialize SQLite shadow trade store.

        Args:
            db_path: Path to SQLite database file
        """
        self._db_path = Path(db_path)
        self._db_path.parent.mkdir(parents=True, exist_ok=True)

        # Thread-local storage for connections
        self._local = threading.local()
        self._write_lock = threading.Lock()

        # Initialize schema
        self._init_schema()

        logger.info(
            f"SQLiteShadowStore initialized: {db_path} "
            f"(schema v{SHADOW_STORE_SCHEMA_VERSION}, db v{SQLITE_SCHEMA_VERSION})"
        )

    @property
    def _store_path(self) -> Path:
        """Return database path for API compatibility with ShadowTradeStore."""
        return self._db_path

    def _get_connection(self) -> sqlite3.Connection:
        """Get thread-local database connection."""
        if not hasattr(self._local, "conn") or self._local.conn is None:
            conn = sqlite3.connect(str(self._db_path), check_same_thread=False, timeout=30.0)
            # Enable WAL mode for better concurrency
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA synchronous=NORMAL")
            conn.execute("PRAGMA foreign_keys=ON")
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
        """Initialize database schema."""
        with self._transaction() as conn:
            conn.execute(self.CREATE_TABLE_SQL)
            for idx_sql in self.CREATE_INDEXES_SQL:
                conn.execute(idx_sql)
            conn.execute(self.SCHEMA_VERSION_TABLE_SQL)

            # Store schema version
            conn.execute(
                "INSERT OR REPLACE INTO schema_meta (key, value) VALUES (?, ?)",
                ("sqlite_schema_version", str(SQLITE_SCHEMA_VERSION)),
            )
            conn.execute(
                "INSERT OR REPLACE INTO schema_meta (key, value) VALUES (?, ?)",
                ("record_schema_version", SHADOW_STORE_SCHEMA_VERSION),
            )

    def append(self, record: ShadowTradeRecord) -> None:
        """
        Append a shadow trade record to the store.

        Args:
            record: Shadow trade record to store
        """
        with self._transaction() as conn:
            conn.execute(
                """
                INSERT INTO shadow_trades (
                    trade_id, timestamp, contract_type, direction,
                    probability, entry_price, reconstruction_error, regime_state,
                    model_version, feature_schema_version,
                    tick_window, candle_window,
                    outcome, exit_price, resolved_at,
                    metadata, schema_version, created_at,
                    barrier_level, barrier2_level
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    record.trade_id,
                    record.timestamp.isoformat(),
                    record.contract_type,
                    record.direction,
                    record.probability,
                    record.entry_price,
                    record.reconstruction_error,
                    record.regime_state,
                    record.model_version,
                    record.feature_schema_version,
                    json.dumps(record.tick_window),
                    json.dumps(record.candle_window),
                    1 if record.outcome is True else (0 if record.outcome is False else None),
                    record.exit_price,
                    record.resolved_at.isoformat() if record.resolved_at else None,
                    json.dumps(record.metadata),
                    record._schema_version,
                    record._created_at,
                    record.barrier_level,
                    record.barrier2_level,
                ),
            )

        logger.debug(f"Appended shadow trade: {record.trade_id}")

    async def append_async(self, record: ShadowTradeRecord) -> None:
        """Async append."""
        import asyncio
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(None, lambda: self.append(record))

    def update_outcome(self, trade: ShadowTradeRecord, outcome: bool, exit_price: float) -> bool:
        """
        Update a trade's outcome atomically.

        This is the KEY IMPROVEMENT over NDJSON - no file rewrite needed.

        Args:
            trade: Trade record to update
            outcome: True for win, False for loss
            exit_price: Actual exit price

        Returns:
            True if trade was found and updated, False otherwise
        """
        trade_id = trade.trade_id
        resolved_at = datetime.now(timezone.utc).isoformat()

        with self._transaction() as conn:
            cursor = conn.execute(
                """
                UPDATE shadow_trades
                SET outcome = ?, exit_price = ?, resolved_at = ?
                WHERE trade_id = ?
                """,
                (1 if outcome else 0, exit_price, resolved_at, trade_id),
            )

            if cursor.rowcount > 0:
                logger.debug(f"Updated outcome for {trade_id}: outcome={outcome}")
                return True
            else:
                logger.warning(f"Trade not found for outcome update: {trade_id}")
                return False

    def mark_stale(self, trade_id: str, exit_price: float) -> bool:
        """
        Mark a trade as stale (unresolvable) without polluting training data.
        
        Sets outcome to -1 (sentinel value indicating "stale/error") and stores
        resolution_status in metadata. Training scripts should filter these out.
        
        Args:
            trade_id: Trade ID to mark stale
            exit_price: Best available price at resolution time
            
        Returns:
            True if trade was found and updated
        """
        resolved_at = datetime.now(timezone.utc).isoformat()
        stale_metadata = json.dumps({"resolution_status": "stale_error"})
        
        with self._transaction() as conn:
            # Set outcome to -1 (sentinel) - distinct from 0 (loss) and 1 (win)
            # The outcome column is INTEGER, so -1 is valid
            cursor = conn.execute(
                """
                UPDATE shadow_trades
                SET outcome = -1, exit_price = ?, resolved_at = ?, metadata = ?
                WHERE trade_id = ?
                """,
                (exit_price, resolved_at, stale_metadata, trade_id),
            )
            
            if cursor.rowcount > 0:
                logger.info(f"Marked trade {trade_id} as stale (excluded from training)")
                return True
            else:
                logger.warning(f"Trade not found for stale marking: {trade_id}")
                return False

    # H08: Async Wrappers
    async def update_outcome_async(self, trade: ShadowTradeRecord, outcome: bool, exit_price: float) -> bool:
        """Async update outcome."""
        import asyncio
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, lambda: self.update_outcome(trade, outcome, exit_price))

    async def mark_stale_async(self, trade_id: str, exit_price: float) -> bool:
        """Async mark stale."""
        import asyncio
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, lambda: self.mark_stale(trade_id, exit_price))
    
    async def query_async(self, **kwargs) -> list[ShadowTradeRecord]:
        """Async query."""
        import asyncio
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, lambda: self.query(**kwargs))

    def prune(self, retention_days: int = 30) -> int:
        """
        Prune records older than retention period.
        
        Args:
            retention_days: Number of days to keep records
            
        Returns:
            Number of deleted records
        """
        from datetime import timedelta
        
        cutoff_date = datetime.now(timezone.utc) - timedelta(days=retention_days)
        cutoff_iso = cutoff_date.isoformat()
        
        with self._transaction() as conn:
            # Delete old records
            cursor = conn.execute(
                "DELETE FROM shadow_trades WHERE timestamp < ?", 
                (cutoff_iso,)
            )
            deleted_count = cursor.rowcount
            
        # Reclaim space
        if deleted_count > 0:
            try:
                with self._get_connection() as conn:
                    conn.execute("VACUUM")
                logger.info(f"Pruned {deleted_count} shadow trades older than {retention_days} days")
            except Exception as e:
                logger.warning(f"Failed to VACUUM after prune: {e}")
                
        return deleted_count

    def query_iter(
        self,
        start: datetime | None = None,
        end: datetime | None = None,
        resolved_only: bool = False,
        unresolved_only: bool = False,
    ):
        """
        Iterate over shadow trades from the store.
        
        Memory-efficient alternative to query() for large result sets.
        Yields ShadowTradeRecord objects one by one.

        Args:
            start: Start of time range (inclusive)
            end: End of time range (exclusive)
            resolved_only: Only return trades with outcomes
            unresolved_only: Only return trades without outcomes

        Yields:
            ShadowTradeRecord matching criteria
        """
        conditions = []
        params = []

        if start:
            conditions.append("timestamp >= ?")
            params.append(start.isoformat())
        if end:
            conditions.append("timestamp < ?")
            params.append(end.isoformat())
        if resolved_only:
            conditions.append("outcome IS NOT NULL")
        if unresolved_only:
            conditions.append("outcome IS NULL")

        where_clause = " AND ".join(conditions) if conditions else "1=1"

        # Use a fresh connection for the iterator to avoid interfering with other ops
        # (though sqlite3.connect is cheap, we can reuse the thread-local one if we are careful)
        conn = self._get_connection()
        
        # Helper to ensure cursor is closed if iteration is interrupted
        try:
            cursor = conn.execute(
                f"SELECT * FROM shadow_trades WHERE {where_clause} ORDER BY timestamp", params
            )
            for row in cursor:
                yield self._row_to_record(row)
        except Exception:
            raise

    def query(
        self,
        start: datetime | None = None,
        end: datetime | None = None,
        resolved_only: bool = False,
        unresolved_only: bool = False,
    ) -> list[ShadowTradeRecord]:
        """
        Query shadow trades from the store.

        Args:
            start: Start of time range (inclusive)
            end: End of time range (exclusive)
            resolved_only: Only return trades with outcomes
            unresolved_only: Only return trades without outcomes

        Returns:
            List of matching shadow trade records
        """
        # Delegating to query_iter to avoid code duplication
        return list(self.query_iter(start, end, resolved_only, unresolved_only))

    def _row_to_record(self, row: sqlite3.Row) -> ShadowTradeRecord:
        """Convert a database row to ShadowTradeRecord."""
        # Parse outcome: NULL -> None, 0 -> False, 1 -> True
        outcome = None
        if row["outcome"] is not None:
            outcome = bool(row["outcome"])

        # Parse timestamps
        timestamp = datetime.fromisoformat(row["timestamp"])
        resolved_at = None
        if row["resolved_at"]:
            resolved_at = datetime.fromisoformat(row["resolved_at"])

        return ShadowTradeRecord(
            trade_id=row["trade_id"],
            timestamp=timestamp,
            contract_type=row["contract_type"],
            direction=row["direction"],
            probability=row["probability"],
            entry_price=row["entry_price"],
            reconstruction_error=row["reconstruction_error"],
            regime_state=row["regime_state"],
            model_version=row["model_version"],
            feature_schema_version=row["feature_schema_version"],
            tick_window=json.loads(row["tick_window"]) if row["tick_window"] else [],
            candle_window=json.loads(row["candle_window"]) if row["candle_window"] else [],
            outcome=outcome,
            exit_price=row["exit_price"],
            resolved_at=resolved_at,
            metadata=json.loads(row["metadata"]) if row["metadata"] else {},
            _schema_version=row["schema_version"],
            _created_at=row["created_at"],
            barrier_level=row["barrier_level"] if "barrier_level" in row.keys() else None,
            barrier2_level=row["barrier2_level"] if "barrier2_level" in row.keys() else None,
        )

    def get_by_id(self, trade_id: str) -> ShadowTradeRecord | None:
        """
        Get a single trade by ID.

        Args:
            trade_id: Trade ID to fetch

        Returns:
            ShadowTradeRecord if found, None otherwise
        """
        conn = self._get_connection()
        cursor = conn.execute("SELECT * FROM shadow_trades WHERE trade_id = ?", (trade_id,))
        row = cursor.fetchone()

        if row:
            return self._row_to_record(row)
        return None

    def export_parquet(self, output_path: Path) -> Path:
        """
        Export shadow trades to Parquet format.

        Maintains API compatibility with NDJSON ShadowTradeStore.

        Args:
            output_path: Path for output Parquet file

        Returns:
            Path to exported file
        """
        try:
            import pandas as pd
        except ImportError:
            raise ImportError("pandas required for Parquet export: pip install pandas pyarrow")

        records = self.query()
        if not records:
            logger.warning("No records to export")
            return Path(output_path)

        # Convert to DataFrame
        data = []
        for r in records:
            row = {
                "trade_id": r.trade_id,
                "timestamp": r.timestamp,
                "contract_type": r.contract_type,
                "direction": r.direction,
                "probability": r.probability,
                "entry_price": r.entry_price,
                "reconstruction_error": r.reconstruction_error,
                "regime_state": r.regime_state,
                "model_version": r.model_version,
                "feature_schema_version": r.feature_schema_version,
                "outcome": r.outcome,
                "exit_price": r.exit_price,
                "resolved_at": r.resolved_at,
                "tick_window": json.dumps(r.tick_window),
                "candle_window": json.dumps(r.candle_window),
                "metadata": json.dumps(r.metadata),
                "_schema_version": r._schema_version,
            }
            data.append(row)

        df = pd.DataFrame(data)

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(output_path, index=False)

        logger.info(f"Exported {len(records)} records to {output_path}")
        return output_path

    def get_statistics(self) -> dict[str, Any]:
        """Get summary statistics of the store."""
        conn = self._get_connection()

        # Use SQL for efficient counting
        total = conn.execute("SELECT COUNT(*) FROM shadow_trades").fetchone()[0]
        resolved = conn.execute(
            "SELECT COUNT(*) FROM shadow_trades WHERE outcome IS NOT NULL"
        ).fetchone()[0]
        wins = conn.execute("SELECT COUNT(*) FROM shadow_trades WHERE outcome = 1").fetchone()[0]

        return {
            "total_records": total,
            "resolved_records": resolved,
            "unresolved_records": total - resolved,
            "wins": wins,
            "losses": resolved - wins,
            "win_rate": wins / resolved if resolved > 0 else 0.0,
            "schema_version": SHADOW_STORE_SCHEMA_VERSION,
            "storage_backend": "sqlite",
        }

    def close(self) -> None:
        """Close database connection."""
        if hasattr(self._local, "conn") and self._local.conn:
            self._local.conn.close()
            self._local.conn = None

    @classmethod
    def from_ndjson(cls, ndjson_path: Path, db_path: Path) -> "SQLiteShadowStore":
        """
        Create SQLite store by importing from existing NDJSON file.

        This is the migration path from the old ShadowTradeStore.

        Args:
            ndjson_path: Path to existing NDJSON file
            db_path: Path for new SQLite database

        Returns:
            New SQLiteShadowStore with imported data
        """
        store = cls(db_path)

        if not ndjson_path.exists():
            logger.warning(f"NDJSON file not found: {ndjson_path}")
            return store

        imported = 0
        with open(ndjson_path) as f:
            for line in f:
                if not line.strip():
                    continue
                try:
                    d = json.loads(line)
                    record = ShadowTradeRecord.from_dict(d)
                    store.append(record)
                    imported += 1
                except (json.JSONDecodeError, TypeError) as e:
                    logger.warning(f"Skipping malformed record: {e}")
                    continue

        logger.info(f"Imported {imported} records from {ndjson_path} to {db_path}")
        return store
