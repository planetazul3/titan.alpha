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
    >>> store = SQLiteShadowStore(Path("data_cache/trading_state.db"))
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
from execution.migrations import MigrationRunner, get_shadow_store_migrations
from execution.sqlite_mixin import SQLiteTransactionMixin

logger = logging.getLogger(__name__)

# SQLite database schema version
SQLITE_SCHEMA_VERSION = 4  # C01: Added resolution_context column


class OptimisticLockError(Exception):
    """Raised when an update fails due to version mismatch (concurrency conflict)."""
    pass

class SQLiteShadowStore(SQLiteTransactionMixin):
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
        barrier2_level REAL,
        duration_minutes INTEGER DEFAULT 1,
        resolution_context TEXT,
        version_number INTEGER DEFAULT 0,
        last_update_attempt TEXT,
        update_conflict_count INTEGER DEFAULT 0
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
        super().__init__(db_path)

        # Initialize schema via migrations
        self._init_migrations()
        
        # CRITICAL-001: Auto-migration for conflict tracking
        self._ensure_conflict_columns()

        logger.info(
            f"SQLiteShadowStore initialized: {db_path} "
            f"(record v{SHADOW_STORE_SCHEMA_VERSION})"
        )

    @property
    def _store_path(self) -> Path:
        """Return database path for API compatibility with ShadowTradeStore."""
        return self._db_path

    def _ensure_conflict_columns(self) -> None:
        """
        Ensure the schema has conflict tracking columns.
        Auto-migrates existing databases if needed.
        """
        required_columns = {
            "last_update_attempt": "TEXT", 
            "update_conflict_count": "INTEGER DEFAULT 0"
        }
        
        with self._transaction() as conn:
            cursor = conn.execute("PRAGMA table_info(shadow_trades)")
            existing_columns = {row[1] for row in cursor.fetchall()}
            
            for col, dtype in required_columns.items():
                if col not in existing_columns:
                    logger.warning(f"Auto-migrating schema: Adding column {col}")
                    try:
                        conn.execute(f"ALTER TABLE shadow_trades ADD COLUMN {col} {dtype}")
                    except sqlite3.OperationalError as e:
                        if "duplicate column" not in str(e).lower():
                            raise

    def _init_migrations(self) -> None:
        """Initialize database schema via migrations."""
        runner = MigrationRunner(str(self._db_path))
        
        from typing import Callable
        # Register shadow store migrations
        for version, steps in get_shadow_store_migrations().items():
            runner.add_migration(version, cast(list[str | Callable[..., Any]], steps))
            
        # Run migrations
        runner.run()

        # Update SHADOW_STORE_SCHEMA_VERSION in schema_meta (record schema vs db schema)
        with self._transaction() as conn:
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
                    barrier_level, barrier2_level, duration_minutes, resolution_context,
                    version_number, update_conflict_count
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 0)
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
                    record.duration_minutes,
                    json.dumps(record.resolution_context),
                    record.version_number,
                ),
            )

        logger.debug(f"Appended shadow trade: {record.trade_id}")

    async def append_async(self, record: ShadowTradeRecord, timeout: float = 5.0) -> None:
        """Async append."""
        import asyncio
        loop = asyncio.get_running_loop()
        await self.run_with_timeout(loop, self.append, timeout, record)

    def update_outcome(self, trade: ShadowTradeRecord, outcome: bool, exit_price: float) -> bool:
        """
        Update a trade's outcome atomically with robust retry logic.

        CRITICAL-001: Implements exponential backoff for optimistic locking conflicts.
        """
        import time
        import random
        
        trade_id = trade.trade_id
        resolved_at = datetime.now(timezone.utc).isoformat()
        current_version = trade.version_number
        
        max_retries = 5
        base_delay = 0.1
        
        for attempt in range(max_retries):
            try:
                with self._transaction() as conn:
                    # Update attempts
                    conn.execute(
                        "UPDATE shadow_trades SET last_update_attempt = ? WHERE trade_id = ?",
                        (datetime.now(timezone.utc).isoformat(), trade_id)
                    )
                    
                    cursor = conn.execute(
                        """
                        UPDATE shadow_trades
                        SET outcome = ?, exit_price = ?, resolved_at = ?, 
                            version_number = version_number + 1,
                            update_conflict_count = update_conflict_count + ?
                        WHERE trade_id = ? AND version_number = ?
                        """,
                        (1 if outcome else 0, exit_price, resolved_at, attempt, trade_id, current_version),
                    )

                    if cursor.rowcount > 0:
                        logger.debug(f"Updated outcome for {trade_id} (v{current_version}->v{current_version+1})")
                        return True
                    else:
                        # Conflict or Not Found
                        # Check current state
                        cursor = conn.execute(
                            "SELECT version_number, outcome FROM shadow_trades WHERE trade_id = ?", 
                            (trade_id,)
                        )
                        row = cursor.fetchone()
                        
                        if not row:
                            logger.warning(f"Trade not found for outcome update: {trade_id}")
                            return False
                        
                        db_version, db_outcome = row[0], row[1]
                        
                        if db_outcome is not None:
                             # Already resolved by concurrent process - Job Done
                             logger.info(f"Trade {trade_id} already resolved (outcome={db_outcome}). Skipping.")
                             return True
                        
                        if db_version != current_version:
                            logger.warning(
                                f"Optimistic lock conflict {trade_id}: v{current_version} vs v{db_version}. "
                                f"Retrying ({attempt+1}/{max_retries})..."
                            )
                            current_version = db_version # Update version for next retry
                            raise OptimisticLockError(f"Version mismatch v{current_version} vs v{db_version}")

            except OptimisticLockError:
                if attempt < max_retries - 1:
                    sleep_time = base_delay * (2 ** attempt) + random.uniform(0, 0.1)
                    time.sleep(sleep_time)
                else:
                    logger.error(f"Failed to update outcome for {trade_id} after {max_retries} attempts.")
                    raise
        
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
    async def update_outcome_async(self, trade: ShadowTradeRecord, outcome: bool, exit_price: float, timeout: float = 5.0) -> bool:
        """Async update outcome."""
        import asyncio
        from typing import cast
        loop = asyncio.get_running_loop()
        res = await self.run_with_timeout(loop, self.update_outcome, timeout, trade, outcome, exit_price)
        return cast(bool, res)

    async def mark_stale_async(self, trade_id: str, exit_price: float, timeout: float = 5.0) -> bool:
        """Async mark stale."""
        import asyncio
        from typing import cast
        loop = asyncio.get_running_loop()
        res = await self.run_with_timeout(loop, self.mark_stale, timeout, trade_id, exit_price)
        return cast(bool, res)

    def update_resolution_context(self, trade_id: str, high: float, low: float, close: float) -> bool:
        """
        C01 Fix: Append a resolution candle to a trade's resolution context.
        
        This accumulates OHLC data observed AFTER trade entry for path-dependent
        contract resolution (TOUCH, RANGE).
        
        Args:
            trade_id: Trade ID to update
            high: High price of the candle
            low: Low price of the candle
            close: Close price of the candle
            
        Returns:
            True if trade was found and updated
        """
        conn = self._get_connection()
        
        # First, get current resolution_context
        cursor = conn.execute(
            "SELECT resolution_context FROM shadow_trades WHERE trade_id = ?",
            (trade_id,)
        )
        row = cursor.fetchone()
        
        if not row:
            logger.warning(f"Trade not found for resolution context update: {trade_id}")
            return False
        
        # Parse existing context and append new candle
        current_context = json.loads(row["resolution_context"]) if row["resolution_context"] else []
        current_context.append([high, low, close])
        
        with self._transaction() as conn:
            cursor = conn.execute(
                """
                UPDATE shadow_trades
                SET resolution_context = ?
                WHERE trade_id = ?
                """,
                (json.dumps(current_context), trade_id),
            )
            
            if cursor.rowcount > 0:
                logger.debug(f"Updated resolution context for {trade_id}: {len(current_context)} candles")
                return True
            return False

    async def update_resolution_context_async(
        self, trade_id: str, high: float, low: float, close: float, timeout: float = 5.0
    ) -> bool:
        """Async update resolution context."""
        import asyncio
        from typing import cast
        loop = asyncio.get_running_loop()
        res = await self.run_with_timeout(
            loop, self.update_resolution_context, timeout, trade_id, high, low, close
        )
        return cast(bool, res)
    
    async def query_async(self, timeout: float = 5.0, **kwargs) -> list[ShadowTradeRecord]:
        """Async query."""
        import asyncio
        from typing import cast
        loop = asyncio.get_running_loop()
        res = await self.run_with_timeout(loop, self.query, timeout, **kwargs)
        return cast(list[ShadowTradeRecord], res)

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
            deleted_count = int(cursor.rowcount)
            
        # Reclaim space
        if deleted_count > 0:
            try:
                with self._get_connection() as conn:
                    # VACUUM reclaims unused pages in the main DB file
                    conn.execute("VACUUM")
                    # TRUNCATE checkpoint reclaims space in the WAL file
                    conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
                logger.info(f"Pruned {deleted_count} shadow trades older than {retention_days} days (VACUUM+Checkpoint complete)")
            except Exception as e:
                logger.warning(f"Failed to VACUUM/Checkpoint after prune: {e}")
                
        return deleted_count

    async def prune_async(self, retention_days: int = 30, timeout: float = 30.0) -> int:
        """Async prune (heavy operation). timeout defaults to 30s."""
        import asyncio
        from typing import cast
        loop = asyncio.get_running_loop()
        res = await self.run_with_timeout(loop, self.prune, timeout, retention_days)
        return cast(int, res)

    def query_iter(
        self,
        start: datetime | None = None,
        end: datetime | None = None,
        resolved_only: bool = False,
        unresolved_only: bool = False,
        limit: int | None = None
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
            # nosec B608 - where_clause is built from code-defined conditions (resolved_only,
            # unresolved_only, time range), not user input. Values use parameterized query.
            limit_clause = f" LIMIT {limit}" if limit else ""
            cursor = conn.execute(
                f"SELECT * FROM shadow_trades WHERE {where_clause} ORDER BY timestamp{limit_clause}", params # nosec
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
        limit: int | None = None
    ) -> list[ShadowTradeRecord]:
        """
        Query shadow trades from the store.

        Args:
            start: Start of time range (inclusive)
            end: End of time range (exclusive)
            resolved_only: Only return trades with outcomes
            unresolved_only: Only return trades without outcomes
            limit: Maximum number of records to return

        Returns:
            List of matching shadow trade records
        """
        # Delegating to query_iter to avoid code duplication
        return list(self.query_iter(start, end, resolved_only, unresolved_only, limit))

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
            duration_minutes=row["duration_minutes"] if "duration_minutes" in row.keys() else 1,
            resolution_context=json.loads(row["resolution_context"]) if "resolution_context" in row.keys() and row["resolution_context"] else [],
            version_number=row["version_number"] if "version_number" in row.keys() else 0,
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

    async def get_statistics_async(self, timeout: float = 5.0) -> dict[str, Any]:
        """Async get_statistics."""
        import asyncio
        from typing import cast
        loop = asyncio.get_running_loop()
        res = await self.run_with_timeout(loop, self.get_statistics, timeout)
        return cast(dict[str, Any], res)



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
