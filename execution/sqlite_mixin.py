"""
SQLite Transaction Mixin.

Provides standardized connection pooling and atomic transaction management
for SQLite-backed stores (ShadowStore, SafetyStateStore).
"""

import sqlite3
import threading
import logging
from contextlib import contextmanager
from pathlib import Path
from typing import Generator, cast

logger = logging.getLogger(__name__)


class SQLiteTransactionMixin:
    """
    Mixin providing thread-safe SQLite connection pooling and atomic transactions.
    
    Attributes:
        _db_path (Path): Path to the database file.
        _write_lock (threading.Lock): Lock for serializing write operations.
        _local (threading.local): Thread-local storage for connections.
    """

    def __init__(self, db_path: str | Path):
        self._db_path = Path(db_path)
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._write_lock = threading.Lock()
        self._local = threading.local()

    def _get_connection(self) -> sqlite3.Connection:
        """
        Get thread-local database connection.
        
        Configures WAL mode for concurrency.
        """
        if not hasattr(self._local, "conn") or self._local.conn is None:
            conn = sqlite3.connect(
                str(self._db_path), 
                check_same_thread=False, 
                timeout=30.0
            )
            # Enable WAL mode for better concurrency
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA synchronous=NORMAL")
            conn.execute("PRAGMA foreign_keys=ON")
            conn.row_factory = sqlite3.Row
            self._local.conn = conn
        return cast(sqlite3.Connection, self._local.conn)

    @contextmanager
    def _transaction(self) -> Generator[sqlite3.Connection, None, None]:
        """
        Context manager for atomic transactions.
        
        Acquires a write lock, yields the connection, and commits.
        Rolls back on exception.
        
        Usage:
            with self._transaction() as conn:
                conn.execute(...)
        """
        conn = self._get_connection()
        with self._write_lock:
            try:
                yield conn
                conn.commit()
            except Exception:
                conn.rollback()
                raise

    @staticmethod
    async def run_with_timeout(loop, func, timeout: float, *args):
        """
        Run a blocking function in executor with a timeout.
        
        Args:
            loop: The asyncio loop.
            func: The blocking function to run.
            timeout: Maximum time in seconds.
            *args: Arguments for the function.
            
        Returns:
            Result of the function.
            
        Raises:
            asyncio.TimeoutError: If execution exceeds timeout.
        """
        import asyncio
        return await asyncio.wait_for(
            loop.run_in_executor(None, lambda: func(*args)),
            timeout=timeout
        )

    def close(self) -> None:
        """Close thread-local connection if it exists."""
        if hasattr(self._local, "conn") and self._local.conn:
            self._local.conn.close()
            self._local.conn = None
