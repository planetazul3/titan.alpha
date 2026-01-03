import logging
import time
import threading
from pathlib import Path
import json
from typing import Any, Optional
from datetime import datetime, timezone


from execution.sqlite_mixin import SQLiteTransactionMixin

logger = logging.getLogger(__name__)

class SQLiteSafetyStateStore(SQLiteTransactionMixin):
    """
    SQLite backend for persisting safety and risk state.
    
    Schema:
        kv_store table: key (TEXT PRIMARY KEY), value (TEXT), updated_at (REAL)
        daily_stats table: date (TEXT PRIMARY KEY), trade_count (INT), daily_pnl (REAL)
        
    We use SQLite for its reliability, ACID properties, and zero-conf nature.
    R07: Uses thread-local connection pooling for high-frequency access.
    """
    
    def __init__(self, db_path: str | Path):
        """
        Initialize store.
        
        Args:
            db_path: Path to SQLite database file
        """
        super().__init__(db_path)
        
        self._init_db()
        
    def _init_db(self):
        """Create tables if they don't exist."""
        try:
            with self._transaction() as conn:
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS kv_store (
                        key TEXT PRIMARY KEY,
                        value TEXT,
                        updated_at REAL
                    )
                """)
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS daily_stats (
                        date TEXT PRIMARY KEY,
                        trade_count INTEGER DEFAULT 0,
                        daily_pnl REAL DEFAULT 0.0,
                        updated_at REAL
                    )
                """)
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS trade_timestamps (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        symbol TEXT,
                        timestamp REAL
                    )
                """)
                conn.execute("CREATE INDEX IF NOT EXISTS idx_trade_ts ON trade_timestamps(timestamp)")
                conn.execute("CREATE INDEX IF NOT EXISTS idx_trade_sym_ts ON trade_timestamps(symbol, timestamp)")
        except Exception as e:
            logger.error(f"Failed to initialize safety DB: {e}")
            raise

    def get_value(self, key: str, default: str | None = None) -> str | None:
        """Get simple key-value pair."""
        try:
            conn = self._get_connection()
            cursor = conn.execute("SELECT value FROM kv_store WHERE key = ?", (key,))
            row = cursor.fetchone()
            return str(row[0]) if row else default
        except Exception as e:
            logger.error(f"DB Read Error (get_value): {e}")
            return default

    def set_value(self, key: str, value: str):
        """Set simple key-value pair."""
        try:
            with self._transaction() as conn:
                conn.execute(
                    "INSERT OR REPLACE INTO kv_store (key, value, updated_at) VALUES (?, ?, ?)",
                    (key, str(value), time.time())
                )
        except Exception as e:
            logger.error(f"DB Write Error (set_value): {e}")

    def get_daily_stats(self) -> tuple[int, float]:
        """
        Get stats for the CURRENT UTC day.
        
        Returns:
            (trade_count, daily_pnl)
        """
        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        try:
            conn = self._get_connection()
            cursor = conn.execute(
                "SELECT trade_count, daily_pnl FROM daily_stats WHERE date = ?", 
                (today,)
            )
            row = cursor.fetchone()
            if row:
                return row[0], row[1]
            return 0, 0.0
        except Exception as e:
            logger.error(f"DB Read Error (get_daily_stats): {e}")
            return 0, 0.0

    def increment_daily_trade_count(self):
        """Increment daily trade counter for today."""
        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        try:
            with self._transaction() as conn:
                conn.execute("""
                    INSERT INTO daily_stats (date, trade_count, daily_pnl, updated_at)
                    VALUES (?, 1, 0.0, ?)
                    ON CONFLICT(date) DO UPDATE SET
                        trade_count = trade_count + 1,
                        updated_at = excluded.updated_at
                """, (today, time.time()))
        except Exception as e:
            logger.error(f"DB Write Error (increment_daily_trade_count): {e}")

    def update_daily_pnl(self, pnl: float):
        """Add pnl to daily total."""
        from utils.numerical_validation import ensure_finite
        pnl = ensure_finite(pnl, "update_daily_pnl", default=0.0)
        
        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        try:
            with self._transaction() as conn:
                conn.execute("""
                    INSERT INTO daily_stats (date, trade_count, daily_pnl, updated_at)
                    VALUES (?, 0, ?, ?)
                    ON CONFLICT(date) DO UPDATE SET
                        daily_pnl = daily_pnl + excluded.daily_pnl,
                        updated_at = excluded.updated_at
                """, (today, pnl, time.time()))
        except Exception as e:
            logger.error(f"DB Write Error (update_daily_pnl): {e}")

    def record_trade_timestamp(self, symbol: str, timestamp: float):
        """Record a trade timestamp for rate limiting."""
        try:
            with self._transaction() as conn:
                conn.execute(
                    "INSERT INTO trade_timestamps (symbol, timestamp) VALUES (?, ?)",
                    (symbol, timestamp)
                )
        except Exception as e:
             logger.error(f"DB Write Error (record_trade_timestamp): {e}")

    def get_trades_in_window(self, symbol: str | None, window_seconds: float) -> int:
        """
        Count trades in the last window_seconds.
        If symbol is None, counts GLOBAL trades.
        """
        import time
        cutoff = time.time() - window_seconds
        try:
            conn = self._get_connection()
            if symbol:
                cursor = conn.execute(
                    "SELECT COUNT(*) FROM trade_timestamps WHERE symbol = ? AND timestamp > ?",
                    (symbol, cutoff)
                )
            else:
                cursor = conn.execute(
                    "SELECT COUNT(*) FROM trade_timestamps WHERE timestamp > ?",
                    (cutoff,)
                )
            row = cursor.fetchone()
            return row[0] if row else 0
        except Exception as e:
            logger.error(f"DB Read Error (get_trades_in_window): {e}")
            return 0
            
    def prune_old_timestamps(self, max_age_seconds: float = 3600):
        """Delete old timestamps to prevent bloat."""
        import time
        cutoff = time.time() - max_age_seconds
        try:
            with self._transaction() as conn:
                conn.execute("DELETE FROM trade_timestamps WHERE timestamp < ?", (cutoff,))
        except Exception as e:
            logger.error(f"DB Write Error (prune_old_timestamps): {e}")
            
    # H04 Helpers for Adaptive Risk
    def get_risk_metrics(self) -> dict:
        """Retrieve persisted risk metrics."""
        from utils.numerical_validation import ensure_finite
        
        try:
            drawdown_raw = float(self.get_value("risk_current_drawdown") or "0.0")
            losses_raw = int(self.get_value("risk_consecutive_losses") or "0")
            peak_equity_raw = float(self.get_value("risk_peak_equity") or "0.0")
            
            drawdown = ensure_finite(drawdown_raw, "loaded_drawdown", 0.0)
            losses = losses_raw # Integer doesn't need is_finite check usually but good to be safe if it was float
            peak_equity = ensure_finite(peak_equity_raw, "loaded_peak_equity", 0.0)
            
            returns_json = self.get_value("risk_returns_history")
            returns = json.loads(returns_json) if returns_json else []
            
            return {
                "current_drawdown": drawdown,
                "consecutive_losses": losses,
                "peak_equity": peak_equity,
                "returns": returns
            }
        except ValueError:
             logger.error("Corrupt risk metrics in DB, returning defaults")
             return {
                "current_drawdown": 0.0,
                "consecutive_losses": 0,
                "peak_equity": 0.0,
                "returns": []
            }

    def update_risk_metrics(self, drawdown: float, losses: int, peak_equity: float, returns: list[float] | None = None):
        """Persist risk metrics."""
        from utils.numerical_validation import ensure_finite
        
        drawdown = ensure_finite(drawdown, "persisting_drawdown", 0.0)
        peak_equity = ensure_finite(peak_equity, "persisting_peak_equity", 0.0)
        
        try:
            ts = time.time()
            data = [
                ("risk_current_drawdown", str(drawdown), ts),
                ("risk_consecutive_losses", str(losses), ts),
                ("risk_peak_equity", str(peak_equity), ts)
            ]
            if returns is not None:
                data.append(("risk_returns_history", json.dumps(returns), ts))
                
            with self._transaction() as conn:
                conn.executemany(
                    "INSERT OR REPLACE INTO kv_store (key, value, updated_at) VALUES (?, ?, ?)",
                    data
                )
        except Exception as e:
            logger.error(f"DB Write Error (update_risk_metrics): {e}")

    # H09: Async Wrappers to prevent blocking event loop
    async def get_daily_stats_async(self, timeout: float = 5.0) -> tuple[int, float]:
        """Async version of get_daily_stats."""
        import asyncio
        loop = asyncio.get_running_loop()
        return await self.run_with_timeout(loop, self.get_daily_stats, timeout)

    async def increment_daily_trade_count_async(self, timeout: float = 5.0):
        """Async version of increment_daily_trade_count."""
        import asyncio
        loop = asyncio.get_running_loop()
        await self.run_with_timeout(loop, self.increment_daily_trade_count, timeout)

    async def update_daily_pnl_async(self, pnl: float, timeout: float = 5.0):
        """Async version of update_daily_pnl."""
        import asyncio
        loop = asyncio.get_running_loop()
        await self.run_with_timeout(loop, self.update_daily_pnl, timeout, pnl)

    async def get_risk_metrics_async(self, timeout: float = 5.0) -> dict:
        """Async version of get_risk_metrics."""
        import asyncio
        loop = asyncio.get_running_loop()
        return await self.run_with_timeout(loop, self.get_risk_metrics, timeout)

    async def update_risk_metrics_async(self, drawdown: float, losses: int, peak_equity: float, returns: list[float] | None = None, timeout: float = 5.0):
        """Async version of update_risk_metrics."""
        import asyncio
        loop = asyncio.get_running_loop()
        await self.run_with_timeout(loop, self.update_risk_metrics, timeout, drawdown, losses, peak_equity, returns)

    async def record_trade_timestamp_async(self, symbol: str, timestamp: float, timeout: float = 5.0):
        """Async version of record_trade_timestamp."""
        import asyncio
        loop = asyncio.get_running_loop()
        await self.run_with_timeout(loop, self.record_trade_timestamp, timeout, symbol, timestamp)

    async def get_trades_in_window_async(self, symbol: str | None, window_seconds: float, timeout: float = 5.0) -> int:
        """Async version of get_trades_in_window."""
        import asyncio
        loop = asyncio.get_running_loop()
        return await self.run_with_timeout(loop, self.get_trades_in_window, timeout, symbol, window_seconds)

    async def prune_old_timestamps_async(self, max_age_seconds: float = 3600, timeout: float = 5.0):
         """Async version of prune_old_timestamps."""
         import asyncio
         loop = asyncio.get_running_loop()
         await self.run_with_timeout(loop, self.prune_old_timestamps, timeout, max_age_seconds)
