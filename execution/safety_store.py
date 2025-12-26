"""
Persistent state store for safety and risk modules using SQLite.

Ensures that critical safety counters (daily trades, loss limits) and
risk metrics (drawdown, consecutive losses) survive application restarts.
"""

import sqlite3
import logging
import time
from pathlib import Path
from typing import Any, Optional
from datetime import datetime, timezone

logger = logging.getLogger(__name__)

class SQLiteSafetyStateStore:
    """
    SQLite backend for persisting safety and risk state.
    
    Schema:
        kv_store table: key (TEXT PRIMARY KEY), value (TEXT), updated_at (REAL)
        daily_stats table: date (TEXT PRIMARY KEY), trade_count (INT), daily_pnl (REAL)
        
    We use SQLite for its reliability, ACID properties, and zero-conf nature.
    """
    
    def __init__(self, db_path: str | Path):
        """
        Initialize store.
        
        Args:
            db_path: Path to SQLite database file
        """
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()
        
    def _init_db(self):
        """Create tables if they don't exist."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("PRAGMA journal_mode=WAL")
                conn.execute("PRAGMA synchronous=NORMAL")
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
                conn.commit()
        except Exception as e:
            logger.error(f"Failed to initialize safety DB: {e}")
            raise

    def get_value(self, key: str, default: Any = None) -> str | None:
        """Get simple key-value pair."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("PRAGMA journal_mode=WAL")
                cursor = conn.execute("SELECT value FROM kv_store WHERE key = ?", (key,))
                row = cursor.fetchone()
                return row[0] if row else default
        except Exception as e:
            logger.error(f"DB Read Error (get_value): {e}")
            return default

    def set_value(self, key: str, value: str):
        """Set simple key-value pair."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute(
                    "INSERT OR REPLACE INTO kv_store (key, value, updated_at) VALUES (?, ?, ?)",
                    (key, str(value), time.time())
                )
                conn.commit()
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
            with sqlite3.connect(self.db_path) as conn:
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
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("""
                    INSERT INTO daily_stats (date, trade_count, daily_pnl, updated_at)
                    VALUES (?, 1, 0.0, ?)
                    ON CONFLICT(date) DO UPDATE SET
                        trade_count = trade_count + 1,
                        updated_at = excluded.updated_at
                """, (today, time.time()))
                conn.commit()
        except Exception as e:
            logger.error(f"DB Write Error (increment_daily_trade_count): {e}")

    def update_daily_pnl(self, pnl: float):
        """Add pnl to daily total."""
        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("""
                    INSERT INTO daily_stats (date, trade_count, daily_pnl, updated_at)
                    VALUES (?, 0, ?, ?)
                    ON CONFLICT(date) DO UPDATE SET
                        daily_pnl = daily_pnl + excluded.daily_pnl,
                        updated_at = excluded.updated_at
                """, (today, pnl, time.time()))
                conn.commit()
        except Exception as e:
            logger.error(f"DB Write Error (update_daily_pnl): {e}")
            
    # H04 Helpers for Adaptive Risk
    def get_risk_metrics(self) -> dict:
        """Retrieve persisted risk metrics."""
        drawdown = float(self.get_value("risk_current_drawdown", "0.0"))
        losses = int(self.get_value("risk_consecutive_losses", "0"))
        peak_equity = float(self.get_value("risk_peak_equity", "0.0"))
        return {
            "current_drawdown": drawdown,
            "consecutive_losses": losses,
            "peak_equity": peak_equity
        }

    def update_risk_metrics(self, drawdown: float, losses: int, peak_equity: float):
        """Persist risk metrics."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("PRAGMA journal_mode=WAL")
                ts = time.time()
                data = [
                    ("risk_current_drawdown", str(drawdown), ts),
                    ("risk_consecutive_losses", str(losses), ts),
                    ("risk_peak_equity", str(peak_equity), ts)
                ]
                conn.executemany(
                    "INSERT OR REPLACE INTO kv_store (key, value, updated_at) VALUES (?, ?, ?)",
                    data
                )
                conn.commit()
        except Exception as e:
            logger.error(f"DB Write Error (update_risk_metrics): {e}")

    # H09: Async Wrappers to prevent blocking event loop
    async def get_daily_stats_async(self) -> tuple[int, float]:
        """Async version of get_daily_stats."""
        import asyncio
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, self.get_daily_stats)

    async def increment_daily_trade_count_async(self):
        """Async version of increment_daily_trade_count."""
        import asyncio
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(None, self.increment_daily_trade_count)

    async def update_daily_pnl_async(self, pnl: float):
        """Async version of update_daily_pnl."""
        import asyncio
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(None, lambda: self.update_daily_pnl(pnl))

    async def get_risk_metrics_async(self) -> dict:
        """Async version of get_risk_metrics."""
        import asyncio
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, self.get_risk_metrics)

    async def update_risk_metrics_async(self, drawdown: float, losses: int, peak_equity: float):
        """Async version of update_risk_metrics."""
        import asyncio
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(None, lambda: self.update_risk_metrics(drawdown, losses, peak_equity))
