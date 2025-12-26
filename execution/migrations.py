"""
Lightweight migration runner for SQLite-backed stores.
"""

import logging
import sqlite3
from typing import Callable, List, Dict

logger = logging.getLogger(__name__)

class MigrationRunner:
    """
    Handles sequential SQL migrations for SQLite databases.
    """
    
    def __init__(self, db_path: str):
        self.db_path = db_path
        self.migrations: Dict[int, List[str | Callable]] = {}

    def add_migration(self, version: int, steps: List[str | Callable]):
        """Add a migration version with a list of SQL strings or callables."""
        self.migrations[version] = steps

    def run(self):
        """Run all pending migrations."""
        conn = sqlite3.connect(self.db_path)
        try:
            # 1. Ensure schema_meta exists
            conn.execute("CREATE TABLE IF NOT EXISTS schema_meta (key TEXT PRIMARY KEY, value TEXT)")
            
            # 2. Get current version
            cursor = conn.execute("SELECT value FROM schema_meta WHERE key = 'sqlite_schema_version'")
            row = cursor.fetchone()
            current_version = int(row[0]) if row else 0
            
            # 3. Apply migrations sequentially
            target_versions = sorted([v for v in self.migrations.keys() if v > current_version])
            
            if not target_versions:
                logger.debug(f"Database at {self.db_path} is up to date (v{current_version})")
                return

            for version in target_versions:
                logger.info(f"Applying migration to version {version}...")
                steps = self.migrations[version]
                
                with conn:
                    for step in steps:
                        if isinstance(step, str):
                            conn.execute(step)
                        elif callable(step):
                            step(conn)
                    
                    # Update version
                    conn.execute(
                        "INSERT OR REPLACE INTO schema_meta (key, value) VALUES (?, ?)",
                        ("sqlite_schema_version", str(version))
                    )
                logger.info(f"Successfully migrated to version {version}")

        finally:
            conn.close()

def get_shadow_store_migrations() -> Dict[int, List[str]]:
    """Return the migration definitions for the shadow store."""
    return {
        1: [
            """
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
                created_at TEXT NOT NULL
            )
            """,
            "CREATE INDEX IF NOT EXISTS idx_timestamp ON shadow_trades(timestamp)",
            "CREATE INDEX IF NOT EXISTS idx_outcome ON shadow_trades(outcome)",
            "CREATE INDEX IF NOT EXISTS idx_regime_state ON shadow_trades(regime_state)"
        ],
        2: [
            "ALTER TABLE shadow_trades ADD COLUMN barrier_level REAL",
            "ALTER TABLE shadow_trades ADD COLUMN barrier2_level REAL",
            "ALTER TABLE shadow_trades ADD COLUMN duration_minutes INTEGER DEFAULT 1"
        ],
        3: [
            "ALTER TABLE shadow_trades ADD COLUMN resolution_context TEXT"
        ],
        4: [
             # Example future migration: index for resolution_context? Or just version bump for sanity check from previous manual changes
             # Actually, version 4 in existing code included resolution_context.
             # To align with current DBs that might already be at v4:
             # We'll keep version 4 as a placeholder if already at v4.
             "SELECT 1" 
        ]
    }
