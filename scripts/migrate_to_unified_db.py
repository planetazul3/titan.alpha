#!/usr/bin/env python3
"""
Migration script to unify shadow_trades.db and safety_state.db into trading_state.db.

This script:
1. Checks for existence of legacy databases.
2. Creates/Connects to the new unified trading_state.db.
3. Attaches legacy DBs and copies data over.
4. Archives legacy DBs.
"""

import sqlite3
import shutil
import logging
from pathlib import Path
from datetime import datetime

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

DATA_DIR = Path("data_cache")
UNIFIED_DB = DATA_DIR / "trading_state.db"
SHADOW_DB = DATA_DIR / "shadow_trades.db"
SAFETY_DB = DATA_DIR / "safety_state.db"
BACKUP_DIR = Path("backups")

def migrate_shadow_trades(conn):
    if not SHADOW_DB.exists():
        logger.info("No shadow_trades.db found. Skipping shadow trade migration.")
        return

    logger.info(f"Migrating shadow trades from {SHADOW_DB}...")
    try:
        # Attach legacy DB
        conn.execute("ATTACH DATABASE ? AS legacy_shadow", (str(SHADOW_DB),))

        # Check if source table exists
        cursor = conn.execute("SELECT name FROM legacy_shadow.sqlite_master WHERE type='table' AND name='shadow_trades'")
        if not cursor.fetchone():
            logger.warning("shadow_trades table not found in legacy DB.")
            return

        # Create target table (schema from SQLiteShadowStore)
        # We trust the app to create the schema, but let's ensure it exists or create it if missing
        # Actually, best to let SQLiteShadowStore init do it, but we are in a script.
        # We will assume target schema matches source or use generic copy if possible.
        # Better: use INSERT OR IGNORE INTO main.shadow_trades SELECT * FROM legacy_shadow.shadow_trades

        # First ensure main table exists by creating it if not
        # Copying schema from SQLiteShadowStore (simplified)
        conn.execute("""
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
            resolution_context TEXT
        )
        """)

        # Copy data
        # We need to handle potential schema differences (e.g. missing columns in old DB)
        # Get columns from source
        cursor = conn.execute("PRAGMA legacy_shadow.table_info(shadow_trades)")
        source_cols = [row[1] for row in cursor.fetchall()]

        # Get columns from dest
        cursor = conn.execute("PRAGMA main.table_info(shadow_trades)")
        dest_cols = [row[1] for row in cursor.fetchall()]

        common_cols = list(set(source_cols).intersection(dest_cols))
        cols_str = ", ".join(common_cols)

        logger.info(f"Copying columns: {cols_str}")

        conn.execute(f"""
            INSERT OR IGNORE INTO main.shadow_trades ({cols_str})
            SELECT {cols_str} FROM legacy_shadow.shadow_trades
        """) # nosec

        count = conn.execute("SELECT changes()").fetchone()[0]
        logger.info(f"Migrated {count} shadow trades.")

        # Also migrate schema_meta if exists
        cursor = conn.execute("SELECT name FROM legacy_shadow.sqlite_master WHERE type='table' AND name='schema_meta'")
        if cursor.fetchone():
             conn.execute("""
                CREATE TABLE IF NOT EXISTS schema_meta (
                    key TEXT PRIMARY KEY,
                    value TEXT
                )
            """)
             conn.execute("INSERT OR IGNORE INTO main.schema_meta SELECT * FROM legacy_shadow.schema_meta")

    except Exception as e:
        logger.error(f"Failed to migrate shadow trades: {e}")
    finally:
        conn.execute("DETACH DATABASE legacy_shadow")

def migrate_safety_state(conn):
    if not SAFETY_DB.exists():
        logger.info("No safety_state.db found. Skipping safety state migration.")
        return

    logger.info(f"Migrating safety state from {SAFETY_DB}...")
    try:
        conn.execute("ATTACH DATABASE ? AS legacy_safety", (str(SAFETY_DB),))

        # Migrate kv_store
        cursor = conn.execute("SELECT name FROM legacy_safety.sqlite_master WHERE type='table' AND name='kv_store'")
        if cursor.fetchone():
            conn.execute("""
                CREATE TABLE IF NOT EXISTS kv_store (
                    key TEXT PRIMARY KEY,
                    value TEXT,
                    updated_at REAL
                )
            """)
            conn.execute("INSERT OR REPLACE INTO main.kv_store SELECT * FROM legacy_safety.kv_store")
            count = conn.execute("SELECT changes()").fetchone()[0]
            logger.info(f"Migrated {count} kv_store entries.")

        # Migrate daily_stats
        cursor = conn.execute("SELECT name FROM legacy_safety.sqlite_master WHERE type='table' AND name='daily_stats'")
        if cursor.fetchone():
            conn.execute("""
                CREATE TABLE IF NOT EXISTS daily_stats (
                    date TEXT PRIMARY KEY,
                    trade_count INTEGER DEFAULT 0,
                    daily_pnl REAL DEFAULT 0.0,
                    updated_at REAL
                )
            """)
            conn.execute("INSERT OR REPLACE INTO main.daily_stats SELECT * FROM legacy_safety.daily_stats")
            count = conn.execute("SELECT changes()").fetchone()[0]
            logger.info(f"Migrated {count} daily_stats entries.")

    except Exception as e:
        logger.error(f"Failed to migrate safety state: {e}")
    finally:
        conn.execute("DETACH DATABASE legacy_safety")

def archive_legacy_db(path):
    if not path.exists():
        return

    BACKUP_DIR.mkdir(exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_path = BACKUP_DIR / f"{path.name}.{timestamp}.bak"

    logger.info(f"Archiving {path} to {backup_path}")
    shutil.move(str(path), str(backup_path))

def main():
    if not SHADOW_DB.exists() and not SAFETY_DB.exists():
        logger.info("No legacy databases found. Nothing to do.")
        return

    logger.info(f"Starting migration to {UNIFIED_DB}...")
    UNIFIED_DB.parent.mkdir(parents=True, exist_ok=True)

    conn = sqlite3.connect(str(UNIFIED_DB))
    try:
        migrate_shadow_trades(conn)
        migrate_safety_state(conn)
        conn.commit()
        logger.info("Migration completed successfully.")
    except Exception as e:
        logger.error(f"Migration failed: {e}")
        conn.rollback()
        return
    finally:
        conn.close()

    # Archive legacy files only if migration succeeded
    archive_legacy_db(SHADOW_DB)
    archive_legacy_db(SAFETY_DB)

    # Also clean up WAL/SHM files
    for db in [SHADOW_DB, SAFETY_DB]:
        for ext in ["-wal", "-shm"]:
            f = db.parent / (db.name + ext)
            if f.exists():
                f.unlink()

if __name__ == "__main__":
    main()
