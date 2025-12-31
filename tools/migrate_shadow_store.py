#!/usr/bin/env python3
"""
Migration tool for converting NDJSON shadow trades to SQLite.

This script migrates existing shadow trade data from the NDJSON format
to the new SQLite-backed store.

Usage:
    python tools/migrate_shadow_store.py                    # Default paths
    python tools/migrate_shadow_store.py --source logs/shadow_trades.ndjson
    python tools/migrate_shadow_store.py --dry-run          # Preview only
"""

import argparse
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import logging


from config.logging_config import setup_logging

log_file = setup_logging(script_name="migrate_shadow", level="INFO")
logger = logging.getLogger(__name__)


def migrate_shadow_store(source_path: Path, target_path: Path, dry_run: bool = False) -> int:
    """
    Migrate shadow trades from NDJSON to SQLite.

    Args:
        source_path: Path to existing NDJSON file
        target_path: Path for new SQLite database
        dry_run: If True, only preview without writing

    Returns:
        Number of records migrated
    """
    import json

    from execution.shadow_store import ShadowTradeRecord

    if not source_path.exists():
        logger.error(f"Source file not found: {source_path}")
        return 0

    # Count and preview records
    records = []
    with open(source_path) as f:
        for line_num, line in enumerate(f, 1):
            if not line.strip():
                continue
            try:
                d = json.loads(line)
                record = ShadowTradeRecord.from_dict(d)
                records.append(record)
            except (json.JSONDecodeError, TypeError) as e:
                logger.warning(f"Line {line_num}: Skipping malformed record: {e}")

    logger.info(f"Found {len(records)} valid records in {source_path}")

    # Show preview
    resolved = [r for r in records if r.is_resolved()]
    unresolved = [r for r in records if not r.is_resolved()]

    logger.info(f"  Resolved: {len(resolved)}")
    logger.info(f"  Unresolved: {len(unresolved)}")

    if resolved:
        wins = sum(1 for r in resolved if r.outcome)
        logger.info(f"  Win rate: {wins}/{len(resolved)} ({100 * wins / len(resolved):.1f}%)")

    if dry_run:
        logger.info("Dry run - no changes made")
        return len(records)

    # Perform migration
    from execution.sqlite_shadow_store import SQLiteShadowStore

    if target_path.exists():
        backup_path = target_path.with_suffix(".bak.db")
        logger.warning(f"Target exists, backing up to {backup_path}")
        import shutil

        shutil.copy(target_path, backup_path)

    store = SQLiteShadowStore(target_path)

    migrated = 0
    for record in records:
        try:
            store.append(record)
            migrated += 1
        except Exception as e:
            logger.error(f"Failed to migrate {record.trade_id}: {e}")

    store.close()

    logger.info(f"Successfully migrated {migrated}/{len(records)} records to {target_path}")

    # Verify migration
    store = SQLiteShadowStore(target_path)
    stats = store.get_statistics()
    logger.info(f"Verification: {stats}")
    store.close()

    return migrated


def main():
    parser = argparse.ArgumentParser(description="Migrate shadow trades from NDJSON to SQLite")
    parser.add_argument(
        "--source",
        type=Path,
        default=Path("logs/shadow_trades.ndjson"),
        help="Source NDJSON file (default: logs/shadow_trades.ndjson)",
    )
    parser.add_argument(
        "--target",
        type=Path,
        default=Path("data_cache/trading_state.db"),
        help="Target SQLite database (default: data_cache/trading_state.db)",
    )
    parser.add_argument("--dry-run", action="store_true", help="Preview migration without writing")

    args = parser.parse_args()

    logger.info("Shadow Trade Store Migration Tool")
    logger.info(f"Source: {args.source}")
    logger.info(f"Target: {args.target}")

    count = migrate_shadow_store(args.source, args.target, args.dry_run)

    if count > 0:
        logger.info("Migration complete!")
    else:
        logger.warning("No records migrated")

    return 0 if count > 0 else 1


if __name__ == "__main__":
    sys.exit(main())
