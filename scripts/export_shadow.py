
#!/usr/bin/env python3
"""
Export shadow trades from SQLite to Parquet for training.

Usage:
    python scripts/export_shadow.py --output data_cache/shadow_replay.parquet
"""

import argparse
import sys
import logging
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from execution.sqlite_shadow_store import SQLiteShadowStore
from utils.logging_setup import setup_logging

logger, log_dir, log_file = setup_logging(script_name="export_shadow")

def main(args):
    db_path = Path(args.db)
    if not db_path.exists():
        logger.error(f"Database not found: {db_path}")
        return 1
        
    store = SQLiteShadowStore(db_path)
    
    # Export resolved trades only by default, unless --all is specified
    # However, store.export_parquet exports ALL by currently. 
    # For training we usually want resolved ones? 
    # Actually, we might want to filter later or export all.
    # The store's export_parquet exports everything returned by query(), which defaults to all.
    
    output_path = Path(args.output)
    logger.info(f"Exporting shadow trades from {db_path} to {output_path}...")
    
    try:
        store.export_parquet(output_path)
        logger.info("Export complete.")
    except Exception as e:
        logger.error(f"Export failed: {e}")
        return 1
    finally:
        store.close()
        
    return 0

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--db", type=str, default="data_cache/shadow_trades.db", help="Path to SQLite DB")
    parser.add_argument("--output", type=str, default="data_cache/shadow_replay.parquet", help="Output Parquet file")
    
    args = parser.parse_args()
    sys.exit(main(args))
