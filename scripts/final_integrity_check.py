
import logging
import sys
from pathlib import Path
import pandas as pd
from datetime import datetime

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from config.settings import load_settings
from data.ingestion.versioning import load_metadata, verify_checksum

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

def check_partitions(data_dir: Path, data_type: str, granularity: int | None = None):
    """Verify integrity of all parquet partitions for a data type."""
    
    label = f"{data_type}" + (f"_{granularity}" if granularity else "")
    partition_dir = data_dir / label
    
    if not partition_dir.exists():
        logger.warning(f"No data found for {label} at {partition_dir}")
        return []

    partitions = sorted(partition_dir.glob("*.parquet"))
    report = []
    
    logger.info(f"Checking {len(partitions)} partitions for {label}...")
    
    for p_file in partitions:
        meta = load_metadata(p_file)
        status = "OK"
        notes = []
        
        if not meta:
            status = "MISSING_META"
            notes.append("No metadata found")
        else:
            # checksum verification
            if meta.sha256:
                if not verify_checksum(p_file, meta.sha256):
                    status = "CORRUPT"
                    notes.append("Checksum mismatch")
            
            # gap check
            if meta.gaps_detected > 0:
                notes.append(f"{meta.gaps_detected} gaps")
            
        report.append({
            "file": p_file.name,
            "type": label,
            "start": datetime.fromtimestamp(meta.start_epoch).strftime('%Y-%m-%d %H:%M') if meta else "N/A",
            "end": datetime.fromtimestamp(meta.end_epoch).strftime('%Y-%m-%d %H:%M') if meta else "N/A",
            "records": meta.record_count if meta else 0,
            "size_mb": p_file.stat().st_size / 1024 / 1024,
            "status": status,
            "notes": ", ".join(notes)
        })

    return report

def main():
    settings = load_settings()
    symbol = settings.trading.symbol
    
    # Construct base cache directory (default matching download_data.py logic)
    # Ideally should be configurable, but assuming data_cache/{symbol} based on current codebase
    cache_root = Path("data_cache") / symbol
    
    if not cache_root.exists():
        logger.error(f"Data directory not found: {cache_root}")
        return

    logger.info(f"Auditing data integrity for {symbol} in {cache_root}")
    
    all_reports = []
    
    # Check Ticks
    all_reports.extend(check_partitions(cache_root, "ticks"))
    
    # Check Candles (Standard 1m)
    all_reports.extend(check_partitions(cache_root, "candles", granularity=60))
    
    if not all_reports:
        logger.warning("No partitions found to audit.")
        return

    df = pd.DataFrame(all_reports)
    
    # Print Summary
    print("\n" + "="*80)
    print(f" INTEGRITY AUDIT REPORT: {symbol}")
    print("="*80)
    
    # Group by Status
    print("\nSummary by Status:")
    print(df.groupby("status")["file"].count())
    
    print(f"\nTotal Records: {df['records'].sum():,}")
    print(f"Total Size: {df['size_mb'].sum():.2f} MB")
    
    if "CORRUPT" in df["status"].values:
        print("\n❌ CRITICAL: CORRUPT FILES DETECTED")
        print(df[df["status"] == "CORRUPT"][["file", "status", "notes"]])
    else:
        print("\n✅ No file corruption detected.")

    if any("gaps" in str(n) for n in df["notes"]):
        print("\n⚠ Data Gaps Detected in some partitions (check logs/metadata for details).")
    
    # Detailed view for last 5 files
    print("\nRecent Partitions:")
    print(df.tail(5).to_string(index=False, columns=["file", "start", "end", "records", "status", "notes"]))
    print("\n" + "="*80)

if __name__ == "__main__":
    main()
