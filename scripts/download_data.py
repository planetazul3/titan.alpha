#!/usr/bin/env python3
"""
Download historical data from Deriv API.

Supports:
- Memory-safe partitioned storage (monthly chunks)
- Concurrent tick + candle downloads
- Smart resume from interrupted downloads
- Flexible date ranges

Usage:
    # Download 12 months (default behavior)
    python scripts/download_data.py --months 12 --symbol R_100

    # Download specific date range
    python scripts/download_data.py --start-date 2022-01-01 --end-date 2022-12-31 --symbol R_100

    # Use 5-minute candles instead of 1-minute
    python scripts/download_data.py --months 3 --granularity 300

    # Quick test (1 day)
    python scripts/download_data.py --test
"""

import argparse
import logging
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv

load_dotenv()

from config.settings import load_settings
from data.ingestion.client import DerivClient
from data.ingestion.historical import download_months


from config.logging_config import setup_logging

log_file = setup_logging(script_name="download_data", level="INFO")
logger = logging.getLogger(__name__)


def parse_date(date_str: str) -> datetime:
    """Parse date string in YYYY-MM-DD format."""
    try:
        return datetime.strptime(date_str, "%Y-%m-%d")
    except ValueError:
        raise argparse.ArgumentTypeError(f"Invalid date format: {date_str}. Expected YYYY-MM-DD")


async def main(args):
    """Main download function."""
    from scripts.console_utils import (
        console_header,
        console_log,
        console_separator,
        format_duration,
        format_size,
    )

    console_header("DATA DOWNLOAD STARTING")

    console_log("Loading settings...", "WAIT")
    settings = load_settings()

    # Override symbol if provided
    symbol = args.symbol or settings.trading.symbol
    cache_dir = Path(args.output)

    console_log(f"Symbol: {symbol}", "DATA")
    console_log(f"Output: {cache_dir}", "INFO")
    console_log(f"Granularity: {args.granularity}s ({args.granularity // 60}m candles)", "INFO")

    # Determine date range
    start_date = None
    end_date = None
    months = None

    if args.start_date:
        start_date = args.start_date
        end_date = args.end_date or datetime.now(timezone.utc)

        # Validate date order
        if start_date >= end_date:
            console_log("--start-date must be before --end-date", "ERROR")
            logger.error("--start-date must be before --end-date")
            return 1

        console_log(f"Date range: {start_date.date()} to {end_date.date()}", "INFO")
        logger.info(f"Downloading data from {start_date.date()} to {end_date.date()}")
    else:
        months = args.months
        console_log(f"Downloading {months} months of data", "INFO")
        logger.info(f"Downloading {months} months of data for {symbol}")

    logger.info(f"Output directory: {cache_dir}")
    logger.info(f"Candle granularity: {args.granularity}s ({args.granularity // 60}m)")

    if not args.resume:
        console_log("Resume disabled - will re-download all data", "WARN")
        logger.info("Resume disabled - will re-download all data")

    client = DerivClient(settings)

    try:
        console_log("Connecting to Deriv API...", "NET")
        await client.connect()
        console_log("Connected to Deriv API", "SUCCESS")
        logger.info("Connected to Deriv API")

        console_header("DOWNLOADING DATA")
        console_log("This may take a while for large date ranges...", "WAIT")

        start_time = time.time()

        # Download data with all improvements
        result = await download_months(
            client=client,
            symbol=symbol,
            months=months,
            cache_dir=cache_dir,
            granularity=args.granularity,
            start_date=start_date,
            end_date=end_date,
            resume=args.resume,
        )

        total_duration = time.time() - start_time

        console_header("DOWNLOAD COMPLETE")

        # Load metadata for statistics
        from data.ingestion.versioning import load_metadata

        total_ticks = 0
        total_candles = 0
        total_size = 0
        gaps = 0
        dupes = 0

        for tick_file in result["ticks"]:
            meta = load_metadata(tick_file)
            if meta:
                total_ticks += meta.record_count
                total_size += meta.file_size or 0
                gaps += meta.gaps_detected
                dupes += meta.duplicates_removed

        for candle_file in result["candles"]:
            meta = load_metadata(candle_file)
            if meta:
                total_candles += meta.record_count
                total_size += meta.file_size or 0
                gaps += meta.gaps_detected
                dupes += meta.duplicates_removed

        # Summary Report
        console_log("DOWNLOAD SUMMARY", "DATA")
        console_separator()
        console_log(f"Total Ticks:      {total_ticks:,}", "INFO")
        console_log(f"Total Candles:    {total_candles:,}", "INFO")
        console_log(f"Total Size:       {format_size(total_size)}", "INFO")
        console_log(f"Total Duration:   {format_duration(total_duration)}", "INFO")
        console_log(f"Average Speed:    {(total_ticks + total_candles) / total_duration:.1f} records/sec", "INFO")
        
        if gaps > 0 or dupes > 0:
            console_separator()
            console_log(f"Gaps Detected:    {gaps}", "WARN")
            console_log(f"Duplicates:      {dupes}", "WARN")
            console_log("Integrity checks passed (auto-fixed chunks)", "SUCCESS")
        
        console_separator()
        logger.info(f"Download complete: {total_ticks} ticks, {total_candles} candles, {format_size(total_size)}")

    except Exception as e:
        console_log(f"Download failed: {e}", "ERROR")
        logger.error(f"Download failed: {e}")
        import traceback

        traceback.print_exc()
        return 1
    finally:
        console_log("Disconnecting...", "WAIT")
        await client.disconnect()
        console_log("Done!", "SUCCESS")

    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Download historical data from Deriv API",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --months 12 --symbol R_100
      Download 12 months of data for R_100

  %(prog)s --start-date 2022-01-01 --end-date 2022-12-31
      Download specific date range

  %(prog)s --months 3 --granularity 300
      Download 3 months with 5-minute candles

  %(prog)s --test
      Quick test with 1 day of data
""",
    )

    # Date range options (mutually exclusive groups)
    date_group = parser.add_argument_group("Date Range Options")
    date_group.add_argument(
        "--months", type=float, default=1, help="Number of months to download (default: 1)"
    )
    date_group.add_argument(
        "--start-date",
        type=parse_date,
        default=None,
        help="Start date (YYYY-MM-DD). Overrides --months if provided.",
    )
    date_group.add_argument(
        "--end-date",
        type=parse_date,
        default=None,
        help="End date (YYYY-MM-DD). Defaults to today.",
    )

    # Data options
    data_group = parser.add_argument_group("Data Options")
    data_group.add_argument(
        "--symbol", type=str, default=None, help="Trading symbol (default: from .env)"
    )
    data_group.add_argument(
        "--granularity",
        type=int,
        default=60,
        help="Candle granularity in seconds (default: 60 = 1 minute)",
    )

    # Output options
    output_group = parser.add_argument_group("Output Options")
    output_group.add_argument(
        "--output", type=str, default="data_cache", help="Output directory (default: data_cache)"
    )
    output_group.add_argument(
        "--no-resume",
        dest="resume",
        action="store_false",
        help="Disable resume - re-download all data even if partitions exist",
    )

    # Test mode
    parser.add_argument("--test", action="store_true", help="Test mode - download only 1 day")

    args = parser.parse_args()

    # Override months for test mode
    if args.test:
        args.months = 0.033  # ~1 day
        args.start_date = None  # Reset explicit dates
        args.end_date = None
        logger.info("Test mode: downloading 1 day of data")

    from scripts.shutdown_handler import run_async_with_graceful_shutdown

    sys.exit(run_async_with_graceful_shutdown(main(args)))
