"""
Historical data downloader for Deriv API.

Downloads ticks and candles with pagination, stores in Parquet format.
Includes post-download integrity checks and versioning metadata.

Improvements:
- Memory-safe: Streams data to disk in monthly partitions
- Smart resume: Detects existing partitions and continues from last epoch
- Concurrent-ready: Async design supports parallel tick/candle downloads
"""

import asyncio
import logging
import os
import time
from collections.abc import Callable
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, cast

import numpy as np

from data.ingestion.integrity import IntegrityChecker
from data.ingestion.versioning import (
    DatasetMetadata,
    compute_checksum,
    create_metadata,
    load_metadata,
    save_metadata,
)

logger = logging.getLogger(__name__)

# Try to import optional dependencies
try:
    import pandas as pd

    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False
    logger.warning("pandas not installed - Parquet storage disabled")


def epoch_to_month_key(epoch: int) -> str:
    """Convert epoch to YYYY-MM format for partitioning."""
    dt = datetime.fromtimestamp(epoch, tz=timezone.utc)
    return dt.strftime("%Y-%m")


def get_month_boundaries(
    start_time: datetime, end_time: datetime
) -> list[tuple[datetime, datetime]]:
    """
    Split a time range into monthly boundaries.

    Returns list of (month_start, month_end) tuples.
    """
    boundaries = []
    current = start_time.replace(day=1, hour=0, minute=0, second=0, microsecond=0)

    while current < end_time:
        # Calculate end of this month
        if current.month == 12:
            next_month = current.replace(year=current.year + 1, month=1)
        else:
            next_month = current.replace(month=current.month + 1)

        month_start = max(current, start_time)
        month_end = min(next_month, end_time)

        boundaries.append((month_start, month_end))
        current = next_month

    return boundaries


class PartitionedDownloader:
    """
    Memory-safe downloader that streams data to disk in monthly partitions.

    Supports:
    - Monthly partitioning to prevent OOM on large downloads
    - Smart resume from existing partitions
    - Integrity checks per partition

    Directory structure:
        data_cache/{symbol}/ticks/{YYYY-MM}.parquet
        data_cache/{symbol}/candles_{granularity}/{YYYY-MM}.parquet
    """

    # API limits (conservative estimates)
    MAX_TICKS_PER_REQUEST = 5000
    MAX_CANDLES_PER_REQUEST = 5000
    CHUNK_SIZE = 100000  # Records per memory chunk before flushing/clearing

    def __init__(self, client, symbol: str, cache_dir: Path):
        """
        Args:
            client: Connected DerivClient instance
            symbol: Trading symbol (e.g., "R_100")
            cache_dir: Base directory for partitioned data
        """
        self.client = client
        self.symbol = symbol
        self.cache_dir = Path(cache_dir)

        # Create symbol-specific directories
        self.symbol_dir = self.cache_dir / symbol
        self.symbol_dir.mkdir(parents=True, exist_ok=True)

    def _get_partition_dir(self, data_type: str, granularity: int | None = None) -> Path:
        """Get the partition directory for a data type."""
        if data_type == "ticks":
            partition_dir = self.symbol_dir / "ticks"
        elif data_type == "candles":
            partition_dir = self.symbol_dir / f"candles_{granularity}"
        else:
            raise ValueError(f"Unknown data_type: {data_type}")

        partition_dir.mkdir(parents=True, exist_ok=True)
        return partition_dir

    def _get_partition_path(
        self, data_type: str, month_key: str, granularity: int | None = None
    ) -> Path:
        """Get the full path for a partition file."""
        partition_dir = self._get_partition_dir(data_type, granularity)
        return partition_dir / f"{month_key}.parquet"

    def find_resume_point(self, data_type: str, granularity: int | None = None) -> int | None:
        """
        Find the latest epoch from existing partitions for resume capability.

        Returns:
            Latest end_epoch + 1 if partitions exist, None otherwise.
        """
        partition_dir = self._get_partition_dir(data_type, granularity)

        latest_epoch = None
        for parquet_file in partition_dir.glob("*.parquet"):
            metadata = load_metadata(parquet_file)
            if metadata and metadata.end_epoch:
                if latest_epoch is None or metadata.end_epoch > latest_epoch:
                    latest_epoch = metadata.end_epoch

        if latest_epoch:
            logger.info(f"Found existing data up to epoch {latest_epoch}, will resume from there")
            return latest_epoch + 1

        return None

    def _save_partition(
        self,
        data: list[dict],
        data_type: str,
        month_key: str,
        granularity: int | None = None,
        download_duration: float | None = None,
    ) -> tuple[Path, DatasetMetadata]:
        """
        Save a monthly partition with integrity checks and metadata.
        """
        if not HAS_PANDAS:
            raise RuntimeError("pandas required for Parquet storage")

        if not data:
            raise ValueError("Cannot save an empty partition: no data provided.")

        # Run integrity checks
        checker = IntegrityChecker()
        if data_type == "ticks":
            cleaned_data, report = checker.generate_report(
                data, data_type="ticks", auto_dedupe=True
            )
        else:
            cleaned_data, report = checker.generate_report(
                data, data_type="candles", granularity=granularity, auto_dedupe=True
            )

        logger.info(f"Partition {month_key}: {report}")

        # Save to Parquet
        filepath = self._get_partition_path(data_type, month_key, granularity)
        df = pd.DataFrame(cleaned_data)
        
        # Optimize dtypes for RAM if possible
        if data_type == "ticks":
            df["epoch"] = df["epoch"].astype("int64")
            df["quote"] = df["quote"].astype("float64")
        elif data_type == "candles":
            df["epoch"] = df["epoch"].astype("int64")
            for col in ["open", "high", "low", "close"]:
                if col in df.columns:
                    df[col] = df[col].astype("float64")

        df.to_parquet(filepath, index=False)
        file_size = os.path.getsize(filepath)
        logger.info(f"Saved {len(cleaned_data)} records to {filepath} ({file_size / 1024 / 1024:.2f} MB)")

        # Create and save metadata
        metadata = create_metadata(
            symbol=self.symbol,
            data_type=data_type,
            records=cleaned_data,
            granularity=granularity,
            gaps_detected=report.gaps_detected,
            duplicates_removed=report.duplicates_removed,
            download_duration=download_duration,
        )
        metadata.file_size = file_size
        save_metadata(filepath, metadata)

        return filepath, metadata

    async def download_ticks_partitioned(
        self,
        start_time: datetime,
        end_time: datetime,
        progress_callback: Callable | None = None,
        resume: bool = True,
    ) -> list[Path]:
        """
        Download tick data with monthly partitioning.

        Streams data to disk monthly to prevent OOM on large downloads.

        Args:
            start_time: Start of range (UTC)
            end_time: End of range (UTC)
            progress_callback: Optional callback(downloaded, total_estimate)
            resume: If True, resume from existing partitions

        Returns:
            List of partition file paths created
        """
        partition_files = []

        # Check for resume point
        if resume:
            resume_epoch = self.find_resume_point("ticks")
            if resume_epoch and resume_epoch > int(start_time.timestamp()):
                new_start = datetime.fromtimestamp(resume_epoch, tz=timezone.utc)
                if new_start < end_time:
                    start_time = new_start
                    logger.info(f"Resuming tick download from {start_time}")
                else:
                    logger.info("Tick data is already up to date, skipping download")
                    return []

        # Get monthly boundaries
        boundaries = get_month_boundaries(start_time, end_time)
        total_months = len(boundaries)

        for month_idx, (month_start, month_end) in enumerate(boundaries):
            month_key = month_start.strftime("%Y-%m")
            logger.info(f"Downloading ticks for {month_key} ({month_idx + 1}/{total_months})")

            partition_start = time.time()
            month_ticks = []

            current_end = int(month_end.timestamp())
            start_epoch = int(month_start.timestamp())
            total_seconds = current_end - start_epoch

            while current_end > start_epoch:
                retry_count = 0
                max_retries = 5

                while retry_count < max_retries:
                    try:
                        # Add timeout to prevent indefinite hangs
                        response = await asyncio.wait_for(
                            self.client.api.ticks_history(
                                {
                                    "ticks_history": self.symbol,
                                    "count": self.MAX_TICKS_PER_REQUEST,
                                    "end": current_end,
                                    "start": start_epoch,
                                    "style": "ticks",
                                }
                            ),
                            timeout=30.0,  # 30 second timeout
                        )
                        break  # Success, exit retry loop

                    except asyncio.TimeoutError:
                        retry_count += 1
                        wait_time = min(2**retry_count, 30)  # Exponential backoff, max 30s
                        logger.warning(
                            f"Tick request timed out (attempt {retry_count}/{max_retries}), retrying in {wait_time}s..."
                        )
                        await asyncio.sleep(wait_time)
                        continue

                    except Exception as e:
                        retry_count += 1
                        wait_time = min(2**retry_count, 30)
                        logger.error(
                            f"Error downloading ticks: {e} (attempt {retry_count}/{max_retries})"
                        )
                        await asyncio.sleep(wait_time)
                        continue

                if retry_count >= max_retries:
                    logger.error(
                        f"Max retries exceeded for ticks at epoch {current_end}, skipping to next batch"
                    )
                    current_end -= 3600  # Skip 1 hour of data if stuck
                    continue

                history = response.get("history", {})
                prices = history.get("prices", [])
                times = history.get("times", [])

                if not prices:
                    break

                # Add ticks
                for price, epoch in zip(prices, times, strict=False):
                    month_ticks.append({"epoch": epoch, "quote": price})

                # Move window back
                if times:
                    current_end = min(times) - 1
                    downloaded_seconds = int(month_end.timestamp()) - current_end

                    if progress_callback:
                        # Report overall progress across all months
                        # Ensure month_progress is within [0, 1] to avoid visual glitches
                        if total_seconds > 0:
                            month_progress = max(0.0, min(1.0, downloaded_seconds / total_seconds))
                        else:
                            month_progress = 1.0
                            
                        overall = (month_idx + month_progress) / total_months
                        progress_callback(overall, 1.0)

                # Save intermediate chunks if necessary to save RAM
                if len(month_ticks) >= self.CHUNK_SIZE:
                    logger.debug(f"Month {month_key} reached {len(month_ticks)} ticks, keeping in memory (chunking logic can be enhanced here if needed)")

                # Adaptive Rate Limiting: 
                # If we got a full response, we might be hitting limits, so sleep just enough.
                # If we got less than full, we can go slightly faster.
                if len(prices) >= self.MAX_TICKS_PER_REQUEST:
                    await asyncio.sleep(0.3)
                else:
                    await asyncio.sleep(0.1)

            # Sort and save partition
            month_ticks.sort(key=lambda x: x["epoch"])
            partition_duration = time.time() - partition_start

            if month_ticks:
                filepath, _ = self._save_partition(
                    month_ticks, "ticks", month_key, download_duration=partition_duration
                )
                partition_files.append(filepath)

            logger.info(
                f"Completed {month_key}: {len(month_ticks)} ticks in {partition_duration:.1f}s"
            )

        return partition_files

    async def download_candles_partitioned(
        self,
        start_time: datetime,
        end_time: datetime,
        granularity: int = 60,
        progress_callback: Callable | None = None,
        resume: bool = True,
    ) -> list[Path]:
        """
        Download candle data with monthly partitioning.

        Args:
            start_time: Start of range (UTC)
            end_time: End of range (UTC)
            granularity: Candle interval in seconds (60=1m)
            progress_callback: Optional callback(downloaded, total_estimate)
            resume: If True, resume from existing partitions

        Returns:
            List of partition file paths created
        """
        partition_files = []

        # Check for resume point
        if resume:
            resume_epoch = self.find_resume_point("candles", granularity)
            if resume_epoch and resume_epoch > int(start_time.timestamp()):
                new_start = datetime.fromtimestamp(resume_epoch, tz=timezone.utc)
                if new_start < end_time:
                    start_time = new_start
                    logger.info(f"Resuming candle download from {start_time}")
                else:
                    logger.info("Candle data is already up to date, skipping download")
                    return []

        # Get monthly boundaries
        boundaries = get_month_boundaries(start_time, end_time)
        total_months = len(boundaries)

        for month_idx, (month_start, month_end) in enumerate(boundaries):
            month_key = month_start.strftime("%Y-%m")
            logger.info(f"Downloading candles for {month_key} ({month_idx + 1}/{total_months})")

            partition_start = time.time()
            month_candles = []

            current_end = int(month_end.timestamp())
            start_epoch = int(month_start.timestamp())
            total_candles_estimate = (current_end - start_epoch) // granularity

            while current_end > start_epoch:
                retry_count = 0
                max_retries = 5

                while retry_count < max_retries:
                    try:
                        # Add timeout to prevent indefinite hangs
                        response = await asyncio.wait_for(
                            self.client.api.ticks_history(
                                {
                                    "ticks_history": self.symbol,
                                    "count": self.MAX_CANDLES_PER_REQUEST,
                                    "end": current_end,
                                    "start": start_epoch,
                                    "style": "candles",
                                    "granularity": granularity,
                                }
                            ),
                            timeout=30.0,  # 30 second timeout
                        )
                        break  # Success, exit retry loop

                    except asyncio.TimeoutError:
                        retry_count += 1
                        wait_time = min(2**retry_count, 30)
                        logger.warning(
                            f"Candle request timed out (attempt {retry_count}/{max_retries}), retrying in {wait_time}s..."
                        )
                        await asyncio.sleep(wait_time)
                        continue

                    except Exception as e:
                        retry_count += 1
                        wait_time = min(2**retry_count, 30)
                        logger.error(
                            f"Error downloading candles: {e} (attempt {retry_count}/{max_retries})"
                        )
                        await asyncio.sleep(wait_time)
                        continue

                if retry_count >= max_retries:
                    logger.error(
                        f"Max retries exceeded for candles at epoch {current_end}, skipping to next batch"
                    )
                    current_end -= 3600  # Skip 1 hour of data if stuck
                    continue

                candles = response.get("candles", [])

                if not candles:
                    break

                month_candles.extend(candles)

                # Move window back
                if candles:
                    current_end = min(c["epoch"] for c in candles) - 1

                    if progress_callback:
                        # Report overall progress
                        if total_candles_estimate > 0:
                            month_progress = max(0.0, min(1.0, len(month_candles) / total_candles_estimate))
                        else:
                            month_progress = 1.0
                            
                        overall = (month_idx + month_progress) / total_months
                        progress_callback(overall, 1.0)

                # Adaptive Rate Limiting
                if len(candles) >= self.MAX_CANDLES_PER_REQUEST:
                    await asyncio.sleep(0.3)
                else:
                    await asyncio.sleep(0.1)

            # Sort and save partition
            month_candles.sort(key=lambda x: x["epoch"])
            partition_duration = time.time() - partition_start

            if month_candles:
                filepath, _ = self._save_partition(
                    month_candles,
                    "candles",
                    month_key,
                    granularity=granularity,
                    download_duration=partition_duration,
                )
                partition_files.append(filepath)

            logger.info(
                f"Completed {month_key}: {len(month_candles)} candles in {partition_duration:.1f}s"
            )

        return partition_files

    def load_all_partitions(self, data_type: str, granularity: int | None = None) -> pd.DataFrame:
        """
        Load all partitions as a single DataFrame.

        Use sparingly - for large datasets, iterate over partitions instead.
        """
        if not HAS_PANDAS:
            raise RuntimeError("pandas required for Parquet storage")

        partition_dir = self._get_partition_dir(data_type, granularity)
        parquet_files = sorted(partition_dir.glob("*.parquet"))

        if not parquet_files:
            return pd.DataFrame()

        dfs = [pd.read_parquet(f) for f in parquet_files]
        return pd.concat(dfs, ignore_index=True)


# -----------------------------------------------------------------------------
# Legacy API - Maintained for backward compatibility
# -----------------------------------------------------------------------------
# Deprecated HistoricalDataDownloader removed in Iteration 1 Cleanup


async def download_months(
    client,
    symbol: str,
    months: float | None = None,
    cache_dir: Path | None = None,
    granularity: int = 60,
    compute_checksums: bool = False,
    start_date: datetime | None = None,
    end_date: datetime | None = None,
    resume: bool = True,
) -> dict[str, Any]:
    """
    Download historical data with memory-safe partitioning and concurrency.

    Downloads both ticks and candles concurrently, saving to monthly
    partitions to prevent OOM on large downloads.

    Args:
        client: Connected DerivClient
        symbol: Trading symbol
        months: Number of months to download (if start_date not provided)
        cache_dir: Directory for cache
        granularity: Candle granularity in seconds
        compute_checksums: Whether to compute SHA-256 checksums
        start_date: Explicit start date (overrides months)
        end_date: Explicit end date (defaults to now)
        resume: If True, resume from existing partitions

    Returns:
        Dict with 'ticks', 'candles' partition lists
    """
    downloader = PartitionedDownloader(client, symbol, cache_dir or Path("data_cache"))

    # Determine time range
    if end_date is None:
        end_time = datetime.now(timezone.utc)
    else:
        end_time = end_date if end_date.tzinfo else end_date.replace(tzinfo=timezone.utc)

    if start_date is not None:
        start_time = start_date if start_date.tzinfo else start_date.replace(tzinfo=timezone.utc)
    elif months is not None:
        start_time = end_time - timedelta(days=30 * months)
    else:
        raise ValueError("Either 'months' or 'start_date' must be provided")

    logger.info(f"Downloading data from {start_time} to {end_time}")
    logger.info(f"Symbol: {symbol}, Granularity: {granularity}s")

    # Progress tracking
    tick_progress = [0.0]
    candle_progress = [0.0]

    def tick_progress_cb(current, total):
        tick_progress[0] = current / total if total > 0 else 1.0
        print(
            f"\r  Ticks: {tick_progress[0] * 100:.1f}%  |  Candles: {candle_progress[0] * 100:.1f}%",
            end="",
        )

    def candle_progress_cb(current, total):
        candle_progress[0] = current / total if total > 0 else 1.0
        print(
            f"\r  Ticks: {tick_progress[0] * 100:.1f}%  |  Candles: {candle_progress[0] * 100:.1f}%",
            end="",
        )

    print("Starting concurrent download...")

    # Run tick and candle downloads concurrently
    tick_task = downloader.download_ticks_partitioned(
        start_time, end_time, progress_callback=tick_progress_cb, resume=resume
    )

    candle_task = downloader.download_candles_partitioned(
        start_time,
        end_time,
        granularity=granularity,
        progress_callback=candle_progress_cb,
        resume=resume,
    )

    # Execute concurrently
    tick_files, candle_files = await asyncio.gather(tick_task, candle_task)

    print()  # Newline after progress
    logger.info(
        f"Download complete: {len(tick_files)} tick partitions, {len(candle_files)} candle partitions"
    )

    return {
        "ticks": tick_files,
        "candles": candle_files,
        "tick_dir": downloader._get_partition_dir("ticks"),
        "candle_dir": downloader._get_partition_dir("candles", granularity),
    }
