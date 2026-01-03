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
from typing import Any

from data.ingestion.integrity import IntegrityChecker
from data.ingestion.versioning import (
    DatasetMetadata,
    create_metadata,
    get_metadata_path,
    load_metadata,
    save_metadata,
)
from data.validation.schemas import TickSchema, CandleSchema

logger = logging.getLogger(__name__)

# Try to import optional dependencies
try:
    import pandas as pd

    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False
    logger.warning("pandas not installed - Parquet storage disabled")

try:
    import pyarrow as pa
    import pyarrow.parquet as pq

    HAS_PYARROW = True
except ImportError:
    HAS_PYARROW = False
    logger.warning("pyarrow not installed - Incremental storage disabled")


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


class TokenBucket:
    """
    Token Bucket algorithm for rate limiting.
    Allows for bursts while maintaining a steady average rate.
    """
    def __init__(self, tokens_per_second: float, burst_multiplier: float = 1.0):
        self.capacity = tokens_per_second * burst_multiplier
        self.tokens = self.capacity
        self.refill_rate = tokens_per_second
        self.last_refill = time.time()

    async def consume(self, amount: float = 1.0):
        while self.tokens < amount:
            self._refill()
            if self.tokens < amount:
                # Wait for just enough time to get one token or the remaining amount
                await asyncio.sleep(max(0.01, (amount - self.tokens) / self.refill_rate))

        self.tokens -= amount

    def _refill(self):
        now = time.time()
        elapsed = now - self.last_refill
        self.tokens = min(self.capacity, self.tokens + elapsed * self.refill_rate)
        self.last_refill = now


class IncrementalWriter:
    """Helper to write Parquet files incrementally using PyArrow."""
    def __init__(self, filepath: Path, data_type: str):
        self.filepath = filepath
        self.data_type = data_type
        self.writer = None
        self.schema = None
        self._temp_path = filepath.with_suffix(".inc.parquet")

    def write_chunk(self, data: list[dict]):
        if not data:
            return

        df = pd.DataFrame(data)
        # Optimize dtypes
        if self.data_type == "ticks":
            # Cast for mypy - we know these cols exist in dicts converted to df
            df["epoch"] = df["epoch"].astype("int64")
            df["quote"] = df["quote"].astype("float64")
        elif self.data_type == "candles":
            df["epoch"] = df["epoch"].astype("int64")
            for col in ["open", "high", "low", "close"]:
                if col in df.columns:
                    df[col] = df[col].astype("float64")

        table = pa.Table.from_pandas(df, preserve_index=False)

        if self.writer is None:
            self.schema = table.schema
            self.writer = pq.ParquetWriter(self._temp_path, self.schema)

        if self.writer is not None:
            self.writer.write_table(table)

    def close(self):
        if self.writer:
            self.writer.close()
            self.writer = None
        return self._temp_path if self._temp_path.exists() else None

    def cleanup(self):
        self.close()
        if self._temp_path.exists():
            self._temp_path.unlink()


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

        self.symbol_dir = self.cache_dir / symbol
        self.symbol_dir.mkdir(parents=True, exist_ok=True)

        # Rate limiter: 5 requests per second (Deriv allows ~10-20, so 5 is safe)
        self.rate_limiter = TokenBucket(tokens_per_second=5.0, burst_multiplier=2.0)

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
        Verifies checksums of existing partitions and checks for continuity gaps.

        Returns:
            Latest end_epoch + 1 if partitions exist, are valid, and continuous.
            If a gap is detected, returns the epoch after the last continuous partition.
            None if no valid partitions exist.
        """
        from data.ingestion.versioning import verify_checksum

        partition_dir = self._get_partition_dir(data_type, granularity)

        # Collect all valid partitions with their metadata
        valid_partitions: list[tuple[Path, DatasetMetadata]] = []

        # Sort partitions by name (YYYY-MM) chronologically
        parquet_files = sorted(partition_dir.glob("*.parquet"))

        for parquet_file in parquet_files:
            metadata = load_metadata(parquet_file)
            if not metadata or not metadata.end_epoch or not metadata.start_epoch:
                logger.debug(f"Skipping {parquet_file.name}: missing metadata or epoch info")
                continue

            # Verify checksum if present in metadata
            if metadata.sha256:
                if not verify_checksum(parquet_file, metadata.sha256):
                    logger.warning(f"Checksum validation failed for {parquet_file.name}, stopping resume chain")
                    break  # Stop at first invalid partition

            valid_partitions.append((parquet_file, metadata))

        if not valid_partitions:
            return None

        # Check continuity between sequential partitions
        # Max allowed gap: 2x granularity for candles, 60 seconds for ticks
        max_gap = (granularity * 2) if granularity else 60

        last_continuous_idx = 0
        for i in range(1, len(valid_partitions)):
            prev_file, prev_meta = valid_partitions[i - 1]
            curr_file, curr_meta = valid_partitions[i]

            gap = curr_meta.start_epoch - prev_meta.end_epoch

            if gap > max_gap:
                logger.warning(
                    f"Gap detected between {prev_file.name} (end: {prev_meta.end_epoch}) "
                    f"and {curr_file.name} (start: {curr_meta.start_epoch}): {gap}s gap"
                )
                # Resume from the end of the last continuous partition before the gap
                resume_point = prev_meta.end_epoch + 1
                logger.info(f"Resuming from gap point: epoch {resume_point}")
                return resume_point

            last_continuous_idx = i

        # All partitions are continuous - resume from the latest
        latest_file, latest_meta = valid_partitions[last_continuous_idx]
        logger.info(f"Found valid continuous data in {latest_file.name} up to epoch {latest_meta.end_epoch}")
        return latest_meta.end_epoch + 1

    def _save_partition(
        self,
        data: list[dict] | pd.DataFrame,
        data_type: str,
        month_key: str,
        granularity: int | None = None,
        download_duration: float | None = None,
        incremental_file: Path | None = None,
    ) -> tuple[Path, DatasetMetadata]:
        """
        Save a monthly partition with integrity checks and metadata.
        Uses the Write-Rename pattern for atomic storage.
        If incremental_file is provided, it merges it with data.
        Accepts both list of dicts and pandas DataFrames.
        """
        if not HAS_PANDAS:
            raise RuntimeError("pandas required for Parquet storage")

        # Convert to DataFrame if needed
        if isinstance(data, list):
            df = pd.DataFrame(data)
        else:
            df = data

        # Merge incremental data if it exists
        if incremental_file and incremental_file.exists():
            try:
                inc_df = pd.read_parquet(incremental_file)
                df = pd.concat([inc_df, df], ignore_index=True)
                # Cleanup temp inc file
                incremental_file.unlink()
            except Exception as e:
                logger.error(f"Failed to read incremental file {incremental_file}: {e}")

        if df.empty:
            raise ValueError("Cannot save an empty partition: no data provided.")

        # Sort by epoch and deduplicate
        df = df.sort_values("epoch").drop_duplicates(subset=["epoch"], keep="first")

        # Create temporary path for atomic write
        filepath = self._get_partition_path(data_type, month_key, granularity)
        temp_filepath = filepath.with_suffix(".tmp.parquet")

        try:
            # Run integrity check on final merged data
            checker = IntegrityChecker()
            cleaned_data, report = checker.generate_report(
                df, data_type=data_type, granularity=granularity
            )

            # Use cleaned data (which might be the same DataFrame or a new one)
            df = cleaned_data
            
            # Ensure schema compliance for missing optional columns
            if data_type == "candles" and "volume" not in df.columns:
                # Deriv API sometimes omits volume for certain instruments/granularities
                df["volume"] = 0.0

            # --- STRICT SCHEMA VALIDATION ---
            try:
                if data_type == "ticks":
                    TickSchema.validate(df, lazy=True)
                elif data_type == "candles":
                    CandleSchema.validate(df, lazy=True)
            except Exception as schema_err:
                logger.error(f"Schema validation failed for {month_key} ({data_type}): {schema_err}")
                # We raise to prevent corrupt data from touching disk
                raise

            # Optimize dtypes before saving
            # Cast df to Any to bypass mypy's confusion about DataFrame/list Union in this scope
            # cleaned_data returns list|Any(df), and we assigned it to df
            df_typed: Any = df
            df_typed["epoch"] = df_typed["epoch"].astype("int64")
            if data_type == "ticks":
                df_typed["quote"] = df_typed["quote"].astype("float64")
            elif data_type == "candles":
                for col in ["open", "high", "low", "close"]:
                    if col in df_typed.columns:
                        df_typed[col] = df_typed[col].astype("float64")

            # Write to temporary file with compression
            df_typed.to_parquet(temp_filepath, index=False, engine="pyarrow", compression="zstd")
            file_size = os.path.getsize(temp_filepath)

            # Create metadata
            from data.ingestion.versioning import compute_checksum

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

            # Compute checksum for integrity tracking
            metadata.sha256 = compute_checksum(temp_filepath)

            # Save metadata first (if it fails, parquet is still temp)
            save_metadata(temp_filepath, metadata)

            # Atomic swap
            os.rename(temp_filepath, filepath)
            os.rename(get_metadata_path(temp_filepath), get_metadata_path(filepath))

            logger.info(f"Saved {len(cleaned_data)} records to {filepath} ({file_size / 1024 / 1024:.2f} MB)")
            return filepath, metadata

        except Exception as e:
            logger.error(f"Failed to save partition {month_key} atomically: {e}")
            if temp_filepath.exists():
                temp_filepath.unlink()
            meta_temp = get_metadata_path(temp_filepath)
            if meta_temp.exists():
                meta_temp.unlink()
            raise

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

        # Get monthly boundaries
        boundaries = get_month_boundaries(start_time, end_time)
        total_months = len(boundaries)

        for month_idx, (month_start, month_end) in enumerate(boundaries):
            month_key = month_start.strftime("%Y-%m")

            # Per-month resume check
            filepath = self._get_partition_path("ticks", month_key)
            if resume and filepath.exists():
                # Verify if it's complete
                metadata = load_metadata(filepath)
                if metadata and metadata.end_epoch >= int(month_end.timestamp()) - 1:
                    logger.info(f"Partition {month_key} already exists and is complete, skipping")
                    partition_files.append(filepath)
                    continue
                else:
                    logger.info(f"Partition {month_key} exists but may be incomplete, re-downloading")

            logger.info(f"Downloading ticks for {month_key} ({month_idx + 1}/{total_months})")

            partition_start = time.time()
            month_ticks = []

            # Setup incremental writer if needed
            inc_file_path = self._get_partition_path("ticks", month_key)
            inc_writer = IncrementalWriter(inc_file_path, "ticks") if HAS_PYARROW else None

            current_end = int(month_end.timestamp())
            start_epoch = int(month_start.timestamp())

            while current_end > start_epoch:
                # Rate limit requests
                await self.rate_limiter.consume(1.0)

                retry_count = 0
                max_retries = 5

                while retry_count < max_retries:
                    try:
                        # Backward download within the month range
                        # Providing BOTH start and end is more robust for historical data
                        response = await asyncio.wait_for(
                            self.client.api.ticks_history(
                                {
                                    "ticks_history": self.symbol,
                                    "start": start_epoch,
                                    "end": str(current_end),
                                    "style": "ticks",
                                    "count": self.MAX_TICKS_PER_REQUEST,
                                }
                            ),
                            timeout=30.0,
                        )
                        break  # Success

                    except asyncio.TimeoutError:
                        retry_count += 1
                        wait_time = min(2**retry_count, 30)
                        logger.warning(f"Timeout (attempt {retry_count}/{max_retries})")
                        await asyncio.sleep(wait_time)
                        continue

                    except Exception as e:
                        retry_count += 1
                        wait_time = min(2**retry_count, 30)
                        logger.error(f"Error: {e} (attempt {retry_count}/{max_retries})")

                        # Fix: Force reconnection on error to ensure healthy socket
                        try:
                            logger.info("Attempting proactive reconnection...")
                            await self.client.disconnect()
                            await self.client.connect()
                        except Exception as rec_e:
                             logger.warning(f"Reconnection attempt failed: {rec_e}")

                        await asyncio.sleep(wait_time)
                        continue

                if retry_count >= max_retries:
                    logger.error(f"Max retries exceeded at {current_end}, skipping")
                    current_end -= 3600
                    continue

                history = response.get("history", {})
                prices = history.get("prices", [])
                times = history.get("times", [])

                logger.debug(f"API returned {len(prices)} ticks for {self.symbol}. First: {times[0] if times else 'N/A'}, Last: {times[-1] if times else 'N/A'}")

                if not prices:
                    break

                # Add ticks
                for price, epoch in zip(prices, times, strict=False):
                    if epoch < start_epoch:
                        continue
                    month_ticks.append({"epoch": epoch, "quote": price})

                # Move window backward
                if times:
                    new_end = min(times) - 1
                    if new_end >= current_end: # Sanity check to prevent infinite loops
                        break
                    current_end = new_end

                    if progress_callback:
                        # Progress is approximate for backward download
                        elapsed_seconds = int(month_end.timestamp()) - current_end
                        total_range = int(month_end.timestamp()) - start_epoch
                        month_progress = min(1.0, elapsed_seconds / total_range) if total_range > 0 else 1.0

                        overall = (month_idx + month_progress) / total_months
                        progress_callback(overall, 1.0, current_count=len(month_ticks))

                # Memory risk check: Flush to intermediate Parquet if chunk size exceeded
                if len(month_ticks) >= self.CHUNK_SIZE and inc_writer:
                    logger.info(f"Flushing {len(month_ticks)} ticks for {month_key} to disk to save RAM")
                    try:
                        inc_writer.write_chunk(month_ticks)
                        month_ticks = []  # Clear memory
                    except Exception as e:
                        logger.error(f"Failed to write incremental chunk: {e}")

                # Adaptive Rate Limiting is now handled by self.rate_limiter
                pass

            # Finalize partition
            inc_path = inc_writer.close() if inc_writer else None
            partition_duration = time.time() - partition_start

            try:
                filepath, _ = self._save_partition(
                    month_ticks,
                    "ticks",
                    month_key,
                    download_duration=partition_duration,
                    incremental_file=inc_path,
                )
                partition_files.append(filepath)
            except Exception as e:
                logger.error(f"Failed to finalize partition {month_key}: {e}")
                if inc_writer:
                    inc_writer.cleanup()

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

        # Get monthly boundaries
        boundaries = get_month_boundaries(start_time, end_time)
        total_months = len(boundaries)

        for month_idx, (month_start, month_end) in enumerate(boundaries):
            month_key = month_start.strftime("%Y-%m")

            # Per-month resume check
            filepath = self._get_partition_path("candles", month_key, granularity)
            if resume and filepath.exists():
                metadata = load_metadata(filepath)
                if metadata and metadata.end_epoch >= int(month_end.timestamp()) - granularity:
                    logger.info(f"Partition {month_key} (candles) already exists and is complete, skipping")
                    partition_files.append(filepath)
                    continue
                else:
                    logger.info(f"Partition {month_key} exists but may be incomplete, re-downloading")

            logger.info(f"Downloading candles for {month_key} ({month_idx + 1}/{total_months})")

            partition_start = time.time()
            month_candles = []

            # Setup incremental writer if needed
            inc_file_path = self._get_partition_path("candles", month_key, granularity)
            inc_writer = IncrementalWriter(inc_file_path, "candles") if HAS_PYARROW else None

            current_end = int(month_end.timestamp())
            start_epoch = int(month_start.timestamp())

            while current_end > start_epoch:
                # Rate limit requests
                await self.rate_limiter.consume(1.0)

                retry_count = 0
                max_retries = 5

                while retry_count < max_retries:
                    try:
                        # Backward download for candles with explicit start/end
                        response = await asyncio.wait_for(
                            self.client.api.ticks_history(
                                {
                                    "ticks_history": self.symbol,
                                    "start": start_epoch,
                                    "end": str(current_end),
                                    "style": "candles",
                                    "granularity": granularity,
                                    "count": self.MAX_CANDLES_PER_REQUEST,
                                }
                            ),
                            timeout=30.0,
                        )
                        break  # Success

                    except asyncio.TimeoutError:
                        retry_count += 1
                        wait_time = min(2**retry_count, 30)
                        logger.warning(f"Timeout (attempt {retry_count}/{max_retries})")
                        await asyncio.sleep(wait_time)
                        continue

                    except Exception as e:
                        retry_count += 1
                        wait_time = min(2**retry_count, 30)
                        logger.error(f"Error: {e} (attempt {retry_count}/{max_retries})")

                        # Fix: Force reconnection on error to ensure healthy socket
                        try:
                            logger.info("Attempting proactive reconnection...")
                            await self.client.disconnect()
                            await self.client.connect()
                        except Exception as rec_e:
                             logger.warning(f"Reconnection attempt failed: {rec_e}")

                        await asyncio.sleep(wait_time)
                        continue

                if retry_count >= max_retries:
                    logger.error(f"Max retries exceeded for candles at {current_end}, skipping")
                    current_end -= granularity * 100
                    continue

                candles = response.get("candles", [])

                logger.debug(f"API returned {len(candles)} candles for {self.symbol}. First epoch: {candles[0]['epoch'] if candles else 'N/A'}")

                if not candles:
                    break

                # Add candles, clipping to month start
                for candle in candles:
                    if candle["epoch"] < start_epoch:
                        continue
                    month_candles.append(candle)

                # Move window backward
                if candles:
                    new_end = min(c["epoch"] for c in candles) - 1
                    if new_end >= current_end:
                        break
                    current_end = new_end

                    if progress_callback:
                        elapsed_seconds = int(month_end.timestamp()) - current_end
                        total_range = int(month_end.timestamp()) - start_epoch
                        month_progress = min(1.0, elapsed_seconds / total_range) if total_range > 0 else 1.0

                        overall = (month_idx + month_progress) / total_months
                        progress_callback(overall, 1.0, current_count=len(month_candles))

                # Memory risk check: Flush to intermediate Parquet if chunk size exceeded
                if len(month_candles) >= self.CHUNK_SIZE and inc_writer:
                    logger.info(f"Flushing {len(month_candles)} candles for {month_key} to disk to save RAM")
                    try:
                        inc_writer.write_chunk(month_candles)
                        month_candles = []  # Clear memory
                    except Exception as e:
                        logger.error(f"Failed to write incremental chunk: {e}")

                # Adaptive Rate Limiting is now handled by self.rate_limiter
                pass

            # Finalize partition
            inc_path = inc_writer.close() if inc_writer else None
            partition_duration = time.time() - partition_start

            try:
                filepath, _ = self._save_partition(
                    month_candles,
                    "candles",
                    month_key,
                    granularity=granularity,
                    download_duration=partition_duration,
                    incremental_file=inc_path,
                )
                partition_files.append(filepath)
            except Exception as e:
                logger.error(f"Failed to finalize partition {month_key}: {e}")
                if inc_writer:
                    inc_writer.cleanup()

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
    compute_checksums: bool = True,
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

    tick_progress = [0.0]
    candle_progress = [0.0]
    tick_start = time.time()
    candle_start = time.time()
    tick_count = [0]
    candle_count = [0]

    def tick_progress_cb(current, total, current_count=0):
        tick_progress[0] = current / total if total > 0 else 1.0
        tick_count[0] = current_count
        elapsed = time.time() - tick_start
        speed = current_count / elapsed if elapsed > 0 else 0

        print(
            f"\r  Ticks: {tick_progress[0] * 100:.1f}% ({speed:.0f} t/s)  |  Candles: {candle_progress[0] * 100:.1f}%",
            end="",
        )

    def candle_progress_cb(current, total, current_count=0):
        candle_progress[0] = current / total if total > 0 else 1.0
        candle_count[0] = current_count
        elapsed = time.time() - candle_start
        speed = current_count / elapsed if elapsed > 0 else 0

        print(
            f"\r  Ticks: {tick_progress[0] * 100:.1f}%  |  Candles: {candle_progress[0] * 100:.1f}% ({speed:.1f} c/s)",
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
