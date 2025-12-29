"""
Dataset versioning and checksum management.

Provides metadata tracking and integrity verification for Parquet datasets.
"""

import hashlib
import json
import logging
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

logger = logging.getLogger(__name__)

# Current metadata schema version
METADATA_SCHEMA_VERSION = "1.0.0"


@dataclass
class DatasetMetadata:
    """
    Versioned metadata for a dataset file.

    Stored as JSON sidecar alongside Parquet files for reproducibility
    and data lineage tracking.
    """

    # Schema version for forward compatibility
    schema_version: str = METADATA_SCHEMA_VERSION

    # Dataset identification
    symbol: str = ""
    data_type: str = ""  # "ticks" or "candles"
    granularity: int | None = None  # For candles only (seconds)

    # Time range
    start_epoch: int = 0
    end_epoch: int = 0

    # Record statistics
    record_count: int = 0

    # Integrity information
    gaps_detected: int = 0
    duplicates_removed: int = 0

    # Checksum (optional)
    sha256: str | None = None

    # Production protection
    locked: bool = False  # Prevent modification when in production use

    # Metadata
    created_at: str = ""
    download_duration_seconds: float | None = None
    file_size: int | None = None

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "DatasetMetadata":
        """Create from dictionary."""
        # Handle schema version migration if needed
        schema_version = data.get("schema_version", "1.0.0")
        if schema_version != METADATA_SCHEMA_VERSION:
            logger.warning(
                f"Metadata schema version mismatch: {schema_version} != {METADATA_SCHEMA_VERSION}"
            )

        # Filter to only known fields
        known_fields = {f.name for f in cls.__dataclass_fields__.values()}
        filtered = {k: v for k, v in data.items() if k in known_fields}

        return cls(**filtered)


def get_metadata_path(parquet_path: Path) -> Path:
    """
    Get the metadata sidecar path for a Parquet file.

    Convention: {filename}.metadata.json
    """
    return parquet_path.with_suffix(".metadata.json")


def save_metadata(parquet_path: Path, metadata: DatasetMetadata) -> Path:
    """
    Save metadata as JSON sidecar file.

    Args:
        parquet_path: Path to the Parquet data file.
        metadata: Metadata to save.

    Returns:
        Path to the saved metadata file.
    """
    metadata_path = get_metadata_path(parquet_path)

    with open(metadata_path, "w") as f:
        json.dump(metadata.to_dict(), f, indent=2)

    logger.info(f"Saved metadata to {metadata_path}")
    return metadata_path


def load_metadata(parquet_path: Path) -> DatasetMetadata | None:
    """
    Load metadata from JSON sidecar file.

    Args:
        parquet_path: Path to the Parquet data file.

    Returns:
        DatasetMetadata if sidecar exists, None otherwise.
    """
    metadata_path = get_metadata_path(parquet_path)

    if not metadata_path.exists():
        logger.debug(f"No metadata file found at {metadata_path}")
        return None

    try:
        with open(metadata_path) as f:
            data = json.load(f)
        return DatasetMetadata.from_dict(data)
    except (json.JSONDecodeError, KeyError) as e:
        logger.error(f"Failed to load metadata from {metadata_path}: {e}")
        return None


def compute_checksum(file_path: Path, algorithm: str = "sha256") -> str:
    """
    Compute hash checksum of a file.

    Args:
        file_path: Path to the file.
        algorithm: Hash algorithm ('sha256', 'md5', etc.)

    Returns:
        Hex digest of the file hash.
    """
    hash_func = hashlib.new(algorithm)

    with open(file_path, "rb") as f:
        # Read in chunks for large files
        for chunk in iter(lambda: f.read(8192), b""):
            hash_func.update(chunk)

    checksum = hash_func.hexdigest()
    logger.debug(f"Computed {algorithm} checksum for {file_path}: {checksum[:16]}...")
    return checksum


def verify_checksum(file_path: Path, expected: str, algorithm: str = "sha256") -> bool:
    """
    Verify file checksum matches expected value.

    Args:
        file_path: Path to the file.
        expected: Expected checksum hex digest.
        algorithm: Hash algorithm used.

    Returns:
        True if checksum matches, False otherwise.
    """
    actual = compute_checksum(file_path, algorithm)
    matches = actual == expected

    if not matches:
        logger.warning(
            f"Checksum mismatch for {file_path}:\n  Expected: {expected}\n  Actual:   {actual}"
        )

    return matches


def create_metadata(
    symbol: str,
    data_type: str,
    records: list,
    granularity: int | None = None,
    gaps_detected: int = 0,
    duplicates_removed: int = 0,
    download_duration: float | None = None,
) -> DatasetMetadata:
    """
    Create metadata for a dataset.

    Args:
        symbol: Trading symbol.
        data_type: 'ticks' or 'candles'.
        records: List of data records with 'epoch' field.
        granularity: Candle interval (seconds), required for candles.
        gaps_detected: Number of gaps found in integrity check.
        duplicates_removed: Number of duplicates removed.
        download_duration: Time taken to download (seconds).

    Returns:
        Populated DatasetMetadata instance.
    """
    if not records:
        epochs = []
    else:
        epochs = [r["epoch"] for r in records]

    return DatasetMetadata(
        symbol=symbol,
        data_type=data_type,
        granularity=granularity,
        start_epoch=min(epochs) if epochs else 0,
        end_epoch=max(epochs) if epochs else 0,
        record_count=len(records),
        gaps_detected=gaps_detected,
        duplicates_removed=duplicates_removed,
        created_at=datetime.now(timezone.utc).isoformat(),
        download_duration_seconds=download_duration,
    )


def is_dataset_locked(parquet_path: Path) -> bool:
    """
    Check if a dataset is locked for production use.
    
    Locked datasets should not be modified or deleted by maintenance scripts.
    
    Args:
        parquet_path: Path to the Parquet data file.
        
    Returns:
        True if dataset is locked, False otherwise.
    """
    metadata = load_metadata(parquet_path)
    return metadata.locked if metadata else False


def lock_dataset(parquet_path: Path, reason: str = "") -> bool:
    """
    Lock a dataset to prevent accidental modification.
    
    Use this when a dataset is being used for production training or trading.
    
    Args:
        parquet_path: Path to the Parquet data file.
        reason: Optional reason for locking (logged but not stored).
        
    Returns:
        True if lock was successful, False if already locked or metadata missing.
    """
    metadata = load_metadata(parquet_path)
    if metadata is None:
        logger.error(f"Cannot lock {parquet_path}: no metadata found")
        return False
    
    if metadata.locked:
        logger.warning(f"Dataset {parquet_path.name} is already locked")
        return False
    
    metadata.locked = True
    save_metadata(parquet_path, metadata)
    logger.info(f"Locked dataset {parquet_path.name}" + (f": {reason}" if reason else ""))
    return True


def unlock_dataset(parquet_path: Path) -> bool:
    """
    Unlock a dataset for maintenance operations.
    
    Args:
        parquet_path: Path to the Parquet data file.
        
    Returns:
        True if unlock was successful, False if not locked or metadata missing.
    """
    metadata = load_metadata(parquet_path)
    if metadata is None:
        logger.error(f"Cannot unlock {parquet_path}: no metadata found")
        return False
    
    if not metadata.locked:
        logger.debug(f"Dataset {parquet_path.name} is not locked")
        return False
    
    metadata.locked = False
    save_metadata(parquet_path, metadata)
    logger.info(f"Unlocked dataset {parquet_path.name}")
    return True
