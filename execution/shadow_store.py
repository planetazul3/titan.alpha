"""
Shadow Trade Store - Immutable storage for shadow trades as data assets.

Shadow trades are first-class data products, not logs. This module provides:
- Immutable append-only storage with rich metadata
- Parquet export for efficient columnar storage
- Schema enforcement and versioning

Usage:
    >>> store = ShadowTradeStore(Path("data_cache/shadow_trades"))
    >>> store.append(ShadowTradeRecord(...))
    >>> trades = store.query(start=yesterday, end=today)
    >>> store.export_parquet(Path("exports/shadow_v1.parquet"))
"""

import json
import logging
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

# Schema version - increment when record format changes
SHADOW_STORE_SCHEMA_VERSION = "2.0"


@dataclass
class ShadowTradeRecord:
    """
    Immutable shadow trade record with rich metadata.

    This is a DATA ASSET, not a log entry. It contains everything needed
    to later:
    1. Resolve the trade outcome from market data
    2. Reconstruct the exact features used for prediction
    3. Generate training labels

    Attributes:
        trade_id: Unique identifier (UUID)
        timestamp: When the signal was generated
        contract_type: Type of contract (RISE_FALL, TOUCH_NO_TOUCH, etc.)
        direction: Trade direction (CALL, PUT, TOUCH, etc.)
        probability: Model's predicted probability
        entry_price: Price at signal generation

        # Risk/Regime metadata
        reconstruction_error: Volatility expert's reconstruction error
        regime_state: TRUSTED, CAUTION, or VETO

        # Versioning
        model_version: Version of the model used
        feature_schema_version: Version of the feature schema

        # Market context (for outcome resolution)
        tick_window: Last N ticks before trade (compressed)
        candle_window: Last M candles before trade (compressed)

        # Outcome (filled later by OutcomeResolver)
        outcome: True (win) / False (loss) / None (unresolved)
        exit_price: Actual exit price (filled by resolver)
        resolved_at: When outcome was resolved
    """

    # Core trade info
    trade_id: str
    timestamp: datetime
    contract_type: str
    direction: str
    probability: float
    entry_price: float

    # Risk/Regime metadata
    reconstruction_error: float
    regime_state: str

    # Versioning
    model_version: str = "unknown"
    feature_schema_version: str = "1.0"

    # Market context (stored as lists for JSON serialization)
    tick_window: list[float] = field(default_factory=list)
    candle_window: list[list[float]] = field(default_factory=list)

    # Outcome (filled by OutcomeResolver)
    outcome: bool | None = None
    exit_price: float | None = None
    resolved_at: datetime | None = None

    # Barrier level (M03: specific barrier for TOUCH/RANGE trades)
    barrier_level: float | None = None
    barrier2_level: float | None = None

    # C02 Fix: Per-trade duration for accurate resolution timing
    duration_minutes: int = 1

    # C01 Fix: Resolution context - accumulates candles AFTER trade entry
    # Each entry is [high, low, close] from candles observed during trade duration
    resolution_context: list[list[float]] = field(default_factory=list)

    # Extra metadata
    metadata: dict[str, Any] = field(default_factory=dict)

    # Concurrency Control (CRITICAL-002)
    version_number: int = 0

    # Schema version (auto-set)
    _schema_version: str = field(default=SHADOW_STORE_SCHEMA_VERSION)
    _created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    @classmethod
    def create(
        cls,
        contract_type: str,
        direction: str,
        probability: float,
        entry_price: float,
        reconstruction_error: float,
        regime_state: str,
        tick_window: np.ndarray | list[float] | None = None,
        candle_window: np.ndarray | list[list[float]] | None = None,
        model_version: str = "unknown",
        feature_schema_version: str = "1.0",
        barrier_level: float | None = None,
        barrier2_level: float | None = None,
        duration_minutes: int = 1,
        metadata: dict[str, Any] | None = None,
    ) -> "ShadowTradeRecord":
        """
        Factory method to create a new shadow trade record.

        Automatically handles:
        - UUID generation
        - Timestamp creation
        - Numpy array to list conversion
        """
        return cls(
            trade_id=str(uuid.uuid4()),
            timestamp=datetime.now(timezone.utc),
            contract_type=contract_type,
            direction=direction,
            probability=probability,
            entry_price=entry_price,
            reconstruction_error=reconstruction_error,
            regime_state=regime_state,
            model_version=model_version,
            feature_schema_version=feature_schema_version,
            tick_window=tick_window.tolist()
            if isinstance(tick_window, np.ndarray)
            else (list(tick_window) if tick_window is not None else []),
            candle_window=candle_window.tolist()
            if isinstance(candle_window, np.ndarray)
            else (list(candle_window) if candle_window is not None else []),
            barrier_level=barrier_level,
            barrier2_level=barrier2_level,
            duration_minutes=duration_minutes,
            metadata=metadata or {},
        )

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        d = asdict(self)
        # Convert datetime to ISO string
        d["timestamp"] = self.timestamp.isoformat()
        if self.resolved_at:
            d["resolved_at"] = self.resolved_at.isoformat()
        return d

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "ShadowTradeRecord":
        """Create from dictionary (JSON deserialization)."""
        # Parse timestamps
        if isinstance(d.get("timestamp"), str):
            d["timestamp"] = datetime.fromisoformat(d["timestamp"])
        if d.get("resolved_at") and isinstance(d["resolved_at"], str):
            d["resolved_at"] = datetime.fromisoformat(d["resolved_at"])

        # Handle missing fields gracefully
        valid_fields = {f.name for f in cls.__dataclass_fields__.values()}
        filtered = {k: v for k, v in d.items() if k in valid_fields}

        return cls(**filtered)

    def with_outcome(self, outcome: bool, exit_price: float) -> "ShadowTradeRecord":
        """
        Create a new record with outcome resolved.

        Returns a NEW record (immutability preserved).
        """
        return ShadowTradeRecord(
            trade_id=self.trade_id,
            timestamp=self.timestamp,
            contract_type=self.contract_type,
            direction=self.direction,
            probability=self.probability,
            entry_price=self.entry_price,
            reconstruction_error=self.reconstruction_error,
            regime_state=self.regime_state,
            model_version=self.model_version,
            feature_schema_version=self.feature_schema_version,
            tick_window=self.tick_window,
            candle_window=self.candle_window,
            outcome=outcome,
            exit_price=exit_price,
            resolved_at=datetime.now(timezone.utc),
            barrier_level=self.barrier_level,
            barrier2_level=self.barrier2_level,
            duration_minutes=self.duration_minutes,
            resolution_context=self.resolution_context,
            metadata=self.metadata,
            version_number=self.version_number,
            _schema_version=self._schema_version,
            _created_at=self._created_at,
        )

    def with_resolution_candle(
        self, high: float, low: float, close: float
    ) -> "ShadowTradeRecord":
        """
        C01 Fix: Create a new record with an additional resolution candle appended.

        This accumulates OHLC data observed AFTER trade entry for path-dependent
        contract resolution (TOUCH, RANGE).

        Returns a NEW record (immutability preserved).
        """
        new_context = self.resolution_context + [[high, low, close]]
        return ShadowTradeRecord(
            trade_id=self.trade_id,
            timestamp=self.timestamp,
            contract_type=self.contract_type,
            direction=self.direction,
            probability=self.probability,
            entry_price=self.entry_price,
            reconstruction_error=self.reconstruction_error,
            regime_state=self.regime_state,
            model_version=self.model_version,
            feature_schema_version=self.feature_schema_version,
            tick_window=self.tick_window,
            candle_window=self.candle_window,
            outcome=self.outcome,
            exit_price=self.exit_price,
            resolved_at=self.resolved_at,
            barrier_level=self.barrier_level,
            barrier2_level=self.barrier2_level,
            duration_minutes=self.duration_minutes,
            resolution_context=new_context,
            metadata=self.metadata,
            version_number=self.version_number,
            _schema_version=self._schema_version,
            _created_at=self._created_at,
        )

    def is_resolved(self) -> bool:
        """Check if outcome has been resolved."""
        return self.outcome is not None


class ShadowTradeStore:
    """
    Immutable append-only store for shadow trades.

    Design principles:
    - APPEND-ONLY: Records are never modified in place
    - IMMUTABLE EXPORTS: Export creates new file, doesn't modify source
    - SCHEMA VERSIONED: Every record has schema version for compatibility
    - RICH METADATA: Includes everything needed for outcome resolution

    Storage format: NDJSON (one JSON object per line)
    Export format: Parquet (columnar, efficient for analysis)
    """

    def __init__(self, store_path: Path):
        """
        Initialize shadow trade store.

        Args:
            store_path: Path to NDJSON file for storage
        """
        self._store_path = Path(store_path)
        self._store_path.parent.mkdir(parents=True, exist_ok=True)

        logger.info(
            f"ShadowTradeStore initialized: {store_path} (schema v{SHADOW_STORE_SCHEMA_VERSION})"
        )

    def append(self, record: ShadowTradeRecord) -> None:
        """
        Append a shadow trade record to the store.

        This is an APPEND-ONLY operation. Records are never modified.

        Args:
            record: Shadow trade record to store
        """
        with open(self._store_path, "a") as f:
            f.write(json.dumps(record.to_dict()) + "\n")

        logger.debug(f"Appended shadow trade: {record.trade_id}")

    async def append_async(self, record: ShadowTradeRecord) -> None:
        """
        Append a shadow trade record to the store asynchronously.

        Offloads the blocking I/O operation to a thread pool executor.

        Args:
            record: Shadow trade record to store
        """
        import asyncio
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(None, lambda: self.append(record))

    def query(
        self,
        start: datetime | None = None,
        end: datetime | None = None,
        resolved_only: bool = False,
        unresolved_only: bool = False,
    ) -> list[ShadowTradeRecord]:
        """
        Query shadow trades from the store.

        Args:
            start: Start of time range (inclusive)
            end: End of time range (exclusive)
            resolved_only: Only return trades with outcomes
            unresolved_only: Only return trades without outcomes

        Returns:
            List of matching shadow trade records
        """
        if not self._store_path.exists():
            return []

        records = []
        with open(self._store_path) as f:
            for line in f:
                if not line.strip():
                    continue
                try:
                    d = json.loads(line)
                    record = ShadowTradeRecord.from_dict(d)

                    # Filter by time range
                    if start and record.timestamp < start:
                        continue
                    if end and record.timestamp >= end:
                        continue

                    # Filter by resolution status
                    if resolved_only and not record.is_resolved():
                        continue
                    if unresolved_only and record.is_resolved():
                        continue

                    records.append(record)
                except (json.JSONDecodeError, TypeError) as e:
                    logger.warning(f"Skipping malformed record: {e}")
                    continue

        return records

    def append_resolved(self, resolved_records: list[ShadowTradeRecord]) -> Path:
        """
        Append resolved records to a separate resolved store.

        This creates a NEW file for resolved trades, preserving immutability.

        Args:
            resolved_records: List of records with outcomes resolved

        Returns:
            Path to the resolved store file
        """
        resolved_path = self._store_path.with_suffix(".resolved.ndjson")

        with open(resolved_path, "a") as f:
            for record in resolved_records:
                if record.is_resolved():
                    f.write(json.dumps(record.to_dict()) + "\n")

        logger.info(f"Appended {len(resolved_records)} resolved trades to {resolved_path}")
        return resolved_path

    def export_parquet(self, output_path: Path) -> Path:
        """
        Export shadow trades to Parquet format.

        Parquet provides:
        - Efficient columnar storage
        - Schema enforcement
        - Compression
        - Fast analytical queries

        Args:
            output_path: Path for output Parquet file

        Returns:
            Path to exported file

        Raises:
            ImportError: If pyarrow/pandas not available
        """
        try:
            import pandas as pd
        except ImportError:
            raise ImportError("pandas required for Parquet export: pip install pandas pyarrow")

        records = self.query()
        if not records:
            logger.warning("No records to export")
            return output_path

        # Convert to DataFrame
        data = []
        for r in records:
            row = {
                "trade_id": r.trade_id,
                "timestamp": r.timestamp,
                "contract_type": r.contract_type,
                "direction": r.direction,
                "probability": r.probability,
                "entry_price": r.entry_price,
                "reconstruction_error": r.reconstruction_error,
                "regime_state": r.regime_state,
                "model_version": r.model_version,
                "feature_schema_version": r.feature_schema_version,
                "outcome": r.outcome,
                "exit_price": r.exit_price,
                "resolved_at": r.resolved_at,
                "_schema_version": r._schema_version,
            }
            data.append(row)

        df = pd.DataFrame(data)

        # Export to Parquet
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(output_path, index=False)

        logger.info(f"Exported {len(records)} records to {output_path}")
        return output_path

    def get_statistics(self) -> dict[str, Any]:
        """Get summary statistics of the store."""
        records = self.query()
        resolved = [r for r in records if r.is_resolved()]
        wins = [r for r in resolved if r.outcome]

        return {
            "total_records": len(records),
            "resolved_records": len(resolved),
            "unresolved_records": len(records) - len(resolved),
            "wins": len(wins),
            "losses": len(resolved) - len(wins),
            "win_rate": len(wins) / len(resolved) if resolved else 0.0,
            "schema_version": SHADOW_STORE_SCHEMA_VERSION,
        }

    def update_outcome(self, trade: ShadowTradeRecord, outcome: bool, exit_price: float) -> None:
        """
        Update the outcome of a trade.
        
        Creates a resolved copy and appends to resolved store.
        """
        resolved_trade = trade.with_outcome(outcome, exit_price)
        self.append_resolved([resolved_trade])
        logger.debug(f"Updated outcome for trade {trade.trade_id}: {outcome}")

    def mark_stale(self, trade: ShadowTradeRecord) -> None:
        """
        Mark a trade as stale (unresolved after timeout).
        
        Treats as a loss (outcome=False) with metadata flag.
        """
        # Create resolved copy marked as stale
        # We treat stale as loss for safety, but note it in metadata
        resolved_trade = trade.with_outcome(outcome=False, exit_price=trade.entry_price)
        resolved_trade.metadata["resolution"] = "stale"
        self.append_resolved([resolved_trade])
        logger.warning(f"Marked trade {trade.trade_id} as stale")
