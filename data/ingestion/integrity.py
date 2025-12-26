"""
Data integrity checking for downloaded tick and candle data.

Provides gap detection, duplicate removal, and integrity reporting.
"""

import logging
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class GapInfo:
    """Information about a detected gap in time series data."""

    start_epoch: int
    end_epoch: int
    expected_interval: int
    actual_interval: int

    @property
    def gap_duration(self) -> int:
        """Duration of gap in seconds."""
        return self.actual_interval - self.expected_interval

    def __str__(self) -> str:
        return f"Gap: {self.start_epoch} -> {self.end_epoch} (expected {self.expected_interval}s, got {self.actual_interval}s)"


@dataclass
class IntegrityReport:
    """Summary of data integrity checks."""

    total_records: int
    duplicates_found: int
    duplicates_removed: int
    gaps_detected: int
    gaps: list[GapInfo] = field(default_factory=list)
    is_clean: bool = True

    def __str__(self) -> str:
        status = "✓ CLEAN" if self.is_clean else "⚠ ISSUES FOUND"
        return (
            f"Integrity Report [{status}]\n"
            f"  Records: {self.total_records}\n"
            f"  Duplicates: {self.duplicates_found} found, {self.duplicates_removed} removed\n"
            f"  Gaps: {self.gaps_detected} detected"
        )


class IntegrityChecker:
    """
    Post-download integrity validation for tick and candle data.

    Detects gaps and duplicates in time series data, providing
    detailed reports for data quality assessment.
    """

    # Default gap threshold for tick data (seconds)
    DEFAULT_TICK_GAP_THRESHOLD = 10

    def __init__(self, tick_gap_threshold: int = DEFAULT_TICK_GAP_THRESHOLD):
        """
        Args:
            tick_gap_threshold: Maximum expected interval between ticks (seconds).
                               Intervals larger than this are flagged as gaps.
        """
        self.tick_gap_threshold = tick_gap_threshold

    def check_gaps_ticks(self, ticks: list[dict[str, Any]]) -> list[GapInfo]:
        """
        Detect gaps in tick data.

        Flags intervals between consecutive ticks that exceed the threshold.

        Args:
            ticks: List of tick records with 'epoch' field, sorted by time.

        Returns:
            List of detected gaps.
        """
        if len(ticks) < 2:
            return []

        gaps = []
        for i in range(len(ticks) - 1):
            current_epoch = ticks[i]["epoch"]
            next_epoch = ticks[i + 1]["epoch"]
            interval = next_epoch - current_epoch

            if interval > self.tick_gap_threshold:
                gaps.append(
                    GapInfo(
                        start_epoch=current_epoch,
                        end_epoch=next_epoch,
                        expected_interval=self.tick_gap_threshold,
                        actual_interval=interval,
                    )
                )

        return gaps

    def check_gaps_candles(self, candles: list[dict[str, Any]], granularity: int) -> list[GapInfo]:
        """
        Detect gaps in candle data.

        Checks if consecutive candles are exactly 'granularity' seconds apart.

        Args:
            candles: List of candle records with 'epoch' field, sorted by time.
            granularity: Expected interval between candles in seconds.

        Returns:
            List of detected gaps.
        """
        if len(candles) < 2:
            return []

        gaps = []
        for i in range(len(candles) - 1):
            current_epoch = candles[i]["epoch"]
            next_epoch = candles[i + 1]["epoch"]
            interval = next_epoch - current_epoch

            # Allow for minor timing variations (±1 second)
            if abs(interval - granularity) > 1:
                gaps.append(
                    GapInfo(
                        start_epoch=current_epoch,
                        end_epoch=next_epoch,
                        expected_interval=granularity,
                        actual_interval=interval,
                    )
                )

        return gaps

    def find_duplicates(self, data: list[dict[str, Any]]) -> list[int]:
        """
        Find indices of duplicate records based on epoch.

        Args:
            data: List of records with 'epoch' field.

        Returns:
            List of indices that are duplicates (keeps first occurrence).
        """
        seen_epochs = set()
        duplicate_indices = []

        for i, record in enumerate(data):
            epoch = record["epoch"]
            if epoch in seen_epochs:
                duplicate_indices.append(i)
            else:
                seen_epochs.add(epoch)

        return duplicate_indices

    def remove_duplicates(self, data: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], int]:
        """
        Remove duplicate records, keeping the first occurrence.

        Args:
            data: List of records with 'epoch' field.

        Returns:
            Tuple of (deduplicated data, count of removed duplicates).
        """
        duplicate_indices = set(self.find_duplicates(data))

        if not duplicate_indices:
            return data, 0

        cleaned = [record for i, record in enumerate(data) if i not in duplicate_indices]

        return cleaned, len(duplicate_indices)

    def generate_report(
        self,
        data: list[dict[str, Any]],
        data_type: str = "ticks",
        granularity: int | None = None,
        auto_dedupe: bool = True,
    ) -> tuple[list[dict[str, Any]], IntegrityReport]:
        """
        Run full integrity check and generate report.

        Args:
            data: List of tick or candle records.
            data_type: 'ticks' or 'candles'.
            granularity: Candle interval in seconds (required for candles).
            auto_dedupe: If True, automatically remove duplicates.

        Returns:
            Tuple of (cleaned data, integrity report).
        """
        len(data)

        # Check duplicates
        duplicate_indices = self.find_duplicates(data)
        duplicates_found = len(duplicate_indices)

        # Optionally remove duplicates
        if auto_dedupe and duplicates_found > 0:
            data, duplicates_removed = self.remove_duplicates(data)
            logger.info(f"Removed {duplicates_removed} duplicate records")
        else:
            duplicates_removed = 0

        # Check gaps
        if data_type == "ticks":
            gaps = self.check_gaps_ticks(data)
        elif data_type == "candles":
            if granularity is None:
                raise ValueError("granularity required for candle gap detection")
            gaps = self.check_gaps_candles(data, granularity)
        else:
            raise ValueError(f"Unknown data_type: {data_type}")

        # Log gap warnings
        if gaps:
            logger.warning(f"Detected {len(gaps)} gaps in {data_type} data")
            for gap in gaps[:5]:  # Show first 5
                logger.warning(f"  {gap}")
            if len(gaps) > 5:
                logger.warning(f"  ... and {len(gaps) - 5} more gaps")

        # Build report
        is_clean = duplicates_found == 0 and len(gaps) == 0
        report = IntegrityReport(
            total_records=len(data),
            duplicates_found=duplicates_found,
            duplicates_removed=duplicates_removed,
            gaps_detected=len(gaps),
            gaps=gaps,
            is_clean=is_clean,
        )

        return data, report
