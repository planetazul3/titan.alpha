"""
Tests for historical data downloader with mocked Deriv API.

Uses pytest-mock to simulate API responses for tick and candle downloads.
This enables testing of:
- Pagination logic
- Resume capability
- Monthly partitioning
- Error handling during download
"""
import pytest
from datetime import datetime, timezone, timedelta
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch
import tempfile


class TestMonthBoundaries:
    """Test time range splitting into monthly boundaries."""

    def test_single_month_range(self):
        """Single month should return one boundary."""
        from data.ingestion.historical import get_month_boundaries

        start = datetime(2024, 1, 15, tzinfo=timezone.utc)
        end = datetime(2024, 1, 20, tzinfo=timezone.utc)

        boundaries = get_month_boundaries(start, end)

        assert len(boundaries) == 1
        assert boundaries[0][0] == start
        assert boundaries[0][1] == end

    def test_multi_month_range(self):
        """Multi-month range should split correctly."""
        from data.ingestion.historical import get_month_boundaries

        start = datetime(2024, 1, 15, tzinfo=timezone.utc)
        end = datetime(2024, 3, 10, tzinfo=timezone.utc)

        boundaries = get_month_boundaries(start, end)

        assert len(boundaries) == 3  # Jan, Feb, Mar
        # First boundary starts at our start date
        assert boundaries[0][0] == start
        # Last boundary ends at our end date
        assert boundaries[-1][1] == end

    def test_year_boundary_crossing(self):
        """Should handle year boundary correctly."""
        from data.ingestion.historical import get_month_boundaries

        start = datetime(2023, 12, 15, tzinfo=timezone.utc)
        end = datetime(2024, 2, 10, tzinfo=timezone.utc)

        boundaries = get_month_boundaries(start, end)

        assert len(boundaries) == 3  # Dec 2023, Jan 2024, Feb 2024


class TestEpochToMonthKey:
    """Test epoch to month key conversion."""

    def test_epoch_conversion(self):
        """Should convert epoch to YYYY-MM format."""
        from data.ingestion.historical import epoch_to_month_key

        epoch = int(datetime(2024, 6, 15, tzinfo=timezone.utc).timestamp())
        key = epoch_to_month_key(epoch)

        assert key == "2024-06"


class TestPartitionedDownloader:
    """Test partitioned downloader with mocked client."""

    @pytest.fixture
    def mock_client(self):
        """Create mock Deriv client."""
        client = MagicMock()
        client.api = AsyncMock()
        return client

    @pytest.fixture
    def temp_cache_dir(self):
        """Create temporary cache directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield Path(tmpdir)

    def test_downloader_initialization(self, mock_client, temp_cache_dir):
        """Downloader should initialize with correct paths."""
        from data.ingestion.historical import PartitionedDownloader

        downloader = PartitionedDownloader(mock_client, "R_100", temp_cache_dir)

        assert downloader.symbol == "R_100"
        assert downloader.cache_dir == temp_cache_dir
        assert (temp_cache_dir / "R_100").exists()

    def test_get_partition_dir_ticks(self, mock_client, temp_cache_dir):
        """Should create tick partition directory."""
        from data.ingestion.historical import PartitionedDownloader

        downloader = PartitionedDownloader(mock_client, "R_100", temp_cache_dir)
        tick_dir = downloader._get_partition_dir("ticks")

        assert tick_dir == temp_cache_dir / "R_100" / "ticks"
        assert tick_dir.exists()

    def test_get_partition_dir_candles(self, mock_client, temp_cache_dir):
        """Should create candle partition directory with granularity."""
        from data.ingestion.historical import PartitionedDownloader

        downloader = PartitionedDownloader(mock_client, "R_100", temp_cache_dir)
        candle_dir = downloader._get_partition_dir("candles", granularity=60)

        assert candle_dir == temp_cache_dir / "R_100" / "candles_60"
        assert candle_dir.exists()

    def test_find_resume_point_empty(self, mock_client, temp_cache_dir):
        """Should return None when no partitions exist."""
        from data.ingestion.historical import PartitionedDownloader

        downloader = PartitionedDownloader(mock_client, "R_100", temp_cache_dir)
        resume_point = downloader.find_resume_point("ticks")

        assert resume_point is None


class TestHistoricalDataDownloader:
    """Test legacy downloader interface."""

    @pytest.fixture
    def mock_client(self):
        """Create mock Deriv client."""
        client = MagicMock()
        client.api = AsyncMock()
        return client

    @pytest.fixture
    def temp_cache_dir(self):
        """Create temporary cache directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield Path(tmpdir)

    def test_legacy_downloader_initialization(self, mock_client, temp_cache_dir):
        """Legacy downloader should initialize correctly."""
        from data.ingestion.historical import HistoricalDataDownloader

        downloader = HistoricalDataDownloader(mock_client, "R_100", temp_cache_dir)

        assert downloader.symbol == "R_100"
        assert downloader.cache_dir == temp_cache_dir

    @pytest.mark.asyncio
    async def test_download_ticks_mock(self, mock_client, temp_cache_dir):
        """Should download ticks with mocked API."""
        from data.ingestion.historical import HistoricalDataDownloader

        # Mock API response
        mock_response = {
            "history": {
                "prices": [100.0, 100.5, 101.0, 100.8, 101.2],
                "times": [1000, 1001, 1002, 1003, 1004],
            }
        }
        mock_client.api.ticks_history = AsyncMock(
            side_effect=[mock_response, {"history": {"prices": [], "times": []}}]
        )

        downloader = HistoricalDataDownloader(mock_client, "R_100", temp_cache_dir)

        start = datetime(2024, 1, 1, tzinfo=timezone.utc)
        end = datetime(2024, 1, 1, 0, 0, 5, tzinfo=timezone.utc)

        ticks = await downloader.download_ticks(start, end)

        assert len(ticks) == 5
        assert all("epoch" in t and "quote" in t for t in ticks)

    @pytest.mark.asyncio
    async def test_download_candles_mock(self, mock_client, temp_cache_dir):
        """Should download candles with mocked API."""
        from data.ingestion.historical import HistoricalDataDownloader

        # Mock API response
        mock_candles = [
            {"epoch": 1000, "open": 100, "high": 101, "low": 99, "close": 100.5},
            {"epoch": 1060, "open": 100.5, "high": 102, "low": 100, "close": 101},
        ]
        mock_client.api.ticks_history = AsyncMock(
            side_effect=[{"candles": mock_candles}, {"candles": []}]
        )

        downloader = HistoricalDataDownloader(mock_client, "R_100", temp_cache_dir)

        start = datetime(2024, 1, 1, tzinfo=timezone.utc)
        end = datetime(2024, 1, 1, 0, 2, tzinfo=timezone.utc)

        candles = await downloader.download_candles(start, end, granularity=60)

        assert len(candles) == 2
        assert all("open" in c and "close" in c for c in candles)


class TestIntegrityChecks:
    """Test integrity checking during save operations."""

    @pytest.fixture
    def temp_cache_dir(self):
        """Create temporary cache directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield Path(tmpdir)

    def test_save_ticks_parquet(self, temp_cache_dir):
        """Should save ticks with integrity checks."""
        pytest.importorskip("pandas")
        from data.ingestion.historical import HistoricalDataDownloader

        mock_client = MagicMock()
        downloader = HistoricalDataDownloader(mock_client, "R_100", temp_cache_dir)

        ticks = [
            {"epoch": 1000, "quote": 100.0},
            {"epoch": 1001, "quote": 100.5},
            {"epoch": 1002, "quote": 101.0},
        ]

        filepath, metadata = downloader.save_ticks_parquet(
            ticks, "test_ticks", check_integrity=True
        )

        assert filepath.exists()
        assert metadata is not None
        assert metadata.record_count == 3

    def test_save_candles_parquet(self, temp_cache_dir):
        """Should save candles with integrity checks."""
        pytest.importorskip("pandas")
        from data.ingestion.historical import HistoricalDataDownloader

        mock_client = MagicMock()
        downloader = HistoricalDataDownloader(mock_client, "R_100", temp_cache_dir)

        candles = [
            {"epoch": 1000, "open": 100, "high": 101, "low": 99, "close": 100.5},
            {"epoch": 1060, "open": 100.5, "high": 102, "low": 100, "close": 101},
        ]

        filepath, metadata = downloader.save_candles_parquet(
            candles, "test_candles", granularity=60, check_integrity=True
        )

        assert filepath.exists()
        assert metadata is not None
        assert metadata.record_count == 2
        assert metadata.granularity == 60
