
import pytest
import asyncio
import pandas as pd
from pathlib import Path
from datetime import datetime, timezone, timedelta
from unittest.mock import AsyncMock, MagicMock
from data.ingestion.historical import PartitionedDownloader

class MockClient:
    def __init__(self):
        self.api = AsyncMock()
        self.symbol = "R_100"

@pytest.fixture
def temp_cache(tmp_path):
    return tmp_path / "data_cache"

@pytest.mark.asyncio
async def test_atomic_write_simulation(temp_cache):
    """Verify that _save_partition uses the Write-Rename pattern."""
    downloader = PartitionedDownloader(MockClient(), "R_100", temp_cache)
    data = [{"epoch": 1700000000, "quote": 100.0}]
    
    # Run save
    filepath, metadata = downloader._save_partition(data, "ticks", "2023-11")
    
    assert filepath.exists()
    assert filepath.suffix == ".parquet"
    assert not filepath.with_suffix(".tmp.parquet").exists()
    assert filepath.with_suffix(".metadata.json").exists()

@pytest.mark.asyncio
async def test_incremental_write_trigger(temp_cache, mocker):
    """Verify that IncrementalWriter is triggered when CHUNK_SIZE is reached."""
    from data.ingestion.historical import HAS_PYARROW
    if not HAS_PYARROW:
        pytest.skip("pyarrow not available")

    downloader = PartitionedDownloader(MockClient(), "R_100", temp_cache)
    downloader.CHUNK_SIZE = 5  # Set low to trigger flush
    
    # Mock API to return 10 ticks
    mock_response = {
        "history": {
            "prices": [100.0] * 10,
            "times": list(range(1700000000, 1700000010))
        }
    }
    downloader.client.api.ticks_history.return_value = mock_response
    
    start_time = datetime.fromtimestamp(1700000000, tz=timezone.utc)
    end_time = start_time + timedelta(seconds=10)
    
    # Run download for 1 month
    # We mock _save_partition to see if it receives an incremental file
    spy = mocker.spy(downloader, "_save_partition")
    
    await downloader.download_ticks_partitioned(start_time, end_time, resume=False)
    
    # Check if _save_partition was called with incremental_file
    assert spy.call_count > 0
    args, kwargs = spy.call_args
    assert "incremental_file" in kwargs
    assert kwargs["incremental_file"] is not None
    assert not kwargs["incremental_file"].exists() # Should be unlinked after merge
