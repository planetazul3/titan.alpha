
import pytest
import numpy as np
import pandas as pd
from pathlib import Path
from unittest.mock import MagicMock
from data.dataset import DerivDataset

class TestDatasetMemory:
    @pytest.fixture
    def mock_data(self, tmp_path):
        """Create dummy parquet files for testing."""
        data_dir = tmp_path / "data"
        symbol_dir = data_dir / "R_100"
        ticks_dir = symbol_dir / "ticks"
        candles_dir = symbol_dir / "candles_60"
        
        ticks_dir.mkdir(parents=True)
        candles_dir.mkdir(parents=True)
        
        # Create dummy ticks
        timestamps = np.arange(1000, 2000, 1.0)
        quotes = np.random.randn(1000) * 10 + 100
        tick_df = pd.DataFrame({"epoch": timestamps, "quote": quotes})
        tick_df.to_parquet(ticks_dir / "ticks.parquet")
        
        # Create dummy candles (every 60s)
        c_times = np.arange(1060, 2000, 60.0)
        candle_df = pd.DataFrame({
            "epoch": c_times,
            "open": np.random.randn(len(c_times)) + 100,
            "high": np.random.randn(len(c_times)) + 101,
            "low": np.random.randn(len(c_times)) + 99,
            "close": np.random.randn(len(c_times)) + 100,
        })
        candle_df.to_parquet(candles_dir / "candles.parquet")
        
        return symbol_dir

    def test_memory_mapping(self, mock_data):
        """Verify that dataset creates and uses memory mapped files."""
        settings = MagicMock()
        settings.data_shapes.sequence_length_ticks = 10
        settings.data_shapes.sequence_length_candles = 5
        settings.data_shapes.warmup_steps = 5
        
        # Initialize dataset
        dataset = DerivDataset(mock_data.parent, settings, mode="train")
        
        # Verify correctness
        assert len(dataset.ticks) == 1000
        # Wait, how many candles? 1000 ticks 1000s -> ~16 candles 
        
        # CRITICAL TEST: Check if ticks and candles are memmapped
        assert isinstance(dataset.ticks, np.memmap)
        assert isinstance(dataset.candles, np.memmap)
        assert isinstance(dataset.tick_epochs, np.memmap)
        
        # Verify .cache directory was created
        cache_dir = mock_data.parent / ".cache"
        assert cache_dir.exists()
        assert len(list(cache_dir.glob("*.npy"))) == 3 # ticks, tick_epochs, candles
        
        # Load again - should be fast/cached (how to verify? simple loading works)
        dataset2 = DerivDataset(mock_data.parent, settings, mode="train")
        assert np.array_equal(dataset.ticks, dataset2.ticks)

if __name__ == "__main__":
    pytest.main([__file__])
