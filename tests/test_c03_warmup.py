
import numpy as np
import pandas as pd
import pytest
import torch
from pathlib import Path
from unittest.mock import MagicMock

from config.settings import Settings
from data.dataset import DerivDataset

@pytest.fixture
def mock_settings():
    settings = MagicMock()
    # Mock data shapes
    settings.data_shapes = MagicMock()
    settings.data_shapes.sequence_length_ticks = 20
    settings.data_shapes.sequence_length_candles = 10
    settings.data_shapes.warmup_steps = 20  # Warmup > RSI period (14)
    settings.data_shapes.label_threshold_touch = 0.005
    settings.data_shapes.label_threshold_range = 0.003
    
    # Mock normalization factors (REQUIRED by VolatilityMetricsExtractor)
    settings.normalization = MagicMock()
    settings.normalization.norm_factor_volatility = 1.0
    settings.normalization.norm_factor_atr = 1.0
    settings.normalization.norm_factor_rsi_std = 1.0
    settings.normalization.norm_factor_bb_width = 1.0
    return settings

@pytest.fixture
def mock_data_dir(tmp_path):
    """Create a temporary directory with mock parquet data."""
    # Create sufficient data: Warmup(20) + Seq(10) + Lookahead(5) + Buffer
    # Need at least 20+10+5 = 35 candles. Let's make 100.
    n_candles = 100
    
    # Create Candles
    dates = pd.date_range(start="2023-01-01", periods=n_candles, freq="1min")
    candles = pd.DataFrame({
        "open": np.linspace(100, 110, n_candles),
        "high": np.linspace(101, 111, n_candles),
        "low": np.linspace(99, 109, n_candles),
        "close": np.linspace(100.5, 110.5, n_candles),
        "volume": np.ones(n_candles) * 100,
        "epoch": dates.astype(np.int64) // 10**9
    })
    
    # Create Ticks (just some dummy ticks aligned with candles)
    n_ticks = n_candles * 60
    ticks = pd.DataFrame({
        "quote": np.linspace(100, 110, n_ticks),
        "epoch": np.linspace(candles["epoch"].min(), candles["epoch"].max(), n_ticks)
    })
    
    # Save parquet in partitioned format: {symbol}/ticks/*.parquet
    symbol_dir = tmp_path / "mock_symbol"
    (symbol_dir / "ticks").mkdir(parents=True)
    (symbol_dir / "candles_60").mkdir(parents=True)
    
    ticks.to_parquet(symbol_dir / "ticks/data.parquet")
    candles.to_parquet(symbol_dir / "candles_60/data.parquet")
    
    return tmp_path

def test_dataset_warmup_calculation(mock_data_dir, mock_settings):
    """Verify that dataset correctly uses warmup steps for index calculation."""
    dataset = DerivDataset(
        data_source=mock_data_dir,
        settings=mock_settings,
        lookahead_candles=5
    )
    
    # valid_indices should start at candle_len + warmup_steps
    # 10 + 20 = 30
    assert dataset.valid_indices[0] == 30
    
    # Check that we have valid samples
    assert len(dataset) > 0

def test_dataset_warmup_features(mock_data_dir, mock_settings):
    """Verify that features returned have correct shape and valid indicators."""
    dataset = DerivDataset(
        data_source=mock_data_dir,
        settings=mock_settings,
        lookahead_candles=5
    )
    
    sample = dataset[0]
    
    # Shape check: Should be sequence_length (10), NOT (10+20)
    # The dataset trims the warmup part before returning
    assert sample["candles"].shape == (10, 10)  # (seq_len, features)
    assert sample["ticks"].shape == (20,)       # (seq_len_ticks,)
    
    # RSI check (Feature index 6 is RSI)
    # With 20 warmup steps, RSI (period 14) should be fully valid for the *entire* sequence of 10
    rsi = sample["candles"][:, 6]
    
    # If we didn't have warmup, the first few RSI values of the chunk passed to preprocessor would be NaN
    # (Preprocessor typically handles NaNs by zero-filling or similar, but let's check values are non-zero/valid)
    # In `data/processor.py`: rsi_norm = rsi_val / 100.0
    # `indicators.rsi` usually has nans at start.
    
    # Since we passed 30 candles (20 warmup + 10 seq) to feature builder,
    # and RSI needs ~14, the last 10 (which we get) should be fully populated.
    assert not torch.isnan(rsi).any()
    assert (rsi > 0).all() # Should be > 0 (dummy data is trending up, RSI > 0)
