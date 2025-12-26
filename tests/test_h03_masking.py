import pytest
import torch
import numpy as np
from pathlib import Path
from unittest.mock import MagicMock, patch
from config.settings import Settings
from models.core import DerivOmniModel
from models.tft import TemporalFusionTransformer
from data.dataset import DerivDataset

@pytest.fixture
def settings():
    return Settings()

def test_tft_masking_integration():
    """Test that TFT accepts and handles mask."""
    # Create TFT
    tft = TemporalFusionTransformer(input_size=10, hidden_size=16, num_heads=4)
    
    # Input data (batch=2, seq=20, features=10)
    x = torch.randn(2, 20, 10)
    
    # Mask (batch=2, seq=20)
    # Mask out first 5 steps
    mask = torch.ones(2, 20)
    mask[:, :5] = 0
    
    # Forward pass
    output, attn_weights, _ = tft(x, mask=mask)
    
    # Check shapes
    assert output.shape == (2, 20, 16)
    assert attn_weights.shape == (2, 20, 20)
    
    # Check if attention weights reflect masking (approximate check)
    # If masked, attention from later steps to masked steps should be very small
    # attn_weights[b, target_t, source_t]
    # Check attention at t=19 (last step) to t=0 (masked)
    # It should be close to 0
    assert torch.all(attn_weights[:, 19, :5] < 0.1) 

def test_deriv_dataset_masks(tmp_path):
    """Test DerivDataset returns masks."""
    # Mock settings
    s = Settings()
    
    # Create dummy parquet
    import pandas as pd
    
    # Causal ticks
    ticks_df = pd.DataFrame({
        "quote": np.abs(np.random.randn(500)) + 100.0,
        "epoch": np.arange(500)
    })
    
    # Candles
    candles_df = pd.DataFrame({
        "open": np.abs(np.random.randn(500)) + 100.0,
        "high": np.abs(np.random.randn(500)) + 100.0,
        "low": np.abs(np.random.randn(500)) + 100.0,
        "close": np.abs(np.random.randn(500)) + 100.0,
        "epoch": np.arange(500)
    })
    
    # Create symbol directory
    symbol_dir = tmp_path / "R_100"
    (symbol_dir / "ticks").mkdir(parents=True)
    (symbol_dir / "candles_60").mkdir(parents=True)
    
    ticks_df.to_parquet(symbol_dir / "ticks" / "ticks.parquet")
    candles_df.to_parquet(symbol_dir / "candles_60" / "candles.parquet")
    
    # Create dataset
    ds = DerivDataset(tmp_path, s)
    
    # Get sample
    sample = ds[0]
    
    assert "ticks_mask" in sample
    assert "candles_mask" in sample
    assert sample["ticks_mask"].shape == sample["ticks"].shape
    assert sample["candles_mask"].shape[0] == sample["candles"].shape[0]

def test_model_forward_with_masks(settings):
    """Test full model forward pass with masks."""
    model = DerivOmniModel(settings)
    
    # Mock inputs
    ticks = torch.randn(2, settings.data_shapes.sequence_length_ticks)
    candles = torch.randn(2, settings.data_shapes.sequence_length_candles, 10)
    vol = torch.randn(2, 4)
    
    # Masks
    masks = {
        "ticks_mask": torch.ones_like(ticks),
        "candles_mask": torch.ones(2, settings.data_shapes.sequence_length_candles)
    }
    
    # Forward
    logits = model(ticks, candles, vol, masks=masks)
    
    assert "rise_fall_logit" in logits
