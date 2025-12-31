import pytest
import numpy as np
import torch
from data.normalizers import z_score_normalize, robust_scale
from data.features import FeatureBuilder
from config.settings import Settings

def test_z_score_constant_input():
    """Test z_score_normalize with constant input (std=0)."""
    # 1. Constant array of zeros
    zeros = np.zeros(100, dtype=np.float32)
    norm_zeros = z_score_normalize(zeros)
    assert np.all(norm_zeros == 0)
    assert not np.any(np.isnan(norm_zeros))
    assert not np.any(np.isinf(norm_zeros))
    
    # 2. Constant array of large numbers
    const = np.full(100, 1e6, dtype=np.float32)
    norm_const = z_score_normalize(const)
    assert np.all(norm_const == 0)
    
    # 3. Rolling window on constant array
    norm_rolling = z_score_normalize(const, window=10)
    assert not np.any(np.isnan(norm_rolling))
    # First 9 are 0 (padded nan -> 0), rest are 0
    assert np.all(norm_rolling == 0)

def test_z_score_near_zero_variance():
    """Test z_score_normalize with extremely small variance."""
    # Fluctuations smaller than epsilon?
    # epsilon is 1e-8.
    val = np.array([1.0, 1.0 + 1e-9, 1.0], dtype=np.float32)
    # std will be very small
    epsilon = 1e-8
    norm = z_score_normalize(val, epsilon=epsilon)
    # Should not crash
    assert not np.any(np.isinf(norm))

def test_feature_builder_constant_data():
    """Test FeatureBuilder with constant market data."""
    settings = Settings(environment="test", deriv_api_token="dummy_token")
    builder = FeatureBuilder(settings)
    
    # Constant ticks
    ticks = np.full(100, 100.0, dtype=np.float32)
    
    # Constant candles (O=H=L=C=100, V=0)
    candles = np.zeros((100, 6), dtype=np.float32)
    candles[:, 0:4] = 100.0
    candles[:, 5] = np.arange(100) # Epochs
    
    # Build features
    features = builder.build(ticks, candles)
    
    # Check for NaNs
    assert not torch.isnan(features['ticks']).any()
    assert not torch.isnan(features['candles']).any()
    assert not torch.isnan(features['vol_metrics']).any()
    
    # Ticks should be normalized. Log returns of constant = 0.
    # z-score of 0s = 0s.
    assert torch.all(features['ticks'] == 0)

if __name__ == "__main__":
    test_z_score_constant_input()
    test_z_score_near_zero_variance()
    try:
        test_feature_builder_constant_data()
        print("FeatureBuilder test passed")
    except Exception as e:
        print(f"FeatureBuilder test failed: {e}")
