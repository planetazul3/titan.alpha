import pytest
import numpy as np
from unittest.mock import Mock, patch
from data.processor import VolatilityMetricsExtractor

@pytest.fixture
def extractor():
    settings = Mock()
    # Mock normalization factors to returns floats, not Mocks
    settings.normalization.norm_factor_volatility = 1.0
    settings.normalization.norm_factor_atr = 1.0
    settings.normalization.norm_factor_rsi_std = 1.0
    settings.normalization.norm_factor_bb_width = 1.0
    return VolatilityMetricsExtractor(settings)

@patch("data.processor.indicators")
def test_extract_handles_nans(mock_indicators, extractor):
    """Verify that extract returns clean floats even when calculations produce NaNs."""
    
    # 1. Setup valid input to pass initial validation checks (positive prices)
    candles = np.ones((20, 6)) * 100.0 
    
    # 2. Mock indicators to return NaNs or Infs
    # This simulates "division by zero" or "instability" deep in the math
    mock_indicators.rsi.return_value = np.array([np.nan] * 20)
    mock_indicators.atr.return_value = np.array([np.inf] * 20)
    mock_indicators.bollinger_bands.return_value = (np.zeros(20), np.zeros(20), np.zeros(20))
    mock_indicators.bollinger_bandwidth.return_value = np.array([np.nan] * 20)
    
    # 3. patch normalizers if needed, but let's see if this is enough.
    # The extractor calls indicators.rsi, indicators.atr, etc.
    
    metrics = extractor.extract(candles)
    
    assert isinstance(metrics, np.ndarray)
    # The nan_to_num should convert NaNs to 0.0 and Inf to 1.0 (posinf arg)
    print(f"Metrics: {metrics}")
    
    assert not np.isnan(metrics).any()
    assert not np.isinf(metrics).any()
    
    # Check bounds (clip ensures [0, 1])
    assert np.all(metrics >= 0.0)
    assert np.all(metrics <= 1.0)
    
def test_extract_clean_data(extractor):
    """Verify standard behavior on clean data."""
    # Random realistic data
    # Price ~ 100
    candles = np.random.normal(100, 1, (30, 6))
    candles[:, 1] = candles[:, 0] + 0.5 # High > Open
    candles[:, 2] = candles[:, 0] - 0.5 # Low < Open
    candles[:, 3] = candles[:, 0] + 0.1 # Close
    
    metrics = extractor.extract(candles)
    
    assert not np.isnan(metrics).any()
    assert np.all(metrics >= 0.0)
    assert np.all(metrics <= 1.0)
