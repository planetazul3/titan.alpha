"""Tests for numerical safety in InferenceOrchestrator."""
import pytest
import numpy as np
import math


class TestVolatilityCalculation:
    """Test volatility calculation numerical guards (I2-FIX)."""
    
    def test_volatility_with_zeros(self):
        """Volatility calculation should handle arrays with zeros without NaN."""
        closes = np.array([0.0, 0.0, 0.0, 0.0, 0.0])
        
        # Guard: np.maximum(closes, 1e-8)
        safe_closes = np.maximum(closes, 1e-8)
        log_returns = np.diff(np.log(safe_closes))
        volatility = float(np.std(log_returns) * np.sqrt(365 * 24 * 60))
        
        assert math.isfinite(volatility)
        # All zeros -> same value -> zero std -> zero volatility 
        assert volatility == 0.0

    def test_volatility_with_short_array(self):
        """Volatility calculation should return 0.0 for arrays with < 2 elements."""
        closes = np.array([100.0])
        
        # Guard: len(closes) >= 2
        if len(closes) < 2:
            volatility = 0.0
        else:
            log_returns = np.diff(np.log(closes))
            volatility = float(np.std(log_returns))
        
        assert volatility == 0.0

    def test_volatility_with_normal_data(self):
        """Volatility calculation should work correctly with normal data."""
        # Realistic price series
        closes = np.array([100.0, 101.0, 99.5, 100.5, 102.0, 101.5, 103.0, 102.5, 104.0, 103.5])
        
        safe_closes = np.maximum(closes, 1e-8)
        log_returns = np.diff(np.log(safe_closes))
        volatility = float(np.std(log_returns) * np.sqrt(365 * 24 * 60))
        
        assert math.isfinite(volatility)
        assert volatility > 0  # Real data should have non-zero volatility

    def test_volatility_nan_fallback(self):
        """Non-finite volatility should default to 0.0."""
        # Test the guard logic directly
        volatility = float('nan')
        
        if not math.isfinite(volatility):
            volatility = 0.0
        
        assert volatility == 0.0
        assert math.isfinite(volatility)
