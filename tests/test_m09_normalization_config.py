
import pytest
import numpy as np
from unittest.mock import MagicMock
from config.settings import Settings, NormalizationConfig
from data.processor import VolatilityMetricsExtractor

class TestNormalizationConfig:
    
    def test_normalization_override(self):
        """Verify that extractor uses values from settings."""
        # Create settings with custom normalization
        settings = Settings(environment="test", deriv_api_token="dummy_token")
        # Set all factors to 1.0 for easy checking
        settings.normalization.norm_factor_volatility = 1.0
        settings.normalization.norm_factor_atr = 1.0
        settings.normalization.norm_factor_rsi_std = 1.0
        settings.normalization.norm_factor_bb_width = 1.0
        
        extractor = VolatilityMetricsExtractor(settings)
        
        # Create dummy candles that would produce non-zero metrics
        # We need > 20 candles
        # Let's mock the internal calculation results to isolate scaling logic?
        # A bit hard since extract() does it all.
        # Instead, let's create simple candles where metrics are calculable.
        
        # Create a sine wave to have predictable vol/rsi
        t = np.linspace(0, 10, 100)
        closes = np.sin(t) + 100
        highs = closes + 1
        lows = closes - 1
        opens = closes # irrelevant
        volumes = np.ones_like(closes)
        times = np.arange(100)
        
        candles = np.stack([opens, highs, lows, closes, volumes, times], axis=1)
        
        # Run extraction with 1.0 scaling
        metrics_1 = extractor.extract(candles)
        
        # Now change settings to 2.0
        settings.normalization.norm_factor_volatility = 2.0
        settings.normalization.norm_factor_atr = 2.0
        settings.normalization.norm_factor_rsi_std = 2.0
        settings.normalization.norm_factor_bb_width = 2.0
        
        extractor_2 = VolatilityMetricsExtractor(settings)
        metrics_2 = extractor_2.extract(candles)
        
        # Check that metrics_2 is roughly 2x metrics_1 (ignoring clipping)
        # Note: extract() clips [0,1]. We must ensure metrics_1 are small enough not to clip.
        # With 1.0 scaling on raw values, they might be small.
        # Volatility of sine wave is ~0.7, so 1.0 scaling -> 0.7. 
        # 2.0 scaling -> 1.4 -> clipped to 1.0. 
        # So we can't just assert doubling.
        
        # Let's use extremely small factors for first run
        settings.normalization.norm_factor_volatility = 0.0001
        settings.normalization.norm_factor_atr = 0.0001
        settings.normalization.norm_factor_rsi_std = 0.0001
        settings.normalization.norm_factor_bb_width = 0.0001
        
        extractor_small = VolatilityMetricsExtractor(settings)
        metrics_small = extractor_small.extract(candles)
        
        # Then 100x factors
        settings.normalization.norm_factor_volatility = 0.01
        extractor_med = VolatilityMetricsExtractor(settings)
        metrics_med = extractor_med.extract(candles)
        
        # First metric (volatility) should be exactly 100x larger
        assert np.isclose(metrics_med[0], metrics_small[0] * 100, rtol=0.01)

if __name__ == "__main__":
    pytest.main([__file__])
