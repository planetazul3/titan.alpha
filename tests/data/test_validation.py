
import unittest
import numpy as np
import pandas as pd
from unittest.mock import MagicMock
from data.features import FeatureBuilder
from config.settings import Settings

class TestFeatureBuilderValidation(unittest.TestCase):
    def setUp(self):
        self.settings = MagicMock(spec=Settings)
        # Mock nested config
        self.settings.data_shapes = MagicMock()
        self.settings.data_shapes.sequence_length_ticks = 10
        self.settings.data_shapes.sequence_length_candles = 5
        self.settings.data_shapes.feature_dim_candles = 6
        self.settings.normalization = MagicMock()
        self.settings.trading = MagicMock()
        self.settings.trading.stale_candle_threshold = 1000.0 # High threshold to avoid staleness error
        
        self.builder = FeatureBuilder(self.settings)
        
        # Valid base data
        # [Open, High, Low, Close, Volume, Timestamp]
        self.valid_candles = np.array([
            [100.0, 105.0, 95.0, 102.0, 1000.0, 10000.0],
            [102.0, 103.0, 101.0, 103.0, 1500.0, 10060.0],
            [103.0, 104.0, 102.0, 102.5, 1200.0, 10120.0],
            [102.5, 106.0, 102.0, 105.0, 1800.0, 10180.0],
            [105.0, 107.0, 104.0, 106.0, 2000.0, 10240.0]
        ])
        
        self.valid_ticks = np.ones(10) * 100.0

    def test_valid_data(self):
        """Test valid data passes."""
        try:
            self.builder.build(self.valid_ticks, self.valid_candles, validate=True)
        except Exception as e:
            self.fail(f"Valid data raised exception: {e}")

    def test_negative_price(self):
        """Test negative price rejection."""
        bad_candles = self.valid_candles.copy()
        bad_candles[0, 0] = -100.0 # Negative open
        
        with self.assertRaises(ValueError) as cm:
            self.builder.build(self.valid_ticks, bad_candles, validate=True)
        
        self.assertIn("Invalid market data structure", str(cm.exception))
        print(f"\nCaught expected negative price error: {cm.exception}")

    def test_high_less_than_low(self):
        """Test High < Low rejection."""
        bad_candles = self.valid_candles.copy()
        # High (105) < Low (106)
        bad_candles[0, 1] = 105.0
        bad_candles[0, 2] = 106.0
        
        with self.assertRaises(ValueError) as cm:
            self.builder.build(self.valid_ticks, bad_candles, validate=True)
            
        self.assertIn("Invalid market data structure", str(cm.exception))
        print(f"\nCaught expected H < L error: {cm.exception}")

    def test_high_less_than_open(self):
        """Test High < Open rejection."""
        bad_candles = self.valid_candles.copy()
        # High (90) < Open (100)
        bad_candles[0, 1] = 90.0
        bad_candles[0, 0] = 100.0
        bad_candles[0, 2] = 80.0 # Low valid below both
        
        with self.assertRaises(ValueError) as cm:
            self.builder.build(self.valid_ticks, bad_candles, validate=True)

        self.assertIn("Invalid market data structure", str(cm.exception))
        print(f"\nCaught expected H < O error: {cm.exception}")

if __name__ == "__main__":
    unittest.main()
