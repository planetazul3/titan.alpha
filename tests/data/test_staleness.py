
import unittest
from unittest.mock import MagicMock
import numpy as np
import time
from data.features import FeatureBuilder, StaleDataError
from config.settings import Settings

class TestFeatureBuilderStaleness(unittest.TestCase):
    def setUp(self):
        # Mock settings
        self.settings = MagicMock(spec=Settings)
        # Mock nested config objects
        self.settings.data_shapes = MagicMock()
        self.settings.data_shapes.sequence_length_ticks = 10
        self.settings.data_shapes.sequence_length_candles = 5
        self.settings.data_shapes.feature_dim_candles = 6
        
        self.settings.normalization = MagicMock()
        
        # Set staleness threshold to 10 seconds
        self.settings.trading = MagicMock()
        self.settings.trading.stale_candle_threshold = 10.0
        
        self.builder = FeatureBuilder(self.settings)
        
        # Create dummy data
        # Candles: [open, high, low, close, volume, timestamp]
        # Create 5 candles ending at t=1000
        self.last_ts = 1000.0
        self.candles = np.zeros((5, 6))
        self.candles[:, 5] = [self.last_ts - i*60 for i in range(5)][::-1] # 1000, 940, ...
        self.candles[:, 0:4] = 100.0 # Open, High, Low, Close = 100
        
        self.ticks = np.ones(10) * 100.0

    def test_fresh_data(self):
        """Test that fresh data passes validation."""
        # Current time is 1005 (latency 5s < 10s)
        current_time = self.last_ts + 5.0
        try:
            self.builder.build(self.ticks, self.candles, validate=False, timestamp=current_time)
        except StaleDataError:
            self.fail("FeatureBuilder raised StaleDataError unexpectedly for fresh data!")

    def test_stale_data(self):
        """Test that stale data raises StaleDataError."""
        # Current time is 1015 (latency 15s > 10s)
        current_time = self.last_ts + 15.0
        with self.assertRaises(StaleDataError) as cm:
            self.builder.build(self.ticks, self.candles, validate=False, timestamp=current_time)
        
        print(f"\nCaught expected error: {cm.exception}")
        self.assertIn("Data is STALE", str(cm.exception))
        self.assertIn("Latency: 15.00s", str(cm.exception))

    def test_no_timestamp(self):
        """Test that omitted timestamp skips check."""
        try:
            self.builder.build(self.ticks, self.candles, validate=False, timestamp=None)
        except StaleDataError:
            self.fail("FeatureBuilder raised StaleDataError when timestamp was None!")

    def test_numpy_method(self):
        """Test that build_numpy also enforces staleness."""
        current_time = self.last_ts + 20.0
        with self.assertRaises(StaleDataError):
            self.builder.build_numpy(self.ticks, self.candles, validate=False, timestamp=current_time)

if __name__ == "__main__":
    unittest.main()
