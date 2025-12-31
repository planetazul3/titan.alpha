import shutil
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pandas as pd

from config.settings import Settings
from data.dataset import DerivDataset
from data.features import reset_feature_builder


class TestDatasetAlignment(unittest.TestCase):
    def setUp(self):
        # Create temp directory
        self.test_dir = Path(tempfile.mkdtemp())
        
        # Reset singleton to ensure fresh settings are used
        reset_feature_builder()
        
        self.settings = Settings(environment="test", deriv_api_token="dummy_token")

        # Override data shapes for testing
        self.settings.data_shapes.sequence_length_ticks = 10
        self.settings.data_shapes.sequence_length_candles = 1
        # Set warmup to 1 to ensure the first valid index targets the second candle (epoch 120)
        # min_candle_idx = candle_len (1) + warmup (1) = 2.
        # Index 2 corresponds to the slice ending at index 2 (exclusive), i.e., candle[1].
        self.settings.data_shapes.warmup_steps = 1

        # Generate synthetic data with IRREGULAR tick rates
        # Candle 1: 0-60s (ticks at 1, 2, ..., 10) -> very sparse
        # Candle 2: 60-120s (ticks at 61, 62, ..., 100) -> dense

        # Ticks:
        # Ticks:
        # 0-60s: 10 ticks (timestamps 1, 5, 10, ..., 45) - gaps of 5s
        t1 = np.arange(1, 50, 5)  # 10 ticks
        # 60-120s: 60 ticks (timestamps 61, 62, ..., 120) - gaps of 1s
        t2 = np.arange(61, 121, 1)  # 60 ticks
        # Rest: Dummy ticks to satisfy length requirements
        # We need enough data for ~60 candles of history
        t3 = np.arange(121, 7000, 1) # generous buffer

        self.epochs = np.concatenate([t1, t2, t3])
        self.quotes = (100.0 + np.random.randn(len(self.epochs))).astype(np.float32)

        # Generate sufficient dummy candles to satisfy DerivDataset requirements
        # Need at least sequence_length_candles + lookahead + indicators warmup (~50-60)
        num_candles = 100
        self.candle_epochs = np.arange(60, 60 * (num_candles + 1), 60, dtype=np.float32)
        # Initialize with positive prices (100.0) to pass log_returns check
        self.candles = np.ones((num_candles, 5), dtype=np.float32) * 100.0
        self.candles[:, 4] = self.candle_epochs

        # Save to parquet
        df_ticks = pd.DataFrame({"epoch": self.epochs, "quote": self.quotes})
        df_candles = pd.DataFrame(self.candles, columns=["open", "high", "low", "close", "epoch"])

        df_ticks.to_parquet(self.test_dir / "synthetic_ticks_data.parquet")
        df_candles.to_parquet(self.test_dir / "synthetic_candles_data.parquet")

    def tearDown(self):
        shutil.rmtree(self.test_dir)

    def test_alignment_logic(self):
        """Verify that dataset pulls ticks based on timestamp, NOT index."""
        dataset = DerivDataset(self.test_dir, self.settings, mode="train")

        # Mock FeatureBuilder to just return inputs
        dataset.feature_builder.build_numpy = MagicMock(
            side_effect=lambda ticks, candles, validate: {
                "ticks": ticks,
                "candles": candles,
                "vol_metrics": np.zeros(4),
            }
        )
        dataset._generate_labels = MagicMock(return_value={})

        # Access index corresponding to 2nd candle (epoch 120)
        # valid_indices starts at min_candle_idx = candle_len (1) + warmup (1) = 2
        # Index 0 in valid_indices corresponds to candle index 2
        # __getitem__ uses candle_end = candle_idx = 2
        # Timestamp comes from candles[candle_end - 1] = candles[1] (epoch 120)
        sample = dataset[0]

        # Assert
        ticks_out = sample["ticks"].numpy()

        # Expected behavior:
        # Candle timestamp = 120
        # Ticks should be those <= 120.
        # Last tick is 120.
        # So expected last tick value is self.quotes[-1] where epoch is 120

        # Check last tick value
        # Find index in self.epochs where epoch is 120
        expected_idx = np.searchsorted(self.epochs, 120, side="right") - 1
        expected_val = self.quotes[expected_idx]

        last_tick_val = ticks_out[-1]
        self.assertEqual(last_tick_val, expected_val, "Last tick value mismatch!")

        # Verify alignment
        # The slice length is 10.
        # Ticks up to epoch 120 are indices 0..69 (10+60).
        # We want the last 10 of that set (indices 60..69).
        expected_slice = self.quotes[60:70]
        np.testing.assert_array_equal(ticks_out[-10:], expected_slice)


if __name__ == "__main__":
    unittest.main()
