import shutil
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pandas as pd

from config.settings import Settings
from data.dataset import DerivDataset


class TestDatasetAlignment(unittest.TestCase):
    def setUp(self):
        # Create temp directory
        self.test_dir = Path(tempfile.mkdtemp())
        self.settings = Settings()

        # Override data shapes for testing
        self.settings.data_shapes.sequence_length_ticks = 10
        self.settings.data_shapes.sequence_length_candles = 1

        # Generate synthetic data with IRREGULAR tick rates
        # Candle 1: 0-60s (ticks at 1, 2, ..., 10) -> very sparse
        # Candle 2: 60-120s (ticks at 61, 62, ..., 100) -> dense

        # Ticks:
        # 0-60s: 10 ticks (timestamps 1, 5, 10, ..., 45) - gaps of 5s
        t1 = np.arange(1, 50, 5)  # 10 ticks
        # 60-120s: 60 ticks (timestamps 61, 62, ..., 120) - gaps of 1s
        t2 = np.arange(61, 121, 1)  # 60 ticks

        self.epochs = np.concatenate([t1, t2])
        self.quotes = np.random.randn(len(self.epochs)).astype(np.float32)

        # Generate sufficient dummy candles to satisfy DerivDataset requirements
        # Need at least sequence_length_candles (5) + lookahead (5) = 10 candles
        # We will create 20 candles, spaced 60s apart
        num_candles = 20
        self.candle_epochs = np.arange(60, 60 * (num_candles + 1), 60, dtype=np.float32)
        self.candles = np.zeros((num_candles, 5), dtype=np.float32)
        self.candles[:, 4] = self.candle_epochs

        # Ticks:
        # Candle 1 (60s): sparse (10 ticks)
        # Candle 2 (120s): dense (60 ticks)
        # We only care about testing alignment for these, but ticks need to cover range
        # Let's generate dummy ticks for all other periods to be safe

        # Specific test data for alignment check:
        # 0-60s: ticks 1..45 (step 5)
        t1 = np.arange(1, 50, 5)
        # 60-120s: ticks 61..120 (step 1)
        t2 = np.arange(61, 121, 1)

        self.epochs = np.concatenate([t1, t2])
        # Generate positive prices (100 +/- random)
        self.quotes = (100.0 + np.random.randn(len(self.epochs))).astype(np.float32)

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

        # Access index corresponding to 2nd candle (epoch 120)
        # Should be index 0 in valid_indices if we have enough history...
        # Wait, the dataset needs minimal history.
        # self.candle_len is 5. We only have 2 candles.
        # Dataset will raise error unless we hack it or make more data.

        dataset.ticks = self.quotes
        dataset.tick_epochs = self.epochs
        # self.candles has shape (20, 5). We want to add volume column (zeros) and duplicate epoch
        # Result shape: (20, 7) -> [OHLC, Vol, Epoch, ExtraEpoch?] -> Actually dataset.py logic for volume is different
        # dataset.py: if shape is 5 cols (OHLC, Epoch), it inserts Volume at index 4.
        # But here we are manually setting dataset.candles.
        # let's just use self.candles directly but ensure it has 6 columns like loaded data?
        # Loaded data: [Open, High, Low, Close, Volume, Epoch]

        # self.candles currently: [Open, High, Low, Close, Epoch]
        # We need to insert Volume before Epoch.

        vol_col = np.zeros((len(self.candles), 1), dtype=np.float32)
        ohlc = self.candles[:, :4]
        epoch = self.candles[:, 4:5]

        dataset.candles = np.hstack([ohlc, vol_col, epoch])

        # Hack lookup indices
        # We want to test candle with epoch 120 (index 1 in our array)
        # Tick epochs go up to 120.
        # So we should get the ticks ending at epoch 120.

        dataset.__getitem__(0)  # This will likely fail due to indices calculation

        # Instead, let's call the logic directly if possible or mock valid_indices
        dataset.valid_indices = [
            2
        ]  # Candle index 2 means we use history ending at candles[1] (epoch 120)

        # Mock FeatureBuilder to just return inputs
        dataset.feature_builder.build_numpy = MagicMock(
            side_effect=lambda ticks, candles, validate: {
                "ticks": ticks,
                "candles": candles,
                "vol_metrics": np.zeros(4),
            }
        )
        dataset._generate_labels = MagicMock(return_value={})

        # Act
        sample = dataset[0]  # uses valid_indices[0] -> 2

        # Assert
        ticks_out = sample["ticks"].numpy()

        # Expected behavior:
        # Candle timestamp = 120
        # Ticks should be those <= 120.
        # Last tick is 120.
        # So expected last tick value is self.quotes[-1]

        # The alignment bug would have picked index = 1 * 60 = 60.
        # Total ticks is 70 (10 + 60).
        # We want the logic to find ALL ticks up to 120.

        # Check last tick value
        last_tick_val = ticks_out[-1]
        expected_val = self.quotes[np.searchsorted(self.epochs, 120, side="right") - 1]

        self.assertEqual(last_tick_val, expected_val, "Last tick value mismatch!")

        # Verify alignment
        # The slice length is 10.
        # The last 10 ticks in our data are 111...120.
        # So we expect the output to strictly match those.
        expected_slice = self.quotes[-10:]
        np.testing.assert_array_equal(ticks_out, expected_slice)


if __name__ == "__main__":
    unittest.main()
