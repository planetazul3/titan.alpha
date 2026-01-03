
import unittest
import os
from pydantic import ValidationError
from config.settings import Settings, Thresholds, ShadowTradeConfig
import logging

class TestConfigValidation(unittest.TestCase):
    def setUp(self):
        # Setup minimal valid env vars
        self.env_patcher = unittest.mock.patch.dict(os.environ, {
            "TRADING__SYMBOL": "R_100",
            "TRADING__STAKE_AMOUNT": "10.0",
            "THRESHOLDS__CONFIDENCE_THRESHOLD_HIGH": "0.8",
            "THRESHOLDS__LEARNING_THRESHOLD_MIN": "0.4",
            "THRESHOLDS__LEARNING_THRESHOLD_MAX": "0.7",
            "HYPERPARAMS__LEARNING_RATE": "0.001",
            "HYPERPARAMS__BATCH_SIZE": "32",
            "HYPERPARAMS__LSTM_HIDDEN_SIZE": "64",
            "HYPERPARAMS__CNN_FILTERS": "32",
            "HYPERPARAMS__LATENT_DIM": "16",
            "DATA_SHAPES__SEQUENCE_LENGTH_TICKS": "60",
            "DATA_SHAPES__SEQUENCE_LENGTH_CANDLES": "60",
            "SHADOW_TRADE__MIN_PROBABILITY_TRACK": "0.4",
            "ENVIRONMENT": "development"
        })
        self.env_patcher.start()

    def tearDown(self):
        self.env_patcher.stop()

    def test_valid_thresholds(self):
        """Test strict validation of threshold ordering."""
        # Valid case: 0.4 < 0.7 < 0.8
        t = Thresholds(
            confidence_threshold_high=0.8,
            learning_threshold_min=0.4,
            learning_threshold_max=0.7
        )
        self.assertIsInstance(t, Thresholds)

    def test_invalid_threshold_order(self):
        """Test failure when max < min."""
        with self.assertRaises(ValidationError) as cm:
            Thresholds(
                confidence_threshold_high=0.8,
                learning_threshold_min=0.6,
                learning_threshold_max=0.5  # Invalid: max < min
            )
        self.assertIn("Thresholds must satisfy", str(cm.exception))

    def test_shadow_consistency_warning(self):
        """Test that inconsistent shadow config logs a warning."""
        # Set shadow tracking higher than learning min (0.5 > 0.4)
        with unittest.mock.patch.dict(os.environ, {"SHADOW_TRADE__MIN_PROBABILITY_TRACK": "0.5"}):
            with self.assertLogs("config.settings", level="WARNING") as log:
                Settings()
                # Check for specific warning message
                self.assertTrue(any("shadow_trade.min_probability_track" in m for m in log.output))

    def test_shadow_consistency_ok(self):
        """Test that consistent config generates no warning."""
        # Set shadow tracking equal to learning min
        with unittest.mock.patch.dict(os.environ, {"SHADOW_TRADE__MIN_PROBABILITY_TRACK": "0.4"}):
            # Assert NO logs at warning level
            with self.assertNoLogs("config.settings", level="WARNING"):
                Settings()

    def test_production_safety_check(self):
        """Test production security constraints."""
        with unittest.mock.patch.dict(os.environ, {
            "ENVIRONMENT": "production", 
            "PYTEST_CURRENT_TEST": "True"
        }):
            with self.assertRaises(RuntimeError) as cm:
                Settings()
            self.assertIn("SECURITY EXCEPTION", str(cm.exception))

if __name__ == '__main__':
    # Need mock import
    import unittest.mock
    unittest.main()
