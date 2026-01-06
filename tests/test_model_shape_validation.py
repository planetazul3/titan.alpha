
import unittest
from unittest.mock import MagicMock
import torch
from models.core import DerivOmniModel

class TestModelShapeValidation(unittest.TestCase):
    def setUp(self):
        self.mock_settings = MagicMock()
        self.mock_settings.data_shapes.sequence_length_ticks = 60
        self.mock_settings.data_shapes.sequence_length_candles = 60
        self.mock_settings.data_shapes.feature_dim_candles = 10
        self.mock_settings.data_shapes.feature_dim_volatility = 4
        
        # Hyperparams needed for init
        self.mock_settings.hyperparams.dropout_rate = 0.1
        self.mock_settings.hyperparams.fusion_dropout = 0.1
        self.mock_settings.hyperparams.head_dropout = 0.1
        self.mock_settings.hyperparams.latent_dim = 16
        self.mock_settings.hyperparams.lstm_hidden_size = 32
        self.mock_settings.hyperparams.cnn_filters = 16
        self.mock_settings.hyperparams.use_tft = False
        self.mock_settings.hyperparams.temporal_embed_dim = 16 # Added mock for embedding dimensions
        self.mock_settings.hyperparams.spatial_embed_dim = 16
        self.mock_settings.hyperparams.fusion_output_dim = 32

    def test_valid_shapes_pass(self):
        """Valid shapes should not raise error."""
        model = DerivOmniModel(self.mock_settings)
        batch = 2
        ticks = torch.randn(batch, 60, dtype=torch.float32)
        candles = torch.randn(batch, 60, 10, dtype=torch.float32)
        vol = torch.randn(batch, 4, dtype=torch.float32)
        
        # Should not raise
        model(ticks, candles, vol)
        
    def test_invalid_ticks_shape_raises(self):
        """Invalid ticks length should raise ValueError."""
        model = DerivOmniModel(self.mock_settings)
        batch = 2
        ticks = torch.randn(batch, 59, dtype=torch.float32) # Wrong length
        candles = torch.randn(batch, 60, 10, dtype=torch.float32)
        vol = torch.randn(batch, 4, dtype=torch.float32)
        
        with self.assertRaises(ValueError) as cm:
            model(ticks, candles, vol)
        self.assertIn("Invalid ticks shape", str(cm.exception))

    def test_invalid_candles_shape_raises(self):
        """Invalid candles dimensions should raise ValueError."""
        model = DerivOmniModel(self.mock_settings)
        batch = 2
        ticks = torch.randn(batch, 60, dtype=torch.float32)
        # Wrong features (9 vs 10)
        candles = torch.randn(batch, 60, 9, dtype=torch.float32)
        vol = torch.randn(batch, 4, dtype=torch.float32)
        
        with self.assertRaises(ValueError) as cm:
            model(ticks, candles, vol)
        self.assertIn("Invalid candles shape", str(cm.exception))
        
    def test_invalid_vol_shape_raises(self):
        """Invalid volatility dimensions should raise ValueError."""
        model = DerivOmniModel(self.mock_settings)
        batch = 2
        ticks = torch.randn(batch, 60, dtype=torch.float32)
        candles = torch.randn(batch, 60, 10, dtype=torch.float32)
        # Wrong vol dims
        vol = torch.randn(batch, 3, dtype=torch.float32)
        
        with self.assertRaises(ValueError) as cm:
            model(ticks, candles, vol)
        self.assertIn("Invalid vol_metrics shape", str(cm.exception))

if __name__ == '__main__':
    unittest.main()
