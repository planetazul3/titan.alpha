"""
Unit tests for neural network models.

Tests model initialization, forward pass, and output shapes.
"""

from unittest.mock import MagicMock

import pytest
import torch


@pytest.fixture
def mock_settings():
    """Create mock settings for models."""
    settings = MagicMock()
    settings.data_shapes.sequence_length_ticks = 100
    settings.data_shapes.sequence_length_ticks = 100
    settings.data_shapes.sequence_length_candles = 50
    settings.data_shapes.feature_dim_ticks = 1
    settings.data_shapes.feature_dim_candles = 10
    settings.hyperparams.dropout_rate = 0.1
    settings.hyperparams.fusion_dropout = 0.2
    settings.hyperparams.head_dropout = 0.1
    settings.hyperparams.latent_dim = 16
    settings.hyperparams.lstm_hidden_size = 64
    settings.hyperparams.cnn_filters = 32
    settings.hyperparams.use_tft = False  # Test BiLSTM path by default
    return settings


@pytest.fixture
def device():
    """Get test device."""
    return torch.device("cpu")


class TestBuildingBlocks:
    """Tests for model building blocks."""

    def test_bilstm_block_output_shape(self):
        """Test BiLSTM block output dimensions."""
        from models.blocks import BiLSTMBlock

        lstm = BiLSTMBlock(input_size=64, hidden_size=32, num_layers=2)
        x = torch.randn(4, 50, 64)  # (batch, seq, features)

        output, (h_n, c_n) = lstm(x)

        assert output.shape == (4, 50, 64)  # 2 * hidden_size due to bidirectional

    def test_bilstm_invalid_input_raises(self):
        """Invalid input size should raise during construction."""
        from models.blocks import BiLSTMBlock

        with pytest.raises(ValueError, match="must be positive"):
            BiLSTMBlock(input_size=-1, hidden_size=32)

    def test_mlp_block_output_shape(self):
        """Test MLP block output dimensions."""
        from models.blocks import MLPBlock

        mlp = MLPBlock([64, 32, 16])
        x = torch.randn(4, 64)

        output = mlp(x)

        assert output.shape == (4, 16)

    def test_mlp_with_dropout(self):
        """Test MLP with dropout active in training mode."""
        from models.blocks import MLPBlock

        mlp = MLPBlock([64, 32, 16], dropout=0.5)
        mlp.train()
        x = torch.randn(4, 64)

        output = mlp(x)

        assert output.shape == (4, 16)

    def test_conv1d_block_output_shape(self):
        """Test Conv1D block output dimensions."""
        from models.blocks import Conv1DBlock

        conv = Conv1DBlock(in_channels=1, out_channels=32, kernel_size=3)
        x = torch.randn(4, 1, 100)  # (batch, channels, seq)

        output = conv(x)

        assert output.shape[0] == 4
        assert output.shape[1] == 32


class TestDerivOmniModel:
    """Tests for the main model."""

    def test_model_initialization(self, mock_settings):
        """Model should initialize without errors."""
        from models.core import DerivOmniModel

        model = DerivOmniModel(mock_settings)

        assert model is not None
        assert model.count_parameters() > 0

    def test_model_forward_pass(self, mock_settings):
        """Forward pass should produce correct output shapes."""
        from models.core import DerivOmniModel

        model = DerivOmniModel(mock_settings)
        model.eval()

        batch_size = 4
        ticks = torch.randn(batch_size, 100)  # seq_len_ticks
        candles = torch.randn(batch_size, 50, 10)  # seq_len_candles, features
        vol_metrics = torch.randn(batch_size, 4)

        outputs = model(ticks, candles, vol_metrics)

        assert "rise_fall_logit" in outputs
        assert "touch_logit" in outputs
        assert "range_logit" in outputs
        assert outputs["rise_fall_logit"].shape == (batch_size, 1)

    def test_model_predict_probs(self, mock_settings):
        """Predict probs should return values in [0, 1]."""
        from models.core import DerivOmniModel

        model = DerivOmniModel(mock_settings)
        model.eval()

        batch_size = 2
        ticks = torch.randn(batch_size, 100)
        candles = torch.randn(batch_size, 50, 10)
        vol_metrics = torch.randn(batch_size, 4)

        probs = model.predict_probs(ticks, candles, vol_metrics)

        assert "rise_fall_prob" in probs
        # Probabilities should be in [0, 1] after sigmoid
        assert (probs["rise_fall_prob"] >= 0).all()
        assert (probs["rise_fall_prob"] <= 1).all()

    def test_model_gradient_flow(self, mock_settings):
        """Gradients should flow back through all parameters."""
        from models.core import DerivOmniModel

        model = DerivOmniModel(mock_settings)
        model.train()

        ticks = torch.randn(2, 100)
        candles = torch.randn(2, 50, 10)
        vol_metrics = torch.randn(2, 4)

        outputs = model(ticks, candles, vol_metrics)

        # Sum all logits to verify gradient flow to all heads
        loss = 0.0
        if "rise_fall_logit" in outputs:
            loss += outputs["rise_fall_logit"].sum()
        if "touch_logit" in outputs:
            loss += outputs["touch_logit"].sum()
        if "range_logit" in outputs:
            loss += outputs["range_logit"].sum()

        loss.backward()

        # Check that parameters have gradients
        grad_count = sum(1 for p in model.parameters() if p.grad is not None)
        total_params = sum(1 for p in model.parameters())

        # 85% of parameters should have gradients (some might be unused in specific forward pass)
        assert grad_count > total_params * 0.85

    def test_model_eval_mode(self, mock_settings):
        """Model in eval mode should have consistent outputs."""
        from models.core import DerivOmniModel

        model = DerivOmniModel(mock_settings)
        model.eval()

        ticks = torch.randn(2, 100)
        candles = torch.randn(2, 50, 10)
        vol_metrics = torch.randn(2, 4)

        with torch.no_grad():
            out1 = model(ticks, candles, vol_metrics)
            out2 = model(ticks, candles, vol_metrics)

        # Outputs should be identical in eval mode
        assert torch.allclose(out1["rise_fall_logit"], out2["rise_fall_logit"])

class TestVolatilityExpert:
    """Tests for VolatilityExpert."""

    def test_volatility_forward_pass(self, mock_settings):
        """Forward pass should return latent embedding."""
        from models.volatility import VolatilityExpert

        expert = VolatilityExpert(input_dim=4, settings=mock_settings, hidden_dim=32)
        x = torch.randn(4, 4)

        encoded = expert(x)
        assert encoded.shape == (4, mock_settings.hyperparams.latent_dim)

    def test_reconstruction_error(self, mock_settings):
        """Reconstruction error should be scalar per sample."""
        from models.volatility import VolatilityExpert

        expert = VolatilityExpert(input_dim=4, settings=mock_settings, hidden_dim=32)
        x = torch.randn(4, 4)

        error = expert.reconstruction_error(x)
        assert error.shape == (4,)
        assert (error >= 0).all()
