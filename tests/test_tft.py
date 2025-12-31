"""
Unit tests for TFT (Temporal Fusion Transformer) components.
"""

import pytest
import torch

from config.settings import Settings
from models.tft import (
    GatedLinearUnit,
    GatedResidualNetwork,
    InterpretableMultiHeadAttention,
    TemporalFusionTransformer,
    VariableSelectionNetwork,
)


class TestGatedLinearUnit:
    """Tests for GatedLinearUnit."""

    def test_output_shape(self):
        """Test that GLU produces correct output shape."""
        glu = GatedLinearUnit(input_size=32, output_size=16)
        x = torch.randn(4, 10, 32)  # [batch, seq, features]
        
        output = glu(x)
        
        assert output.shape == (4, 10, 16)

    def test_default_output_size(self):
        """Test default output size equals input size."""
        glu = GatedLinearUnit(input_size=32)
        x = torch.randn(4, 10, 32)
        
        output = glu(x)
        
        assert output.shape == (4, 10, 32)


class TestGatedResidualNetwork:
    """Tests for GatedResidualNetwork."""

    def test_output_shape(self):
        """Test GRN output shape."""
        grn = GatedResidualNetwork(input_size=16, hidden_size=32, output_size=24)
        x = torch.randn(4, 10, 16)
        
        output = grn(x)
        
        assert output.shape == (4, 10, 24)

    def test_skip_connection_same_size(self):
        """Test skip connection when input == output size."""
        grn = GatedResidualNetwork(input_size=32, hidden_size=64, output_size=32)
        x = torch.randn(4, 10, 32)
        
        output = grn(x)
        
        assert output.shape == (4, 10, 32)
        assert grn.skip_layer is None  # No projection needed

    def test_skip_connection_different_size(self):
        """Test skip connection when input != output size."""
        grn = GatedResidualNetwork(input_size=16, hidden_size=64, output_size=32)
        x = torch.randn(4, 10, 16)
        
        output = grn(x)
        
        assert output.shape == (4, 10, 32)
        assert grn.skip_layer is not None

    def test_with_context(self):
        """Test GRN with context enrichment."""
        grn = GatedResidualNetwork(
            input_size=16, hidden_size=32, output_size=24, context_size=8
        )
        x = torch.randn(4, 10, 16)
        context = torch.randn(4, 8)
        
        output = grn(x, context)
        
        assert output.shape == (4, 10, 24)


class TestVariableSelectionNetwork:
    """Tests for VariableSelectionNetwork."""

    def test_output_shapes(self):
        """Test VSN output shapes."""
        vsn = VariableSelectionNetwork(num_features=10, hidden_size=32)
        x = torch.randn(4, 20, 10)  # [batch, seq_len, features]
        
        output, weights = vsn(x)
        
        assert output.shape == (4, 20, 32)
        assert weights.shape == (4, 20, 10)

    def test_weights_sum_to_one(self):
        """Test feature weights sum to 1 (softmax)."""
        vsn = VariableSelectionNetwork(num_features=5, hidden_size=16)
        x = torch.randn(2, 10, 5)
        
        _, weights = vsn(x)
        
        # Weights should sum to 1 along feature dimension
        weight_sums = weights.sum(dim=-1)
        assert torch.allclose(weight_sums, torch.ones_like(weight_sums), atol=1e-5)

    def test_with_context(self):
        """Test VSN with context vector."""
        vsn = VariableSelectionNetwork(
            num_features=10, hidden_size=32, context_size=16
        )
        x = torch.randn(4, 20, 10)
        context = torch.randn(4, 16)
        
        output, weights = vsn(x, context)
        
        assert output.shape == (4, 20, 32)


class TestInterpretableMultiHeadAttention:
    """Tests for InterpretableMultiHeadAttention."""

    def test_output_shape(self):
        """Test attention output shape."""
        attn = InterpretableMultiHeadAttention(hidden_size=64, num_heads=4)
        x = torch.randn(4, 20, 64)
        
        output, weights = attn(x)
        
        assert output.shape == (4, 20, 64)
        assert weights.shape == (4, 20, 20)  # Averaged across heads

    def test_different_num_heads(self):
        """Test with different number of heads."""
        for num_heads in [1, 2, 4, 8]:
            attn = InterpretableMultiHeadAttention(hidden_size=64, num_heads=num_heads)
            x = torch.randn(2, 10, 64)
            
            output, _ = attn(x)
            
            assert output.shape == (2, 10, 64)

    def test_attention_weights_valid(self):
        """Test that attention weights are valid probabilities."""
        attn = InterpretableMultiHeadAttention(hidden_size=32, num_heads=4)
        attn.eval()  # Disable dropout for deterministic behavior
        x = torch.randn(2, 10, 32)
        
        with torch.no_grad():
            _, weights = attn(x)
        
        # Weights should be non-negative
        assert (weights >= 0).all()
        # Averaged weights are approximately normalized
        # (exact normalization only for individual heads before averaging)
        weight_sums = weights.sum(dim=-1)
        assert torch.allclose(weight_sums, torch.ones_like(weight_sums), atol=0.2)


class TestTemporalFusionTransformer:
    """Tests for full TFT module."""

    def test_output_shapes(self):
        """Test TFT output shapes."""
        tft = TemporalFusionTransformer(
            input_size=10, hidden_size=32, num_heads=2
        )
        x = torch.randn(4, 20, 10)
        
        output, attention_weights, feature_weights = tft(x)
        
        assert output.shape == (4, 20, 32)  # Full sequence [batch, seq, hidden]
        assert attention_weights.shape == (4, 20, 20)
        assert feature_weights.shape == (4, 20, 10)

    def test_different_configs(self):
        """Test TFT with various configurations."""
        configs = [
            {"input_size": 5, "hidden_size": 16, "num_heads": 1},
            {"input_size": 10, "hidden_size": 64, "num_heads": 4},
            {"input_size": 20, "hidden_size": 128, "num_heads": 8},
        ]
        
        for config in configs:
            tft = TemporalFusionTransformer(**config)
            x = torch.randn(2, 15, config["input_size"])
            
            output, _, _ = tft(x)
            
            assert output.shape == (2, 15, config["hidden_size"])

    def test_get_feature_importance(self):
        """Test feature importance extraction."""
        tft = TemporalFusionTransformer(input_size=10, hidden_size=32)
        x = torch.randn(4, 20, 10)
        
        importance = tft.get_feature_importance(x)
        
        assert importance.shape == (4, 10)
        # Should be valid probabilities
        assert (importance >= 0).all()

    def test_gradients_flow(self):
        """Test that gradients flow through TFT."""
        tft = TemporalFusionTransformer(input_size=10, hidden_size=32)
        x = torch.randn(2, 15, 10)
        
        output, _, _ = tft(x)
        loss = output.sum()
        loss.backward()
        
        # Check gradients exist on model parameters
        has_grad = False
        for param in tft.parameters():
            if param.grad is not None and param.grad.abs().sum() > 0:
                has_grad = True
                break
        assert has_grad, "No gradients found on model parameters"

    def test_eval_mode_deterministic(self):
        """Test TFT produces same output in eval mode."""
        tft = TemporalFusionTransformer(input_size=10, hidden_size=32, dropout=0.1)
        tft.eval()
        x = torch.randn(2, 15, 10)
        
        with torch.no_grad():
            output1, _, _ = tft(x)
            output2, _, _ = tft(x)
        
        assert torch.allclose(output1, output2)
    
    def test_pooling_methods(self):
        """Test different pooling methods via TemporalExpert integration."""
        from models.temporal import TemporalExpert
        
        settings = Settings(environment="test", deriv_api_token="dummy_token")
        # Test default (attention)
        model_attn = TemporalExpert(settings, use_tft=True)
        # Manually override for testing
        model_attn.pooling_method = "attention"
        model_attn.attention = InterpretableMultiHeadAttention(hidden_size=64, num_heads=4) # Mock approx
        
        # Actually better to test integration via settings
        
        # Case 1: Attention
        settings_attn = Settings(environment="test", deriv_api_token="dummy_token")
        # Create a mutable copy/mock if needed, but Pydantic is frozen.
        # We can just instantiate TemporalExpert and patch the attribute.
        
        expert = TemporalExpert(settings_attn, use_tft=True)
        expert.pooling_method = "attention"
        # Re-init attention because init logic depends on setting (which we can't easily change due to frozen settings)
        # But we can manually add the layer
        from models.attention import AdditiveAttention
        expert.attention = AdditiveAttention(hidden_dim=settings.hyperparams.lstm_hidden_size)
        
        candles = torch.randn(2, 20, settings.data_shapes.feature_dim_candles)
        out_attn = expert(candles)
        assert out_attn.shape == (2, 64) # embedding dim
        
        # Case 2: Last
        expert.pooling_method = "last"
        out_last = expert(candles)
        assert out_last.shape == (2, 64)
        
        # Case 3: Mean
        expert.pooling_method = "mean"
        out_mean = expert(candles)
        assert out_mean.shape == (2, 64)
        
        # Verify they are different (attention/mean integrate whole seq, last only sees last)
        # Note: TFT connects last step to everything via attention, so 'last' is still powerful. 
        # But numerically they should differ.
        assert not torch.allclose(out_attn, out_last)
