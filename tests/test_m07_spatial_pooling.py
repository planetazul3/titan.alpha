
import pytest
import torch
import torch.nn as nn
from models.spatial import SpatialExpert
from config.settings import Settings

class TestSpatialPooling:
    def test_attention_pooling_shape_and_flow(self):
        """Verify SpatialExpert uses attention and outputs correct shape."""
        settings = Settings()
        model = SpatialExpert(settings, embedding_dim=16)
        
        # Mocking convolution blocks to return deterministic sequences to checking attention?
        # Hard to test attention "correctness" without training.
        # But we can verify it runs and respects the shape.
        
        batch = 2
        seq_len = 50
        ticks = torch.randn(batch, seq_len)
        
        # Forward pass
        emb = model(ticks)
        
        assert emb.shape == (batch, 16)
        assert isinstance(model.attention, nn.Module)
        assert not hasattr(model, 'pool'), "Pooling layer should be removed"

    def test_temporal_sensitivity(self):
        """Verify that output changes if we swap time steps (unlike global average pooling)."""
        settings = Settings()
        model = SpatialExpert(settings, embedding_dim=16)
        
        # Create a sequence
        seq_len = 50
        x = torch.randn(1, seq_len)
        
        # Create a permuted sequence (swap start and end)
        x_permuted = x.clone()
        x_permuted[0, 0] = x[0, -1]
        x_permuted[0, -1] = x[0, 0]
        
        # If strictly Global Avg Pooling, and if convolution padding allows, 
        # small localized changes might wash out.
        # But actually, even GAP with valid convolution is sensitive to position if padding=0.
        # However, attention explicitly weights positions.
        # Let's just check that outputs are stable and valid.
        
        out1 = model(x)
        out2 = model(x_permuted)
        
        # They should be different
        assert not torch.allclose(out1, out2)

if __name__ == "__main__":
    pytest.main([__file__])
