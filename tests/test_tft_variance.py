
import torch
import pytest
from models.tft import TemporalFusionTransformer
from config.settings import Settings

class TestTFTVariance:
    def test_temporal_variance(self):
        """
        Verify that TFT outputs vary across time steps.
        
        The previous implementation pooled the output (mean), causing 
        identical predictions for all time steps. The fix should 
        result in distinct outputs for each time step given varying input.
        """
        batch_size = 2
        seq_len = 10
        input_size = 4
        hidden_size = 8
        
        model = TemporalFusionTransformer(
            input_size=input_size,
            hidden_size=hidden_size,
            num_heads=2,
            num_lstm_layers=1
        )
        model.eval()
        
        # Create input with temporal variation
        # [batch, seq_len, input_size]
        x = torch.randn(batch_size, seq_len, input_size)
        
        # Get output
        output, _, _ = model(x)
        
        # Check shape: [batch, seq_len, hidden_size]
        assert output.shape == (batch_size, seq_len, hidden_size)
        
        # Check that output at t != output at t-1
        # Calculate difference between adjacent time steps
        diffs = output[:, 1:, :] - output[:, :-1, :]
        max_diff = diffs.abs().max().item()
        
        print(f"Max temporal difference: {max_diff}")
        
        # Assert there is variation
        assert max_diff > 1e-6, "TFT output is constant across time! Pooling bug still present."
        
        # Also ensure batch items are different
        batch_diff = (output[0] - output[1]).abs().max().item()
        assert batch_diff > 1e-6, "Batch items are identical!"

if __name__ == "__main__":
    test = TestTFTVariance()
    test.test_temporal_variance()
    print("Test Passed!")
