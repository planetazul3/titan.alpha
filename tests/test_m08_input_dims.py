
import pytest
import torch
from config.settings import Settings
from models.temporal import TemporalExpert

class TestInputDimensions:
    def test_configurable_dimensions(self):
        """Verify model respects configured input dimensions."""
        # Create settings with custom dimension
        custom_dim = 15
        settings = Settings(environment="test", deriv_api_token="dummy_token")
        settings.data_shapes.feature_dim_candles = custom_dim
        
        # Instantiate model
        model = TemporalExpert(settings)
        
        # Verify internal initialization
        if model.use_tft:
            assert model.tft.input_size == custom_dim
        else:
             # Should check LSTM input size but currently obscured in BiLSTMBlock
             pass
             
        # Create input with matching dimension
        batch = 2
        seq_len = 20
        x_correct = torch.randn(batch, seq_len, custom_dim)
        
        # Should run without error
        out = model(x_correct)
        assert out.shape == (batch, 64) # default embedding dim
        
        # Create input with WRONG dimension (legacy default 10)
        x_wrong = torch.randn(batch, seq_len, 10)
        
        # Should fail
        with pytest.raises(RuntimeError):
            model(x_wrong)

if __name__ == "__main__":
    pytest.main([__file__])
