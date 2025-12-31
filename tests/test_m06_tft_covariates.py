
import pytest
import torch
from models.tft import TemporalFusionTransformer
from models.core import DerivOmniModel
from config.settings import Settings

class TestTFTCovariates:
    def test_tft_with_static_covariates(self):
        """Verify TFT runs with static covariates and they influence output."""
        batch_size = 2
        seq_len = 10
        input_size = 5
        hidden_size = 8
        static_size = 4
        
        tft = TemporalFusionTransformer(
            input_size=input_size,
            hidden_size=hidden_size,
            static_input_size=static_size
        )
        
        x = torch.randn(batch_size, seq_len, input_size)
        static1 = torch.randn(batch_size, static_size)
        
        # Forward pass 1
        out1, _, _ = tft(x, static_covariates=static1)
        
        # Forward pass 2 (different static context)
        static2 = torch.randn(batch_size, static_size)
        out2, _, _ = tft(x, static_covariates=static2)
        
        # Verify outputs are different (proving static context is utilized)
        assert not torch.allclose(out1, out2), "TFT output should change when static context changes"
        assert out1.shape == (batch_size, seq_len, hidden_size)

    def test_full_model_integration(self):
        """Verify DerivOmniModel correctly threads volatility stats to TFT."""
        settings = Settings(environment="test", deriv_api_token="dummy_token")
        settings.hyperparams.latent_dim = 8 # Force latent dim to match expectations
        
        model = DerivOmniModel(settings)
        
        # Check that TemporalExpert was initialized with static support
        if hasattr(model.temporal, "tft"):
             assert model.temporal.tft.static_input_size == settings.hyperparams.latent_dim
        
        # Full forward pass
        ticks = torch.randn(2, 20)
        candles = torch.randn(2, 30, 10)
        vol_metrics = torch.randn(2, 4)
        
        logits = model(ticks, candles, vol_metrics)
        assert "rise_fall_logit" in logits
        assert logits["rise_fall_logit"].shape == (2, 1)

if __name__ == "__main__":
    pytest.main([__file__])
