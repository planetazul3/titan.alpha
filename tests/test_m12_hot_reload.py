
import pytest
import torch
import time
from pathlib import Path
from unittest.mock import MagicMock
from models.core import DerivOmniModel
from config.settings import Settings

class TestModelHotReload:
    
    def test_reload_mechanism(self, tmp_path):
        """Verify we can detect file changes and reload weights."""
        
        # Setup
        model_path = tmp_path / "test_model.pt"
        settings = Settings()
        model = DerivOmniModel(settings)
        
        # Save initial weights (all zeros)
        params_0 = {k: torch.zeros_like(v) for k, v in model.state_dict().items()}
        torch.save({"model_state_dict": params_0, "manifest": {"model_version": "v1"}}, model_path)
        
        # Simulating live.py logic
        checkpoint_path = model_path
        last_ckpt_mtime = checkpoint_path.stat().st_mtime
        
        # Load initial
        ckpt = torch.load(checkpoint_path, weights_only=True)
        model.load_state_dict(ckpt["model_state_dict"])
        assert torch.all(list(model.parameters())[0] == 0)
        
        # Wait a bit to ensure mtime diff
        time.sleep(1.1)
        
        # Create new weights (all ones)
        params_1 = {k: torch.ones_like(v) for k, v in model.state_dict().items()}
        torch.save({"model_state_dict": params_1, "manifest": {"model_version": "v2"}}, model_path)
        
        # Check logic
        current_mtime = checkpoint_path.stat().st_mtime
        assert current_mtime > last_ckpt_mtime
        
        # "Reload"
        if current_mtime > last_ckpt_mtime:
            new_ckpt = torch.load(checkpoint_path, weights_only=True)
            model.load_state_dict(new_ckpt["model_state_dict"])
            last_ckpt_mtime = current_mtime
            
        # Verify
        assert torch.all(list(model.parameters())[0] == 1)

if __name__ == "__main__":
    pytest.main([__file__])
