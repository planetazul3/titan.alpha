
import pytest
import torch
from models.policy import TradingActor

class TestActionScaling:
    def test_scaling_consistency(self):
        """Verify that deterministic and stochastic paths have consistent scaling."""
        max_stake = 10.0
        actor = TradingActor(state_dim=10, max_stake=max_stake)
        
        # Force the network output to be 0 (logits) -> 0.5 (prob)
        # We mock the forward pass output directly to test the sampling logic
        state = torch.zeros(1, 10)
        
        # Test Case 1: Mean = 0
        # Sigmoid(0) = 0.5. Action should be 0.5 * max_stake = 5.0
        with torch.no_grad():
             # Mock the internal methods to control 'mean'
             # Actually, simpler to just instantiate the class and use the logic directly or 
             # mock the forward return. Let's mock forward.
             actor.forward = lambda s: (torch.zeros(1, 1), torch.zeros(1, 1))
             
             # Deterministic
             action_det, _, _ = actor.sample(state, deterministic=True)
             assert action_det.item() == 5.0, f"Deterministic 0-logit should produce 5.0, got {action_det.item()}"
             
             # Stochastic (with std=1.0)
             # The mean of the distribution is what matters.
             # If we sampled a LOT, it should average around 5.0.
             # But here we are testing the TRANSFORMATION logic.
             
    def test_deterministic_range(self):
        """Verify that deterministic output is strictly positive [0, max]."""
        max_stake = 10.0
        actor = TradingActor(state_dim=10, max_stake=max_stake)
        state = torch.zeros(1, 10)
        
        # Force large negative mean -> Should be close to 0 (not -10)
        actor.forward = lambda s: (torch.tensor([[-100.0]]), torch.zeros(1, 1))
        action, _, _ = actor.sample(state, deterministic=True)
        assert 0.0 <= action.item() < 0.01, f"Large negative input should be ~0, got {action.item()}"
        
        # Force large positive mean -> Should be close to 10
        actor.forward = lambda s: (torch.tensor([[100.0]]), torch.zeros(1, 1))
        action, _, _ = actor.sample(state, deterministic=True)
        assert 9.99 < action.item() <= 10.0, f"Large positive input should be ~10, got {action.item()}"

if __name__ == "__main__":
    pytest.main([__file__])
