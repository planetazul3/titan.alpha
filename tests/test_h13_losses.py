import torch
import unittest
from training.losses import MultiTaskLoss

class TestLossBalance(unittest.TestCase):
    def test_loss_magnitude_balance(self):
        """Verify that reconstruction loss is scaled up to be comparable to classification loss."""
        loss_fn = MultiTaskLoss()
        
        # 1. Simulate typical Classification Inputs (Batch Size 2)
        # Target: 1.0, Logit: 0.0 (sigmoid(0) = 0.5) -> Error ~0.69
        logits = {
            "rise_fall_logit": torch.tensor([[0.0], [0.0]]),
            "touch_logit": torch.tensor([[0.0], [0.0]]),
            "range_logit": torch.tensor([[0.0], [0.0]])
        }
        targets = {
            "rise_fall": torch.tensor([1.0, 1.0]),
            "touch": torch.tensor([1.0, 1.0]),
            "range": torch.tensor([1.0, 1.0])
        }
        
        # 2. Simulate typical Reconstruction Inputs (Batch Size 2)
        # Normalized input: 0.5, Reconstruction: 0.4 -> MSE = 0.01
        vol_input = torch.tensor([[0.5], [0.5]])
        vol_recon = torch.tensor([[0.4], [0.4]])
        
        losses = loss_fn(logits, targets, vol_input, vol_recon)
        
        # Expected components
        # BCE(0.0, 1.0) approx 0.693
        # rise_fall: 0.693 * 1.0 = 0.693
        # touch: 0.693 * 0.5 = 0.346
        # range: 0.693 * 0.5 = 0.346
        
        # Reconstruction:
        # MSE(0.5, 0.4) = 0.01
        # Weighted: 0.01 * 50.0 = 0.5
        
        print(f"Losses: {losses}")
        
        # Verify magnitudes
        self.assertAlmostEqual(losses["reconstruction"].item(), 0.5, delta=0.05)
        self.assertGreater(losses["reconstruction"].item(), 0.1, "Reconstruction loss should be significant")
        
        # Check that reconstruction is not overwhelmed (e.g. at least 20% of rise_fall)
        ratio = losses["reconstruction"].item() / losses["rise_fall"].item()
        self.assertGreater(ratio, 0.2, f"Reconstruction/Classification ratio {ratio:.2f} is too low")

if __name__ == "__main__":
    unittest.main()
