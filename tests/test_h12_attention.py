import torch
import unittest
from models.attention import AdditiveAttention

class TestAdditiveAttention(unittest.TestCase):
    def test_forward_pass_no_u(self):
        """Verify AdditiveAttention works without self.U and self.U is gone."""
        hidden_dim = 32
        batch_size = 4
        seq_len = 10
        
        attn = AdditiveAttention(hidden_dim)
        
        # Check self.U is gone
        self.assertFalse(hasattr(attn, 'U'), "self.U should be removed")
        
        # Check forward pass
        encoder_outputs = torch.randn(batch_size, seq_len, hidden_dim)
        context, weights = attn(encoder_outputs)
        
        # Check shapes
        self.assertEqual(context.shape, (batch_size, hidden_dim))
        self.assertEqual(weights.shape, (batch_size, seq_len))
        
        # Check weights sum to 1
        self.assertTrue(torch.allclose(weights.sum(dim=1), torch.tensor(1.0)), "Weights must sum to 1")

if __name__ == "__main__":
    unittest.main()
