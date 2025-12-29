"""
Tests for Training Loop Audit Fixes.
"""
import pytest
import torch
import torch.nn as nn
from unittest.mock import MagicMock
from training.losses import MultiTaskLoss
from training.trainer import BatchProfiler

class TestMultiTaskLossAudit:
    def test_uncertainty_ignores_manual_weights(self):
        # Setup: Huge manual weights that would dominate if used
        loss_fn = MultiTaskLoss(
            rise_fall_weight=1000.0, 
            touch_weight=1000.0,
            range_weight=1000.0, 
            reconstruction_weight=1000.0
        )
        
        # Enable uncertainty
        # log_vars init to 0. precision = exp(0) = 1.
        # Loss formula: exp(-s)*L + 0.5*s
        # If s=0, loss = 1*L + 0 = L.
        
        logits = {"rise_fall_logit": torch.tensor([10.0])} # Sigmoid(10) ~ 1.0
        targets = {"rise_fall": torch.tensor([1.0])}
        
        # logits.shape should be [Batch, 1]
        # targets.shape should be [Batch]
        
        # Use Batch Size 2 to avoid squeeze() collapsing to scalar
        logits = {"rise_fall_logit": torch.tensor([[0.0], [0.0]])} 
        targets = {"rise_fall": torch.tensor([1.0, 1.0])}
        
        # Calculate expected "Uncertainty" loss
        # raw loss per item = 0.6931
        # uncertainty loss = 1.0 * 0.6931 + 0 = 0.6931
        
        # Calculate expected "Manual" loss
        # manual loss = 0.6931 * 1000.0 = 693.1
        
        loss_dict = loss_fn(logits, targets)
        total_loss = loss_dict["total"].item()
        
        # Should match uncertainty path, NOT manual path
        assert total_loss < 1.0
        assert abs(total_loss - 0.6931) < 0.001

    def test_manual_weights_used_if_uncertainty_disabled(self):
        loss_fn = MultiTaskLoss(rise_fall_weight=10.0)
        # Disable uncertainty (simulate by removing attr or requiring_grad=False?)
        # Class checks: hasattr(self, "log_vars") and self.log_vars.requires_grad
        loss_fn.log_vars.requires_grad = False
        
        # Class checks: hasattr(self, "log_vars") and self.log_vars.requires_grad
        loss_fn.log_vars.requires_grad = False
        
        logits = {"rise_fall_logit": torch.tensor([[0.0], [0.0]])}
        targets = {"rise_fall": torch.tensor([1.0, 1.0])}
        
        # raw = 0.6931
        # weighted = 6.931
        
        loss_dict = loss_fn(logits, targets)
        total_loss = loss_dict["total"].item()
        
        assert total_loss > 1.0
        assert abs(total_loss - 6.931) < 0.01

class TestBatchProfiler:
    def test_profiler_steps(self):
        profiler = BatchProfiler()
        assert profiler.data_time == 0.0
        assert profiler.compute_time == 0.0
        
        # Mock time
        with pytest.MonkeyPatch.context() as m:
            mock_time = MagicMock(side_effect=[100.0, 101.0, 103.0]) # start, data_end, compute_end
            m.setattr("time.time", mock_time)
            
            # Reset triggers first side_effect (100.0)
            profiler.reset()
            
            # Step data: now=101.0. data_time += 1.0. last=101.0
            profiler.step_data()
            
            # Step compute: now=103.0. compute_time += 2.0. last=103.0
            profiler.step_compute()
            
            assert profiler.data_time == 1.0
            assert profiler.compute_time == 2.0
