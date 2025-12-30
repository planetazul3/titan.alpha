"""
Unit tests for online learning module.
"""

import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from unittest.mock import MagicMock, patch

from training.online_learning import (
    EWCLoss,
    Experience,
    FisherInformation,
    OnlineLearningModule,
    ReplayBuffer,
)


class SimpleModel(nn.Module):
    """Simple model for testing."""
    
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(10, 1)
    
    def forward(self, x, *args, **kwargs):
        return self.fc(x)


class TestReplayBuffer:
    """Tests for ReplayBuffer."""

    def test_add_and_len(self):
        """Test adding experiences and length."""
        buffer = ReplayBuffer(capacity=100)
        
        for i in range(10):
            exp = Experience(
                ticks=torch.randn(50),
                candles=torch.randn(30, 10),
                vol_metrics=torch.randn(4),
                contract_type="rise_fall",
                probability=0.6,
                outcome=1 if i % 2 == 0 else 0,
                reconstruction_error=0.1,
                timestamp=float(i),
            )
            buffer.add(exp)
        
        assert len(buffer) == 10

    def test_capacity_limit(self):
        """Test that buffer respects capacity."""
        buffer = ReplayBuffer(capacity=5)
        
        for i in range(10):
            exp = Experience(
                ticks=torch.randn(50),
                candles=torch.randn(30, 10),
                vol_metrics=torch.randn(4),
                contract_type="rise_fall",
                probability=0.6,
                outcome=1,
                reconstruction_error=0.1,
                timestamp=float(i),
            )
            buffer.add(exp)
        
        assert len(buffer) == 5

    def test_sample(self):
        """Test random sampling."""
        buffer = ReplayBuffer(capacity=100)
        
        for i in range(20):
            exp = Experience(
                ticks=torch.randn(50),
                candles=torch.randn(30, 10),
                vol_metrics=torch.randn(4),
                contract_type="rise_fall",
                probability=0.6,
                outcome=1,
                reconstruction_error=0.1,
                timestamp=float(i),
            )
            buffer.add(exp)
        
        sample = buffer.sample(5)
        assert len(sample) == 5

    def test_get_resolved_experiences(self):
        """Test filtering resolved experiences."""
        buffer = ReplayBuffer(capacity=100)
        
        # Add some resolved and unresolved
        for i in range(5):
            buffer.add(Experience(
                ticks=torch.randn(50),
                candles=torch.randn(30, 10),
                vol_metrics=torch.randn(4),
                contract_type="rise_fall",
                probability=0.6,
                outcome=1,  # Resolved
                reconstruction_error=0.1,
                timestamp=float(i),
            ))
        
        for i in range(5):
            buffer.add(Experience(
                ticks=torch.randn(50),
                candles=torch.randn(30, 10),
                vol_metrics=torch.randn(4),
                contract_type="rise_fall",
                probability=0.6,
                outcome=-1,  # Unresolved
                reconstruction_error=0.1,
                timestamp=float(i + 5),
            ))
        
        resolved = buffer.get_resolved_experiences()
        assert len(resolved) == 5


class TestFisherInformation:
    """Tests for FisherInformation."""

    def test_initialization(self):
        """Test initialization."""
        model = SimpleModel()
        fisher = FisherInformation(model)
        
        assert not fisher.is_computed()

    def test_compute_fisher(self):
        """Test Fisher computation."""
        model = SimpleModel()
        fisher = FisherInformation(model)
        
        # Create simple dataset
        x = torch.randn(100, 10)
        y = torch.randint(0, 2, (100,)).float()
        dataset = TensorDataset(x, y)
        dataloader = DataLoader(dataset, batch_size=10)
        
        fisher.compute(dataloader, nn.BCEWithLogitsLoss())
        
        assert fisher.is_computed()
        
        # Fisher values should exist for parameters
        for name, param in model.named_parameters():
            if param.requires_grad:
                f = fisher.get_fisher(name)
                assert f is not None
                assert f.shape == param.shape

    def test_optimal_params_stored(self):
        """Test that optimal parameters are stored."""
        model = SimpleModel()
        fisher = FisherInformation(model)
        
        x = torch.randn(20, 10)
        y = torch.randint(0, 2, (20,)).float()
        dataset = TensorDataset(x, y)
        dataloader = DataLoader(dataset, batch_size=10)
        
        fisher.compute(dataloader, nn.BCEWithLogitsLoss())
        
        for name, param in model.named_parameters():
            if param.requires_grad:
                optimal = fisher.get_optimal_params(name)
                assert optimal is not None
                assert torch.allclose(optimal, param.data)


class TestEWCLoss:
    """Tests for EWCLoss."""

    def test_zero_without_fisher(self):
        """Test EWC loss is zero without computed Fisher."""
        model = SimpleModel()
        fisher = FisherInformation(model)
        ewc = EWCLoss(ewc_lambda=0.4)
        
        loss = ewc(model, fisher)
        
        assert loss.item() == 0.0

    def test_zero_at_optimal(self):
        """Test EWC loss is zero when at optimal params."""
        model = SimpleModel()
        fisher = FisherInformation(model)
        
        # Compute Fisher
        x = torch.randn(20, 10)
        y = torch.randint(0, 2, (20,)).float()
        dataset = TensorDataset(x, y)
        dataloader = DataLoader(dataset, batch_size=10)
        fisher.compute(dataloader, nn.BCEWithLogitsLoss())
        
        ewc = EWCLoss(ewc_lambda=0.4)
        loss = ewc(model, fisher)
        
        # At optimal, loss should be very small
        assert loss.item() < 1e-6

    def test_nonzero_after_change(self):
        """Test EWC loss is nonzero after params change."""
        model = SimpleModel()
        fisher = FisherInformation(model)
        
        # Compute Fisher
        x = torch.randn(20, 10)
        y = torch.randint(0, 2, (20,)).float()
        dataset = TensorDataset(x, y)
        dataloader = DataLoader(dataset, batch_size=10)
        fisher.compute(dataloader, nn.BCEWithLogitsLoss())
        
        # Change parameters
        with torch.no_grad():
            for param in model.parameters():
                param.add_(torch.randn_like(param) * 0.1)
        
        ewc = EWCLoss(ewc_lambda=0.4)
        loss = ewc(model, fisher)
        
        # After change, loss should be nonzero
        assert loss.item() > 0


class TestOnlineLearningModule:
    """Tests for OnlineLearningModule."""

    def test_initialization(self):
        """Test module initialization."""
        model = SimpleModel()
        online = OnlineLearningModule(model, ewc_lambda=0.4)
        
        assert online.ewc_lambda == 0.4
        assert len(online.replay_buffer) == 0

    def test_add_experience(self):
        """Test adding experiences."""
        model = SimpleModel()
        online = OnlineLearningModule(model)
        
        exp = Experience(
            ticks=torch.randn(50),
            candles=torch.randn(30, 10),
            vol_metrics=torch.randn(4),
            contract_type="rise_fall",
            probability=0.6,
            outcome=1,
            reconstruction_error=0.1,
            timestamp=0.0,
        )
        
        online.add_experience(exp)
        
        assert len(online.replay_buffer) == 1

    def test_should_update(self):
        """Test update triggering logic."""
        model = SimpleModel()
        online = OnlineLearningModule(
            model,
            update_interval=5,
            min_experiences=3,
        )
        
        # Not enough experiences
        assert not online.should_update()
        
        # Add resolved experiences
        for i in range(6):
            exp = Experience(
                ticks=torch.randn(50),
                candles=torch.randn(30, 10),
                vol_metrics=torch.randn(4),
                contract_type="rise_fall",
                probability=0.6,
                outcome=1,
                reconstruction_error=0.1,
                timestamp=float(i),
            )
            online.add_experience(exp)
        
        # Now should update
        assert online.should_update()

    def test_register_task(self):
        """Test task registration."""
        model = SimpleModel()
        online = OnlineLearningModule(model)
        
        # Register without dataloader (just snapshots params)
        online.register_task()
        
        # Optimal params should be stored
        for name, param in model.named_parameters():
            if param.requires_grad:
                assert online.fisher._optimal_params.get(name) is not None

    def test_get_statistics(self):
        """Test statistics retrieval."""
        model = SimpleModel()
        online = OnlineLearningModule(model)
        
        stats = online.get_statistics()
        
        assert "buffer_size" in stats
        assert "total_updates" in stats
        assert stats["buffer_size"] == 0
        assert stats["total_updates"] == 0

@pytest.fixture
def mock_experience():
    def _create(vol_val):
        return Experience(
            ticks=torch.randn(10),
            candles=torch.randn(10, 5),
            vol_metrics=torch.tensor([vol_val, vol_val, vol_val, vol_val]),
            contract_type="RISE_FALL",
            probability=0.5,
            outcome=1,
            reconstruction_error=0.1,
            timestamp=1234567890.0
        )
    return _create

class TestStratifiedSampling:
    def test_stratified_sampling_balances_skewed_buffer(self, mock_experience):
        buffer = ReplayBuffer(capacity=100)
        
        # Create SKEWED buffer: 90 Low Vol, 5 Med Vol, 5 High Vol
        # Low Vol: 0.1
        # Med Vol: 0.5
        # High Vol: 0.9
        
        for _ in range(90):
            buffer.add(mock_experience(0.1))
        for _ in range(5):
             buffer.add(mock_experience(0.5))
        for _ in range(5):
             buffer.add(mock_experience(0.9))
             
        # Random sampling should be dominated by Low Vol (approx 90%)
        random_batch = buffer.sample(batch_size=30, stratified=False)
        low_count_random = sum(1 for e in random_batch if e.vol_metrics[0] < 0.2)
        assert low_count_random > 20, "Random sampling should reflect skew"
        
        # Stratified sampling should ideally have equal representation (10 Low, 10 Med, 10 High)
        # But we only have 5 Med and 5 High available.
        # So it should take all 5 Med, all 5 High, and fill rest with Low (or duplicate? implementation details)
        # My implementation takes min(available, needed).
        # desired per stratum = 30 / 3 = 10.
        # Low: takes 10.
        # Med: takes 5 (all).
        # High: takes 5 (all).
        # Total 20. Remainder 10.
        # Fills remainder from rest. Remaining pool is mostly Low.
        # So we expect roughly 20 Low, 5 Med, 5 High.
        # This is strictly better than 27 Low, 1.5 Med, 1.5 High.
        
        stratified_batch = buffer.sample(batch_size=30, stratified=True)
        low_count = sum(1 for e in stratified_batch if e.vol_metrics[0] < 0.2)
        med_count = sum(1 for e in stratified_batch if 0.4 < e.vol_metrics[0] < 0.6)
        high_count = sum(1 for e in stratified_batch if e.vol_metrics[0] > 0.8)
        
        assert med_count == 5, f"Should capture all medium exp, got {med_count}"
        assert high_count == 5, f"Should capture all high exp, got {high_count}"
        assert low_count == 20, f"Should fill remainder with low, got {low_count}"
        
    def test_stratified_sampling_even_buffer(self, mock_experience):
        buffer = ReplayBuffer(capacity=100)
        # Even buffer
        for _ in range(10): buffer.add(mock_experience(0.1))
        for _ in range(10): buffer.add(mock_experience(0.5))
        for _ in range(10): buffer.add(mock_experience(0.9))
        
        batch = buffer.sample(batch_size=15, stratified=True)
        
        low = sum(1 for e in batch if e.vol_metrics[0] < 0.2)
        med = sum(1 for e in batch if 0.4 < e.vol_metrics[0] < 0.6)
        high = sum(1 for e in batch if e.vol_metrics[0] > 0.8)
        
        # Should be roughly equal (5, 5, 5)
        # Since logic splits evenly and all are available
        assert abs(low - 5) <= 1
        assert abs(med - 5) <= 1
        assert abs(high - 5) <= 1

class TestFisherAccumulation:
    def test_fisher_accumulation_alpha(self):
        """Test Fisher accumulation using deterministic mocks."""
        model = SimpleModel()
        fisher = FisherInformation(model)
        
        # Fake dataloader - content doesn't matter because we mock backward
        dataloader = [
             (torch.randn(1, 10), torch.tensor([1.0]))
        ]
        
        # 1. Compute Initial Fisher
        # We want specific gradients.
        # Let's say we want gradient 2.0 for all params.
        # Fisher = 2.0**2 = 4.0
        
        def mock_backward_1():
            for param in model.parameters():
                if param.requires_grad:
                    param.grad = torch.ones_like(param) * 2.0

        loss_mock_1 = MagicMock()
        loss_mock_1.backward.side_effect = mock_backward_1
        loss_fn_1 = MagicMock(return_value=loss_mock_1)

        fisher.compute(dataloader, loss_fn_1)
        initial_fisher = fisher.get_fisher("fc.weight").clone()
        
        assert torch.all(initial_fisher == 4.0)

        # 2. Compute Update with Alpha = 0.5
        # We want new gradients to be 4.0.
        # New Fisher calc = 4.0**2 = 16.0
        # Accumulated = 0.5 * 16.0 + 0.5 * 4.0 = 8.0 + 2.0 = 10.0
        
        def mock_backward_2():
             for param in model.parameters():
                if param.requires_grad:
                    param.grad = torch.ones_like(param) * 4.0

        loss_mock_2 = MagicMock()
        loss_mock_2.backward.side_effect = mock_backward_2
        loss_fn_2 = MagicMock(return_value=loss_mock_2)
        
        fisher.compute(dataloader, loss_fn_2, alpha=0.5)
        
        accumulated_fisher = fisher.get_fisher("fc.weight")
        
        # Note: atol=1e-5 is used to handle floating point inaccuracies
        assert torch.allclose(accumulated_fisher, torch.tensor(10.0), atol=1e-5)
        
        # 3. Test Alpha=1.0 (Overwrite)
        # Gradient = 3.0 -> Fisher = 9.0
        def mock_backward_3():
             for param in model.parameters():
                if param.requires_grad:
                    param.grad = torch.ones_like(param) * 3.0
        
        loss_mock_3 = MagicMock()
        loss_mock_3.backward.side_effect = mock_backward_3
        loss_fn_3 = MagicMock(return_value=loss_mock_3)
        
        fisher.compute(dataloader, loss_fn_3, alpha=1.0)
        
        assert torch.allclose(fisher.get_fisher("fc.weight"), torch.tensor(9.0), atol=1e-5)
