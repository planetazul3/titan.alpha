"""
Unit tests for online learning module.
"""

import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

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
    
    def forward(self, x):
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
