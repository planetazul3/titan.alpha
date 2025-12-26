"""
Unit tests for RL trainer module.
"""

import pytest
import torch

from models.policy import TemperatureParameter, TradingActor, TradingCritic
from training.rl_trainer import ReplayMemory, RLTrainer, Transition


class TestReplayMemory:
    """Tests for ReplayMemory."""

    def test_push_and_len(self):
        """Test adding to memory and length."""
        memory = ReplayMemory(capacity=100)
        
        for i in range(10):
            t = Transition(
                state=torch.randn(32),
                action=torch.randn(1),
                reward=float(i),
                next_state=torch.randn(32),
                done=False,
            )
            memory.push(t)
        
        assert len(memory) == 10

    def test_capacity_limit(self):
        """Test capacity is respected."""
        memory = ReplayMemory(capacity=5)
        
        for i in range(10):
            t = Transition(
                state=torch.randn(32),
                action=torch.randn(1),
                reward=float(i),
                next_state=torch.randn(32),
                done=False,
            )
            memory.push(t)
        
        assert len(memory) == 5

    def test_sample(self):
        """Test random sampling."""
        memory = ReplayMemory(capacity=100)
        
        for i in range(20):
            t = Transition(
                state=torch.randn(32),
                action=torch.randn(1),
                reward=float(i),
                next_state=torch.randn(32),
                done=False,
            )
            memory.push(t)
        
        batch = memory.sample(5)
        assert len(batch) == 5

    def test_clear(self):
        """Test clearing memory."""
        memory = ReplayMemory(capacity=100)
        
        for i in range(10):
            t = Transition(
                state=torch.randn(32),
                action=torch.randn(1),
                reward=float(i),
                next_state=torch.randn(32),
                done=False,
            )
            memory.push(t)
        
        memory.clear()
        assert len(memory) == 0


class TestRLTrainer:
    """Tests for RLTrainer."""

    @pytest.fixture
    def trainer(self):
        """Create trainer for tests."""
        actor = TradingActor(state_dim=32, action_dim=1)
        critic = TradingCritic(state_dim=32, action_dim=1)
        temp = TemperatureParameter(action_dim=1)
        
        return RLTrainer(
            actor=actor,
            critic=critic,
            temperature=temp,
            batch_size=8,
            memory_size=100,
        )

    def test_add_experience(self, trainer):
        """Test adding experience."""
        state = torch.randn(32)
        action = torch.randn(1)
        next_state = torch.randn(32)
        
        trainer.add_experience(state, action, 1.0, next_state, False)
        
        assert len(trainer.memory) == 1

    def test_should_update(self, trainer):
        """Test update trigger logic."""
        # Not enough samples
        assert trainer.should_update() is False
        
        # Add enough samples
        for _ in range(10):
            trainer.add_experience(
                torch.randn(32),
                torch.randn(1),
                1.0,
                torch.randn(32),
                False,
            )
        
        assert trainer.should_update() is True

    def test_update(self, trainer):
        """Test update step."""
        # Fill memory
        for _ in range(20):
            trainer.add_experience(
                torch.randn(32),
                torch.randn(1),
                1.0,
                torch.randn(32),
                False,
            )
        
        metrics = trainer.update()
        
        assert "critic_loss" in metrics
        assert "actor_loss" in metrics
        assert "alpha" in metrics

    def test_statistics(self, trainer):
        """Test statistics retrieval."""
        stats = trainer.get_statistics()
        
        assert "memory_size" in stats
        assert "step_count" in stats
        assert "alpha" in stats


class TestTransition:
    """Tests for Transition dataclass."""

    def test_creation(self):
        """Test transition creation."""
        t = Transition(
            state=torch.randn(32),
            action=torch.randn(1),
            reward=5.0,
            next_state=torch.randn(32),
            done=False,
        )
        
        assert t.reward == 5.0
        assert t.done is False
