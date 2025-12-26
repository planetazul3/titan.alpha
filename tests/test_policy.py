"""
Unit tests for RL policy module.
"""

import pytest
import torch

from models.policy import (
    MLP,
    TemperatureParameter,
    TradingActor,
    TradingCritic,
    TradingReward,
    TradingState,
)


class TestMLP:
    """Tests for MLP utility."""

    def test_output_shape(self):
        """Test MLP output shape."""
        mlp = MLP(10, [32, 64], 5)
        x = torch.randn(8, 10)
        
        output = mlp(x)
        
        assert output.shape == (8, 5)

    def test_single_hidden_layer(self):
        """Test MLP with single hidden layer."""
        mlp = MLP(10, [32], 5)
        x = torch.randn(4, 10)
        
        output = mlp(x)
        
        assert output.shape == (4, 5)


class TestTradingActor:
    """Tests for TradingActor."""

    def test_forward_shape(self):
        """Test actor forward pass shapes."""
        actor = TradingActor(state_dim=32, action_dim=1)
        state = torch.randn(4, 32)
        
        mean, log_std = actor(state)
        
        assert mean.shape == (4, 1)
        assert log_std.shape == (4, 1)

    def test_sample_shapes(self):
        """Test action sampling shapes."""
        actor = TradingActor(state_dim=32, action_dim=1, max_stake=10.0)
        state = torch.randn(4, 32)
        
        action, log_prob, mean = actor.sample(state)
        
        assert action.shape == (4, 1)
        assert log_prob.shape == (4, 1)

    def test_action_bounds(self):
        """Test actions are within bounds."""
        actor = TradingActor(state_dim=32, action_dim=1, max_stake=10.0)
        state = torch.randn(100, 32)
        
        action, _, _ = actor.sample(state)
        
        assert (action >= 0).all()
        assert (action <= 10.0).all()

    def test_deterministic_action(self):
        """Test deterministic action is consistent."""
        actor = TradingActor(state_dim=32, action_dim=1)
        actor.eval()
        state = torch.randn(1, 32)
        
        with torch.no_grad():
            action1 = actor.get_action(state, deterministic=True)
            action2 = actor.get_action(state, deterministic=True)
        
        assert torch.allclose(action1, action2)

    def test_gradients_flow(self):
        """Test gradients flow through actor."""
        actor = TradingActor(state_dim=32, action_dim=1)
        state = torch.randn(4, 32, requires_grad=True)
        
        action, log_prob, _ = actor.sample(state)
        loss = (action + log_prob).sum()
        loss.backward()
        
        assert state.grad is not None


class TestTradingCritic:
    """Tests for TradingCritic."""

    def test_forward_shapes(self):
        """Test critic forward pass shapes."""
        critic = TradingCritic(state_dim=32, action_dim=1)
        state = torch.randn(4, 32)
        action = torch.randn(4, 1)
        
        q1, q2 = critic(state, action)
        
        assert q1.shape == (4, 1)
        assert q2.shape == (4, 1)

    def test_twin_critics_different(self):
        """Test twin critics produce different values."""
        critic = TradingCritic(state_dim=32, action_dim=1)
        state = torch.randn(4, 32)
        action = torch.randn(4, 1)
        
        q1, q2 = critic(state, action)
        
        # They should be different (different random init)
        assert not torch.allclose(q1, q2)

    def test_q1_forward(self):
        """Test single Q1 forward."""
        critic = TradingCritic(state_dim=32, action_dim=1)
        state = torch.randn(4, 32)
        action = torch.randn(4, 1)
        
        q1 = critic.q1_forward(state, action)
        
        assert q1.shape == (4, 1)


class TestTemperatureParameter:
    """Tests for TemperatureParameter."""

    def test_initialization(self):
        """Test temperature initialization."""
        temp = TemperatureParameter(action_dim=1, initial_temperature=0.2)
        
        assert temp.alpha.item() == pytest.approx(0.2, rel=1e-4)

    def test_target_entropy(self):
        """Test target entropy calculation."""
        temp = TemperatureParameter(action_dim=2)
        
        assert temp.target_entropy == -2.0

    def test_compute_loss(self):
        """Test temperature loss computation."""
        temp = TemperatureParameter(action_dim=1)
        log_prob = torch.randn(4, 1)
        
        loss = temp.compute_loss(log_prob)
        
        assert loss.dim() == 0  # Scalar


class TestTradingState:
    """Tests for TradingState."""

    def test_from_context(self):
        """Test state construction from context."""
        state = TradingState.from_context(
            model_probs={"rise_fall_prob": 0.65, "touch_prob": 0.5, "range_prob": 0.4},
            reconstruction_error=0.1,
            trust_score=0.8,
            account_balance=500.0,
            current_drawdown=0.05,
            recent_win_rate=0.55,
            volatility_regime="low",
        )
        
        assert state.dim() == 1
        assert state.shape[0] == 12  # Updated: added edge feature  # 3 probs + 5 features + 3 vol encoding

    def test_volatility_encoding(self):
        """Test volatility regime encoding."""
        state_low = TradingState.from_context(
            model_probs={"rise_fall_prob": 0.5},
            reconstruction_error=0.1,
            trust_score=0.8,
            account_balance=1000.0,
            current_drawdown=0.0,
            recent_win_rate=0.5,
            volatility_regime="low",
        )
        
        state_high = TradingState.from_context(
            model_probs={"rise_fall_prob": 0.5},
            reconstruction_error=0.1,
            trust_score=0.8,
            account_balance=1000.0,
            current_drawdown=0.0,
            recent_win_rate=0.5,
            volatility_regime="high",
        )
        
        # Should have different encodings
        assert not torch.allclose(state_low, state_high)


class TestTradingReward:
    """Tests for TradingReward."""

    def test_positive_pnl_positive_reward(self):
        """Test positive PnL gives positive reward."""
        reward_fn = TradingReward()
        
        reward = reward_fn.compute(
            pnl=5.0,
            stake=1.0,
            max_stake=10.0,
            trust_score=0.8,
            current_drawdown=0.0,
            trade_won=True,
        )
        
        assert reward > 0

    def test_negative_pnl_negative_reward(self):
        """Test negative PnL gives negative reward."""
        reward_fn = TradingReward()
        
        reward = reward_fn.compute(
            pnl=-5.0,
            stake=1.0,
            max_stake=10.0,
            trust_score=0.8,
            current_drawdown=0.0,
            trade_won=False,
        )
        
        assert reward < 0

    def test_high_risk_penalty(self):
        """Test high stake with low trust gets penalized."""
        reward_fn = TradingReward(risk_weight=1.0)
        
        # Same PnL, but one with high risk during low trust
        reward_safe = reward_fn.compute(
            pnl=5.0,
            stake=1.0,
            max_stake=10.0,
            trust_score=0.9,
            current_drawdown=0.0,
            trade_won=True,
        )
        
        reward_risky = reward_fn.compute(
            pnl=5.0,
            stake=9.0,  # High stake
            max_stake=10.0,
            trust_score=0.2,  # Low trust
            current_drawdown=0.0,
            trade_won=True,
        )
        
        # Risky trade should have lower reward
        assert reward_safe > reward_risky

    def test_drawdown_penalty(self):
        """Test drawdown increases penalty."""
        reward_fn = TradingReward(drawdown_weight=1.0)
        
        reward_no_dd = reward_fn.compute(
            pnl=5.0,
            stake=5.0,
            max_stake=10.0,
            trust_score=0.8,
            current_drawdown=0.0,
            trade_won=True,
        )
        
        reward_with_dd = reward_fn.compute(
            pnl=5.0,
            stake=5.0,
            max_stake=10.0,
            trust_score=0.8,
            current_drawdown=0.5,  # In drawdown
            trade_won=True,
        )
        
        # Reward during drawdown should be lower
        assert reward_no_dd > reward_with_dd
