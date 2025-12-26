"""
Reinforcement Learning Policy Module for Trading Decisions.

Implements Soft Actor-Critic (SAC) components for autonomous
position sizing and contract selection decisions.

Reference: "Soft Actor-Critic: Off-Policy Maximum Entropy Deep 
Reinforcement Learning with a Stochastic Actor" (Haarnoja et al., 2018)
https://arxiv.org/abs/1801.01290

Key components:
- Actor: Policy network that outputs action distributions
- Twin Critics: Q-function estimators to reduce overestimation
- Temperature: Automatic entropy tuning for exploration

ARCHITECTURAL PRINCIPLE:
The RL policy SUPPLEMENTS the prediction model rather than replacing it.
The prediction model outputs probabilities, and the RL policy decides
HOW MUCH to bet based on the probability, regime, and account state.

Example:
    >>> from models.policy import TradingActor, TradingCritic
    >>> actor = TradingActor(state_dim=64, action_dim=2)
    >>> critic = TradingCritic(state_dim=64, action_dim=2)
    >>> action_mean, action_std = actor(state)
    >>> q_value = critic(state, action)
"""

import logging
import math
from typing import Any, cast

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal

logger = logging.getLogger(__name__)


class MLP(nn.Module):
    """Multi-layer perceptron with configurable architecture."""
    
    def __init__(
        self,
        input_dim: int,
        hidden_dims: list[int],
        output_dim: int,
        activation: nn.Module = nn.ReLU(),
        output_activation: nn.Module | None = None,
    ):
        super().__init__()
        
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                activation,
            ])
            prev_dim = hidden_dim
        
        layers.append(nn.Linear(prev_dim, output_dim))
        
        if output_activation:
            layers.append(output_activation)
        
        self.net = nn.Sequential(*layers)
        
        # Apply Kaiming initialization for better training stability
        self._init_weights()
    
    def _init_weights(self) -> None:
        """Initialize weights using Kaiming (He) initialization."""
        for module in self.net.modules():
            if isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(module.weight, nonlinearity='relu')
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return cast(torch.Tensor, self.net(x))


class TradingActor(nn.Module):
    """
    SAC Actor (Policy) Network for trading decisions.
    
    Outputs:
    - Position size (continuous, 0 to max_stake)
    - Action probabilities for contract selection (if multiple types)
    
    Uses reparameterization trick for gradient propagation through
    stochastic sampling.
    """
    
    LOG_STD_MIN = -20
    LOG_STD_MAX = 2
    
    def __init__(
        self,
        state_dim: int,
        action_dim: int = 1,  # Just position size by default
        hidden_dims: list[int] | None = None,
        max_stake: float = 10.0,
    ):
        """
        Initialize actor.
        
        Args:
            state_dim: Dimension of state input
            action_dim: Dimension of action output
            hidden_dims: Hidden layer dimensions
            max_stake: Maximum stake for position sizing
        """
        super().__init__()
        
        if hidden_dims is None:
            hidden_dims = [256, 256]
        
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.max_stake = max_stake
        
        # Shared trunk
        self.trunk = MLP(
            state_dim, 
            hidden_dims[:-1] if len(hidden_dims) > 1 else hidden_dims,
            hidden_dims[-1],
        )
        
        # Output heads (mean and log_std for position size)
        self.mean_head = nn.Linear(hidden_dims[-1], action_dim)
        self.log_std_head = nn.Linear(hidden_dims[-1], action_dim)
        
        logger.info(f"TradingActor initialized: state={state_dim}, action={action_dim}")
    
    def forward(
        self, 
        state: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Compute action distribution parameters.
        
        Args:
            state: State tensor [batch, state_dim]
        
        Returns:
            Tuple of (mean, log_std) for action distribution
        """
        features = self.trunk(state)
        mean = self.mean_head(features)
        log_std = self.log_std_head(features)
        
        # Clamp log_std for stability
        log_std = torch.clamp(log_std, self.LOG_STD_MIN, self.LOG_STD_MAX)
        
        return mean, log_std
    
    def sample(
        self, 
        state: torch.Tensor,
        deterministic: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Sample action with reparameterization trick.
        
        Args:
            state: State tensor
            deterministic: If True, return mean action
        
        Returns:
            Tuple of (action, log_prob, mean)
        """
        mean, log_std = self.forward(state)
        std = log_std.exp()
        
        if deterministic:
            # M05: Fix action scaling. Use sigmoid to match stochastic path range [0, max_stake].
            # Previously tanh mapped to [-max, max] which is invalid for stake.
            action = torch.sigmoid(mean) * self.max_stake
            log_prob = torch.zeros_like(action)
        else:
            # Reparameterization trick
            normal = Normal(mean, std)
            x = normal.rsample()
            
            # Squash to [0, max_stake]
            action = torch.sigmoid(x) * self.max_stake
            
            # Compute log probability with Jacobian correction
            log_prob = normal.log_prob(x)
            log_prob -= torch.log(self.max_stake * torch.sigmoid(x) * (1 - torch.sigmoid(x)) + 1e-6)
            log_prob = log_prob.sum(dim=-1, keepdim=True)
        
        return action, log_prob, mean
    
    def get_action(self, state: torch.Tensor, deterministic: bool = False) -> torch.Tensor:
        """Get action for inference."""
        action, _, _ = self.sample(state, deterministic)
        return action


class TradingCritic(nn.Module):
    """
    SAC Twin Critic (Q-function) Networks.
    
    Uses two Q-networks to reduce overestimation bias.
    Estimates expected return from a state-action pair.
    """
    
    def __init__(
        self,
        state_dim: int,
        action_dim: int = 1,
        hidden_dims: list[int] | None = None,
    ):
        """
        Initialize twin critics.
        
        Args:
            state_dim: Dimension of state input
            action_dim: Dimension of action input
            hidden_dims: Hidden layer dimensions
        """
        super().__init__()
        
        if hidden_dims is None:
            hidden_dims = [256, 256]
        
        input_dim = state_dim + action_dim
        
        # Twin Q-networks
        self.q1 = MLP(input_dim, hidden_dims, 1)
        self.q2 = MLP(input_dim, hidden_dims, 1)
        
        logger.info(f"TradingCritic initialized: state={state_dim}, action={action_dim}")
    
    def forward(
        self, 
        state: torch.Tensor, 
        action: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Compute Q-values for state-action pairs.
        
        Args:
            state: State tensor [batch, state_dim]
            action: Action tensor [batch, action_dim]
        
        Returns:
            Tuple of (Q1, Q2) values
        """
        sa = torch.cat([state, action], dim=-1)
        return self.q1(sa), self.q2(sa)
    
    def q1_forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """Compute only Q1 (for policy optimization)."""
        sa = torch.cat([state, action], dim=-1)
        return cast(torch.Tensor, self.q1(sa))


class TemperatureParameter(nn.Module):
    """
    Learnable temperature parameter for entropy tuning.
    
    The temperature α controls the exploration/exploitation tradeoff.
    Higher α = more exploration, lower α = more exploitation.
    """
    
    def __init__(
        self, 
        action_dim: int,
        initial_temperature: float = 0.2,
        target_entropy: float | None = None,
    ):
        """
        Initialize temperature.
        
        Args:
            action_dim: Used to compute target entropy
            initial_temperature: Starting temperature value
            target_entropy: Target entropy (defaults to -action_dim)
        """
        super().__init__()
        
        self.log_alpha = nn.Parameter(
            torch.tensor(math.log(initial_temperature))
        )
        
        # Target entropy: typically -dim(A) for continuous actions
        self.target_entropy = target_entropy or -float(action_dim)
    
    @property
    def alpha(self) -> torch.Tensor:
        """Get current temperature value."""
        return self.log_alpha.exp()
    
    def compute_loss(self, log_prob: torch.Tensor) -> torch.Tensor:
        """
        Compute temperature loss for automatic tuning.
        
        Args:
            log_prob: Log probability of sampled actions
        
        Returns:
            Temperature loss
        """
        return -(self.log_alpha * (log_prob + self.target_entropy).detach()).mean()


class TradingState:
    """
    Construct trading state from market context.
    
    Combines multiple signals into RL state vector:
    - Model prediction probabilities
    - Reconstruction error (regime signal)
    - Account state (balance, drawdown)
    - Recent performance metrics
    """
    
    def __init__(self, state_dim: int = 64):
        self.state_dim = state_dim
    
    @staticmethod
    def from_context(
        model_probs: dict[str, float],
        reconstruction_error: float,
        trust_score: float,
        account_balance: float,
        current_drawdown: float,
        recent_win_rate: float,
        volatility_regime: str = "medium",
    ) -> torch.Tensor:
        """
        Create state tensor from trading context.
        
        Args:
            model_probs: Model probability outputs
            reconstruction_error: Volatility expert error
            trust_score: Hierarchical regime trust score
            account_balance: Current account balance
            current_drawdown: Current drawdown (0-1)
            recent_win_rate: Recent trade win rate
            volatility_regime: Volatility regime string
        
        Returns:
            State tensor [state_dim]
        """
        # Extract probabilities
        probs = [
            model_probs.get("rise_fall_prob", 0.5),
            model_probs.get("touch_prob", 0.5),
            model_probs.get("range_prob", 0.5),
        ]
        
        # Volatility regime encoding
        vol_encoding = {
            "low": [1.0, 0.0, 0.0],
            "medium": [0.0, 1.0, 0.0],
            "high": [0.0, 0.0, 1.0],
        }.get(volatility_regime, [0.0, 1.0, 0.0])
        
        # Combine into state vector (12 dimensions)
        state = torch.tensor([
            *probs,                                    # 3 values
            reconstruction_error,                      # 1
            trust_score,                               # 1
            # Log scaling for balance: handles $1 to $1M without saturation
            # log10(1) = 0, log10(1000) = 3, log10(1M) = 6 -> scales to 0-1
            math.log10(max(account_balance, 1.0)) / 6.0,  # 1 Log-normalized
            current_drawdown,                          # 1
            recent_win_rate,                           # 1
            *vol_encoding,                             # 3
            max(probs) - 0.5,                         # 1 Edge (padding)
        ], dtype=torch.float32)
        
        return state


class TradingReward:
    """
    Multi-objective reward function for trading RL.
    
    Components:
    - Profit/Loss: Primary signal from trade outcome
    - Risk penalty: Penalize large positions during uncertainty
    - Drawdown penalty: Extra penalty during drawdown
    - Regime alignment: Bonus for trading with favorable regime
    """
    
    def __init__(
        self,
        profit_weight: float = 1.0,
        risk_weight: float = 0.1,
        drawdown_weight: float = 0.2,
        regime_weight: float = 0.1,
    ):
        self.profit_weight = profit_weight
        self.risk_weight = risk_weight
        self.drawdown_weight = drawdown_weight
        self.regime_weight = regime_weight
    
    def compute(
        self,
        pnl: float,
        stake: float,
        max_stake: float,
        trust_score: float,
        current_drawdown: float,
        trade_won: bool,
    ) -> float:
        """
        Compute reward for a trade.
        
        Args:
            pnl: Profit/loss from trade
            stake: Stake used
            max_stake: Maximum allowed stake
            trust_score: Regime trust score (0-1)
            current_drawdown: Current drawdown (0-1)
            trade_won: Whether trade was successful
        
        Returns:
            Scalar reward value
        """
        # Base profit reward
        profit_reward = pnl * self.profit_weight
        
        # Risk penalty (larger stakes during low trust = bad)
        stake_ratio = stake / max_stake
        risk_penalty = -self.risk_weight * stake_ratio * (1 - trust_score)
        
        # Drawdown penalty (larger stakes during drawdown = bad)
        drawdown_penalty = -self.drawdown_weight * stake_ratio * current_drawdown
        
        # Regime alignment bonus
        regime_bonus = self.regime_weight * trust_score if trade_won else 0.0
        
        return profit_reward + risk_penalty + drawdown_penalty + regime_bonus
