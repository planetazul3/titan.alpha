"""
Reinforcement Learning Integration for Live Trading.

Integrates the SAC policy with the live trading loop for
RL-based position sizing and decision enhancement.

ARCHITECTURAL PRINCIPLE:
The RL policy operates ALONGSIDE the prediction model, not replacing it.
The prediction model says WHAT to trade, the RL policy refines HOW MUCH.

Integration points:
1. State construction from live market data
2. Action sampling from trained policy
3. Reward computation from trade outcomes
4. Online experience collection for continued learning

Example:
    >>> from execution.rl_integration import RLTradingIntegration
    >>> rl = RLTradingIntegration(actor, critic)
    >>> stake = rl.get_recommended_stake(market_state)
"""

import logging
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch

from models.policy import TradingActor, TradingCritic, TradingReward, TradingState

logger = logging.getLogger(__name__)


@dataclass 
class RLDecision:
    """
    RL-informed trading decision.
    
    Attributes:
        recommended_stake: RL-recommended stake amount
        confidence: RL confidence in recommendation
        action_raw: Raw action from policy
        state_embedding: State used for decision
        use_rl_sizing: Whether to use RL sizing vs default
    """
    recommended_stake: float
    confidence: float
    action_raw: float
    state_embedding: list[float]
    use_rl_sizing: bool


class RLTradingIntegration:
    """
    Integrates RL policy with live trading decisions.
    
    Provides:
    - State construction from market data
    - Stake recommendation from trained policy
    - Experience collection for online learning
    - Fallback to rule-based sizing
    
    Example:
        >>> rl = RLTradingIntegration.from_checkpoint("checkpoints/rl_policy.pt")
        >>> decision = rl.get_decision(
        ...     model_probs={"rise_fall_prob": 0.7},
        ...     reconstruction_error=0.02,
        ...     account_balance=1000.0,
        ... )
        >>> stake = decision.recommended_stake
    """
    
    def __init__(
        self,
        actor: TradingActor,
        max_stake: float = 10.0,
        min_confidence_for_rl: float = 0.6,
        fallback_fraction: float = 0.02,
    ):
        """
        Initialize RL integration.
        
        Args:
            actor: Trained TradingActor policy
            max_stake: Maximum allowed stake
            min_confidence_for_rl: Minimum model confidence to use RL sizing
            fallback_fraction: Fraction of balance for fallback sizing
        """
        self.actor = actor
        self.actor.eval()  # Ensure inference mode
        self.max_stake = max_stake
        self.min_confidence_for_rl = min_confidence_for_rl
        self.fallback_fraction = fallback_fraction
        
        self._experience_buffer: list[dict] = []
        self._max_buffer_size = 1000
        
        logger.info(
            f"RLTradingIntegration initialized: "
            f"max_stake={max_stake}, min_conf={min_confidence_for_rl}"
        )
    
    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_path: str,
        state_dim: int = 12,
        **kwargs,
    ) -> "RLTradingIntegration":
        """
        Load RL integration from checkpoint.
        
        Args:
            checkpoint_path: Path to saved checkpoint
            state_dim: State dimension
            **kwargs: Additional init arguments
        
        Returns:
            RLTradingIntegration instance
        """
        actor = TradingActor(state_dim=state_dim, action_dim=1)
        
        # Security: Use weights_only=True as checkpoint contains only state_dict
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        if "actor" in checkpoint:
            actor.load_state_dict(checkpoint["actor"])
        else:
            actor.load_state_dict(checkpoint)
        
        actor.eval()
        
        logger.info(f"Loaded RL policy from {checkpoint_path}")
        return cls(actor=actor, **kwargs)
    
    def construct_state(
        self,
        model_probs: dict[str, float],
        reconstruction_error: float,
        trust_score: float,
        account_balance: float,
        current_drawdown: float,
        recent_win_rate: float,
        volatility_regime: str = "medium",
    ) -> torch.Tensor:
        """
        Construct state tensor for policy.
        
        Args:
            model_probs: Model probability outputs
            reconstruction_error: Volatility model error
            trust_score: Regime trust score
            account_balance: Current account balance
            current_drawdown: Current drawdown percentage
            recent_win_rate: Recent win rate
            volatility_regime: Current volatility regime
        
        Returns:
            State tensor for policy
        """
        return TradingState.from_context(
            model_probs=model_probs,
            reconstruction_error=reconstruction_error,
            trust_score=trust_score,
            account_balance=account_balance,
            current_drawdown=current_drawdown,
            recent_win_rate=recent_win_rate,
            volatility_regime=volatility_regime,
        )
    
    def get_decision(
        self,
        model_probs: dict[str, float],
        reconstruction_error: float,
        trust_score: float = 0.5,
        account_balance: float = 1000.0,
        current_drawdown: float = 0.0,
        recent_win_rate: float = 0.5,
        volatility_regime: str = "medium",
        deterministic: bool = True,
    ) -> RLDecision:
        """
        Get RL-informed decision.
        
        Args:
            model_probs: Model probability outputs
            reconstruction_error: Volatility model error
            trust_score: Regime trust score
            account_balance: Current account balance
            current_drawdown: Current drawdown percentage
            recent_win_rate: Recent win rate
            volatility_regime: Current volatility regime
            deterministic: Use deterministic policy (no exploration)
        
        Returns:
            RLDecision with stake recommendation
        """
        # Check if model confidence is sufficient for RL
        max_prob = max(model_probs.values(), default=0.5)
        use_rl = max_prob >= self.min_confidence_for_rl
        
        # Construct state
        state = self.construct_state(
            model_probs=model_probs,
            reconstruction_error=reconstruction_error,
            trust_score=trust_score,
            account_balance=account_balance,
            current_drawdown=current_drawdown,
            recent_win_rate=recent_win_rate,
            volatility_regime=volatility_regime,
        )
        
        if use_rl:
            # Get RL recommendation
            with torch.inference_mode():  # I4: More memory-efficient
                action, log_prob, mean = self.actor.sample(
                    state.unsqueeze(0),
                    deterministic=deterministic,
                )
            
            recommended_stake = action.item()
            confidence = torch.sigmoid(mean).item()
            action_raw = action.item()
        else:
            # Fallback to rule-based sizing
            recommended_stake = min(
                account_balance * self.fallback_fraction * max_prob,
                self.max_stake,
            )
            confidence = max_prob
            action_raw = 0.0
        
        return RLDecision(
            recommended_stake=min(recommended_stake, self.max_stake),
            confidence=confidence,
            action_raw=action_raw,
            state_embedding=state.tolist(),
            use_rl_sizing=use_rl,
        )
    
    def record_outcome(
        self,
        decision: RLDecision,
        pnl: float,
        won: bool,
    ) -> None:
        """
        Record trade outcome for learning.
        
        Args:
            decision: The decision that was made
            pnl: Profit/loss from trade
            won: Whether trade won
        """
        experience = {
            "state": decision.state_embedding,
            "action": decision.action_raw,
            "reward": pnl,
            "won": won,
            "rl_sizing_used": decision.use_rl_sizing,
        }
        
        self._experience_buffer.append(experience)
        
        # Trim buffer
        if len(self._experience_buffer) > self._max_buffer_size:
            self._experience_buffer.pop(0)
    
    def get_experiences(self, limit: int | None = None) -> list[dict]:
        """Get collected experiences."""
        if limit:
            return self._experience_buffer[-limit:]
        return self._experience_buffer.copy()
    
    def get_statistics(self) -> dict[str, Any]:
        """Get integration statistics."""
        if not self._experience_buffer:
            return {"experiences": 0}
        
        wins = sum(1 for e in self._experience_buffer if e["won"])
        rl_used = sum(1 for e in self._experience_buffer if e["rl_sizing_used"])
        
        return {
            "experiences": len(self._experience_buffer),
            "win_rate": wins / len(self._experience_buffer),
            "rl_sizing_rate": rl_used / len(self._experience_buffer),
            "avg_reward": np.mean([e["reward"] for e in self._experience_buffer]),
        }
