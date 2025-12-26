"""
Dynamic Position Sizing Module.

Implements optimal position sizing using the Kelly Criterion with safety
margins and adaptive adjustments based on model confidence, market regime,
and portfolio state.

Reference: Kelly, J.L. "A New Interpretation of Information Rate" (1956)

ARCHITECTURAL PRINCIPLE:
Position sizing is SEPARATE from signal generation. The model produces
probabilities, the decision engine produces signals, and this module
determines HOW MUCH to stake on each signal.

This separation enables:
- Independent optimization of sizing strategy
- Easy A/B testing of sizing approaches
- Risk management independent of model predictions

Example:
    >>> from execution.position_sizer import KellyPositionSizer
    >>> sizer = KellyPositionSizer(base_stake=1.0, safety_factor=0.5)
    >>> stake = sizer.compute_stake(
    ...     probability=0.65,
    ...     payout_ratio=0.9,
    ...     model_confidence=0.8,
    ...     current_drawdown=0.05
    ... )
"""

import logging
import threading
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class PositionSizeResult:
    """Result of position sizing calculation.
    
    Attributes:
        stake: Computed stake amount (account currency)
        kelly_fraction: Raw Kelly fraction before adjustments
        adjusted_fraction: Kelly fraction after safety/drawdown adjustments
        confidence_multiplier: Multiplier from model confidence
        drawdown_multiplier: Multiplier from current drawdown
        reason: Human-readable explanation of sizing decision
    """
    stake: float
    kelly_fraction: float
    adjusted_fraction: float
    confidence_multiplier: float
    drawdown_multiplier: float
    reason: str
    
    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for logging/serialization."""
        return {
            "stake": self.stake,
            "kelly_fraction": self.kelly_fraction,
            "adjusted_fraction": self.adjusted_fraction,
            "confidence_multiplier": self.confidence_multiplier,
            "drawdown_multiplier": self.drawdown_multiplier,
            "reason": self.reason,
        }


class KellyPositionSizer:
    """
    Optimal position sizing using Kelly Criterion with safety margins.
    
    The Kelly Criterion provides the mathematically optimal bet size to
    maximize long-term growth rate. However, full Kelly is aggressive and
    can lead to large drawdowns, so we apply a safety factor (fractional Kelly).
    
    Kelly Formula for binary outcomes:
        f* = (p * b - q) / b
        
    Where:
        f* = Optimal fraction of bankroll to bet
        p = Probability of winning
        q = 1 - p = Probability of losing
        b = Payout odds (e.g., 0.9 for 90% payout)
    
    For binary options:
        Win: +payout_ratio (e.g., +0.9)
        Loss: -1.0 (lose stake)
    
    Adjustments applied:
        1. Safety factor (e.g., 0.5 = half-Kelly)
        2. Model confidence scaling
        3. Drawdown reduction (reduce exposure during losing streaks)
        4. Volatility scaling (optional)
    
    Example:
        >>> sizer = KellyPositionSizer(base_stake=1.0, safety_factor=0.5)
        >>> result = sizer.compute_stake(probability=0.65, payout_ratio=0.9)
        >>> print(f"Stake: ${result.stake:.2f}")
    """
    
    def __init__(
        self,
        base_stake: float = 1.0,
        safety_factor: float = 0.5,
        max_stake: float = 10.0,
        min_stake: float = 0.35,
        drawdown_scale_threshold: float = 0.1,
        high_volatility_reduction: float = 0.5,
        default_payout_ratio: float = 0.9,
    ):
        """
        Initialize Kelly position sizer.
        
        Args:
            base_stake: Base stake amount (multiplied by Kelly fraction)
            safety_factor: Fraction of Kelly to use (0.5 = half-Kelly)
            max_stake: Maximum allowed stake per trade
            min_stake: Minimum viable stake (broker minimum)
            drawdown_scale_threshold: Drawdown level at which to start reducing
            high_volatility_reduction: Multiplier when volatility regime is high
            default_payout_ratio: Default payout ratio if not specified in signal
        """
        if base_stake <= 0:
            raise ValueError(f"base_stake must be positive, got {base_stake}")
        if not 0 < safety_factor <= 1:
            raise ValueError(f"safety_factor must be in (0, 1], got {safety_factor}")
        if max_stake < min_stake:
            raise ValueError(f"max_stake ({max_stake}) must be >= min_stake ({min_stake})")
        
        self.base_stake = base_stake
        self.safety_factor = safety_factor
        self.max_stake = max_stake
        self.min_stake = min_stake
        self.drawdown_scale_threshold = drawdown_scale_threshold
        self.high_volatility_reduction = high_volatility_reduction
        self.default_payout_ratio = default_payout_ratio
        
        logger.info(
            f"KellyPositionSizer initialized: base=${base_stake}, "
            f"safety={safety_factor}, max=${max_stake}, min=${min_stake}, "
            f"payout={default_payout_ratio}"
        )
    
    def compute_kelly_fraction(
        self, 
        probability: float, 
        payout_ratio: float | None = None
    ) -> float:
        """
        Compute raw Kelly fraction for given probability and payout.
        
        Args:
            probability: Estimated win probability (0 to 1)
            payout_ratio: Payout on win (e.g., 0.9 for 90%)
        
        Returns:
            Kelly fraction (can be negative if edge is negative)
        """
        payout_ratio = payout_ratio if payout_ratio is not None else self.default_payout_ratio

        if not 0 <= probability <= 1:
            logger.warning(f"Probability {probability} out of range, clamping to [0, 1]")
            probability = max(0, min(1, probability))
        
        # Kelly formula: f* = (p * b - q) / b
        # Where q = 1 - p (loss probability)
        # For binary options: win = +b, lose = -1
        win_prob = probability
        loss_prob = 1 - probability
        
        if payout_ratio <= 1e-8:
            logger.warning(f"Suspicious payout_ratio: {payout_ratio}, returning 0")
            return 0.0
        
        kelly = (win_prob * payout_ratio - loss_prob) / payout_ratio
        return kelly
    
    def compute_stake(
        self,
        probability: float,
        payout_ratio: float | None = None,
        model_confidence: float = 1.0,
        current_drawdown: float = 0.0,
        volatility_regime: str = "normal",
        account_balance: float | None = None,
    ) -> PositionSizeResult:
        """
        Compute optimal stake with all adjustments applied.
        
        Args:
            probability: Model's predicted win probability
            payout_ratio: Binary option payout ratio (e.g., 0.9)
            model_confidence: Confidence score from model (0 to 1)
            current_drawdown: Current account drawdown as fraction (0 to 1)
            volatility_regime: Market volatility state ("low", "normal", "high")
            account_balance: Optional current balance for dynamic sizing
        
        Returns:
            PositionSizeResult with computed stake and diagnostics
        """
        payout_ratio = payout_ratio if payout_ratio is not None else self.default_payout_ratio

        # 1. Compute raw Kelly fraction
        kelly = self.compute_kelly_fraction(probability, payout_ratio)
        
        # 2. If Kelly is negative or zero, no bet
        if kelly <= 0:
            return PositionSizeResult(
                stake=0.0,
                kelly_fraction=kelly,
                adjusted_fraction=0.0,
                confidence_multiplier=model_confidence,
                drawdown_multiplier=1.0,
                reason=f"Negative edge: Kelly={kelly:.4f} (prob={probability:.3f}, payout={payout_ratio})"
            )
        
        # 3. Apply safety factor (fractional Kelly)
        adjusted = kelly * self.safety_factor
        
        # 4. Apply model confidence scaling
        confidence_mult = max(0.1, min(1.0, model_confidence))
        adjusted *= confidence_mult
        
        # 5. Apply drawdown reduction
        drawdown_mult = 1.0
        if current_drawdown > self.drawdown_scale_threshold:
            # Linear reduction: at 10% drawdown reduce to 90%, at 20% to 80%, etc.
            drawdown_mult = max(0.3, 1.0 - current_drawdown)
            adjusted *= drawdown_mult
        
        # 6. Apply volatility regime adjustment
        if volatility_regime == "high":
            adjusted *= self.high_volatility_reduction
        
        # 7. Convert to stake amount
        if account_balance is not None:
            if account_balance <= 0:
                return PositionSizeResult(
                    stake=0.0,
                    kelly_fraction=kelly,
                    adjusted_fraction=adjusted,
                    confidence_multiplier=confidence_mult,
                    drawdown_multiplier=drawdown_mult,
                    reason="Insufficient account balance (<= 0)"
                )
            # Dynamic sizing based on current balance
            stake = adjusted * account_balance
        else:
            # Fixed base stake approach
            stake = adjusted * self.base_stake * 10  # Scale up from fraction
        
        # 8. Apply min/max bounds
        if stake < self.min_stake:
            if kelly > 0:
                # Positive edge but stake too small - use minimum
                stake = self.min_stake
                reason = f"Min stake: Kelly={kelly:.4f} -> ${stake:.2f} (at minimum)"
            else:
                stake = 0.0
                reason = "Below minimum stake, no trade"
        elif stake > self.max_stake:
            stake = self.max_stake
            reason = f"Max stake capped: Kelly={kelly:.4f} -> ${stake:.2f} (capped at max)"
        else:
            reason = f"Kelly optimal: f={kelly:.4f}, adjusted={adjusted:.4f} -> ${stake:.2f}"
        
        return PositionSizeResult(
            stake=round(stake, 2),
            kelly_fraction=kelly,
            adjusted_fraction=adjusted,
            confidence_multiplier=confidence_mult,
            drawdown_multiplier=drawdown_mult,
            reason=reason,
        )
    
    def suggest_stake_for_signal(
        self,
        signal: Any,
        model_confidence: float = 1.0,
        current_drawdown: float = 0.0,
        volatility_regime: str = "normal",
        account_balance: float | None = None,
    ) -> float:
        """
        Convenience method to compute stake from a TradeSignal object.
        
        Args:
            signal: TradeSignal object with probability attribute
            model_confidence: Confidence score
            current_drawdown: Current drawdown
            volatility_regime: Volatility state
            account_balance: Optional balance for dynamic sizing
        
        Returns:
            Stake amount (float)
        """
        probability = getattr(signal, "probability", 0.5)
        # Use signal payout if present, else fallback to class default
        payout_ratio = getattr(signal, "payout_ratio", self.default_payout_ratio)
        
        result = self.compute_stake(
            probability=probability,
            payout_ratio=payout_ratio,
            model_confidence=model_confidence,
            current_drawdown=current_drawdown,
            volatility_regime=volatility_regime,
            account_balance=account_balance,
        )
        
        logger.debug(f"Position sizing for signal: {result.reason}")
        return result.stake


class CompoundingPositionSizer:
    """
    Confidence-based Profit Reinvestment Sizer (Compounding).

    Implements a streak-based anti-martingale strategy where profit (or a multiplier)
    is reinvested into the next trade as long as the model maintains 
    a high confidence level and the maximum streak count is not exceeded.

    Strategy:
        Trade 1 (Base Stake) -> Win -> Trade 2 (Base * Multiplier) -> Win -> ... 
        
    Conditions for Compounding:
        1. Previous trade must be a WIN.
        2. Current streak count < max_consecutive_wins.
        3. Current model probability >= min_confidence_to_compound.
    
    Reset Conditions:
        1. A loss occurs (return to base stake).
        2. Streak cap reached (bank profit, return to base stake).
        3. Model probability drops below confidence threshold (safety reset).

    Attributes:
        base_stake: Starting stake amount.
        max_consecutive_wins: Maximum streak length before forced reset.
        min_confidence_to_compound: Probability threshold to allow stake increase.
        max_stake_cap: Hard ceiling on stake size regardless of compounding.
        streak_multiplier: Factor to multiply stake by after a win. 
                           If None, defaults to "reinvest full profit" (Classic compounding).
    """

    def __init__(
        self,
        base_stake: float,
        max_consecutive_wins: int = 5,
        min_confidence_to_compound: float = 0.70,
        max_stake_cap: float = 50.0,
        streak_multiplier: float | None = None,
    ):
        """
        Initialize compounding sizer.

        Args:
            base_stake: The initial stake amount.
            max_consecutive_wins: Number of wins before taking profit and resetting (e.g. 5).
            min_confidence_to_compound: Model probability required to increase stake.
            max_stake_cap: Absolute maximum allowed stake (safety break).
            streak_multiplier: Multiplier for next stake (e.g. 2.0 for 2x). 
                               If None, defaults to reinvesting principle + profit.
        """
        if base_stake <= 0:
            raise ValueError(f"base_stake must be positive, got {base_stake}")
        if max_consecutive_wins < 1:
            raise ValueError("max_consecutive_wins must be at least 1")

        self.base_stake = base_stake
        self.max_consecutive_wins = max_consecutive_wins
        self.min_confidence_to_compound = min_confidence_to_compound
        self.max_stake_cap = max_stake_cap
        self.streak_multiplier = streak_multiplier

        # State tracking
        self._current_streak = 0
        self._next_stake = base_stake
        self._lock = threading.Lock()
    
    def compute_stake(
        self,
        probability: float,
        payout_ratio: float = 0.9,  # Used only for logging context here
        **kwargs  # Accept kwargs to be compatible with other sizer interfaces
    ) -> PositionSizeResult:
        """
        Determine the stake for the *next* trade.

        Logic:
        - If streak is 0, use base_stake.
        - If streak > 0, check if probability >= min_confidence.
          - If yes, use the compounded _next_stake (limited by caps).
          - If no, RESET to base_stake (confidence not high enough to risk profits).
        """
        with self._lock:
            # 1. Check Confidence Safety Reset
            # If we have profits on the table but the model is unsure, take the profit now.
            if self._current_streak > 0 and probability < self.min_confidence_to_compound:
                logger.info(
                    f"Compounding reset due to low confidence ({probability:.2f} < {self.min_confidence_to_compound}). "
                    f"Banking profits from {self._current_streak} streak."
                )
                self._reset_streak_locked()
                # Note: reset_streak sets _next_stake to base_stake

            # 2. Check Max Stake Cap (Safety)
            final_stake = min(self._next_stake, self.max_stake_cap)
            if final_stake < self._next_stake:
                reason = f"Streak {self._current_streak}/{self.max_consecutive_wins} (Capped at ${self.max_stake_cap})"
            else:
                if self._current_streak == 0:
                    reason = f"Base Stake (Streak 0)"
                else:
                    reason = f"Compounding: Streak {self._current_streak}/{self.max_consecutive_wins} (Prob {probability:.2f})"

            return PositionSizeResult(
                stake=round(final_stake, 2),
                kelly_fraction=0.0, # Not applicable
                adjusted_fraction=0.0,
                confidence_multiplier=1.0,
                drawdown_multiplier=1.0,
                reason=reason,
            )

    def record_outcome(self, pnl: float, won: bool) -> None:
        """
        Update the internal state based on the completed trade's result.

        Args:
            pnl: Profit/Loss amount (positive for win, negative for loss).
            won: Boolean indicating win status.
        """
        with self._lock:
            if not won:
                # Loss: Reset immediately to base stake
                logger.info(f"Streak broken at {self._current_streak}. Resetting to base stake.")
                self._reset_streak_locked()
                return

            # Trade Won
            current_stake_used = self._next_stake
            
            # Increment streak
            self._current_streak += 1

            # Check if we hit the limit
            if self._current_streak >= self.max_consecutive_wins:
                logger.info(f"Max streak of {self.max_consecutive_wins} reached! Banking profits. Resetting.")
                self._reset_streak_locked()
            else:
                # Calculate next stake
                if self.streak_multiplier:
                    # Explicit multiplier (e.g. 2x, 1.5x)
                    # Next stake = Base Stake * (Multiplier ^ Streak) ??? 
                    # OR is it Previous Stake * Multiplier? -> "2x options". Usually means 2x previous.
                    # Standard implementation: Previous * Multiplier.
                    self._next_stake = current_stake_used * self.streak_multiplier
                else:
                    # Reinvest Principle + Profit (Standard Compounding)
                    # New Stake = Old Stake + Profit (P&L)
                    self._next_stake = current_stake_used + pnl
                
                logger.debug(
                    f"Win recorded. Compounding to level {self._current_streak}. "
                    f"Next stake: ${self._next_stake:.2f}"
                )

    def reset_streak(self) -> None:
        """Force reset to base settings."""
        with self._lock:
            self._reset_streak_locked()

    def _reset_streak_locked(self) -> None:
        """Internal uncached reset."""
        self._current_streak = 0
        self._next_stake = self.base_stake

    def get_current_streak(self) -> int:
        """Return current win streak count."""
        with self._lock:
            return self._current_streak
    
    def suggest_stake_for_signal(self, signal: Any, **kwargs) -> float:
        """Convenience method."""
        probability = getattr(signal, "probability", 0.5)
        res = self.compute_stake(probability=probability)
        return res.stake


class MartingalePositionSizer:
    """
    Martingale Position Sizer (Double Down on LOSS).

    WARNING: EXTREMELY HIGH RISK.
    Doubles the stake after every loss to recover previous losses.
    Resets to base stake after a win.

    Includes safety caps to prevent account blowouts.
    """

    def __init__(
        self,
        base_stake: float,
        multiplier: float = 2.0,
        max_streak: int = 5,
        max_stake_cap: float = 50.0,
    ):
        """
        Initialize Martingale sizer.

        Args:
            base_stake: Initial stake.
            multiplier: Multiplier after loss (default 2.0).
            max_streak: Max consecutive losses to double down before giving up.
            max_stake_cap: Hard cap on stake.
        """
        if base_stake <= 0:
            raise ValueError("base_stake must be positive")
        
        self.base_stake = base_stake
        self.multiplier = multiplier
        self.max_streak = max_streak
        self.max_stake_cap = max_stake_cap
        
        self._loss_streak = 0
        self._next_stake = base_stake
        self._lock = threading.Lock()

    def compute_stake(self, probability: float = 0.5, **kwargs) -> PositionSizeResult:
        """Compute stake based on loss streak."""
        
        with self._lock:
            # Apply strict max stake cap
            final_stake = min(self._next_stake, self.max_stake_cap)
            if final_stake < self._next_stake:
                reason = f"Martingale (Loss Streak {self._loss_streak}) - Capped at ${self.max_stake_cap}"
            else:
                if self._loss_streak == 0:
                    reason = "Martingale Base Stake"
                else:
                    reason = f"Martingale Recovery (Loss Streak {self._loss_streak})"

            return PositionSizeResult(
                stake=round(final_stake, 2),
                kelly_fraction=0.0,
                adjusted_fraction=0.0,
                confidence_multiplier=1.0,
                drawdown_multiplier=1.0,
                reason=reason,
            )

    def record_outcome(self, pnl: float, won: bool) -> None:
        """Update state: Reset on Win, Increase on Loss."""
        with self._lock:
            if won:
                logger.info(f"Martingale: Win at streak {self._loss_streak}. Resetting to base.")
                self._loss_streak = 0
                self._next_stake = self.base_stake
            else:
                # Loss
                self._loss_streak += 1
                if self._loss_streak >= self.max_streak:
                    logger.info(f"Martingale: Max loss streak {self.max_streak} hit. Resetting to base (Stop Loss).")
                    self._loss_streak = 0
                    self._next_stake = self.base_stake
                else:
                    # Double down
                    self._next_stake = self._next_stake * self.multiplier
                    logger.info(f"Martingale: Loss recorded. Increasing stake to ${self._next_stake:.2f}")

    def suggest_stake_for_signal(self, signal: Any, **kwargs) -> float:
        """Convenience method."""
        return self.compute_stake().stake


class FixedStakeSizer:
    """
    Simple fixed stake position sizer for baseline comparison.
    
    Always returns the same stake regardless of probability or conditions.
    Useful for testing and comparison against Kelly-based sizing.
    """
    
    def __init__(self, stake: float = 1.0):
        """
        Initialize fixed stake sizer.
        
        Args:
            stake: Fixed stake amount for all trades
        """
        self.stake = stake
    
    def compute_stake(self, **kwargs) -> PositionSizeResult:
        """Return fixed stake regardless of inputs."""
        return PositionSizeResult(
            stake=self.stake,
            kelly_fraction=0.0,
            adjusted_fraction=0.0,
            confidence_multiplier=1.0,
            drawdown_multiplier=1.0,
            reason=f"Fixed stake: ${self.stake:.2f}",
        )
    
    def suggest_stake_for_signal(self, signal: Any, **kwargs) -> float:
        """Return fixed stake."""
        return self.stake
