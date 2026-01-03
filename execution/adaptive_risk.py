"""
Adaptive Risk Limits Module.

Implements dynamic risk limit adjustment based on model
performance, market conditions, and trading outcomes.

Key capabilities:
- Sharpe-based limit scaling
- Regime-aware risk adjustment
- Automatic recovery after drawdown
- Performance-based stake scaling

ARCHITECTURAL PRINCIPLE:
Risk limits should adapt to market conditions. During high
uncertainty (low trust score, high volatility), limits tighten.
During favorable conditions with proven performance, limits
can relax within configured bounds.

Example:
    >>> from execution.adaptive_risk import AdaptiveRiskManager
    >>> risk_mgr = AdaptiveRiskManager(base_daily_limit=20.0)
    >>> adjusted_limit = risk_mgr.get_adjusted_limits(trust_score=0.8)
"""

import logging
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


class RiskLevel(Enum):
    """Risk level states."""
    CONSERVATIVE = "conservative"  # Reduced limits
    NORMAL = "normal"              # Standard limits
    AGGRESSIVE = "aggressive"      # Increased limits (during favorable conditions)


@dataclass
class RiskLimits:
    """
    Current risk limits.
    
    Attributes:
        daily_loss_limit: Maximum daily loss allowed
        max_stake: Maximum stake per trade
        max_trades_per_hour: Rate limit
        max_drawdown: Maximum drawdown before pause
        risk_level: Current risk level
    """
    daily_loss_limit: float
    max_stake: float
    max_trades_per_hour: int
    max_drawdown: float
    risk_level: RiskLevel = RiskLevel.NORMAL
    
    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "daily_loss_limit": self.daily_loss_limit,
            "max_stake": self.max_stake,
            "max_trades_per_hour": self.max_trades_per_hour,
            "max_drawdown": self.max_drawdown,
            "risk_level": self.risk_level.value,
        }


class PerformanceTracker:
    """
    Track trading performance for risk adjustment.
    
    Computes rolling metrics like Sharpe ratio, win rate,
    and drawdown for risk decisions.
    """
    
    def __init__(self, window_size: int = 100):
        self.window_size = window_size
        self._returns: deque = deque(maxlen=window_size)
        self._peak_equity = 0.0
        self._current_drawdown = 0.0
        self._consecutive_losses = 0
    
    def record(
        self,
        pnl: float,
        current_equity: float | None = None,
    ) -> None:
        """
        Record trade outcome.
        
        Args:
            pnl: Profit/loss from trade
            current_equity: Current account equity
        """
        # Centralized validation
        from utils.numerical_validation import ensure_finite
        
        # Validate P&L (default safely to 0.0)
        pnl = ensure_finite(pnl, "PerformanceTracker.pnl", default=0.0)
        
        # Validate Equity (default to existing peak to avoid drawdown spikes, or 0 if none)
        if current_equity is not None:
             # If bad equity comes in, use peak equity to 'flatten' the curve rather than crash it
             # or use 0.0 if we have no history.
             safe_equity_default = self._peak_equity if self._peak_equity > 0 else 0.0
             current_equity = ensure_finite(
                 current_equity, 
                 "PerformanceTracker.current_equity", 
                 default=safe_equity_default
             )

        self._returns.append(pnl)
        
        if pnl < 0:
            self._consecutive_losses += 1
        else:
            self._consecutive_losses = 0

        if current_equity is not None:
            if current_equity > self._peak_equity:
                self._peak_equity = current_equity
            
            if self._peak_equity > 0:
                self._current_drawdown = (self._peak_equity - current_equity) / self._peak_equity
    
    def get_sharpe_ratio(self, risk_free_rate: float = 0.0) -> float:
        """
        Compute rolling Sharpe ratio.
        
        Returns:
            Annualized Sharpe ratio (approximate)
        """
        if len(self._returns) < 10:
            return 0.0
        
        returns = np.array(self._returns)
        mean_return = np.mean(returns)
        std_return = np.std(returns)
        
        if std_return == 0:
            return 0.0
        
        # Approximate annualization (assume ~10 trades/day, 252 days/year)
        sharpe = (mean_return - risk_free_rate) / std_return
        
        from utils.numerical_validation import ensure_finite
        result = float(sharpe * np.sqrt(252 * 10))
        return ensure_finite(result, "SharpeRatio", default=0.0)

    def get_win_rate(self) -> float:
        """Get rolling win rate."""
        if not self._returns:
            return 0.5
        wins = sum(1 for r in self._returns if r > 0)
        return wins / len(self._returns)
    
    def get_drawdown(self) -> float:
        """Get current drawdown."""
        from utils.numerical_validation import ensure_finite
        return ensure_finite(self._current_drawdown, "Drawdown", default=0.0)
    
    def get_consecutive_losses(self) -> int:
        """Get current consecutive losses."""
        return self._consecutive_losses

    def get_total_losses(self) -> float:
        """Get total magnitude of losses in the current window."""
        if not self._returns:
            return 0.0
        return float(abs(sum(r for r in self._returns if r < 0)))

    def get_profit_factor(self) -> float:
        """Get profit factor (gains / losses)."""
        if not self._returns:
            return 1.0
        
        gains = sum(r for r in self._returns if r > 0)
        losses = abs(sum(r for r in self._returns if r < 0))
        
        if losses < 1e-8:
            return 100.0 if gains > 0 else 1.0  # Cap at 100x instead of inf
            
        from utils.numerical_validation import ensure_finite
        return ensure_finite(float(gains / losses), "ProfitFactor", default=1.0)


class AdaptiveRiskManager:
    """
    Manages adaptive risk limits based on performance and conditions.
    
    Adjusts limits dynamically:
    - Tighten during drawdown or low trust
    - Relax during favorable conditions with good Sharpe
    - Implements recovery logic after hitting limits
    
    Example:
        >>> risk_mgr = AdaptiveRiskManager(base_daily_limit=20.0)
        >>> limits = risk_mgr.get_adjusted_limits(trust_score=0.8)
        >>> if risk_mgr.should_pause():
        ...     stop_trading()
    """
    
    def __init__(
        self,
        base_daily_limit: float = 20.0,
        base_max_stake: float = 5.0,
        base_trades_per_hour: int = 10,
        base_max_drawdown: float = 0.15,
        min_scale: float = 0.3,
        max_scale: float = 1.5,
        sharpe_threshold_aggressive: float = 1.5,
        sharpe_threshold_conservative: float = 0.5,
        drawdown_threshold: float = 0.1,
        recovery_period_trades: int = 20,
        state_store: Any = None,  # H04: Inject persistence store
    ):
        """
        Initialize risk manager.
        
        Args:
           ...
           state_store: Optional SQLiteSafetyStateStore for persistence
        """
        self.base_daily_limit = base_daily_limit
        self.base_max_stake = base_max_stake
        self.base_trades_per_hour = base_trades_per_hour
        self.base_max_drawdown = base_max_drawdown
        
        self.min_scale = min_scale
        self.max_scale = max_scale
        self.sharpe_threshold_aggressive = sharpe_threshold_aggressive
        self.sharpe_threshold_conservative = sharpe_threshold_conservative
        self.drawdown_threshold = drawdown_threshold
        self.recovery_period_trades = recovery_period_trades
        
        self.state_store = state_store
        self.performance = PerformanceTracker()
        
        self._daily_pnl = 0.0
        self._pause_until: datetime | None = None
        self._last_limit_hit: datetime | None = None
        self._trades_since_limit_hit = 0
        
        # Load persisted state if available
        if self.state_store:
            self._load_state()

        logger.info(
            f"AdaptiveRiskManager initialized: "
            f"base_limit={base_daily_limit}, base_stake={base_max_stake}, persisted={bool(state_store)}"
        )
    
    def _load_state(self):
        """Load state from persistent store."""
        try:
            metrics = self.state_store.get_risk_metrics()
            
            # Use centralized validation for loaded metrics
            from utils.numerical_validation import validate_numeric_dict
            metrics = validate_numeric_dict(metrics)
            
            self.performance._current_drawdown = metrics.get("current_drawdown", 0.0)
            self.performance._peak_equity = metrics.get("peak_equity", 0.0)
            self.performance._consecutive_losses = int(metrics.get("consecutive_losses", 0))
            logger.info(f"Restored risk state: drawdown={self.performance._current_drawdown:.3f}, losses={self.performance._consecutive_losses}")
        except Exception as e:
            logger.error(f"Failed to load risk state: {e}")

    def _save_state(self):
        """Save state to persistent store."""
        if not self.state_store:
            return
        
        # Ensure values are finite before saving
        from utils.numerical_validation import ensure_finite
        
        try:
            drawdown = ensure_finite(self.performance.get_drawdown(), "drawdown", 0.0)
            peak_equity = ensure_finite(self.performance._peak_equity, "peak_equity", 0.0)
            
            self.state_store.update_risk_metrics(
                drawdown=drawdown,
                losses=self.performance.get_consecutive_losses(),
                peak_equity=peak_equity
            )
        except Exception as e:
            logger.error(f"Failed to save risk state: {e}")

    def record_trade(
        self,
        pnl: float,
        current_equity: float | None = None,
    ) -> None:
        """
        Record trade outcome.
        
        Args:
            pnl: Profit/loss from trade
            current_equity: Current account equity
        """
        from utils.numerical_validation import ensure_finite
        
        pnl = ensure_finite(pnl, "AdaptiveRiskManager.pnl", default=0.0)

        self._daily_pnl += pnl
        self.performance.record(pnl, current_equity)
        self._trades_since_limit_hit += 1
        
        # Check if limit hit
        if self._daily_pnl <= -self.base_daily_limit:
            self._last_limit_hit = datetime.now(timezone.utc)
            self._trades_since_limit_hit = 0
            logger.warning(f"Daily loss limit hit: {self._daily_pnl}")
            
        # H04: Persist State
        self._save_state()
    
    def reset_daily(self) -> None:
        """Reset daily tracking (call at day start)."""
        self._daily_pnl = 0.0
        logger.info("Daily risk tracking reset")
    
    def get_adjusted_limits(
        self,
        trust_score: float = 1.0,
        volatility_regime: str = "medium",
    ) -> RiskLimits:
        """
        Get current adjusted risk limits.
        
        Args:
            trust_score: Regime trust score (0-1)
            volatility_regime: Current volatility regime
        
        Returns:
            Adjusted RiskLimits
        """
        # Base scaling factors
        sharpe = self.performance.get_sharpe_ratio()
        drawdown = self.performance.get_drawdown()
        win_rate = self.performance.get_win_rate()
        
        # Determine risk level
        if drawdown > self.drawdown_threshold or trust_score < 0.4:
            risk_level = RiskLevel.CONSERVATIVE
            base_scale = self.min_scale
        elif sharpe > self.sharpe_threshold_aggressive and trust_score > 0.7:
            risk_level = RiskLevel.AGGRESSIVE
            base_scale = min(self.max_scale, 1.0 + sharpe * 0.2)
        else:
            risk_level = RiskLevel.NORMAL
            base_scale = 1.0
        
        # Apply trust score modifier
        trust_modifier = 0.5 + 0.5 * trust_score
        
        # Apply volatility modifier
        vol_modifier = {
            "low": 1.2,
            "medium": 1.0,
            "high": 0.7,
        }.get(volatility_regime, 1.0)
        
        # Combined scale
        scale = base_scale * trust_modifier * vol_modifier
        scale = max(self.min_scale, min(self.max_scale, scale))
        
        # Apply recovery penalty if recently hit limit
        if self._last_limit_hit and self._trades_since_limit_hit < self.recovery_period_trades:
            recovery_factor = 0.5 + 0.5 * (self._trades_since_limit_hit / self.recovery_period_trades)
            scale *= recovery_factor
        
        return RiskLimits(
            daily_loss_limit=self.base_daily_limit * scale,
            max_stake=self.base_max_stake * scale,
            max_trades_per_hour=max(1, int(self.base_trades_per_hour * scale)),
            max_drawdown=self.base_max_drawdown,
            risk_level=risk_level,
        )
    
    def should_pause(self) -> bool:
        """
        Check if trading should be paused.
        
        Returns:
            True if trading should pause
        """
        # Daily limit exceeded
        if self._daily_pnl <= -self.base_daily_limit:
            return True
        
        # Maximum drawdown exceeded
        if self.performance.get_drawdown() > self.base_max_drawdown:
            return True
        
        # Explicit pause set
        if self._pause_until:
            if datetime.now(timezone.utc) < self._pause_until:
                return True
            self._pause_until = None
        
        return False
    
    def pause_until(self, until: datetime) -> None:
        """
        Pause trading until specified time.
        
        Args:
            until: Datetime to resume trading
        """
        self._pause_until = until
        logger.warning(f"Trading paused until {until}")
    
    def get_statistics(self) -> dict[str, Any]:
        """Get risk manager statistics."""
        return {
            "daily_pnl": self._daily_pnl,
            "sharpe_ratio": self.performance.get_sharpe_ratio(),
            "win_rate": self.performance.get_win_rate(),
            "drawdown": self.performance.get_drawdown(),
            "profit_factor": self.performance.get_profit_factor(),
            "total_losses": self.performance.get_total_losses(),
            "consecutive_losses": self.performance.get_consecutive_losses(),
            "should_pause": self.should_pause(),
            "trades_since_limit": self._trades_since_limit_hit,
        }
