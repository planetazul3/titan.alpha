"""
Shadow Trade Performance Metrics.

This module provides real-time performance tracking for shadow trades,
treating them as first-class citizens for model improvement diagnostics.

Based on industry research:
- Binary options typically pay ~95% on wins (Deriv standard)
- Shadow trades should model realistic slippage and execution costs
- Performance metrics enable data-driven model retraining decisions

Usage:
    >>> from observability.shadow_metrics import ShadowTradeMetrics
    >>> metrics = ShadowTradeMetrics()
    >>> metrics.update_from_store(shadow_store)
    >>> print(f"Win rate: {metrics.win_rate * 100:.1f}%")
    >>> print(f"Simulated P&L: ${metrics.simulated_pnl:.2f}")
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from execution.sqlite_shadow_store import SQLiteShadowStore

logger = logging.getLogger(__name__)

# Industry standard payout for binary options (Deriv: ~95%)
# NOTE: For consistency, use settings.trading.payout_ratio when available.


@dataclass
class ShadowTradeMetrics:
    """
    Real-time shadow trade performance metrics.
    
    Shadow trades are treated as first-class citizens for model improvement.
    All metrics simulate real trading conditions with industry-standard payouts.
    
    Attributes:
        total_trades: Total shadow trades created
        resolved_trades: Shadow trades with determined outcomes
        unresolved_trades: Shadow trades still pending resolution
        wins: Number of winning trades
        losses: Number of losing trades
        win_rate: Win percentage (wins / resolved_trades)
        by_contract_type: Performance breakdown by contract type
        avg_probability: Average model probability across all trades
        confidence_distribution: Binned probability distribution
        simulated_pnl: Simulated profit/loss with $1 stake per trade
        simulated_roi: Return on investment percentage
        wins_by_regime: Win count grouped by regime state
        losses_by_regime: Loss count grouped by regime state
        last_hour_win_rate: Rolling win rate for last 60 minutes
        last_24h_win_rate: Rolling win rate for last 24 hours
    """
    
    total_trades: int = 0
    resolved_trades: int = 0
    unresolved_trades: int = 0
    
    # Win/Loss tracking (treat as real trades for metrics)
    wins: int = 0
    losses: int = 0
    win_rate: float = 0.0
    
    # By contract type
    by_contract_type: dict[str, dict] = field(default_factory=dict)
    
    # Probability analysis
    avg_probability: float = 0.0
    confidence_distribution: dict[str, int] = field(default_factory=dict)
    
    # ROI simulation (as if real trades with $1 stake)
    simulated_pnl: float = 0.0
    simulated_roi: float = 0.0
    
    # Regime correlation
    wins_by_regime: dict[str, int] = field(default_factory=dict)
    losses_by_regime: dict[str, int] = field(default_factory=dict)
    
    # Time-based metrics
    last_hour_win_rate: float = 0.0
    last_24h_win_rate: float = 0.0
    
    def update_from_store(self, shadow_store: "SQLiteShadowStore", payout_ratio: float = 0.95, max_detailed_trades: int = 1000) -> None:
        """
        Update metrics from shadow trade store (optimized).
        
        Args:
            shadow_store: SQLiteShadowStore instance
            payout_ratio: Payout ratio for P&L simulation (e.g., 0.95)
            max_detailed_trades: Max trades to analyze in detail
        """
        # Reset counters for fresh calculation
        self.wins = 0
        self.losses = 0
        self.by_contract_type = {}
        self.wins_by_regime = {}
        self.losses_by_regime = {}
        self.confidence_distribution = {
            "0.50-0.60": 0,
            "0.60-0.70": 0,
            "0.70-0.80": 0,
            "0.80-0.90": 0,
            "0.90-1.00": 0,
        }
        
        # Use efficient SQL statistics for summary metrics
        db_stats = shadow_store.get_statistics()
        self.total_trades = db_stats["total_records"]
        self.resolved_trades = db_stats["resolved_records"]
        self.unresolved_trades = db_stats["unresolved_records"]
        self.wins = db_stats["wins"]
        self.losses = db_stats["losses"]
        self.win_rate = db_stats["win_rate"]
        
        if self.total_trades == 0:
            logger.debug("No shadow trades to analyze")
            return
        
        # Calculate simulated P&L from summary stats (no iteration needed)
        if self.resolved_trades > 0:
            # Wins pay payout_ratio profit, losses lose 100% of stake
            self.simulated_pnl = (self.wins * payout_ratio) - self.losses
            self.simulated_roi = (self.simulated_pnl / self.resolved_trades) * 100
        
        # Only query recent trades for detailed analysis (limit to avoid unbounded growth)
        # This is sufficient for confidence distribution and regime correlation
        now = datetime.now()
        one_week_ago = now - timedelta(days=7)
        
        # Query recent trades (last 7 days, limited to max_detailed_trades)
        recent_all = shadow_store.query(start=one_week_ago, resolved_only=False)
        recent_resolved = shadow_store.query(start=one_week_ago, resolved_only=True)
        
        # Apply limit to prevent unbounded memory usage
        if len(recent_all) > max_detailed_trades:
            logger.warning(
                f"Limiting detailed metrics to {max_detailed_trades} most recent trades "
                f"(total: {len(recent_all)})"
            )
            recent_all = recent_all[-max_detailed_trades:]  # Most recent
        
        if len(recent_resolved) > max_detailed_trades:
            recent_resolved = recent_resolved[-max_detailed_trades:]
        
        # Calculate average probability from recent trades only
        if recent_all:
            total_prob = sum(t.probability for t in recent_all)
            self.avg_probability = total_prob / len(recent_all)
            
            # Bin probabilities for confidence distribution
            for trade in recent_all:
                prob = trade.probability
                if 0.50 <= prob < 0.60:
                    self.confidence_distribution["0.50-0.60"] += 1
                elif 0.60 <= prob < 0.70:
                    self.confidence_distribution["0.60-0.70"] += 1
                elif 0.70 <= prob < 0.80:
                    self.confidence_distribution["0.70-0.80"] += 1
                elif 0.80 <= prob < 0.90:
                    self.confidence_distribution["0.80-0.90"] += 1
                elif 0.90 <= prob <= 1.00:
                    self.confidence_distribution["0.90-1.00"] += 1
        
        # Calculate contract type breakdown and regime correlation from recent resolved trades
        for trade in recent_resolved:
            if trade.outcome is None:
                continue
            
            # Regime correlation (recent trends)
            if trade.outcome:
                self.wins_by_regime[trade.regime_state] = (
                    self.wins_by_regime.get(trade.regime_state, 0) + 1
                )
            else:
                self.losses_by_regime[trade.regime_state] = (
                    self.losses_by_regime.get(trade.regime_state, 0) + 1
                )
            
            # Contract type breakdown (recent performance)
            ct = trade.contract_type
            if ct not in self.by_contract_type:
                self.by_contract_type[ct] = {"wins": 0, "losses": 0, "total": 0}
            
            self.by_contract_type[ct]["total"] += 1
            if trade.outcome:
                self.by_contract_type[ct]["wins"] += 1
            else:
                self.by_contract_type[ct]["losses"] += 1
        
        # Time-based metrics (last hour and 24h)
        one_hour_ago = now - timedelta(hours=1)
        one_day_ago = now - timedelta(days=1)
        
        # Query trades from last hour
        last_hour_trades = shadow_store.query(start=one_hour_ago, resolved_only=True)
        if last_hour_trades:
            last_hour_wins = sum(1 for t in last_hour_trades if t.outcome)
            self.last_hour_win_rate = last_hour_wins / len(last_hour_trades)
        
        # Query trades from last 24 hours
        last_24h_trades = shadow_store.query(start=one_day_ago, resolved_only=True)
        if last_24h_trades:
            last_24h_wins = sum(1 for t in last_24h_trades if t.outcome)
            self.last_24h_win_rate = last_24h_wins / len(last_24h_trades)
        
        logger.debug(
            f"Shadow metrics updated: {self.resolved_trades} resolved (total), "
            f"{len(recent_resolved)} analyzed (recent 7d), "
            f"{self.win_rate * 100:.1f}% win rate, "
            f"P&L: ${self.simulated_pnl:.2f}"
        )
    
    def get_summary(self) -> dict:
        """
        Get summary dictionary for logging and observability.
        
        Returns:
            Dict with key performance metrics
        """
        return {
            "total_trades": self.total_trades,
            "resolved_trades": self.resolved_trades,
            "unresolved_trades": self.unresolved_trades,
            "wins": self.wins,
            "losses": self.losses,
            "win_rate": round(self.win_rate, 4),
            "avg_probability": round(self.avg_probability, 4),
            "simulated_pnl": round(self.simulated_pnl, 2),
            "simulated_roi_pct": round(self.simulated_roi, 2),
            "last_hour_win_rate": round(self.last_hour_win_rate, 4),
            "last_24h_win_rate": round(self.last_24h_win_rate, 4),
            "contract_types": len(self.by_contract_type),
        }
    
    def should_trigger_retraining(
        self,
        min_resolved_trades: int = 100,
        min_win_rate: float = 0.45,
    ) -> tuple[bool, str]:
        """
        Determine if model retraining should be triggered based on metrics.
        
        Based on industry best practices:
        - Require minimum sample size before making retraining decisions
        - Trigger on sustained poor performance (win rate < 45%)
        - Consider regime correlation for debugging
        
        Args:
            min_resolved_trades: Minimum trades before making decision
            min_win_rate: Minimum acceptable win rate
        
        Returns:
            Tuple of (should_retrain, reason)
        """
        if self.resolved_trades < min_resolved_trades:
            return False, f"Insufficient data ({self.resolved_trades}/{min_resolved_trades})"
        
        if self.win_rate < min_win_rate:
            return True, f"Win rate {self.win_rate * 100:.1f}% below threshold {min_win_rate * 100:.1f}%"
        
        # Check if any contract type has extremely poor performance
        for ct, stats in self.by_contract_type.items():
            if stats["total"] >= 20:  # Need minimum sample
                ct_win_rate = stats["wins"] / stats["total"]
                if ct_win_rate < 0.30:  # Critical threshold
                    return True, f"Contract type {ct} has {ct_win_rate * 100:.1f}% win rate"
        
        return False, "Performance within acceptable range"
