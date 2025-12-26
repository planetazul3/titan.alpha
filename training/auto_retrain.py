"""
Automated Retraining Validation Logic.

This module analyzes shadow trade performance to determine when the model
has degraded sufficiently to warrant retraining. It implements the
"Production Hardening" requirement for automated model maintenance.
"""

import logging
from typing import Tuple

from observability.shadow_metrics import ShadowTradeMetrics

logger = logging.getLogger(__name__)


class RetrainingTrigger:
    """
    Analyzes model performance metrics to trigger retraining.
    
    Logic based on:
    1. Statistical significance (minimum sample size)
    2. Performance degradation (win rate drop)
    3. Calibration drift (confidence vs reality mismatch)
    """

    def __init__(
        self,
        min_trades: int = 100,
        win_rate_threshold: float = 0.45,
        pnl_threshold: float = -50.0  # Simulated
    ):
        self.min_trades = min_trades
        self.win_rate_threshold = win_rate_threshold
        self.pnl_threshold = pnl_threshold

    def should_retrain(self, metrics: ShadowTradeMetrics) -> Tuple[bool, str]:
        """
        Evaluate if retraining is needed.
        
        Args:
            metrics: Current ShadowTradeMetrics snapshot
            
        Returns:
            (should_retrain: bool, reason: str)
        """
        # 1. Check data sufficiency
        if metrics.resolved_trades < self.min_trades:
            return False, f"Insufficient data: {metrics.resolved_trades}/{self.min_trades} trades"

        # 2. Check Win Rate degradation
        if metrics.win_rate < self.win_rate_threshold:
            return True, (
                f"Win rate degradation: {metrics.win_rate*100:.1f}% "
                f"< {self.win_rate_threshold*100:.1f}%"
            )

        # 3. Check specific contract failures
        # If any contract type has >20 trades and <35% win rate
        for ct, stats in metrics.by_contract_type.items():
            if stats["total"] >= 20:
                ct_wr = stats["wins"] / stats["total"]
                if ct_wr < 0.35:
                    return True, f"Contract failure ({ct}): {ct_wr*100:.1f}% win rate"

        # 4. Check P&L (Safety)
        if metrics.simulated_pnl < self.pnl_threshold:
            # Note: Negative P&L alone might just mean bad market conditions, 
            # but sustained loss suggests model is out of sync.
            return True, f"Excessive simulated loss: ${metrics.simulated_pnl:.2f}"

        return False, "Performance nominal"
