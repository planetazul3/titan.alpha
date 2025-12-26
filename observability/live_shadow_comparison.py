"""
Live vs Shadow Trade Comparison.

Comparator logic to identify "execution gap" - the difference between
theoretical model performance (shadow) and actual realized performance (live).
Tracks slippage, missed entries, and veto impacts.
"""

import logging
from dataclasses import dataclass, field
from typing import List, Optional

from execution.signals import ShadowTrade, TradeSignal

logger = logging.getLogger(__name__)


@dataclass
class ComparisonMetric:
    real_trades_count: int = 0
    shadow_trades_count: int = 0
    vetoed_count: int = 0
    start_time: float = 0.0


class LiveShadowComparison:
    """Tracks divergence between real and shadow trades."""
    
    def __init__(self):
        self.metrics = ComparisonMetric()
        
    def record_iteration(
        self, 
        real_signals: List[TradeSignal], 
        shadow_signals: List[ShadowTrade], 
        vetoed: bool
    ) -> None:
        """Record results of a single decision cycle."""
        self.metrics.real_trades_count += len(real_signals)
        self.metrics.shadow_trades_count += len(shadow_signals)
        if vetoed:
            self.metrics.vetoed_count += 1
            
    def get_summary(self) -> dict:
        """Return summary of live vs shadow activity."""
        total = self.metrics.real_trades_count + self.metrics.shadow_trades_count
        ratio = 0.0
        if total > 0:
            ratio = self.metrics.real_trades_count / total
            
        return {
            "real_vs_shadow_ratio": ratio,
            "total_vetoes": self.metrics.vetoed_count,
            "real_total": self.metrics.real_trades_count,
            "shadow_total": self.metrics.shadow_trades_count
        }
