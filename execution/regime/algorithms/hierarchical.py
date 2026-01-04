import numpy as np
from typing import Any

from ..types import (
    MacroRegime,
    MicroRegime,
    VolatilityRegime,
    HierarchicalRegimeAssessment,
)
from ..tracker import WindowedPercentileTracker

from .trend import TrendDetector
from .volatility import VolatilityRegimeDetector
from .hurst import HurstExponentEstimator


class HierarchicalRegimeDetector:
    """Multi-level regime detector combining macro, volatility, and micro signals."""

    def __init__(
        self,
        trend_short_window: int = 20,
        trend_long_window: int = 50,
        vol_low_percentile: float = 30,
        vol_high_percentile: float = 70,
        hurst_min_window: int = 8,
        reconstruction_weight: float = 0.3,
    ):
        self.trend_detector = TrendDetector(trend_short_window, trend_long_window)
        self.vol_detector = VolatilityRegimeDetector(vol_low_percentile, vol_high_percentile)
        self.hurst_estimator = HurstExponentEstimator(hurst_min_window)
        self.reconstruction_weight = reconstruction_weight
        # CRITICAL-003: Dynamic normalization
        self.recon_tracker = WindowedPercentileTracker(window_size=2000)

    def assess(self, prices: np.ndarray, reconstruction_error: float = 0.0) -> HierarchicalRegimeAssessment:
        macro, trend_strength = self.trend_detector.detect(prices)
        volatility, vol_percentile = self.vol_detector.detect(prices)
        hurst = self.hurst_estimator.estimate(prices)

        # Update tracker and get percentile
        recon_percentile = self.recon_tracker.update(reconstruction_error)

        if hurst > 0.55:
            micro = MicroRegime.TRENDING
        elif hurst < 0.45:
            micro = MicroRegime.MEAN_REVERTING
        else:
            micro = MicroRegime.RANDOM

        trust_score = self._calculate_trust_score(
            macro, volatility, micro,
            trend_strength, vol_percentile, hurst,
            recon_percentile
        )

        details = {
            "trend_strength": trend_strength,
            "vol_percentile": vol_percentile,
            "hurst_exponent": hurst,
            "recon_percentile": recon_percentile,
        }

        return HierarchicalRegimeAssessment(
            macro=macro,
            volatility=volatility,
            micro=micro,
            trust_score=trust_score,
            reconstruction_error=reconstruction_error,
            details=details,
        )

    def _calculate_trust_score(
        self, macro, volatility, micro, trend_strength, vol_percentile, hurst, recon_percentile
    ) -> float:
        trust = 1.0

        if volatility == VolatilityRegime.HIGH:
            trust *= 0.6
        elif volatility == VolatilityRegime.MEDIUM:
            trust *= 0.85

        if macro == MacroRegime.SIDEWAYS:
            trust *= 0.8
        elif abs(trend_strength) > 0.02:
            trust *= 1.0

        if micro == MicroRegime.RANDOM:
            trust *= 0.7
        else:
            trust *= 0.9

        # CRITICAL-003: Penalize based on dynamic percentile
        # 99th percentile -> 0.1 penalty factor (severe veto)
        # 95th percentile -> 0.5 penalty
        # 50th percentile -> 1.0 (no penalty)
        if recon_percentile > 99:
             trust *= 0.0  # Hard Veto (Regime Mismatch)
        elif recon_percentile > 95:
             trust *= 0.5
        elif recon_percentile > 80:
             trust *= 0.8

        return float(np.clip(trust, 0.0, 1.0))

    def assess_from_reconstruction_error(
        self, 
        reconstruction_error: float, 
        prices: np.ndarray | None = None,
        threshold_veto: float = 0.3,
        threshold_caution: float = 0.1
    ) -> HierarchicalRegimeAssessment:
        """Backward compatibility for reconstruction-only assessment."""
        if prices is not None and len(prices) >= 50:
            return self.assess(prices, reconstruction_error)

        # Fallback heuristics
        if reconstruction_error >= threshold_veto:
            volatility = VolatilityRegime.HIGH
            trust = 0.2
        elif reconstruction_error >= threshold_caution:
            volatility = VolatilityRegime.MEDIUM
            trust = 0.5 # 0.5 is within [0.3, 0.6) range for CAUTION
        else:
            volatility = VolatilityRegime.LOW
            trust = 0.9

        return HierarchicalRegimeAssessment(
            macro=MacroRegime.SIDEWAYS,
            volatility=volatility,
            micro=MicroRegime.RANDOM,
            trust_score=trust,
            reconstruction_error=reconstruction_error,
            details={"fallback_mode": True},
        )
