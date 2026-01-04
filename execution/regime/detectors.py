"""
Market Regime Detectors.

Algorithms for estimating Hurst exponent, volatility percentiles, and trend strength.
"""

from typing import Any

import numpy as np

from .types import (
    MacroRegime,
    MicroRegime,
    VolatilityRegime,
    VolatilityRegime,
    HierarchicalRegimeAssessment,
)
from .tracker import WindowedPercentileTracker


class HurstExponentEstimator:
    """Estimate the Hurst exponent for regime detection."""

    def __init__(self, min_window: int = 8, max_window: int | None = None):
        self.min_window = min_window
        self.max_window = max_window

    def estimate(self, prices: np.ndarray) -> float:
        """Estimate Hurst exponent from price series."""
        if len(prices) < self.min_window * 2:
            return 0.5  # Default to random walk

        returns = np.diff(np.log(prices + 1e-10))

        if len(returns) < self.min_window:
            return 0.5

        max_window = self.max_window or len(returns) // 2

        window_sizes = []
        rs_values = []

        for window in range(self.min_window, min(max_window + 1, len(returns) + 1)):
            n_windows = len(returns) // window
            if n_windows < 1:
                continue

            truncated_len = n_windows * window
            segments = returns[:truncated_len].reshape(n_windows, window)
            means = np.mean(segments, axis=1, keepdims=True)
            deviations = segments - means
            cumdev = np.cumsum(deviations, axis=1)
            R = np.max(cumdev, axis=1) - np.min(cumdev, axis=1)
            S = np.std(segments, axis=1, ddof=1)

            valid = S > 0
            if np.any(valid):
                avg_rs = np.mean(R[valid] / S[valid])
                window_sizes.append(window)
                rs_values.append(avg_rs)

        if len(window_sizes) < 2:
            return 0.5

        log_n = np.log(window_sizes)
        log_rs = np.log(rs_values)

        n = len(log_n)
        sum_x = np.sum(log_n)
        sum_y = np.sum(log_rs)
        sum_xy = np.sum(log_n * log_rs)
        sum_xx = np.sum(log_n ** 2)

        denominator = n * sum_xx - sum_x ** 2
        if abs(denominator) < 1e-10:
            return 0.5

        hurst = (n * sum_xy - sum_x * sum_y) / denominator
        return float(np.clip(hurst, 0.0, 1.0))


class VolatilityRegimeDetector:
    """Detect volatility regime using rolling volatility."""

    def __init__(self, low_percentile: float = 30, high_percentile: float = 70, lookback: int = 100):
        self.low_percentile = low_percentile
        self.high_percentile = high_percentile
        self.lookback = lookback

    def detect(self, prices: np.ndarray) -> tuple[VolatilityRegime, float]:
        if len(prices) < 20:
            return VolatilityRegime.MEDIUM, 50.0

        returns = np.diff(np.log(prices + 1e-10))
        lookback = min(len(returns), self.lookback)
        window = min(20, lookback // 2)

        if window < 5:
            return VolatilityRegime.MEDIUM, 50.0

        rolling_vol = np.array([
            np.std(returns[max(0, i - window):i + 1])
            for i in range(len(returns))
        ])

        current_vol = rolling_vol[-1]
        historical_vol = rolling_vol[-lookback:]
        percentile = 100 * np.sum(historical_vol < current_vol) / len(historical_vol)

        if percentile < self.low_percentile:
            regime = VolatilityRegime.LOW
        elif percentile > self.high_percentile:
            regime = VolatilityRegime.HIGH
        else:
            regime = VolatilityRegime.MEDIUM

        return regime, float(percentile)


class TrendDetector:
    """Detect macro trend regime using MA crossovers."""

    def __init__(self, short_window: int = 20, long_window: int = 50):
        self.short_window = short_window
        self.long_window = long_window

    def detect(self, prices: np.ndarray) -> tuple[MacroRegime, float]:
        if len(prices) < self.long_window:
            return MacroRegime.SIDEWAYS, 0.0

        short_ma = np.mean(prices[-self.short_window:])
        long_ma = np.mean(prices[-self.long_window:])
        current_price = prices[-1]

        trend_strength = (short_ma - long_ma) / long_ma if long_ma > 0 else 0.0

        bull_threshold = 0.01
        bear_threshold = -0.01

        if trend_strength > bull_threshold and current_price > short_ma:
            regime = MacroRegime.BULL
        elif trend_strength < bear_threshold and current_price < short_ma:
            regime = MacroRegime.BEAR
        else:
            regime = MacroRegime.SIDEWAYS

        return regime, float(trend_strength)


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
