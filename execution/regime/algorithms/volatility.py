import numpy as np
from ..types import VolatilityRegime

class VolatilityRegimeDetector:
    """Detect volatility regime using rolling volatility."""

    def __init__(self, low_percentile: float = 30, high_percentile: float = 70, lookback: int = 100, min_warmup: int = 120):
        self.low_percentile = low_percentile
        self.high_percentile = high_percentile
        self.lookback = lookback
        self.min_warmup = min_warmup

    def detect(self, prices: np.ndarray) -> tuple[VolatilityRegime, float]:
        # H4: Warmup Veto - Enforce minimum history
        if len(prices) < self.min_warmup:
            # Insufficient data to determine regime -> Treat as High Risk
            return VolatilityRegime.HIGH, 100.0

        returns = np.diff(np.log(prices + 1e-10))
        lookback = min(len(returns), self.lookback)
        window = min(20, lookback // 2)

        if window < 5:
            return VolatilityRegime.HIGH, 100.0

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
