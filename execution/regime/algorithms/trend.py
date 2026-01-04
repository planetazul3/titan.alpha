import numpy as np
from ..types import MacroRegime

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
