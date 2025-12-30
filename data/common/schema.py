"""
Shared types and constants for data package to break circular dependencies.
"""

from dataclasses import dataclass
from typing import Any
import numpy as np

# Feature schema version - increment when feature definitions change
FEATURE_SCHEMA_VERSION = "1.1"  # v1.1: Normalized volatility metrics for autoencoder

@dataclass(frozen=True)
class FeatureSchema:
    """
    Immutable schema defining the exact feature structure.

    This is the CONTRACT that all data paths must honor.
    Changes to this schema require incrementing FEATURE_SCHEMA_VERSION.
    """

    tick_length: int
    candle_length: int
    candle_features: int = 10  # O, H, L, C, V, T, RSI, BB_width, BB_%b, ATR
    volatility_features: int = 4  # realized_vol, atr_mean, rsi_std, bb_w_mean

    def validate_ticks(self, ticks: np.ndarray) -> None:
        """Validate tick feature shape."""
        if ticks.shape != (self.tick_length,):
            raise ValueError(
                f"Tick shape mismatch: expected ({self.tick_length},), got {ticks.shape}"
            )

    def validate_candles(self, candles: np.ndarray) -> None:
        """Validate candle feature shape."""
        expected = (self.candle_length, self.candle_features)
        if candles.shape != expected:
            raise ValueError(f"Candle shape mismatch: expected {expected}, got {candles.shape}")

    def validate_volatility(self, vol: np.ndarray) -> None:
        """Validate volatility feature shape."""
        if vol.shape != (self.volatility_features,):
            raise ValueError(
                f"Volatility shape mismatch: expected ({self.volatility_features},), "
                f"got {vol.shape}"
            )
