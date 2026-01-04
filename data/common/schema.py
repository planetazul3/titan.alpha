"""
Shared types and constants for data package to break circular dependencies.
"""

from dataclasses import dataclass
from typing import Any
import numpy as np
import pandas as pd
import pandera as pa
from pandera.typing import DataFrame, Series

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


class CandleInputSchema(pa.DataFrameModel):
    """
    Pandera schema for validating raw candle inputs.
    
    Enforces statistical sanity before feature engineering:
    - Non-negative prices/volumes
    - High >= Low
    - High >= Open/Close
    - Low <= Open/Close
    """
    
    open: Series[float] = pa.Field(ge=0.0)
    high: Series[float] = pa.Field(ge=0.0)
    low: Series[float] = pa.Field(ge=0.0)
    close: Series[float] = pa.Field(ge=0.0)
    volume: Series[float] = pa.Field(ge=0.0)
    timestamp: Series[float] = pa.Field(gt=0.0)

    @pa.dataframe_check
    def high_gte_low(cls, df: DataFrame) -> Series[bool]:
        """Ensure High is greater than or equal to Low."""
        return df["high"] >= df["low"]

    @pa.dataframe_check
    def high_gte_open_close(cls, df: DataFrame) -> Series[bool]:
        """Ensure High is the maximum of the bar."""
        return (df["high"] >= df["open"]) & (df["high"] >= df["close"])

    @pa.dataframe_check
    def low_lte_open_close(cls, df: DataFrame) -> Series[bool]:
        """Ensure Low is the minimum of the bar."""
        return (df["low"] <= df["open"]) & (df["low"] <= df["close"])

