"""
Shared types and constants for data package to break circular dependencies.
"""

from dataclasses import dataclass
from typing import Any
import numpy as np
import pandas as pd
import pandera.pandas as pa
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
    
    open: Series[np.float32] = pa.Field(ge=0.0, coerce=True)
    high: Series[np.float32] = pa.Field(ge=0.0, coerce=True)
    low: Series[np.float32] = pa.Field(ge=0.0, coerce=True)
    close: Series[np.float32] = pa.Field(ge=0.0, coerce=True)
    volume: Series[np.float32] = pa.Field(ge=0.0, coerce=True)
    timestamp: Series[np.float64] = pa.Field(gt=0.0, coerce=True) # Timestamp usually float64 for precision

    @pa.dataframe_check
    def high_gte_low(cls, df: DataFrame) -> Series[bool]:  # type: ignore
        """Ensure High is greater than or equal to Low."""
        from typing import cast
        return cast(Series[bool], df["high"] >= df["low"])

    @pa.dataframe_check
    def high_gte_open_close(cls, df: DataFrame) -> Series[bool]:  # type: ignore
        """Ensure High is the maximum of the bar."""
        from typing import cast
        return cast(Series[bool], (df["high"] >= df["open"]) & (df["high"] >= df["close"]))

    @pa.dataframe_check
    def low_lte_open_close(cls, df: DataFrame) -> Series[bool]:  # type: ignore
        """Ensure Low is the minimum of the bar."""
        from typing import cast
        return cast(Series[bool], (df["low"] <= df["open"]) & (df["low"] <= df["close"]))


@dataclass(frozen=True)
class FeatureOutputSchema:
    """
    Schema for validating final feature tensors before model input.
    
    Enforces:
    1. Shape consistency
    2. Data type consistency (float32)
    3. Finiteness (no NaNs/Infs)
    """
    
    tick_length: int
    candle_length: int
    candle_features: int = 10
    volatility_features: int = 4
    
    def validate(self, features: dict[str, torch.Tensor]) -> None:
        """
        Validate feature dictionary against schema.
        
        Args:
            features: Dictionary containing 'ticks', 'candles', 'vol_metrics' tensors
            
        Raises:
            ValueError: If validation fails
            TypeError: If dtypes are incorrect
        """
        # 1. Dtype Validation (CRITICAL-005)
        for key, tensor in features.items():
            if tensor.dtype != torch.float32:
                 raise TypeError(f"Tensor '{key}' has invalid dtype {tensor.dtype}. Expected float32.")
            
            if not torch.isfinite(tensor).all():
                 raise ValueError(f"Tensor '{key}' contains non-finite values (NaN/Inf).")

        # 2. Shape Validation
        ticks = features["ticks"]
        candles = features["candles"]
        vol_metrics = features["vol_metrics"]
        
        if ticks.shape != (self.tick_length,):
            raise ValueError(f"Ticks shape mismatch: {ticks.shape} != ({self.tick_length},)")
            
        if candles.shape != (self.candle_length, self.candle_features):
             raise ValueError(f"Candles shape mismatch: {candles.shape} != ({self.candle_length}, {self.candle_features})")
             
        if vol_metrics.shape != (self.volatility_features,):
             raise ValueError(f"Vol metrics shape mismatch: {vol_metrics.shape} != ({self.volatility_features},)")


