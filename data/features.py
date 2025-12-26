"""
Canonical Feature Builder - THE ONLY way to prepare features for models.

This module provides a single, unified entry point for all feature engineering.
Every data path in the system (training, replay, shadow, live inference) MUST
use this FeatureBuilder to ensure semantic consistency.

ARCHITECTURAL PRINCIPLE:
There is ONE way to transform raw market data into model-ready features.
No alternative paths are allowed. This prevents the common failure mode where
models receive "valid-shaped" data that is semantically inconsistent across runs.

Usage:
    >>> from data.features import FeatureBuilder
    >>> builder = FeatureBuilder(settings)
    >>> features = builder.build(ticks=raw_ticks, candles=raw_candles)
    >>> # features is a dict ready for model input
"""

import logging
from dataclasses import dataclass

import numpy as np
import torch

from config.settings import Settings
from data.processor import CandlePreprocessor, TickPreprocessor, VolatilityMetricsExtractor

logger = logging.getLogger(__name__)


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


class FeatureBuilder:
    """
    CANONICAL FEATURE BUILDER - The single source of truth for feature engineering.

    This class is THE ONLY way to transform raw market data into model-ready features.
    All code paths (training, validation, shadow trades, live inference) MUST use
    this builder to ensure:

    1. Consistent feature engineering across all runs
    2. Identical normalization procedures
    3. Validated shape contracts
    4. Reproducible preprocessing

    DO NOT:
    - Create TickPreprocessor/CandlePreprocessor directly (deprecated pattern)
    - Apply custom normalization to raw data
    - Bypass this builder for "quick" feature creation

    Attributes:
        settings: Configuration settings
        schema: Feature schema defining exact shapes

    Example:
        >>> builder = FeatureBuilder(settings)
        >>>
        >>> # For training/inference
        >>> features = builder.build(ticks=raw_ticks, candles=raw_candles)
        >>> model_output = model(
        ...     features['ticks'],
        ...     features['candles'],
        ...     features['vol_metrics']
        ... )
        >>>
        >>> # For batched training
        >>> batch = builder.build_batch(tick_list, candle_list)
    """

    def __init__(self, settings: Settings):
        """
        Initialize the canonical feature builder.

        Args:
            settings: Configuration settings with data shape parameters
        """
        self.settings = settings

        # Create preprocessors (internal use only)
        self._tick_pp = TickPreprocessor(settings)
        self._candle_pp = CandlePreprocessor(settings)
        self._vol_ext = VolatilityMetricsExtractor()

        # Define schema from settings
        self.schema = FeatureSchema(
            tick_length=settings.data_shapes.sequence_length_ticks,
            candle_length=settings.data_shapes.sequence_length_candles,
        )

        logger.info(
            f"FeatureBuilder initialized (schema v{FEATURE_SCHEMA_VERSION}): "
            f"ticks={self.schema.tick_length}, candles={self.schema.candle_length}x{self.schema.candle_features}"
        )

    def build(
        self, ticks: np.ndarray, candles: np.ndarray, validate: bool = True
    ) -> dict[str, torch.Tensor]:
        """
        Build features from raw market data.

        This is THE ONLY method that should be used to create model-ready features.

        Args:
            ticks: Raw tick prices, shape (N,) where N >= tick_length
            candles: Raw OHLCVT data, shape (M, 6) where M >= candle_length
            validate: Whether to validate output shapes (recommended True)

        Returns:
            Dict with:
            - 'ticks': torch.Tensor, shape (tick_length,)
            - 'candles': torch.Tensor, shape (candle_length, 10)
            - 'vol_metrics': torch.Tensor, shape (4,)

        Raises:
            ValueError: If input data is insufficient or output fails validation

        Example:
            >>> features = builder.build(raw_ticks, raw_candles)
            >>> probs = model.predict_probs(
            ...     features['ticks'].unsqueeze(0),
            ...     features['candles'].unsqueeze(0),
            ...     features['vol_metrics'].unsqueeze(0)
            ... )
        """
        # Process through canonical preprocessors
        tick_features = self._tick_pp.process(ticks)
        candle_features = self._candle_pp.process(candles)
        vol_features = self._vol_ext.extract(candles)

        # Validate shapes
        if validate:
            self.schema.validate_ticks(tick_features)
            self.schema.validate_candles(candle_features)
            self.schema.validate_volatility(vol_features)

        return {
            "ticks": torch.from_numpy(tick_features),
            "candles": torch.from_numpy(candle_features),
            "vol_metrics": torch.from_numpy(vol_features),
        }

    def build_numpy(
        self, ticks: np.ndarray, candles: np.ndarray, validate: bool = True
    ) -> dict[str, np.ndarray]:
        """
        Build features as numpy arrays (for dataset creation).

        Args:
            ticks: Raw tick prices
            candles: Raw OHLCVT data
            validate: Whether to validate shapes

        Returns:
            Dict with numpy arrays instead of tensors
        """
        tick_features = self._tick_pp.process(ticks)
        candle_features = self._candle_pp.process(candles)
        vol_features = self._vol_ext.extract(candles)

        if validate:
            self.schema.validate_ticks(tick_features)
            self.schema.validate_candles(candle_features)
            self.schema.validate_volatility(vol_features)

        return {"ticks": tick_features, "candles": candle_features, "vol_metrics": vol_features}

    def build_batch(self, ticks_list: list, candles_list: list) -> dict[str, torch.Tensor]:
        """
        Build batched features from lists of raw data.

        Args:
            ticks_list: List of raw tick arrays
            candles_list: List of raw candle arrays (must match length)

        Returns:
            Dict with batched tensors, each with shape (batch_size, ...)

        Raises:
            ValueError: If list lengths don't match
        """
        if len(ticks_list) != len(candles_list):
            raise ValueError(
                f"Mismatched batch sizes: {len(ticks_list)} ticks vs {len(candles_list)} candles"
            )

        batch_ticks = []
        batch_candles = []
        batch_vol = []

        for ticks, candles in zip(ticks_list, candles_list, strict=False):
            features = self.build(ticks, candles)
            batch_ticks.append(features["ticks"])
            batch_candles.append(features["candles"])
            batch_vol.append(features["vol_metrics"])

        return {
            "ticks": torch.stack(batch_ticks),
            "candles": torch.stack(batch_candles),
            "vol_metrics": torch.stack(batch_vol),
        }

    def get_schema_version(self) -> str:
        """Get current feature schema version."""
        return FEATURE_SCHEMA_VERSION

    def get_schema(self) -> FeatureSchema:
        """Get the feature schema."""
        return self.schema


# Singleton-like access for convenience (initialized on first use)
_default_builder: FeatureBuilder | None = None


def get_feature_builder(settings: Settings) -> FeatureBuilder:
    """
    Get the default FeatureBuilder instance.

    This ensures all code uses the same builder instance within a process.

    Args:
        settings: Configuration settings

    Returns:
        FeatureBuilder instance
    """
    global _default_builder
    if _default_builder is None:
        _default_builder = FeatureBuilder(settings)
    return _default_builder


def reset_feature_builder() -> None:
    """
    Reset the default builder (for testing).

    This should NOT be called in production code.
    """
    global _default_builder
    _default_builder = None
    logger.warning("FeatureBuilder reset - this should only happen in tests")
