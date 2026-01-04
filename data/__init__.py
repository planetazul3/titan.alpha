"""
Data module for the DerivOmniModel trading system.

CANONICAL FEATURE PIPELINE:
All feature engineering MUST go through FeatureBuilder. This ensures
consistent preprocessing across training, validation, shadow trades,
and live inference. Direct use of preprocessors is deprecated.

>>> from data import FeatureBuilder
>>> builder = FeatureBuilder(settings)
>>> features = builder.build(ticks=raw_ticks, candles=raw_candles)

This module provides:
- FeatureBuilder: CANONICAL entry point for feature engineering
- DerivDataset: PyTorch dataset for training (uses FeatureBuilder internally)
- ShadowTradeDataset: Shadow trade dataset (uses FeatureBuilder internally)
- create_dataloaders: Factory for train/val data loaders

Low-level utilities (use FeatureBuilder instead):
- TickPreprocessor, CandlePreprocessor, VolatilityMetricsExtractor
- Normalization functions, technical indicators
"""

# DEPRECATED: Low-level preprocessors - use FeatureBuilder instead
import warnings as _warnings

from data.common.schema import FEATURE_SCHEMA_VERSION
from data.indicators import (
    adx,
    atr,
    bollinger_bands,
    bollinger_bandwidth,
    bollinger_percent_b,
    ema,
    macd,
    rsi,
    sma,
    stochastic,
)
from data.ingestion.client import DerivClient
from data.ingestion.historical import download_months
from data.normalizers import (
    log_returns,
    min_max_normalize,
    robust_scale,
    z_score_normalize,
)

# High-level modules (dependent on above)
from data.features import FeatureBuilder, get_feature_builder
from data.dataset import DerivDataset
from data.loader import create_dataloaders
from data.shadow_dataset import ShadowTradeDataset


def _get_deprecated_preprocessor(name: str):
    """Lazy import with deprecation warning for legacy preprocessors."""
    _warnings.warn(
        f"{name} is deprecated. Use FeatureBuilder for consistent feature engineering.",
        DeprecationWarning,
        stacklevel=3,
    )
    from data import processor

    return getattr(processor, name)


class _DeprecatedImport:
    """Wrapper that emits deprecation warning on access."""

    def __init__(self, name: str):
        self._name = name
        self._obj = None

    def __call__(self, *args, **kwargs):
        if self._obj is None:
            self._obj = _get_deprecated_preprocessor(self._name)
        return self._obj(*args, **kwargs)


# Deprecated exports (emit warning on use)
TickPreprocessor = _DeprecatedImport("TickPreprocessor")
CandlePreprocessor = _DeprecatedImport("CandlePreprocessor")
VolatilityMetricsExtractor = _DeprecatedImport("VolatilityMetricsExtractor")

__all__ = [
    # CANONICAL ENTRY POINT
    "FeatureBuilder",
    "get_feature_builder",
    "FEATURE_SCHEMA_VERSION",
    # Datasets (use FeatureBuilder internally)
    "DerivDataset",
    "ShadowTradeDataset",
    "create_dataloaders",
    # Data ingestion
    "DerivClient",
    "download_months",
    # Low-level preprocessors (deprecated, use FeatureBuilder)
    "TickPreprocessor",
    "CandlePreprocessor",
    "VolatilityMetricsExtractor",
    # Normalization utilities
    "log_returns",
    "z_score_normalize",
    "min_max_normalize",
    "robust_scale",
    # Technical indicators
    "rsi",
    "bollinger_bands",
    "atr",
    "bollinger_bandwidth",
    "bollinger_percent_b",
    "ema",
    "sma",
    "macd",
    "stochastic",
    "adx",
]
