"""
Data preprocessing utilities for trading features.

This module provides preprocessors that transform raw market data into
normalized feature vectors suitable for neural network input.

Classes:
    - TickPreprocessor: Process raw tick prices into normalized sequences
    - CandlePreprocessor: Process OHLCV candles with technical indicators
    - VolatilityMetricsExtractor: Extract aggregated volatility metrics

All preprocessors handle edge cases (insufficient data, NaN values) and
ensure consistent output shapes for model consumption.

Example:
    >>> from data.preprocessor import TickPreprocessor, CandlePreprocessor
    >>> from config.settings import load_settings
    >>> settings = load_settings()
    >>> tick_pp = TickPreprocessor(settings)
    >>> processed_ticks = tick_pp.process(raw_ticks)
"""

import logging

import numpy as np
from typing import cast

from config.settings import Settings
from data import indicators, normalizers

logger = logging.getLogger(__name__)


class TickPreprocessor:
    """
    Preprocessor for raw tick price data.

    Transforms raw tick prices through:
    1. Log returns calculation
    2. Z-score normalization
    3. Padding/truncation to target length

    Attributes:
        target_length: Desired sequence length from settings
    """

    def __init__(self, settings: Settings):
        """
        Initialize tick preprocessor.

        Args:
            settings: Configuration settings containing sequence length
        """
        self.target_length = settings.data_shapes.sequence_length_ticks
        logger.debug(f"TickPreprocessor initialized with target_length={self.target_length}")

    def process(self, ticks: np.ndarray) -> np.ndarray:
        """
        Transform raw tick prices to normalized sequence.

        Pipeline:
        1. Calculate log returns
        2. Apply z-score normalization
        3. Pad/truncate to target length

        Args:
            ticks: 1D array of tick prices, shape (N,) where N >= 2

        Returns:
            Normalized tick sequence, shape (target_length,), dtype float32

        Raises:
            TypeError: If ticks is not a numpy array
            ValueError: If ticks has incorrect shape or insufficient data

        Example:
            >>> processor = TickPreprocessor(settings)
            >>> raw_ticks = np.array([100.1, 100.2, 100.15, ...])
            >>> normalized = processor.process(raw_ticks)
            >>> normalized.shape  # (sequence_length_ticks,)
        """
        if not isinstance(ticks, np.ndarray):
            raise TypeError(f"ticks must be np.ndarray, got {type(ticks)}")

        if ticks.ndim != 1:
            raise ValueError(f"ticks must be 1D array, got shape {ticks.shape}")

        if len(ticks) < 2:
            logger.warning(f"Insufficient ticks ({len(ticks)}), returning zero-padded array")
            return np.zeros(self.target_length, dtype=np.float32)

        try:
            # 1. Log returns
            returns = normalizers.log_returns(ticks)

            # 2. Z-score normalize
            normalized = normalizers.z_score_normalize(returns)

            # 3. Pad/Truncate
            curr_len = len(normalized)
            if curr_len >= self.target_length:
                result = normalized[-self.target_length :]
            else:
                pad_width = self.target_length - curr_len
                result = np.pad(normalized, (pad_width, 0), mode="constant", constant_values=0)

            return result.astype(np.float32)

        except Exception as e:
            logger.error(f"Error processing ticks: {e}")
            raise


class CandlePreprocessor:
    """
    Preprocessor for OHLCV candle data with technical indicators.

    Transforms OHLCV candles into a 10-feature representation:
    [O_norm, H_norm, L_norm, C_norm, V_norm, Time_norm, RSI, BB_width, BB_%b, ATR]

    Attributes:
        target_length: Desired sequence length from settings
    """

    def __init__(self, settings: Settings):
        """
        Initialize candle preprocessor.

        Args:
            settings: Configuration settings containing sequence length
        """
        self.target_length = settings.data_shapes.sequence_length_candles
        logger.debug(f"CandlePreprocessor initialized with target_length={self.target_length}")

    def process(self, ohlcv: np.ndarray) -> np.ndarray:
        """
        Transform OHLCV data with technical indicators.

        Input columns: [Open, High, Low, Close, Volume, Time]
        Output features: [O_norm, H_norm, L_norm, C_norm, V_norm, T_norm,
                         RSI_norm, BB_width_norm, BB_%b, ATR_norm]

        Args:
            ohlcv: OHLCV data, shape (N, 6) where N >= 20 (for indicators)

        Returns:
            Normalized feature array, shape (target_length, 10), dtype float32

        Raises:
            TypeError: If ohlcv is not a numpy array
            ValueError: If ohlcv has incorrect shape

        Example:
            >>> processor = CandlePreprocessor(settings)
            >>> candles = np.array([[100, 101, 99, 100.5, 1000, 1234567], ...])
            >>> features = processor.process(candles)
            >>> features.shape  # (sequence_length_candles, 10)
        """
        if not isinstance(ohlcv, np.ndarray):
            raise TypeError(f"ohlcv must be np.ndarray, got {type(ohlcv)}")

        if ohlcv.ndim != 2:
            raise ValueError(f"ohlcv must be 2D array, got shape {ohlcv.shape}")

        if ohlcv.shape[1] != 6:
            raise ValueError(f"ohlcv must have 6 columns [O,H,L,C,V,T], got {ohlcv.shape[1]}")

        if len(ohlcv) < 20:
            logger.warning(f"Insufficient candles ({len(ohlcv)}), returning zero-padded array")
            return np.zeros((self.target_length, 10), dtype=np.float32)

        try:
            # Extract columns
            opens = ohlcv[:, 0]
            highs = ohlcv[:, 1]
            lows = ohlcv[:, 2]
            closes = ohlcv[:, 3]
            volumes = ohlcv[:, 4]
            times = ohlcv[:, 5]

            # 1. Normalize OHLC with log returns
            o_norm = normalizers.log_returns(opens)
            h_norm = normalizers.log_returns(highs)
            l_norm = normalizers.log_returns(lows)
            c_norm = normalizers.log_returns(closes)

            # 2. Normalize Volume with z_score
            v_norm = normalizers.z_score_normalize(volumes)

            # Normalize Time - use time deltas
            time_diffs = np.diff(times)
            if len(time_diffs) > 0:
                median_diff = np.median(time_diffs)
                time_diffs = np.concatenate([[median_diff], time_diffs])
            else:
                time_diffs = np.zeros_like(times)
            t_norm = normalizers.z_score_normalize(time_diffs)

            # 3. Technical indicators
            rsi_val = indicators.rsi(closes)

            bb_upper, bb_mid, bb_lower = indicators.bollinger_bands(closes)
            bb_width = indicators.bollinger_bandwidth(bb_upper, bb_lower, bb_mid)
            bb_pct = indicators.bollinger_percent_b(closes, bb_lower, bb_upper)

            atr_val = indicators.atr(highs, lows, closes)

            # Normalize indicators
            rsi_norm = rsi_val / 100.0  # RSI 0-100 -> 0-1
            bb_w_norm = normalizers.z_score_normalize(bb_width)
            atr_norm = normalizers.z_score_normalize(atr_val)

            # 4. Stack features
            features = np.stack(
                [
                    o_norm,
                    h_norm,
                    l_norm,
                    c_norm,
                    v_norm,
                    t_norm,
                    rsi_norm,
                    bb_w_norm,
                    bb_pct,
                    atr_norm,
                ],
                axis=1,
            )

            # 5. Pad/Truncate
            curr_len = len(features)
            if curr_len >= self.target_length:
                result = features[-self.target_length :]
            else:
                pad_len = self.target_length - curr_len
                pad = np.zeros((pad_len, 10))
                result = np.vstack([pad, features])

            return result.astype(np.float32)

        except Exception as e:
            logger.error(f"Error processing candles: {e}")
            raise


class VolatilityMetricsExtractor:
    """
    Extractor for aggregated volatility metrics.

    Computes summary statistics from candle data:
    - Realized volatility (std of log returns)
    - Mean ATR (Average True Range)
    - RSI standard deviation
    - Mean Bollinger Band width

    These metrics feed into the volatility autoencoder expert.
    """

    def __init__(self, settings: Settings):
        """
        Initialize volatility extractor.

        Args:
            settings: Configuration settings containing normalization factors
        """
        self.settings = settings
        logger.debug("VolatilityMetricsExtractor initialized with config settings")

    def extract(self, candles: np.ndarray) -> np.ndarray:
        """
        Extract aggregated volatility metrics for autoencoder input.

        All metrics are normalized to approximately [0, 1] range for stable
        autoencoder reconstruction. This ensures reconstruction error is
        comparable to regime veto thresholds (typically 0.1-0.3).

        Args:
            candles: OHLCV data, shape (N, 6) where N >= 20

        Returns:
            Normalized volatility metrics, shape (4,) -> [realized_vol, atr_mean, rsi_std, bb_width_mean]
            All values clipped to [0, 1] range.

        Raises:
            TypeError: If candles is not numpy array
            ValueError: If candles has incorrect shape or insufficient data

        Example:
            >>> extractor = VolatilityMetricsExtractor(settings)
            >>> metrics = extractor.extract(candles)
            >>> metrics.shape  # (4,)
            >>> assert 0 <= metrics.min() and metrics.max() <= 1
        """
        if not isinstance(candles, np.ndarray):
            raise TypeError(f"candles must be np.ndarray, got {type(candles)}")

        if candles.ndim != 2:
            raise ValueError(f"candles must be 2D array, got shape {candles.shape}")

        if candles.shape[1] < 4:
            raise ValueError(
                f"candles must have at least 4 columns [O,H,L,C,...], got {candles.shape[1]}"
            )

        if len(candles) < 20:
            logger.warning(f"Insufficient candles ({len(candles)}) for volatility metrics")
            return np.zeros(4, dtype=np.float32)

        try:
            closes = candles[:, 3]
            highs = candles[:, 1]
            lows = candles[:, 2]

            # Realized volatility (std of log returns)
            # Typical range: 0.001 - 0.05 for normal markets
            returns = normalizers.log_returns(closes)
            realized_vol = np.std(returns)

            # ATR mean (in price units)
            # For synthetic indices like R_100 (~10000), ATR might be 10-100
            atr_val = indicators.atr(highs, lows, closes)
            atr_mean = np.mean(atr_val)

            # Normalize ATR relative to price level
            price_level = np.mean(closes)
            if price_level > 0:
                atr_normalized = atr_mean / price_level
            else:
                atr_normalized = 0.0

            # RSI std (RSI is 0-100, std typically 5-20)
            rsi_val = indicators.rsi(closes)
            rsi_std = np.std(rsi_val)

            # BB Width mean (already a ratio, typically 0.01-0.1)
            bb_u, bb_m, bb_l = indicators.bollinger_bands(closes)
            bb_w = indicators.bollinger_bandwidth(bb_u, bb_l, bb_m)
            bb_w_mean = np.mean(bb_w)

            # ═══════════════════════════════════════════════════════════════
            # NORMALIZATION: Scale all metrics to approximately [0, 1] range
            # This ensures reconstruction error is in expected range for
            # regime veto thresholds (typically 0.1-0.3)
            # ═══════════════════════════════════════════════════════════════
            
            norm = self.settings.normalization

            # Normalization factors loaded from settings
            # - realized_vol: typically 0.001-0.05 -> * 20 -> 0.02-1.0
            # - atr_normalized: typically 0.001-0.01 -> * 100 -> 0.1-1.0
            # - rsi_std: typically 5-25 -> * 0.02 (1/50) -> 0.1-0.5
            # - bb_w_mean: typically 0.01-0.1 -> * 10 -> 0.1-1.0

            metrics = np.array(
                [
                    realized_vol * norm.norm_factor_volatility,
                    atr_normalized * norm.norm_factor_atr,
                    rsi_std * norm.norm_factor_rsi_std,
                    bb_w_mean * norm.norm_factor_bb_width,
                ]
            )

            # NaN handling - MUST be done before clip, otherwise clip(NaN) -> NaN persists
            metrics = np.nan_to_num(metrics, nan=0.0, posinf=1.0, neginf=0.0)

            # Clip to [0, 1] to ensure stable autoencoder behavior
            metrics = np.clip(metrics, 0.0, 1.0)
            
            return cast(np.ndarray, metrics.astype(np.float32))

        except Exception as e:
            logger.error(f"Error extracting volatility metrics: {e}")
            raise
