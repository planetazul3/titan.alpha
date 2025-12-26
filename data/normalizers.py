"""
Data normalization utilities for financial time series.

This module provides various normalization techniques optimized for
financial data including prices, returns, and technical indicators.

Functions:
    - log_returns: Calculate logarithmic returns
    - z_score_normalize: Z-score (standard) normalization
    - min_max_normalize: Min-max scaling to specific range
    - robust_scale: Scaling robust to outliers using IQR

All functions use vectorized NumPy operations for performance and
handle edge cases (NaN, inf, division by zero) gracefully.

Example:
    >>> import numpy as np
    >>> from data.normalizers import log_returns, z_score_normalize
    >>> prices = np.array([100, 102, 101, 103, 105])
    >>> returns = log_returns(prices)
    >>> normalized = z_score_normalize(returns)
"""

import logging

from typing import cast, Any

import numpy as np

logger = logging.getLogger(__name__)


def log_returns(prices: np.ndarray, fill_first: bool = True) -> np.ndarray:
    """
    Compute log returns: ln(p_t / p_{t-1}).

    Log returns are preferred over simple returns for financial data because:
    - They are time-additive
    - More suitable for statistical analysis
    - Better handling of compounding

    Args:
        prices: 1D array of prices (must be positive)
        fill_first: If True, replace initial NaN with 0

    Returns:
        Log returns array of same shape as input

    Raises:
        ValueError: If prices contain non-positive values or NaN/inf

    Example:
        >>> prices = np.array([100, 102, 101])
        >>> returns = log_returns(prices)
        >>> # returns[0] = 0, returns[1] = ln(102/100), returns[2] = ln(101/102)
    """
    if not isinstance(prices, np.ndarray):
        raise TypeError(f"prices must be np.ndarray, got {type(prices)}")

    if prices.ndim != 1:
        raise ValueError(f"prices must be 1D array, got shape {prices.shape}")

    if len(prices) < 2:
        logger.warning("Insufficient data for log returns (need >= 2 points)")
        return np.zeros_like(prices, dtype=np.float32)

    # Check for invalid values
    if np.any(prices <= 0):
        raise ValueError("Prices must be positive for log returns")

    if np.any(~np.isfinite(prices)):
        raise ValueError("Prices contain NaN or inf values")

    # Calculate log returns
    returns = np.diff(np.log(prices, dtype=np.float64), prepend=prices[0])

    if fill_first:
        returns[0] = 0.0

    return cast(np.ndarray, returns.astype(np.float32))


def z_score_normalize(
    values: np.ndarray, window: int | None = None, epsilon: float = 1e-6
) -> np.ndarray:
    """
    Apply Z-score normalization: (x - mean) / std.

    Z-score normalization centers data around zero with unit variance,
    making it suitable for neural network inputs.

    Args:
        values: Input array to normalize
        window: Rolling window size for local normalization.
                If None, uses global statistics.
        epsilon: Small value to avoid division by zero (default 1e-6)

    Returns:
        Normalized array of same shape, with NaN filled as 0

    Raises:
        ValueError: If window is larger than array length
        TypeError: If values is not a numpy array

    Example:
        >>> values = np.array([1, 2, 3, 4, 5])
        >>> normalized = z_score_normalize(values)
        >>> # Mean ≈ 0, std ≈ 1
    """
    if not isinstance(values, np.ndarray):
        raise TypeError(f"values must be np.ndarray, got {type(values)}")

    if len(values) == 0:
        logger.warning("Empty array passed to z_score_normalize")
        return np.array([], dtype=np.float32)

    if values.ndim != 1:
        raise ValueError(f"values must be 1D array, got shape {values.shape}")

    # Global normalization
    if window is None:
        mean = np.mean(values)
        std = np.std(values)
        normalized = (values - mean) / (std + epsilon)
        return cast(np.ndarray, np.nan_to_num(normalized, nan=0.0).astype(np.float32))

    # Rolling window normalization
    if window > len(values):
        raise ValueError(f"Window size ({window}) cannot exceed array length ({len(values)})")

    if window < 2:
        raise ValueError(f"Window size must be >= 2, got {window}")

    # Use stride tricks for efficient rolling window
    pad_width = window - 1
    shape = (len(values) - window + 1, window)
    strides = (values.strides[0], values.strides[0])

    try:
        strided = np.lib.stride_tricks.as_strided(
            values, shape=shape, strides=strides, writeable=False
        )
    except ValueError as e:
        logger.error(f"Failed to create strided array: {e}")
        # Fallback to global normalization
        return z_score_normalize(values, window=None, epsilon=epsilon)

    rolling_mean = np.mean(strided, axis=1)
    rolling_std = np.std(strided, axis=1)

    # Pad beginning with NaN
    pad = np.full(pad_width, np.nan)
    rolling_mean = np.concatenate((pad, rolling_mean))
    rolling_std = np.concatenate((pad, rolling_std))

    # Normalize
    result = (values - rolling_mean) / (rolling_std + epsilon)

    # Fill NaN (from padding) with 0
    return cast(np.ndarray, np.nan_to_num(result, nan=0.0).astype(np.float32))


def min_max_normalize(
    values: np.ndarray, feature_range: tuple[float, float] = (0, 1)
) -> np.ndarray:
    """
    Apply Min-Max scaling to specific range.

    Scales features to a given range, typically [0, 1] or [-1, 1].

    Formula: (x - min) / (max - min) * (max_range - min_range) + min_range

    Args:
        values: Input array to scale
        feature_range: Target (min, max) range for scaling

    Returns:
        Scaled array. If max == min, returns zeros.

    Raises:
        TypeError: If values is not numpy array
        ValueError: If feature_range is invalid

    Example:
        >>> values = np.array([1, 2, 3, 4, 5])
        >>> scaled = min_max_normalize(values, feature_range=(-1, 1))
        >>> # Result in [-1, 1] range
    """
    if not isinstance(values, np.ndarray):
        raise TypeError(f"values must be np.ndarray, got {type(values)}")

    if len(values) == 0:
        return np.array([], dtype=np.float32)

    min_range, max_range = feature_range
    if min_range >= max_range:
        raise ValueError(f"feature_range must be (min, max) with min < max, got {feature_range}")

    min_val = np.min(values)
    max_val = np.max(values)

    # Handle constant array
    if np.isclose(max_val - min_val, 0):
        logger.warning("Constant array in min_max_normalize, returning zeros")
        return np.zeros_like(values, dtype=np.float32)

    scaled = (values - min_val) / (max_val - min_val) * (max_range - min_range) + min_range
    return cast(np.ndarray, scaled.astype(np.float32))


def robust_scale(
    values: np.ndarray, quantile_range: tuple[float, float] = (25.0, 75.0)
) -> np.ndarray:
    """
    Scale features using statistics that are robust to outliers.

    Uses median and inter-quartile range (IQR) instead of mean and std,
    making it more robust to outliers in financial data.

    Formula: (x - median) / IQR

    Args:
        values: Input array to scale
        quantile_range: (lower, upper) percentiles for IQR calculation
                       Default (25, 75) uses standard IQR

    Returns:
        Robustly scaled array

    Raises:
        TypeError: If values is not numpy array
        ValueError: If quantile_range is invalid

    Example:
        >>> values = np.array([1, 2, 3, 4, 100])  # Contains outlier
        >>> scaled = robust_scale(values)
        >>> # Outlier has less impact than with z-score
    """
    if not isinstance(values, np.ndarray):
        raise TypeError(f"values must be np.ndarray, got {type(values)}")

    if len(values) == 0:
        return np.array([], dtype=np.float32)

    q_min, q_max = quantile_range
    if not (0 <= q_min < q_max <= 100):
        raise ValueError(
            f"quantile_range must be (min, max) with 0 <= min < max <= 100, got {quantile_range}"
        )

    q25 = np.percentile(values, q_min)
    q75 = np.percentile(values, q_max)
    median = np.median(values)

    iqr = q75 - q25

    # Handle zero IQR (constant or near-constant data)
    if np.isclose(iqr, 0):
        logger.warning("Zero IQR in robust_scale, using zero-centering only")
        return cast(np.ndarray, (values - median).astype(np.float32))

    scaled = (values - median) / iqr
    return cast(np.ndarray, scaled.astype(np.float32))
