"""
Configuration constants for the DerivOmniModel trading system.

This module defines core constants used throughout the application including:
- Contract types for binary options trading
- Signal classification types for trade decisions
- Sequence length constraints for model inputs
- Default configuration values

Type aliases are provided for improved type safety in function signatures.

Example:
    >>> from config.constants import CONTRACT_TYPES, SIGNAL_TYPES
    >>> contract = CONTRACT_TYPES.RISE_FALL
    >>> signal = SIGNAL_TYPES.REAL_TRADE
"""

from enum import Enum
from typing import Final

# Type aliases for better type safety
ContractType = str
SignalType = str


class CONTRACT_TYPES(str, Enum):
    """
    Supported contract types for Deriv binary options trading.

    Attributes:
        RISE_FALL: Predict if price will rise or fall
        TOUCH_NO_TOUCH: Predict if price will touch a barrier
        STAYS_BETWEEN: Predict if price stays within a range

    Example:
        >>> contract = CONTRACT_TYPES.RISE_FALL
        >>> assert contract.value == "RISE_FALL"
    """

    RISE_FALL = "RISE_FALL"
    TOUCH_NO_TOUCH = "TOUCH_NO_TOUCH"
    STAYS_BETWEEN = "STAYS_BETWEEN"


class SIGNAL_TYPES(str, Enum):
    """
    Classification of trade signals based on model confidence and filters.

    Signals are classified into three categories based on probability
    thresholds and risk management filters.

    Attributes:
        REAL_TRADE: High confidence signal, approved for real execution
        SHADOW_TRADE: Medium confidence signal, tracked but not executed
        IGNORE: Low confidence signal, discarded

    Example:
        >>> signal = SIGNAL_TYPES.REAL_TRADE
        >>> if signal == SIGNAL_TYPES.REAL_TRADE:
        ...     # Execute trade
        ...     pass
    """

    REAL_TRADE = "REAL_TRADE"
    SHADOW_TRADE = "SHADOW_TRADE"
    IGNORE = "IGNORE"


# Sequence Length Constraints
MIN_SEQUENCE_LENGTH: Final[int] = 16
"""
Minimum sequence length for model input.

Sequences shorter than this will be rejected during validation.
This constraint ensures the model has sufficient context for predictions.
"""

MAX_SEQUENCE_LENGTH: Final[int] = 2000
"""
Maximum sequence length for model input.

Sequences longer than this will be truncated to prevent memory issues
and maintain reasonable inference times.
"""

# Default Configuration Values
DEFAULT_SEED: Final[int] = 42
"""
Default random seed for reproducibility.

Used for initializing random number generators in Python, NumPy, and PyTorch
to ensure deterministic behavior across runs.
"""

# Network Configuration
DEFAULT_MAX_RETRIES: Final[int] = 3
"""Default retry attempts for API operations."""

DEFAULT_INGESTION_RETRIES: Final[int] = 5
"""Retry attempts for data ingestion operations (more resilient)."""

# Timeout Configuration
DEFAULT_DB_TIMEOUT: Final[float] = 30.0
"""SQLite connection timeout in seconds."""

DEFAULT_API_TIMEOUT: Final[float] = 10.0
"""API request timeout in seconds."""

CONTRACT_SETTLE_TIMEOUT: Final[float] = 180.0
"""Maximum wait time for contract settlement in seconds."""

# Feature Engineering
CANDLE_FEATURE_COUNT: Final[int] = 10
"""Number of features per candle: O, H, L, C, V, T, RSI, BB_width, BB_%b, ATR."""

# Training
MAX_CONSECUTIVE_NANS: Final[int] = 10
"""Maximum consecutive NaN batches before halting training."""

# Data Streaming
DEFAULT_CANDLE_INTERVAL: Final[int] = 60
"""Default candle interval in seconds (1 minute)."""

