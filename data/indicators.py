"""
Technical indicators using TA-Lib for robust, production-ready calculations.
"""

import numpy as np
import talib
from typing import cast


def rsi(prices: np.ndarray, period: int = 14) -> np.ndarray:
    """
    Relative Strength Index using TA-Lib.

    Args:
        prices: Close prices array
        period: RSI period (default 14)

    Returns:
        RSI values (0-100), NaN filled with 50
    """
    if len(prices) < period + 1:
        return np.full_like(prices, 50.0)

    result = talib.RSI(prices.astype(np.float64), timeperiod=period)
    return np.nan_to_num(result, nan=50.0)


def bollinger_bands(
    prices: np.ndarray, period: int = 20, std_dev: float = 2.0
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Bollinger Bands using TA-Lib.

    Returns:
        (upper, middle, lower) bands
    """
    if len(prices) < period:
        return np.zeros_like(prices), np.zeros_like(prices), np.zeros_like(prices)

    upper, middle, lower = talib.BBANDS(
        prices.astype(np.float64),
        timeperiod=period,
        nbdevup=std_dev,
        nbdevdn=std_dev,
        matype=talib.MA_Type.SMA,  # Simple moving average
    )

    # Fill NaN with first price
    fill_val = prices[0]
    upper = np.nan_to_num(upper, nan=fill_val)
    middle = np.nan_to_num(middle, nan=fill_val)
    lower = np.nan_to_num(lower, nan=fill_val)

    return upper, middle, lower


def atr(high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int = 14) -> np.ndarray:
    """
    Average True Range using TA-Lib.
    """
    if len(close) < 2:
        return np.zeros_like(close)

    result = talib.ATR(
        high.astype(np.float64), low.astype(np.float64), close.astype(np.float64), timeperiod=period
    )
    return np.nan_to_num(result, nan=0.0)


def bollinger_bandwidth(upper: np.ndarray, lower: np.ndarray, middle: np.ndarray) -> np.ndarray:
    """
    Bollinger Band Width: (Upper - Lower) / Middle
    """
    with np.errstate(divide="ignore", invalid="ignore"):
        bw = (upper - lower) / middle
    return cast(np.ndarray, np.nan_to_num(bw, nan=0.0, posinf=0.0, neginf=0.0))


def bollinger_percent_b(prices: np.ndarray, lower: np.ndarray, upper: np.ndarray) -> np.ndarray:
    """
    Bollinger %B: (Price - Lower) / (Upper - Lower)
    """
    with np.errstate(divide="ignore", invalid="ignore"):
        pct_b = (prices - lower) / (upper - lower)
    return cast(np.ndarray, np.nan_to_num(pct_b, nan=0.5, posinf=1.0, neginf=0.0))


# Additional TA-Lib indicators that could be useful


def ema(prices: np.ndarray, period: int = 20) -> np.ndarray:
    """Exponential Moving Average."""
    result = talib.EMA(prices.astype(np.float64), timeperiod=period)
    return np.nan_to_num(result, nan=prices[0] if len(prices) > 0 else 0.0)


def sma(prices: np.ndarray, period: int = 20) -> np.ndarray:
    """Simple Moving Average."""
    result = talib.SMA(prices.astype(np.float64), timeperiod=period)
    return np.nan_to_num(result, nan=prices[0] if len(prices) > 0 else 0.0)


def macd(
    prices: np.ndarray, fast_period: int = 12, slow_period: int = 26, signal_period: int = 9
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    MACD indicator.

    Returns:
        (macd_line, signal_line, histogram)
    """
    macd_line, signal, hist = talib.MACD(
        prices.astype(np.float64),
        fastperiod=fast_period,
        slowperiod=slow_period,
        signalperiod=signal_period,
    )
    return (
        np.nan_to_num(macd_line, nan=0.0),
        np.nan_to_num(signal, nan=0.0),
        np.nan_to_num(hist, nan=0.0),
    )


def stochastic(
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    fastk_period: int = 14,
    slowk_period: int = 3,
    slowd_period: int = 3,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Stochastic Oscillator.

    Returns:
        (slowk, slowd)
    """
    slowk, slowd = talib.STOCH(
        high.astype(np.float64),
        low.astype(np.float64),
        close.astype(np.float64),
        fastk_period=fastk_period,
        slowk_period=slowk_period,
        slowd_period=slowd_period,
    )
    return (np.nan_to_num(slowk, nan=50.0), np.nan_to_num(slowd, nan=50.0))


def adx(high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int = 14) -> np.ndarray:
    """Average Directional Index - measures trend strength."""
    result = talib.ADX(
        high.astype(np.float64), low.astype(np.float64), close.astype(np.float64), timeperiod=period
    )
    return np.nan_to_num(result, nan=0.0)
