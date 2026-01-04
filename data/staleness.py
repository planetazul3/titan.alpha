import logging
from typing import Optional
import numpy as np
from config.settings import Settings

logger = logging.getLogger(__name__)

class StaleDataError(Exception):
    """Raised when market data is too old for safe inference."""
    pass

def check_data_staleness(
    timestamp: Optional[float], 
    last_candle_ts: float, 
    threshold: float
) -> None:
    """
    Validate that data is fresh enough for trading.
    
    Args:
        timestamp: Current system timestamp (or None to skip).
        last_candle_ts: Timestamp of the most recent candle.
        threshold: Max allowed latency in seconds.
        
    Raises:
        StaleDataError: If latency exceeds threshold.
    """
    if timestamp is None:
        return
        
    latency = timestamp - last_candle_ts
    
    if latency > threshold:
        msg = f"Data is STALE! Latency: {latency:.2f}s (Threshold: {threshold}s). Last candle: {last_candle_ts}"
        logger.error(msg)
        raise StaleDataError(msg)
    elif latency < -1.0: # Clock shift tolerance
         logger.warning(f"Future data detected? Latency: {latency:.2f}s")
