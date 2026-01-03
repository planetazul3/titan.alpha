"""
Centralized numerical validation module for the x.titan trading system.
Provides utilities to ensure that critical numerical values (P&L, risk metrics)
are finite before they are propagated or persisted.
"""

import math
import logging
from typing import Union, Dict, Any, Optional

logger = logging.getLogger(__name__)

Numeric = Union[float, int]

def ensure_finite(
    value: Numeric, 
    name: str, 
    default: Optional[Numeric] = 0.0,
    log_level: int = logging.WARNING
) -> Numeric:
    """
    Validates that a numeric value is finite (not NaN or Inf).
    
    Args:
        value: The numeric value to check.
        name: Name of the variable for logging context.
        default: fallback value to return if check fails. Defaults to 0.0.
        log_level: Logging level to use if check fails.
        
    Returns:
        The original value if finite, otherwise the default value.
    """
    # Handle non-numeric types gracefully if they slip through
    if not isinstance(value, (int, float)):
        logger.log(
            log_level, 
            f"Non-numeric value detected for {name}: {type(value)}. Using default: {default}"
        )
        return default

    if not math.isfinite(value):
        logger.log(
            log_level, 
            f"Non-finite value detected for {name}: {value}. Using default: {default}"
        )
        return default
        
    return value

def validate_numeric_dict(
    metrics: Dict[str, Any], 
    defaults: Optional[Dict[str, Numeric]] = None
) -> Dict[str, Any]:
    """
    Validates a dictionary of metrics, ensuring all numeric values are finite.
    
    Args:
        metrics: Dictionary containing metrics to validate.
        defaults: Optional dict mapping keys to specific default values.
                  If a key isn't in defaults, 0.0 is used.
                  
    Returns:
        A new dictionary with validated values.
    """
    validated = {}
    defaults = defaults or {}
    
    for k, v in metrics.items():
        if isinstance(v, (int, float)):
            default_val = defaults.get(k, 0.0)
            validated[k] = ensure_finite(v, k, default=default_val)
        else:
            validated[k] = v
            
    return validated
