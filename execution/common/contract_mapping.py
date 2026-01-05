"""
Contract Type Mapping Utility.

Centralizes the logic for mapping generic internal signal types (e.g., RISE_FALL)
to specific execution contract types (e.g., CALL/PUT).
"""

from config.constants import CONTRACT_TYPES
from execution.signals import TradeSignal


def map_signal_to_contract_type(signal: TradeSignal) -> str:
    """
    Map generic signal types to concrete execution contract types.
    
    Args:
        signal: The trade signal containing contract type and direction.
        
    Returns:
        str: The API-specific contract type string (e.g., "CALL", "ONETOUCH").
    """
    if signal.contract_type == CONTRACT_TYPES.RISE_FALL:
        return "CALL" if signal.direction == "CALL" else "PUT"
    elif signal.contract_type == CONTRACT_TYPES.TOUCH_NO_TOUCH:
        return "ONETOUCH" if signal.direction == "TOUCH" else "NOTOUCH"
    elif signal.contract_type == CONTRACT_TYPES.STAYS_BETWEEN:
        return "RANGE" if signal.direction == "IN" else "UPORDOWN"
    else:
        # Fallback: Assuming direction is the contract type if unknown generic
        return signal.direction or signal.contract_type
