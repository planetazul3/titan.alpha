"""
Contract Parameters - Centralized logic for contract configuration.

IMPORTANT-001: Ensures consistent durations between real and shadow trades.
"""

from typing import Tuple
from config.constants import CONTRACT_TYPES
from config.settings import Settings

class ContractDurationResolver:
    """
    Resolves contract duration and unit based on type and settings.
    """
    
    def __init__(self, settings: Settings):
        self.settings = settings
        
    def resolve_duration(self, contract_type: str) -> Tuple[int, str]:
        """
        Get duration and duration unit for a contract type.
        
        Returns:
            Tuple of (duration_value, duration_unit)
        """
        # We use the shadow_trade configuration as the source of truth
        # for both real and shadow trades to ensure perfect alignment.
        config = self.settings.shadow_trade
        
        if contract_type == CONTRACT_TYPES.RISE_FALL:
            return config.duration_rise_fall, "m"
        elif contract_type == CONTRACT_TYPES.TOUCH_NO_TOUCH:
            return config.duration_touch, "m"
        elif contract_type == CONTRACT_TYPES.STAYS_BETWEEN:
            return config.duration_range, "m"
            
        # Fallback
        return getattr(config, "duration_minutes", 1), "m"
