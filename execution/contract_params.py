"""
Contract Parameters - Centralized logic for contract configuration.

IMPORTANT-001: Ensures consistent durations between real and shadow trades.
"""

from typing import Tuple
from config.constants import CONTRACT_TYPES
from config.settings import Settings

class ContractParameterService:
    """
    Centralized service for resolving all contract parameters (duration, barriers, stake).
    
    CRITICAL-003: Unifies scattered logic from executor and adapters.
    """
    
    def __init__(self, settings: Settings):
        self.settings = settings
        
    def resolve_duration(self, contract_type: str) -> Tuple[int, str]:
        """
        Get duration and duration unit for a contract type.
        
        Returns:
            Tuple of (duration_value, duration_unit)
        """
        config = self.settings.contracts
        
        if contract_type == CONTRACT_TYPES.RISE_FALL:
            return config.duration_rise_fall, "m"
        elif contract_type == CONTRACT_TYPES.TOUCH_NO_TOUCH:
            return config.duration_touch, "m"
        elif contract_type == CONTRACT_TYPES.STAYS_BETWEEN:
            return config.duration_range, "m"
            
        return getattr(config, "duration_minutes", 1), "m"

    def resolve_barriers(self, contract_type: str, current_price: float = 0.0) -> Tuple[str | None, str | None]:
        """
        Resolve barrier levels for the contract.
        
        Args:
            contract_type: Type of contract
            current_price: Current spot price (optional, for absolute barriers)
            
        Returns:
            Tuple of (barrier, barrier2) as strings or None
        """
        # CRITICAL-003: Centralized barrier logic
        if contract_type == CONTRACT_TYPES.TOUCH_NO_TOUCH:
            # For Touch/No Touch, we typically use a relative barrier offset
            offset = self.settings.trading.barrier_offset
            # If using relative barriers (e.g. "+0.5"), return directly.
            # If using absolute, we'd need current_price.
            # Assuming relative for now as per Deriv API standard for relative.
            return f"+{offset}", None
            
        elif contract_type == CONTRACT_TYPES.STAYS_BETWEEN:
            # Range contracts usually need two barriers
            offset = self.settings.trading.barrier_offset
            # Example: +offset and -offset
            return f"+{offset}", f"-{offset}"
            
        return None, None



# Backward compatibility alias
ContractDurationResolver = ContractParameterService
