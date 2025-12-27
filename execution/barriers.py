"""
Barrier Resolver - Centralized logic for contract barrier offsets.

REC-004: Standardizes how barriers are calculated across different components.
"""

from typing import Optional, Tuple
from config.settings import Settings

class BarrierResolver:
    """
    Resolves trade barriers based on contract type, price, and settings.
    """
    
    def __init__(self, settings: Settings):
        self.settings = settings
        
    def resolve_barriers(
        self, 
        contract_type: str, 
        entry_price: float,
        manual_barrier: Optional[str] = None,
        manual_barrier2: Optional[str] = None
    ) -> Tuple[Optional[str], Optional[str]]:
        """
        Calculate absolute or offset barriers for a trade.
        
        Args:
            contract_type: RISE_FALL, ONETOUCH, etc.
            entry_price: Current market price.
            manual_barrier: Optional manual override.
            manual_barrier2: Optional second manual override.
            
        Returns:
            Tuple of (barrier, barrier2) strings.
        """
        # If manual barriers provided, they take priority
        if manual_barrier is not None:
            return manual_barrier, manual_barrier2
            
        # Default offsets from settings
        offset1 = self.settings.trading.barrier_offset
        offset2 = self.settings.trading.barrier2_offset
        
        # RISE_FALL doesn't use barriers (it's implicit from entry price)
        from config.constants import CONTRACT_TYPES
        if contract_type == CONTRACT_TYPES.RISE_FALL:
            return None, None
            
        # TOUCH/NO_TOUCH use a single barrier
        if contract_type == CONTRACT_TYPES.TOUCH_NO_TOUCH:
            return offset1, None
            
        # RANGE use two barriers
        if contract_type == CONTRACT_TYPES.STAYS_BETWEEN:
            return offset1, offset2
            
        return None, None

    def calculate_absolute_barrier(self, entry_price: float, offset_str: str) -> float:
        """Helper to convert "+0.50" style offset to absolute price."""
        try:
            offset = float(offset_str)
            return entry_price + offset
        except (ValueError, TypeError):
            return entry_price
