from dataclasses import dataclass
from typing import Any, Optional, Tuple
from config.settings import Settings

@dataclass
class BarrierLevels:
    """Container for resolved upper and lower barriers."""
    upper: float
    lower: float

class BarrierCalculator:
    """
    Standardized utility for trade barrier calculations.
    
    R05: Centralized logic for all contract types.
    """
    
    def calculate(
        self, 
        entry_price: float, 
        contract_type: str, 
        barrier_offset: float | str | None = None,
        barrier2_offset: float | str | None = None
    ) -> BarrierLevels:
        """
        Calculate absolute upper/lower barriers from numeric or string offsets.
        
        Args:
            entry_price: Market price at entry
            contract_type: RISE_FALL, TOUCH_NO_TOUCH, etc.
            barrier_offset: Primary offset (numeric or "+0.50")
            barrier2_offset: Secondary offset (numeric or "-0.50")
            
        Returns:
            BarrierLevels with absolute price targets
        """
        # Parse offsets to floats
        off1 = self._to_float(barrier_offset, 0.0)
        off2 = self._to_float(barrier2_offset, 0.0)
        
        # Default symmetric bands if one is missing but expected
        from config.constants import CONTRACT_TYPES
        
        if contract_type == CONTRACT_TYPES.RISE_FALL:
            return BarrierLevels(upper=entry_price, lower=entry_price)
            
        if contract_type == CONTRACT_TYPES.TOUCH_NO_TOUCH:
            # Usually symmetric around entry if only one offset
            return BarrierLevels(
                upper=entry_price + abs(off1),
                lower=entry_price - abs(off1)
            )
            
        if contract_type == CONTRACT_TYPES.STAYS_BETWEEN:
            # Usually off1 is upper (+), off2 is lower (-)
            u = entry_price + max(off1, off2)
            l = entry_price + min(off1, off2)
            return BarrierLevels(upper=u, lower=l)
            
        return BarrierLevels(upper=entry_price + off1, lower=entry_price + off2)

    def calculate_from_percentage(
        self, 
        entry_price: float, 
        contract_type: str, 
        barrier_pct: float
    ) -> BarrierLevels:
        """Calculate barriers using a percentage of entry price."""
        offset = entry_price * barrier_pct
        return self.calculate(entry_price, contract_type, offset, -offset)

    def _to_float(self, val: Any, default: float) -> float:
        if val is None:
            return default
        try:
            if isinstance(val, str):
                return float(val.replace('+', ''))
            return float(val)
        except (ValueError, TypeError):
            return default

class BarrierResolver:
    """
    Legacy-compatible wrapper for BarrierCalculator (REC-004).
    """
    def __init__(self, settings: Settings):
        self.settings = settings
        self.calc = BarrierCalculator()
        
    def resolve_barriers(
        self, 
        contract_type: str, 
        entry_price: float,
        manual_barrier: Optional[str | float] = None,
        manual_barrier2: Optional[str | float] = None
    ) -> Tuple[Optional[float], Optional[float]]:
        # Use settings if no manual provided
        m1 = manual_barrier if manual_barrier is not None else self.settings.trading.barrier_offset
        m2 = manual_barrier2 if manual_barrier2 is not None else self.settings.trading.barrier2_offset
        
        levels = self.calc.calculate(entry_price, contract_type, m1, m2)
        return levels.upper, levels.lower
