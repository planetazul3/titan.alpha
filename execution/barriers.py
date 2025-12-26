"""
Barrier Calculation Utility.

Centralizes all barrier calculation logic for trading contracts.
Used by executor, shadow resolution, and decision engine.

R05: Consolidate Barrier Calculation Logic
"""

import logging
from dataclasses import dataclass
from typing import Literal

from config.constants import CONTRACT_TYPES

logger = logging.getLogger(__name__)


@dataclass
class BarrierLevels:
    """Calculated barrier levels for a trade."""
    upper: float
    lower: float | None = None  # Only used for STAYS_BETWEEN


class BarrierCalculator:
    """
    Centralized barrier calculation for all contract types.
    
    Takes configuration as input and provides consistent barrier prices
    given entry price and contract type.
    """
    
    def __init__(
        self,
        default_touch_barrier_offset: float = 0.50,
        default_range_high_offset: float = 0.50,
        default_range_low_offset: float = -0.50,
    ):
        """
        Initialize barrier calculator.
        
        Args:
            default_touch_barrier_offset: Default offset for TOUCH/NO_TOUCH contracts
            default_range_high_offset: Default high barrier offset for STAYS_BETWEEN
            default_range_low_offset: Default low barrier offset for STAYS_BETWEEN
        """
        self.default_touch_barrier_offset = default_touch_barrier_offset
        self.default_range_high_offset = default_range_high_offset
        self.default_range_low_offset = default_range_low_offset
    
    def calculate(
        self,
        entry_price: float,
        contract_type: str,
        barrier_offset: float | None = None,
        barrier2_offset: float | None = None,
    ) -> BarrierLevels:
        """
        Calculate barrier levels for a given contract.
        
        Args:
            entry_price: Entry price of the trade
            contract_type: Type of contract (TOUCH_NO_TOUCH, STAYS_BETWEEN, etc.)
            barrier_offset: Custom barrier offset (overrides default)
            barrier2_offset: Custom second barrier offset for STAYS_BETWEEN
            
        Returns:
            BarrierLevels with calculated upper and lower barriers
        """
        if contract_type == CONTRACT_TYPES.TOUCH_NO_TOUCH or contract_type == "TOUCH_NO_TOUCH":
            offset = barrier_offset if barrier_offset is not None else self.default_touch_barrier_offset
            return BarrierLevels(
                upper=entry_price + offset,
                lower=entry_price - abs(offset),
            )
        
        elif contract_type == CONTRACT_TYPES.STAYS_BETWEEN or contract_type == "STAYS_BETWEEN":
            high_offset = barrier_offset if barrier_offset is not None else self.default_range_high_offset
            low_offset = barrier2_offset if barrier2_offset is not None else self.default_range_low_offset
            return BarrierLevels(
                upper=entry_price + high_offset,
                lower=entry_price + low_offset,  # low_offset is typically negative
            )
        
        else:
            # RISE_FALL and other types don't use barriers
            return BarrierLevels(upper=entry_price, lower=None)
    
    def calculate_from_percentage(
        self,
        entry_price: float,
        contract_type: str,
        barrier_pct: float = 0.005,
        barrier2_pct: float = 0.003,
    ) -> BarrierLevels:
        """
        Calculate barrier levels using percentage offsets.
        
        Args:
            entry_price: Entry price of the trade
            contract_type: Type of contract
            barrier_pct: Barrier percentage (0.005 = 0.5%)
            barrier2_pct: Second barrier percentage for ranges
            
        Returns:
            BarrierLevels with calculated barriers
        """
        if contract_type == CONTRACT_TYPES.TOUCH_NO_TOUCH or contract_type == "TOUCH_NO_TOUCH":
            offset = entry_price * barrier_pct
            return BarrierLevels(
                upper=entry_price + offset,
                lower=entry_price - offset,
            )
        
        elif contract_type == CONTRACT_TYPES.STAYS_BETWEEN or contract_type == "STAYS_BETWEEN":
            offset = entry_price * barrier2_pct
            return BarrierLevels(
                upper=entry_price + offset,
                lower=entry_price - offset,
            )
        
        else:
            return BarrierLevels(upper=entry_price, lower=None)
