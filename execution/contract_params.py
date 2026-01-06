"""
Contract Parameters - Centralized logic for contract configuration.

IMPORTANT-001: Ensures consistent durations between real and shadow trades.
"""

import logging
from typing import Tuple
from config.constants import CONTRACT_TYPES
from config.settings import Settings

logger = logging.getLogger(__name__)

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
            duration = config.duration_rise_fall
            unit = "m"
        elif contract_type == CONTRACT_TYPES.TOUCH_NO_TOUCH:
            duration = config.duration_touch
            unit = "m"
        elif contract_type == CONTRACT_TYPES.STAYS_BETWEEN:
            duration = config.duration_range
            unit = "m"
        else:
            duration = getattr(config, "duration_minutes", 1)
            unit = "m"

        # C-002: Check for mismatch with global timeframe
        self._check_timeframe_consistency(duration, unit, contract_type)
        
        return duration, unit

    def _check_timeframe_consistency(self, duration: int, unit: str, contract_type: str):
        """Log warning if duration deviates significantly from trading timeframe."""
        try:
            tf_duration, tf_unit = self._parse_timeframe(self.settings.trading.timeframe)
            if unit == tf_unit and duration != tf_duration:
                 # Only warn for simple mismatches (e.g. 1m vs 5m)
                 # Some strategies INTENTIONALLY differ (e.g. 5m candles, 15m expiry)
                 # So we log INFO/DEBUG, or WARNING if it looks like default-accident.
                 logger.debug(
                     f"Duration mismatch for {contract_type}: Config={duration}{unit}, "
                     f"Timeframe={tf_duration}{tf_unit}. Ensure this is intentional."
                 )
        except Exception:
            pass # Ignore parsing errors for custom timeframes

    def _parse_timeframe(self, timeframe: str) -> Tuple[int, str]:
        """Parse '1m', '5m', '1h' into (value, unit)."""
        if timeframe.endswith("m"):
            return int(timeframe[:-1]), "m"
        if timeframe.endswith("h"):
            return int(timeframe[:-1]), "h"
        if timeframe.endswith("d"):
            return int(timeframe[:-1]), "d"
        return 1, "m" # Fallback

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
