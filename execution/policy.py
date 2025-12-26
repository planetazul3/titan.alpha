"""
Execution Policy with Veto Precedence Hierarchy.

This module implements a centralized execution policy framework that enforces
a strict veto hierarchy for trade execution decisions. Based on algorithmic
trading best practices research, vetoes are ordered by criticality:

1. Kill Switch (L0): Emergency manual halt - highest precedence
2. Circuit Breaker (L1): Consecutive failures threshold
3. Daily Loss Limit (L2): P&L protection cap
4. Calibration Issues (L3): Model reconstruction error threshold
5. Regime Veto (L4): Market anomaly detection
6. Confidence Threshold (L5): Minimum probability requirement

Usage:
    >>> from execution.policy import ExecutionPolicy, VetoPrecedence
    >>> policy = ExecutionPolicy()
    >>> policy.register_veto(VetoPrecedence.KILL_SWITCH, lambda: kill_switch_active, "Manual halt")
    >>> veto = policy.check_vetoes()
    >>> if veto:
    ...     print(f"Trade blocked by {veto.reason}")
"""

import logging
from dataclasses import dataclass
from enum import IntEnum
from typing import Callable, Optional

logger = logging.getLogger(__name__)


class VetoPrecedence(IntEnum):
    """
    Veto precedence levels (lower number = higher precedence).
    
    Based on algorithmic trading risk management hierarchy:
    - Emergency controls (kill switch, circuit breakers) have highest precedence
    - Risk limits (P&L caps) come next
    - Model quality controls (calibration, regime) follow
    - Signal quality (confidence) has lowest precedence
    
    This ensures that critical safety mechanisms cannot be bypassed by
    any other conditions.
    """
    
    KILL_SWITCH = 0  # Emergency manual halt (highest precedence)
    CIRCUIT_BREAKER = 1  # Consecutive failures threshold
    DAILY_LOSS = 2  # Daily P&L limit exceeded
    CALIBRATION = 3  # Reconstruction error threshold
    REGIME = 4  # Market anomaly detection
    CONFIDENCE = 5  # Minimum probability threshold (lowest precedence)


@dataclass
class VetoDecision:
    """
    Result of a veto check.
    
    Attributes:
        level: Precedence level that triggered veto
        reason: Human-readable explanation
        details: Optional additional context for logging
    """
    
    level: VetoPrecedence
    reason: str
    details: Optional[dict] = None
    
    def __str__(self) -> str:
        """Format for logging."""
        return f"[VETO L{self.level}] {self.reason}"


class ExecutionPolicy:
    """
    Centralized execution policy with explicit veto hierarchy.
    
    This class provides a single authority for trade execution decisions,
    enforcing veto precedence to ensure critical safety controls cannot
    be bypassed.
    
    Design principles:
    - Vetoes are checked in precedence order (highest to lowest)
    - First blocking veto terminates the check
    - All vetoes are logged for observability
    - Statistics track veto frequency by type
    
    Example:
        >>> policy = ExecutionPolicy()
        >>> policy.register_veto(
        ...     VetoPrecedence.KILL_SWITCH,
        ...     lambda: settings.execution_safety.kill_switch_enabled,
        ...     "Kill switch active"
        ... )
        >>> veto = policy.check_vetoes()
        >>> if veto:
        ...     logger.warning(f"Trade blocked: {veto}")
    """
    
    def __init__(self):
        """Initialize empty execution policy."""
        # Veto registry: precedence level -> list of (check_fn, reason, details_fn)
        self._vetoes: dict[VetoPrecedence, list[tuple[Callable[[], bool], str, Optional[Callable[[], dict]]]]] = {
            level: [] for level in VetoPrecedence
        }
        
        # Statistics tracking
        self._veto_counts: dict[VetoPrecedence, int] = {
            level: 0 for level in VetoPrecedence
        }
        self._total_checks = 0
    
    def register_veto(
        self,
        level: VetoPrecedence,
        check_fn: Callable[[], bool],
        reason: str,
        details_fn: Optional[Callable[[], dict]] = None,
    ) -> None:
        """
        Register a veto condition.
        
        Args:
            level: Precedence level for this veto
            check_fn: Callable that returns True if veto should trigger
            reason: Human-readable explanation (e.g., "Daily loss limit exceeded")
            details_fn: Optional callable that returns context dict for logging
        """
        self._vetoes[level].append((check_fn, reason, details_fn))
        logger.debug(f"Registered veto: L{level} - {reason}")
    
    def check_vetoes(self) -> Optional[VetoDecision]:
        """
        Check all vetoes in precedence order.
        
        Returns first blocking veto encountered, or None if all clear.
        This ensures highest-precedence vetoes always take priority.
        
        Returns:
            VetoDecision if any veto triggers, None otherwise
        """
        self._total_checks += 1
        
        # Check vetoes in precedence order (L0 -> L5)
        for level in sorted(VetoPrecedence):
            for check_fn, reason, details_fn in self._vetoes[level]:
                try:
                    if check_fn():
                        # Veto triggered
                        self._veto_counts[level] += 1
                        
                        # Get optional details
                        details = details_fn() if details_fn else None
                        
                        veto = VetoDecision(level=level, reason=reason, details=details)
                        logger.info(f"Trade vetoed: {veto}")
                        
                        return veto
                except Exception as e:
                    # Don't let broken veto checks crash the system
                    logger.error(f"Veto check failed (L{level} - {reason}): {e}", exc_info=True)
                    continue
        
        # No vetoes triggered
        return None
    
    def get_veto_statistics(self) -> dict:
        """
        Get veto statistics for observability.
        
        Returns:
            Dict with veto counts by level and total checks
        """
        return {
            "total_checks": self._total_checks,
            "veto_counts": {
                f"L{level}_{ level.name}": count
                for level, count in self._veto_counts.items()
            },
            "veto_rate": (
                sum(self._veto_counts.values()) / self._total_checks
                if self._total_checks > 0
                else 0.0
            ),
        }
    
    def clear_statistics(self) -> None:
        """Reset veto statistics (useful for testing)."""
        self._veto_counts = {level: 0 for level in VetoPrecedence}
        self._total_checks = 0
        logger.debug("Veto statistics cleared")
    
    def unregister_all_vetoes(self, level: Optional[VetoPrecedence] = None) -> None:
        """
        Unregister vetoes for testing/debugging.
        
        Args:
            level: If specified, only clear this level. Otherwise clear all.
        """
        if level is not None:
            self._vetoes[level] = []
            logger.debug(f"Cleared vetoes for L{level}")
        else:
            for lvl in VetoPrecedence:
                self._vetoes[lvl] = []
            logger.debug("Cleared all vetoes")
