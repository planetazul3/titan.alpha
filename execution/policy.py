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

from __future__ import annotations
import logging
from dataclasses import dataclass
from enum import IntEnum
from typing import Callable, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from config.settings import Settings

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
    """
    
    def __init__(self):
        """Initialize empty execution policy."""
        # Veto registry: precedence level -> list of (check_fn, reason_provider, details_fn)
        # reason_provider can be a str or a Callable[[], str]
        self._vetoes: dict[VetoPrecedence, list[tuple[Callable[[], bool], str | Callable[[], str], Optional[Callable[[], dict]]]]] = {
            level: [] for level in VetoPrecedence
        }
        
        # Statistics tracking
        self._veto_counts: dict[VetoPrecedence, int] = {
            level: 0 for level in VetoPrecedence
        }
        self._total_checks = 0

        # Circuit Breaker state (L1)
        self._circuit_breaker_active = False
        self._circuit_breaker_reason = ""
        
        # Register standard circuit breaker check
        self.register_veto(
            VetoPrecedence.CIRCUIT_BREAKER,
            lambda: self._circuit_breaker_active,
            lambda: f"Circuit breaker active: {self._circuit_breaker_reason}" if self._circuit_breaker_reason else "Circuit breaker active"
        )
    
    def trigger_circuit_breaker(self, reason: str) -> None:
        """
        Manually trigger the circuit breaker veto (L1).
        
        Pauses all trading decisions until explicitly reset.
        """
        self._circuit_breaker_active = True
        self._circuit_breaker_reason = reason
        logger.warning(f"CIRCUIT BREAKER TRIGGERED: {reason}")

    def reset_circuit_breaker(self) -> None:
        """Reset the circuit breaker veto."""
        if self._circuit_breaker_active:
            logger.info("CIRCUIT BREAKER RESET")
        self._circuit_breaker_active = False
        self._circuit_breaker_reason = ""

    def register_veto(
        self,
        level: VetoPrecedence,
        check_fn: Callable[[], bool],
        reason: str | Callable[[], str],
        details_fn: Optional[Callable[[], dict]] = None,
    ) -> None:
        """
        Register a veto condition.
        
        Args:
            level: Precedence level for this veto
            check_fn: Callable that returns True if veto should trigger
            reason: Human-readable explanation or callable returning one
            details_fn: Optional callable that returns context dict for logging
        """
        self._vetoes[level].append((check_fn, reason, details_fn))
        display_reason = reason() if callable(reason) else reason
        logger.debug(f"Registered veto: L{level} - {display_reason}")
    
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
            for check_fn, reason_provider, details_fn in self._vetoes[level]:
                try:
                    if check_fn():
                        # Veto triggered
                        self._veto_counts[level] += 1
                        
                        # Get optional details
                        details = details_fn() if details_fn else None
                        
                        # Resolve reason
                        reason = reason_provider() if callable(reason_provider) else reason_provider
                        
                        veto = VetoDecision(level=level, reason=reason, details=details)
                        logger.info(f"Trade vetoed: {veto}")
                        
                        return veto
                except Exception as e:
                    # Don't let broken veto checks crash the system
                    logger.error(f"Veto check failed (L{level}): {e}", exc_info=True)
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
            "circuit_breaker": {
                "active": self._circuit_breaker_active,
                "reason": self._circuit_breaker_reason
            }
        }
    
    def clear_statistics(self) -> None:
        """Reset veto statistics (useful for testing)."""
        self._veto_counts = {level: 0 for level in VetoPrecedence}
        self._total_checks = 0
        logger.debug("Veto statistics cleared")
    
    def unregister_all_vetoes(self, level: Optional[VetoPrecedence] = None, include_system: bool = False) -> None:
        """
        Unregister vetoes for testing/debugging.
        
        Args:
            level: If specified, only clear this level. Otherwise clear all.
            include_system: If True, also clear system-registered vetoes (like circuit breaker).
        """
        if level is not None:
            self._vetoes[level] = []
            logger.debug(f"Cleared vetoes for L{level}")
        else:
            for lvl in VetoPrecedence:
                self._vetoes[lvl] = []
            logger.debug("Cleared all vetoes")
            
        # Re-register system vetoes if we cleared all
        if not include_system and (level is None or level == VetoPrecedence.CIRCUIT_BREAKER):
            self.register_veto(
                VetoPrecedence.CIRCUIT_BREAKER,
                lambda: self._circuit_breaker_active,
                lambda: f"Circuit breaker active: {self._circuit_breaker_reason}" if self._circuit_breaker_reason else "Circuit breaker active"
            )


class SafetyProfile:
    """
    Centralized safety configuration for the trading system.
    
    Acts as a bridge between Settings.execution_safety and the 
    ExecutionPolicy/SafeTradeExecutor components.
    """
    
    @staticmethod
    def apply(policy: ExecutionPolicy, settings: Settings) -> None:
        """
        Apply safety settings to an execution policy.
        
        Registers standard vetoes for the Decision layer.
        """
        config = settings.execution_safety
        
        # L0: Kill Switch
        policy.register_veto(
            level=VetoPrecedence.KILL_SWITCH,
            check_fn=lambda: config.kill_switch_enabled,
            reason="Kill switch enabled (manual halt)"
        )
        
        # Note: Other vetoes like DAILY_LOSS (L2) require live P&L data
        # which is usually managed by the Execution layer's SafeTradeExecutor.
        # DecisionEngine focus is on Model Safety and Regime.
