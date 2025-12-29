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

Configuration:
- Circuit breaker auto-reset timeout configurable via settings.execution_safety.circuit_breaker_reset_minutes
- All vetoes support optional details_fn for structured observability logging

Usage:
    >>> from execution.policy import ExecutionPolicy, VetoPrecedence
    >>> policy = ExecutionPolicy(circuit_breaker_reset_minutes=15)
    >>> policy.register_veto(VetoPrecedence.KILL_SWITCH, lambda: kill_switch_active, "Manual halt")
    >>> veto = policy.check_vetoes()
    >>> if veto:
    ...     print(f"Trade blocked by {veto.reason}, details={veto.details}")
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
    - Risk limits (P&L caps, rate limits) come next
    - Model quality controls (calibration, regime) follow
    - Signal quality (confidence) has lowest precedence
    
    RISK-ARCH-REVIEW: Rate limits are now registered here for unified veto logging.
    SafeTradeExecutor registers its rate-limit checks as RATE_LIMIT vetoes.
    
    This ensures that critical safety mechanisms cannot be bypassed by
    any other conditions.
    """
    
    KILL_SWITCH = 0  # Emergency manual halt (highest precedence)
    CIRCUIT_BREAKER = 1  # Consecutive failures threshold
    DAILY_LOSS = 2  # Daily P&L limit exceeded
    RATE_LIMIT = 3  # Per-minute/symbol rate limits (RISK-ARCH-REVIEW)
    CALIBRATION = 4  # Reconstruction error threshold
    REGIME = 5  # Market anomaly detection
    CONFIDENCE = 6  # Minimum probability threshold (lowest precedence)


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
    - All vetoes are logged for observability (with optional details_fn)
    - Statistics track veto frequency by type
    
    Configuration:
    - circuit_breaker_reset_minutes: Auto-reset timeout (from settings or default 15)
    """
    
    def __init__(self, circuit_breaker_reset_minutes: int = 15):
        """
        Initialize execution policy.
        
        Args:
            circuit_breaker_reset_minutes: Minutes before circuit breaker auto-resets.
                                           Use settings.execution_safety.circuit_breaker_reset_minutes.
        """
        # Veto registry: precedence level -> list of (check_fn, reason_provider, details_fn, use_context)
        # reason_provider can be a str or a Callable[[], str]
        self._vetoes: dict[VetoPrecedence, list[tuple[Callable[..., bool], str | Callable[[], str], Optional[Callable[[], dict]], bool]]] = {
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
        self._circuit_breaker_triggered_at: Optional[float] = None
        
        # AUDIT-FIX: Externalized circuit breaker timeout from settings
        self._circuit_breaker_reset_minutes = circuit_breaker_reset_minutes
        
        # Register standard circuit breaker check with details_fn for observability
        self.register_veto(
            VetoPrecedence.CIRCUIT_BREAKER,
            lambda: self._circuit_breaker_active,
            lambda: f"Circuit breaker active: {self._circuit_breaker_reason}" if self._circuit_breaker_reason else "Circuit breaker active",
            details_fn=lambda: {
                "triggered_at": self._circuit_breaker_triggered_at,
                "reason": self._circuit_breaker_reason,
                "reset_minutes": self._circuit_breaker_reset_minutes,
            }
        )
    
    def trigger_circuit_breaker(self, reason: str) -> None:
        """
        Manually trigger the circuit breaker veto (L1).
        
        Pauses all trading decisions until explicitly reset.
        """
        import time
        self._circuit_breaker_active = True
        self._circuit_breaker_reason = reason
        self._circuit_breaker_triggered_at = time.time()
        logger.warning(f"CIRCUIT BREAKER TRIGGERED: {reason}")

    def reset_circuit_breaker(self) -> None:
        """Reset the circuit breaker veto."""
        if self._circuit_breaker_active:
            logger.info("CIRCUIT BREAKER RESET")
        self._circuit_breaker_active = False
        self._circuit_breaker_reason = ""
        self._circuit_breaker_triggered_at = None

    def register_veto(
        self,
        level: VetoPrecedence,
        check_fn: Callable[..., bool],
        reason: str | Callable[[], str],
        details_fn: Optional[Callable[[], dict]] = None,
        use_context: bool = False,
    ) -> None:
        """
        Register a veto condition.
        
        Args:
            level: Precedence level for this veto
            check_fn: Callable that returns True if veto should trigger.
                      Can accept kwargs if use_context=True.
            reason: Human-readable explanation or callable returning one
            details_fn: Optional callable that returns context dict for logging
            use_context: If True, check_fn will be called with kwargs passed to check_vetoes
        """
        self._vetoes[level].append((check_fn, reason, details_fn, use_context))
        display_reason = reason() if callable(reason) else reason
        logger.debug(f"Registered veto: L{level} - {display_reason}")
    
    def check_vetoes(self, **kwargs) -> Optional[VetoDecision]:
        """
        Check all vetoes in precedence order.
        
        Returns first blocking veto encountered, or None if all clear.
        This ensures highest-precedence vetoes always take priority.
        
        Args:
            **kwargs: Context arguments passed to vetoes registered with use_context=True.

        Returns:
            VetoDecision if any veto triggers, None otherwise
        """
        self._total_checks += 1
        
        # Check vetoes in precedence order (L0 -> L5)
        for level in sorted(VetoPrecedence):
            # REC-003: Check for Circuit Breaker auto-reset
            if level == VetoPrecedence.CIRCUIT_BREAKER and self._circuit_breaker_active:
                self._maybe_auto_reset_circuit_breaker()

            for check_fn, reason_provider, details_fn, use_context in self._vetoes[level]:
                try:
                    is_vetoed = False
                    if use_context:
                        is_vetoed = check_fn(**kwargs)
                    else:
                        is_vetoed = check_fn()

                    if is_vetoed:
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
                "reason": self._circuit_breaker_reason,
                "triggered_at": self._circuit_breaker_triggered_at
            }
        }

    def _maybe_auto_reset_circuit_breaker(self) -> None:
        """
        Check if circuit breaker should be automatically reset based on configured timeout.
        
        Uses _circuit_breaker_reset_minutes which is configured at initialization
        from settings.execution_safety.circuit_breaker_reset_minutes.
        """
        import time
        if not self._circuit_breaker_active or self._circuit_breaker_triggered_at is None:
            return
        
        # AUDIT-FIX: Use configurable timeout instead of hardcoded 15 minutes
        timeout_seconds = self._circuit_breaker_reset_minutes * 60
        
        if time.time() - self._circuit_breaker_triggered_at >= timeout_seconds:
            logger.info(f"Auto-resetting circuit breaker after {self._circuit_breaker_reset_minutes} minutes")
            self.reset_circuit_breaker()
    
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
    
    All vetoes registered here include details_fn for structured observability logging.
    """
    
    @staticmethod
    def apply(
        policy: ExecutionPolicy, 
        settings: Settings,
        pnl_provider: Optional[Callable[[], float]] = None,
        calibration_provider: Optional[Callable[[], float]] = None
    ) -> None:
        """
        Apply safety settings to an execution policy.
        
        Registers standard vetoes for the system-wide protection hierarchy.
        All vetoes include details_fn to capture raw metrics for post-mortem analysis.
        
        Args:
            policy: The ExecutionPolicy to configure.
            settings: Central settings containing thresholds.
            pnl_provider: Callable returning current daily P&L.
            calibration_provider: Callable returning current calibration error (reconstruction error).
        """
        config = settings.execution_safety
        
        # L0: Kill Switch (with details for observability)
        policy.register_veto(
            level=VetoPrecedence.KILL_SWITCH,
            check_fn=lambda: config.kill_switch_enabled,
            reason="Kill switch enabled (manual halt)",
            details_fn=lambda: {"kill_switch_enabled": config.kill_switch_enabled}
        )
        
        # L2: Daily Loss Limit (with raw P&L in details)
        if pnl_provider:
            limit = config.max_daily_loss
            policy.register_veto(
                level=VetoPrecedence.DAILY_LOSS,
                check_fn=lambda: pnl_provider() <= -limit if limit > 0 else False,
                reason=lambda: f"Daily loss limit hit: {pnl_provider():.2f} <= -{limit:.2f}",
                details_fn=lambda: {
                    "current_pnl": pnl_provider(),
                    "limit": limit,
                    "threshold": -limit,
                }
            )
            
        # L3: Calibration / Stability (with raw metrics in details)
        # High-order stability check (e.g. model completely failing to reconstruct normal data)
        if calibration_provider:
            # We use a very high threshold for absolute calibration failure
            # compared to normal regime caution.
            cal_threshold = getattr(settings.hyperparams, "calibration_failure_threshold", 5.0)
            policy.register_veto(
                level=VetoPrecedence.CALIBRATION,
                check_fn=lambda: calibration_provider() >= cal_threshold,
                reason=lambda: f"Model calibration failure: {calibration_provider():.3f} >= {cal_threshold:.3f}",
                details_fn=lambda: {
                    "reconstruction_error": calibration_provider(),
                    "threshold": cal_threshold,
                }
            )
