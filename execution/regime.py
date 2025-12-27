"""
Regime detection and veto authority.

This module provides first-class authority for the volatility/anomaly detection
system to unconditionally veto trading decisions during unstable regimes.

The regime veto operates independently of confidence scores and can override
any trading signal to prevent loss clustering during anomalous market conditions.
"""

import logging
from dataclasses import dataclass
from enum import Enum
from typing import Protocol, runtime_checkable

import torch

logger = logging.getLogger(__name__)


@runtime_checkable
class RegimeAssessmentProtocol(Protocol):
    """
    Protocol defining the common interface for regime assessments.
    
    Both RegimeAssessment and HierarchicalRegimeAssessment implement this
    protocol, enabling proper type compatibility in subclasses like EnhancedRegimeVeto.
    """
    
    reconstruction_error: float
    
    def is_vetoed(self) -> bool:
        """Check if regime has vetoed trading."""
        ...
    
    def requires_caution(self) -> bool:
        """Check if regime requires cautious trading."""
        ...





class TrustState(Enum):
    """Market regime trust states for trading decisions."""

    TRUSTED = "trusted"  # Normal regime, trust predictions
    CAUTION = "caution"  # Elevated uncertainty, reduce stakes
    VETO = "veto"  # Anomalous regime, no trades allowed


@dataclass
class RegimeAssessment:
    """
    Assessment of current market regime trustworthiness.

    Attributes:
        trust_state: Current trust level (TRUSTED/CAUTION/VETO)
        reconstruction_error: Raw reconstruction error from volatility expert
        threshold_low: Threshold for CAUTION state
        threshold_high: Threshold for VETO state
    """

    trust_state: TrustState
    reconstruction_error: float
    threshold_low: float
    threshold_high: float

    def is_vetoed(self) -> bool:
        """Check if regime has vetoed trading."""
        return self.trust_state == TrustState.VETO

    def requires_caution(self) -> bool:
        """Check if regime requires cautious trading."""
        return self.trust_state == TrustState.CAUTION


class RegimeVeto:
    """
    First-class authority that can unconditionally block trades.

    This class analyzes the reconstruction error from the volatility autoencoder
    expert and determines whether the current market regime is trustworthy enough
    for trading.

    Unlike confidence thresholds that can be overridden by strong signals, the
    regime veto is absolute: when the regime is deemed anomalous, NO trades are
    permitted regardless of model confidence.

    This prevents the common failure mode where models hallucinate confidence
    during regime shifts or unprecedented market conditions.

    Attributes:
        threshold_caution: Reconstruction error threshold for CAUTION state
        threshold_veto: Reconstruction error threshold for VETO state

    Example:
        >>> regime_veto = RegimeVeto(threshold_caution=0.1, threshold_veto=0.3)
        >>> assessment = regime_veto.assess(reconstruction_error)
        >>> if assessment.is_vetoed():
        ...     logger.warning("REGIME VETO: No trades allowed")
        ...     return []
    """

    def __init__(self, threshold_caution: float = 0.1, threshold_veto: float = 0.3):
        """
        Initialize regime veto authority.

        Args:
            threshold_caution: Reconstruction error above which CAUTION is triggered
            threshold_veto: Reconstruction error above which trading is VETOED

        Raises:
            ValueError: If thresholds are invalid or misordered
        """
        if threshold_caution <= 0 or threshold_veto <= 0:
            raise ValueError("Thresholds must be positive")

        if threshold_caution >= threshold_veto:
            raise ValueError(
                f"threshold_caution ({threshold_caution}) must be < "
                f"threshold_veto ({threshold_veto})"
            )

        self.threshold_caution = threshold_caution
        self.threshold_veto = threshold_veto

        logger.info(
            f"RegimeVeto initialized: CAUTION={threshold_caution:.3f}, VETO={threshold_veto:.3f}"
        )

    def assess(self, reconstruction_error: torch.Tensor | float) -> RegimeAssessmentProtocol:
        """
        Assess regime trust based on reconstruction error.

        The reconstruction error from the volatility autoencoder indicates
        how well the model understands the current market behavior. High
        reconstruction error suggests the market is behaving anomalously
        relative to historical patterns.

        Args:
            reconstruction_error: Tensor containing reconstruction error
                from volatility expert (typically MSE loss)

        Returns:
            RegimeAssessment with trust_state and threshold information


        Example:
            >>> error_tensor = torch.tensor(0.15)
            >>> assessment = regime_veto.assess(error_tensor)
            >>> print(assessment.trust_state)  # TrustState.CAUTION
        """
        # Extract scalar value
        if isinstance(reconstruction_error, torch.Tensor):
            error = reconstruction_error.item()
        else:
            error = float(reconstruction_error)

        # Determine trust state
        if error >= self.threshold_veto:
            state = TrustState.VETO
            # logger.warning(f"REGIME VETO: {error} >= {self.threshold_veto}")
        elif error >= self.threshold_caution:
            state = TrustState.CAUTION
        else:
            state = TrustState.TRUSTED

        return RegimeAssessment(
            trust_state=state,
            reconstruction_error=error,
            threshold_low=self.threshold_caution,
            threshold_high=self.threshold_veto,
        )

    def update_thresholds(self, threshold_caution: float, threshold_veto: float):
        """
        Update veto thresholds dynamically.

        This allows for adaptive regime detection based on observed
        market conditions or learned parameters.

        Args:
            threshold_caution: New CAUTION threshold
            threshold_veto: New VETO threshold

        Raises:
            ValueError: If new thresholds are invalid
        """
        if threshold_caution >= threshold_veto:
            raise ValueError(
                f"threshold_caution ({threshold_caution}) must be < "
                f"threshold_veto ({threshold_veto})"
            )

        logger.info(
            f"Updating RegimeVeto thresholds: CAUTION {self.threshold_caution:.3f} → "
            f"{threshold_caution:.3f}, VETO {self.threshold_veto:.3f} → "
            f"{threshold_veto:.3f}"
        )

        self.threshold_caution = threshold_caution
        self.threshold_veto = threshold_veto
