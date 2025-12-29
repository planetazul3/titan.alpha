"""
Regime detection and veto authority.

This module provides first-class authority for the volatility/anomaly detection
system to unconditionally veto trading decisions during unstable regimes.

The regime veto operates independently of confidence scores and can override
any trading signal to prevent loss clustering during anomalous market conditions.

Key Features:
- **Threshold Calibration**: Thresholds can be loaded from model checkpoints via `from_checkpoint`
  to ensure alignment with training normalization (AUDIT-FIX: normalization/threshold coupling)
- **Regime Confidence**: `RegimeAssessment.regime_confidence` provides 0-1 score indicating
  confidence in the current regime assessment (AUDIT-FIX: confidence of regime metric)
- **Adaptive Thresholding**: `set_volatility_scaler` enables dynamic threshold adjustment
  based on trailing market variance (AUDIT-FIX: non-stationary volatility handling)

Calibration Sources:
- CHECKPOINT: Thresholds loaded from model checkpoint (preferred)
- SETTINGS: Thresholds from settings.hyperparams (fallback)
- MANUAL: Thresholds set manually via update_thresholds (testing)
- DEFAULT: Hardcoded defaults (not recommended for production)
"""

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Protocol, runtime_checkable, Callable, Optional

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
        regime_confidence: Confidence in regime assessment (0.0 = boundary, 1.0 = certain)
    """

    trust_state: TrustState
    reconstruction_error: float
    threshold_low: float
    threshold_high: float
    regime_confidence: float = field(default=1.0)  # AUDIT-FIX: Confidence of regime metric

    def is_vetoed(self) -> bool:
        """Check if regime has vetoed trading."""
        return self.trust_state == TrustState.VETO

    def requires_caution(self) -> bool:
        """Check if regime requires cautious trading."""
        return self.trust_state == TrustState.CAUTION
    
    def to_details_dict(self) -> dict:
        """
        Convert to details dict for structured logging/observability.
        
        AUDIT-FIX: Enables integration with VetoDecision.details for post-mortem analysis.
        """
        return {
            "trust_state": self.trust_state.value,
            "reconstruction_error": self.reconstruction_error,
            "threshold_low": self.threshold_low,
            "threshold_high": self.threshold_high,
            "regime_confidence": self.regime_confidence,
        }


class CalibrationSource(Enum):
    """Source of regime threshold calibration.
    
    AUDIT-FIX: Tracks provenance of thresholds for debugging and auditability.
    """
    CHECKPOINT = "checkpoint"   # Loaded from model checkpoint (preferred)
    SETTINGS = "settings"       # From settings.hyperparams
    MANUAL = "manual"           # Set via update_thresholds
    DEFAULT = "default"         # Hardcoded defaults (not recommended)


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

    Calibration Features (AUDIT-FIX):
    - Thresholds can be loaded from model checkpoints via `from_checkpoint`
    - `calibration_source` tracks provenance for debugging
    - `set_volatility_scaler` enables adaptive thresholding

    Attributes:
        threshold_caution: Reconstruction error threshold for CAUTION state
        threshold_veto: Reconstruction error threshold for VETO state
        calibration_source: Where thresholds were loaded from

    Example:
        >>> regime_veto = RegimeVeto(threshold_caution=0.1, threshold_veto=0.3)
        >>> assessment = regime_veto.assess(reconstruction_error)
        >>> if assessment.is_vetoed():
        ...     logger.warning("REGIME VETO: No trades allowed")
        ...     return []
    """

    def __init__(
        self, 
        threshold_caution: float = 0.1, 
        threshold_veto: float = 0.3,
        calibration_source: CalibrationSource = CalibrationSource.DEFAULT,
    ):
        """
        Initialize regime veto authority.

        Args:
            threshold_caution: Reconstruction error above which CAUTION is triggered
            threshold_veto: Reconstruction error above which trading is VETOED
            calibration_source: Where these thresholds originated from

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
        self.calibration_source = calibration_source
        
        # AUDIT-FIX: Optional volatility scaler for adaptive thresholding
        self._volatility_scaler: Optional[Callable[[], float]] = None
        self._base_threshold_caution = threshold_caution
        self._base_threshold_veto = threshold_veto

        logger.info(
            f"RegimeVeto initialized: CAUTION={threshold_caution:.3f}, VETO={threshold_veto:.3f}, "
            f"source={calibration_source.value}"
        )
    
    @classmethod
    def from_checkpoint(
        cls, 
        checkpoint: dict, 
        fallback_caution: float = 0.1, 
        fallback_veto: float = 0.3
    ) -> "RegimeVeto":
        """
        Create RegimeVeto from model checkpoint with calibrated thresholds.
        
        AUDIT-FIX: Formalizes handshake between model training and live regime thresholds.
        Thresholds stored in checkpoint during training ensure alignment with normalization.
        
        Args:
            checkpoint: Model checkpoint dict (from torch.load)
            fallback_caution: Fallback if checkpoint lacks regime_caution_threshold
            fallback_veto: Fallback if checkpoint lacks regime_veto_threshold
            
        Returns:
            RegimeVeto with calibrated thresholds from checkpoint
        """
        caution = checkpoint.get("regime_caution_threshold", fallback_caution)
        veto = checkpoint.get("regime_veto_threshold", fallback_veto)
        
        source = CalibrationSource.CHECKPOINT if "regime_caution_threshold" in checkpoint else CalibrationSource.DEFAULT
        
        logger.info(f"Loading regime thresholds from checkpoint: caution={caution}, veto={veto}")
        return cls(threshold_caution=caution, threshold_veto=veto, calibration_source=source)
    
    def set_volatility_scaler(self, scaler_fn: Callable[[], float]) -> None:
        """
        Set a volatility scaler for adaptive thresholding.
        
        AUDIT-FIX: Addresses non-stationary market volatility by allowing thresholds
        to scale based on trailing market variance.
        
        Args:
            scaler_fn: Callable returning scale factor (1.0 = normal, >1 = higher vol)
        
        Example:
            >>> def get_24h_volatility_ratio():
            ...     return current_vol / baseline_vol  # e.g., 1.5 for 50% higher vol
            >>> regime_veto.set_volatility_scaler(get_24h_volatility_ratio)
        """
        self._volatility_scaler = scaler_fn
        logger.info("VolatilityScaler set for adaptive regime thresholds")
    
    def _get_scaled_thresholds(self) -> tuple[float, float]:
        """
        Get current thresholds, scaled by volatility if scaler is set.
        
        Returns:
            Tuple of (threshold_caution, threshold_veto)
        """
        if self._volatility_scaler is None:
            return self.threshold_caution, self.threshold_veto
        
        try:
            scale = self._volatility_scaler()
            # Clamp scale to reasonable range [0.5, 2.0]
            scale = max(0.5, min(2.0, scale))
            return self._base_threshold_caution * scale, self._base_threshold_veto * scale
        except Exception as e:
            logger.warning(f"VolatilityScaler failed, using base thresholds: {e}")
            return self.threshold_caution, self.threshold_veto

    def assess(self, reconstruction_error: torch.Tensor | float) -> RegimeAssessmentProtocol:
        """
        Assess regime trust based on reconstruction error.

        The reconstruction error from the volatility autoencoder indicates
        how well the model understands the current market behavior. High
        reconstruction error suggests the market is behaving anomalously
        relative to historical patterns.
        
        AUDIT-FIX: Now includes regime_confidence metric for observability.

        Args:
            reconstruction_error: Tensor containing reconstruction error
                from volatility expert (typically MSE loss)

        Returns:
            RegimeAssessment with trust_state, threshold info, and regime_confidence

        Example:
            >>> error_tensor = torch.tensor(0.15)
            >>> assessment = regime_veto.assess(error_tensor)
            >>> print(assessment.trust_state)  # TrustState.CAUTION
            >>> print(assessment.regime_confidence)  # 0.75 (distance from boundaries)
        """
        # Extract scalar value
        if isinstance(reconstruction_error, torch.Tensor):
            error = reconstruction_error.item()
        else:
            error = float(reconstruction_error)

        # Get scaled thresholds (adaptive if scaler is set)
        threshold_caution, threshold_veto = self._get_scaled_thresholds()

        # Determine trust state
        if error >= threshold_veto:
            state = TrustState.VETO
        elif error >= threshold_caution:
            state = TrustState.CAUTION
        else:
            state = TrustState.TRUSTED
        
        # AUDIT-FIX: Calculate regime confidence (distance from state boundaries)
        regime_confidence = self._calculate_regime_confidence(error, threshold_caution, threshold_veto)

        return RegimeAssessment(
            trust_state=state,
            reconstruction_error=error,
            threshold_low=threshold_caution,
            threshold_high=threshold_veto,
            regime_confidence=regime_confidence,
        )
    
    def _calculate_regime_confidence(self, error: float, caution: float, veto: float) -> float:
        """
        Calculate confidence in regime assessment (0.0 = boundary, 1.0 = certain).
        
        AUDIT-FIX: Provides metric for "Confidence of Regime" per audit.
        
        A high confidence means the error is far from any threshold boundary.
        A low confidence means the error is near a state transition point.
        
        Returns:
            float between 0.0 and 1.0
        """
        if error < caution:
            # TRUSTED zone: confidence = distance to caution threshold
            return min(1.0, (caution - error) / caution) if caution > 0 else 1.0
        elif error < veto:
            # CAUTION zone: confidence = distance to both boundaries
            range_size = veto - caution
            mid_point = (caution + veto) / 2
            distance_to_boundary = abs(error - mid_point)
            return min(1.0, distance_to_boundary / (range_size / 2)) if range_size > 0 else 0.5
        else:
            # VETO zone: confidence = how far past veto threshold
            overshoot = error - veto
            return min(1.0, 0.5 + overshoot / veto) if veto > 0 else 1.0

    def update_thresholds(self, threshold_caution: float, threshold_veto: float) -> None:
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
        self._base_threshold_caution = threshold_caution
        self._base_threshold_veto = threshold_veto
        self.calibration_source = CalibrationSource.MANUAL
    
    def get_calibration_info(self) -> dict:
        """
        Get calibration information for observability/debugging.
        
        Returns:
            Dict with threshold values and calibration source
        """
        scaled_caution, scaled_veto = self._get_scaled_thresholds()
        return {
            "threshold_caution": self.threshold_caution,
            "threshold_veto": self.threshold_veto,
            "scaled_threshold_caution": scaled_caution,
            "scaled_threshold_veto": scaled_veto,
            "calibration_source": self.calibration_source.value,
            "volatility_scaler_active": self._volatility_scaler is not None,
        }
