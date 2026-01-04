"""
Regime Veto Authority.

The single source of truth for market regime safety checks.
"""

import logging
from typing import Callable, Optional

import numpy as np
import torch

from execution.common.types import TrustState
from .types import CalibrationSource, RegimeAssessmentProtocol, RegimeAssessment, TrustState
from .detectors import HierarchicalRegimeDetector

logger = logging.getLogger(__name__)


class RegimeVeto:
    """
    Unified authority for market regime detection and trading vetoes.

    Combines hierarchical detection with threshold-based logic.
    Maintains compatibility with legacy RegimeVeto interface while
    providing enhanced capabilities.

    Usage:
        >>> veto = RegimeVeto()
        >>> veto.update_prices(recent_prices)
        >>> assessment = veto.assess(reconstruction_error)
        >>> if assessment.is_vetoed():
        ...     stop_trading()
    """

    def __init__(
        self, 
        threshold_caution: float = 0.1, 
        threshold_veto: float = 0.3,
        calibration_source: CalibrationSource = CalibrationSource.DEFAULT,
        use_hierarchical: bool = True
    ):
        if threshold_caution >= threshold_veto:
            raise ValueError("Caution threshold must be less than veto threshold")
            
        self.threshold_caution = threshold_caution
        self.threshold_veto = threshold_veto
        self.calibration_source = calibration_source
        self.use_hierarchical = use_hierarchical
        
        self.hierarchical_detector = HierarchicalRegimeDetector()
        self._price_cache: np.ndarray | None = None
        self._volatility_scaler: Optional[Callable[[], float]] = None

        logger.info(
            f"RegimeVeto initialized (Hierarchical={use_hierarchical}): "
            f"CAUTION={threshold_caution:.3f}, VETO={threshold_veto:.3f}"
        )

    @classmethod
    def from_checkpoint(cls, checkpoint: dict, fallback_caution: float = 0.1, fallback_veto: float = 0.3) -> "RegimeVeto":
        caution = checkpoint.get("regime_caution_threshold", fallback_caution)
        veto = checkpoint.get("regime_veto_threshold", fallback_veto)
        source = CalibrationSource.CHECKPOINT if "regime_caution_threshold" in checkpoint else CalibrationSource.DEFAULT
        return cls(threshold_caution=caution, threshold_veto=veto, calibration_source=source)

    def update_prices(self, prices: np.ndarray) -> None:
        """Update cached prices for hierarchical detection."""
        self._price_cache = prices

    def set_volatility_scaler(self, scaler_fn: Callable[[], float]) -> None:
        """Set volatility scaler for adaptive thresholds."""
        self._volatility_scaler = scaler_fn

    def _get_scaled_thresholds(self) -> tuple[float, float]:
        if self._volatility_scaler is None:
            return self.threshold_caution, self.threshold_veto
        try:
            scale = max(0.5, min(2.0, self._volatility_scaler()))
            return self.threshold_caution * scale, self.threshold_veto * scale
        except Exception:
            return self.threshold_caution, self.threshold_veto

    def assess(self, reconstruction_error: torch.Tensor | float) -> RegimeAssessmentProtocol:
        """
        Assess regime trust.
        
        If hierarchical detection is enabled and price data is available,
        uses full multi-level assessment. Otherwise falls back to
        threshold-based logic.
        """
        if isinstance(reconstruction_error, torch.Tensor):
            error = reconstruction_error.item()
        else:
            error = float(reconstruction_error)
            
        threshold_caution, threshold_veto = self._get_scaled_thresholds()

        # Try hierarchical assessment first
        if self.use_hierarchical:
            return self.hierarchical_detector.assess_from_reconstruction_error(
                error, 
                self._price_cache,
                threshold_veto=threshold_veto,
                threshold_caution=threshold_caution
            )

        # Fallback to simple threshold logic
        threshold_caution, threshold_veto = self._get_scaled_thresholds()

        if error >= threshold_veto:
            state = TrustState.VETO
        elif error >= threshold_caution:
            state = TrustState.CAUTION
        else:
            state = TrustState.TRUSTED

        # Calculate confidence
        if error < threshold_caution:
            conf = min(1.0, (threshold_caution - error) / threshold_caution) if threshold_caution > 0 else 1.0
        elif error < threshold_veto:
            range_size = threshold_veto - threshold_caution
            mid = (threshold_caution + threshold_veto) / 2
            conf = min(1.0, abs(error - mid) / (range_size / 2)) if range_size > 0 else 0.5
        else:
            conf = min(1.0, 0.5 + (error - threshold_veto) / threshold_veto) if threshold_veto > 0 else 1.0

        return RegimeAssessment(
            trust_state=state,
            reconstruction_error=error,
            threshold_low=threshold_caution,
            threshold_high=threshold_veto,
            regime_confidence=conf
        )

    def update_thresholds(self, threshold_caution: float, threshold_veto: float) -> None:
        if threshold_caution >= threshold_veto:
            raise ValueError("Caution threshold must be less than veto threshold")
        self.threshold_caution = threshold_caution
        self.threshold_veto = threshold_veto
        self.calibration_source = CalibrationSource.MANUAL

    def get_calibration_info(self) -> dict:
        scaled_caution, scaled_veto = self._get_scaled_thresholds()
        return {
            "threshold_caution": self.threshold_caution,
            "threshold_veto": self.threshold_veto,
            "scaled_threshold_caution": scaled_caution,
            "scaled_threshold_veto": scaled_veto,
            "calibration_source": self.calibration_source.value,
            "volatility_scaler_active": self._volatility_scaler is not None,
            "use_hierarchical": self.use_hierarchical,
        }
