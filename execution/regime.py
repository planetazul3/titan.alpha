"""
Hierarchical Regime Detection Module.

Provides multi-level market regime detection for smarter trading decisions.
Instead of a single reconstruction error threshold, this module detects
regimes at multiple time scales:

1. **Macro Level**: Bull/Bear/Sideways market trend (long lookback)
2. **Meso Level**: Volatility regime (Low/Medium/High)
3. **Micro Level**: Microstructure regime (Trending/Mean-reverting via Hurst)

This module consolidates the previous `RegimeVeto` logic with the enhanced
hierarchical detection, serving as the single source of truth for regime
assessments.

References:
- Hidden Markov Models in Finance (Nystrup et al., 2015)
- Changepoint Detection Survey (Aminikhanghahi & Cook, 2017)
- Hurst exponent for market regime (Mandelbrot, 1971)
"""

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Protocol, runtime_checkable, Callable, Optional

import numpy as np
import torch

from execution.common.types import TrustState

logger = logging.getLogger(__name__)


# -----------------------------------------------------------------------------
# Base Protocols and Enums
# -----------------------------------------------------------------------------

class MacroRegime(Enum):
    """Macro-level market trend regime."""
    BULL = "bull"       # Uptrending market
    BEAR = "bear"       # Downtrending market
    SIDEWAYS = "sideways"  # Range-bound market


class VolatilityRegime(Enum):
    """Meso-level volatility regime."""
    LOW = "low"         # Quiet market
    MEDIUM = "medium"   # Normal volatility
    HIGH = "high"       # Elevated volatility


class MicroRegime(Enum):
    """Micro-level market microstructure."""
    TRENDING = "trending"       # Momentum-driven (Hurst > 0.5)
    RANDOM = "random"          # Random walk (Hurst ≈ 0.5)
    MEAN_REVERTING = "mean_reverting"  # Anti-persistent (Hurst < 0.5)


class CalibrationSource(Enum):
    """Source of regime threshold calibration."""
    CHECKPOINT = "checkpoint"   # Loaded from model checkpoint (preferred)
    SETTINGS = "settings"       # From settings.hyperparams
    MANUAL = "manual"           # Set via update_thresholds
    DEFAULT = "default"         # Hardcoded defaults (not recommended)


@runtime_checkable
class RegimeAssessmentProtocol(Protocol):
    """
    Protocol defining the common interface for regime assessments.
    
    Ensures compatibility between simple and hierarchical assessments.
    """
    
    reconstruction_error: float
    
    def is_vetoed(self) -> bool:
        """Check if regime has vetoed trading."""
        ...
    
    def requires_caution(self) -> bool:
        """Check if regime requires cautious trading."""
        ...


@dataclass
class RegimeAssessment:
    """
    Basic assessment of current market regime trustworthiness.

    Retained for backward compatibility and simple veto logic.

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
    regime_confidence: float = field(default=1.0)

    def is_vetoed(self) -> bool:
        return self.trust_state == TrustState.VETO

    def requires_caution(self) -> bool:
        return self.trust_state == TrustState.CAUTION
    
    def to_details_dict(self) -> dict:
        return {
            "trust_state": self.trust_state.value,
            "reconstruction_error": self.reconstruction_error,
            "threshold_low": self.threshold_low,
            "threshold_high": self.threshold_high,
            "regime_confidence": self.regime_confidence,
        }


@dataclass
class HierarchicalRegimeAssessment:
    """
    Complete hierarchical regime assessment.
    
    Attributes:
        macro: Macro-level regime (Bull/Bear/Sideways)
        volatility: Meso-level volatility regime (Low/Medium/High)
        micro: Micro-level microstructure (Trending/Random/Mean-reverting)
        trust_score: Overall regime trust score (0 to 1, higher is safer)
        reconstruction_error: Original reconstruction error (for compatibility)
        details: Additional diagnostic information
    """
    macro: MacroRegime
    volatility: VolatilityRegime
    micro: MicroRegime
    trust_score: float
    reconstruction_error: float
    details: dict[str, Any]

    # Compatibility properties for TrustState interface
    @property
    def trust_state(self) -> TrustState:
        if self.is_vetoed():
            return TrustState.VETO
        elif self.requires_caution():
            return TrustState.CAUTION
        else:
            return TrustState.TRUSTED

    def is_vetoed(self) -> bool:
        """Check if regime warrants trade veto (trust_score < 0.3)."""
        return self.trust_score < 0.3

    def requires_caution(self) -> bool:
        """Check if regime requires cautious trading (0.3 <= trust_score < 0.6)."""
        return 0.3 <= self.trust_score < 0.6

    def is_favorable(self) -> bool:
        """Check if regime is favorable for aggressive trading (trust_score >= 0.8)."""
        return self.trust_score >= 0.8

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for logging/serialization."""
        return {
            "macro": self.macro.value,
            "volatility": self.volatility.value,
            "micro": self.micro.value,
            "trust_score": self.trust_score,
            "reconstruction_error": self.reconstruction_error,
            "is_vetoed": self.is_vetoed(),
            "requires_caution": self.requires_caution(),
            **self.details,
        }

    def to_details_dict(self) -> dict[str, Any]:
        """Alias for to_dict to match RegimeAssessment interface."""
        return self.to_dict()


# -----------------------------------------------------------------------------
# Detection Components
# -----------------------------------------------------------------------------

class HurstExponentEstimator:
    """Estimate the Hurst exponent for regime detection."""

    def __init__(self, min_window: int = 8, max_window: int | None = None):
        self.min_window = min_window
        self.max_window = max_window

    def estimate(self, prices: np.ndarray) -> float:
        """Estimate Hurst exponent from price series."""
        if len(prices) < self.min_window * 2:
            return 0.5  # Default to random walk

        returns = np.diff(np.log(prices + 1e-10))

        if len(returns) < self.min_window:
            return 0.5

        max_window = self.max_window or len(returns) // 2

        window_sizes = []
        rs_values = []

        for window in range(self.min_window, min(max_window + 1, len(returns) + 1)):
            n_windows = len(returns) // window
            if n_windows < 1:
                continue

            truncated_len = n_windows * window
            segments = returns[:truncated_len].reshape(n_windows, window)
            means = np.mean(segments, axis=1, keepdims=True)
            deviations = segments - means
            cumdev = np.cumsum(deviations, axis=1)
            R = np.max(cumdev, axis=1) - np.min(cumdev, axis=1)
            S = np.std(segments, axis=1, ddof=1)

            valid = S > 0
            if np.any(valid):
                avg_rs = np.mean(R[valid] / S[valid])
                window_sizes.append(window)
                rs_values.append(avg_rs)

        if len(window_sizes) < 2:
            return 0.5

        log_n = np.log(window_sizes)
        log_rs = np.log(rs_values)

        n = len(log_n)
        sum_x = np.sum(log_n)
        sum_y = np.sum(log_rs)
        sum_xy = np.sum(log_n * log_rs)
        sum_xx = np.sum(log_n ** 2)

        denominator = n * sum_xx - sum_x ** 2
        if abs(denominator) < 1e-10:
            return 0.5

        hurst = (n * sum_xy - sum_x * sum_y) / denominator
        return float(np.clip(hurst, 0.0, 1.0))


class VolatilityRegimeDetector:
    """Detect volatility regime using rolling volatility."""

    def __init__(self, low_percentile: float = 30, high_percentile: float = 70, lookback: int = 100):
        self.low_percentile = low_percentile
        self.high_percentile = high_percentile
        self.lookback = lookback

    def detect(self, prices: np.ndarray) -> tuple[VolatilityRegime, float]:
        if len(prices) < 20:
            return VolatilityRegime.MEDIUM, 50.0

        returns = np.diff(np.log(prices + 1e-10))
        lookback = min(len(returns), self.lookback)
        window = min(20, lookback // 2)

        if window < 5:
            return VolatilityRegime.MEDIUM, 50.0

        rolling_vol = np.array([
            np.std(returns[max(0, i - window):i + 1])
            for i in range(len(returns))
        ])

        current_vol = rolling_vol[-1]
        historical_vol = rolling_vol[-lookback:]
        percentile = 100 * np.sum(historical_vol < current_vol) / len(historical_vol)

        if percentile < self.low_percentile:
            regime = VolatilityRegime.LOW
        elif percentile > self.high_percentile:
            regime = VolatilityRegime.HIGH
        else:
            regime = VolatilityRegime.MEDIUM

        return regime, float(percentile)


class TrendDetector:
    """Detect macro trend regime using MA crossovers."""

    def __init__(self, short_window: int = 20, long_window: int = 50):
        self.short_window = short_window
        self.long_window = long_window

    def detect(self, prices: np.ndarray) -> tuple[MacroRegime, float]:
        if len(prices) < self.long_window:
            return MacroRegime.SIDEWAYS, 0.0

        short_ma = np.mean(prices[-self.short_window:])
        long_ma = np.mean(prices[-self.long_window:])
        current_price = prices[-1]

        trend_strength = (short_ma - long_ma) / long_ma if long_ma > 0 else 0.0

        bull_threshold = 0.01
        bear_threshold = -0.01

        if trend_strength > bull_threshold and current_price > short_ma:
            regime = MacroRegime.BULL
        elif trend_strength < bear_threshold and current_price < short_ma:
            regime = MacroRegime.BEAR
        else:
            regime = MacroRegime.SIDEWAYS

        return regime, float(trend_strength)


class HierarchicalRegimeDetector:
    """Multi-level regime detector combining macro, volatility, and micro signals."""

    def __init__(
        self,
        trend_short_window: int = 20,
        trend_long_window: int = 50,
        vol_low_percentile: float = 30,
        vol_high_percentile: float = 70,
        hurst_min_window: int = 8,
        reconstruction_weight: float = 0.3,
    ):
        self.trend_detector = TrendDetector(trend_short_window, trend_long_window)
        self.vol_detector = VolatilityRegimeDetector(vol_low_percentile, vol_high_percentile)
        self.hurst_estimator = HurstExponentEstimator(hurst_min_window)
        self.reconstruction_weight = reconstruction_weight

    def assess(self, prices: np.ndarray, reconstruction_error: float = 0.0) -> HierarchicalRegimeAssessment:
        macro, trend_strength = self.trend_detector.detect(prices)
        volatility, vol_percentile = self.vol_detector.detect(prices)
        hurst = self.hurst_estimator.estimate(prices)

        if hurst > 0.55:
            micro = MicroRegime.TRENDING
        elif hurst < 0.45:
            micro = MicroRegime.MEAN_REVERTING
        else:
            micro = MicroRegime.RANDOM

        trust_score = self._calculate_trust_score(
            macro, volatility, micro,
            trend_strength, vol_percentile, hurst,
            reconstruction_error
        )

        details = {
            "trend_strength": trend_strength,
            "vol_percentile": vol_percentile,
            "hurst_exponent": hurst,
        }

        return HierarchicalRegimeAssessment(
            macro=macro,
            volatility=volatility,
            micro=micro,
            trust_score=trust_score,
            reconstruction_error=reconstruction_error,
            details=details,
        )

    def _calculate_trust_score(
        self, macro, volatility, micro, trend_strength, vol_percentile, hurst, reconstruction_error
    ) -> float:
        trust = 1.0

        if volatility == VolatilityRegime.HIGH:
            trust *= 0.6
        elif volatility == VolatilityRegime.MEDIUM:
            trust *= 0.85

        if macro == MacroRegime.SIDEWAYS:
            trust *= 0.8
        elif abs(trend_strength) > 0.02:
            trust *= 1.0

        if micro == MicroRegime.RANDOM:
            trust *= 0.7
        else:
            trust *= 0.9

        recon_penalty = 1.0 - min(1.0, reconstruction_error * 2) * self.reconstruction_weight
        trust *= recon_penalty

        return float(np.clip(trust, 0.0, 1.0))

    def assess_from_reconstruction_error(
        self, 
        reconstruction_error: float, 
        prices: np.ndarray | None = None,
        threshold_veto: float = 0.3,
        threshold_caution: float = 0.1
    ) -> HierarchicalRegimeAssessment:
        """Backward compatibility for reconstruction-only assessment."""
        if prices is not None and len(prices) >= 50:
            return self.assess(prices, reconstruction_error)

        # Fallback heuristics
        if reconstruction_error >= threshold_veto:
            volatility = VolatilityRegime.HIGH
            trust = 0.2
        elif reconstruction_error >= threshold_caution:
            volatility = VolatilityRegime.MEDIUM
            trust = 0.5 # 0.5 is within [0.3, 0.6) range for CAUTION
        else:
            volatility = VolatilityRegime.LOW
            trust = 0.9

        return HierarchicalRegimeAssessment(
            macro=MacroRegime.SIDEWAYS,
            volatility=volatility,
            micro=MicroRegime.RANDOM,
            trust_score=trust,
            reconstruction_error=reconstruction_error,
            details={"fallback_mode": True},
        )


# -----------------------------------------------------------------------------
# Main Authority Class
# -----------------------------------------------------------------------------

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
