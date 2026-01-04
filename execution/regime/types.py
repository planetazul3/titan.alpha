"""
Regime Identification Types.

Enums and data structures for market regime classification.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Protocol, runtime_checkable

from execution.common.types import TrustState


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
