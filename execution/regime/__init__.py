"""
Regime Detection and Safety Subsystem.

This package provides:
1. RegimeVeto: The authoritative gatekeeper for volatility safety.
2. Hierarchical Detection: Macro context (Trend), Meso context (Vol), Micro context (Hurst).
"""

from execution.common.types import TrustState
from .types import (
    RegimeAssessment, 
    RegimeAssessmentProtocol, 
    HierarchicalRegimeAssessment,
    CalibrationSource,
    MacroRegime,
    VolatilityRegime,
    MicroRegime
)
from .detectors import (
    HurstExponentEstimator,
    VolatilityRegimeDetector,
    TrendDetector,
    HierarchicalRegimeDetector
)
from .veto import RegimeVeto

__all__ = [
    "RegimeVeto",
    "RegimeAssessment",
    "RegimeAssessmentProtocol",
    "HierarchicalRegimeAssessment",
    "HurstExponentEstimator",
    "VolatilityRegimeDetector",
    "TrendDetector",
    "HierarchicalRegimeDetector",
    "CalibrationSource",
    "MacroRegime",
    "VolatilityRegime",
    "MicroRegime",
    "TrustState"
]
