"""
Shared constants and enums for execution package.
"""

from enum import Enum

class TrustState(Enum):
    """Market regime trust states for trading decisions."""

    TRUSTED = "trusted"  # Normal regime, trust predictions
    CAUTION = "caution"  # Elevated uncertainty, reduce stakes
    VETO = "veto"  # Anomalous regime, no trades allowed
