"""
Neural network models for the DerivOmniModel trading system.

This module provides the multi-expert architecture for binary options trading:
- Temporal Expert (BiLSTM with attention)
- Spatial Expert (Pyramidal CNN)
- Volatility Expert (Autoencoder)
- Expert Fusion layer
- Contract-specific prediction heads

Public API:
    - DerivOmniModel: Main model combining all experts
    - TemporalExpert, SpatialExpert, VolatilityExpert: Individual experts
    - ExpertFusion: Fusion layer
    - Contract heads: RiseFallHead, TouchHead, RangeHead
    - Attention mechanisms
    - Building blocks (BiLSTMBlock, MLPBlock, etc.)
"""

from models.core import DerivOmniModel
from models.fusion import ExpertFusion
from models.spatial import SpatialExpert
from models.temporal import TemporalExpert
from models.volatility import VolatilityExpert

__all__ = [
    "DerivOmniModel",
    "TemporalExpert",
    "SpatialExpert",
    "VolatilityExpert",
    "ExpertFusion",
]
