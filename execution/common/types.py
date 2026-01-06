"""
Shared constants and enums for execution package.
"""

from enum import Enum

class TrustState(Enum):
    """Market regime trust states for trading decisions."""

    TRUSTED = "trusted"  # Normal regime, trust predictions
    CAUTION = "caution"  # Elevated uncertainty, reduce stakes
    VETO = "veto"  # Anomalous regime, no trades allowed

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional

@dataclass
class ExecutionRequest:
    """
    Strict contract for trade execution.
    
    Decouples decision logic (Signal) from execution mechanics (Trade).
    The Executor strictly obeys this request without applying sizing or duration logic.
    """
    signal_id: str
    symbol: str
    contract_type: str  # e.g. "CALL", "PUT", "TOUCH", etc. (Mapped from Signal)
    stake: float
    duration: int
    duration_unit: str
    barrier: Optional[str] = None
    barrier2: Optional[str] = None
    
    # CRITICAL-004: Pass regime context for execution-time safety checks
    regime_state: Optional[str] = None
    reconstruction_error: Optional[float] = None
    
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    
    def __post_init__(self):
        if self.stake <= 0:
            raise ValueError(f"ExecutionRequest stake must be positive, got {self.stake}")
        if self.duration <= 0:
            raise ValueError(f"ExecutionRequest duration must be positive, got {self.duration}")
        if not self.contract_type:
             raise ValueError("ExecutionRequest must have a valid contract_type")

