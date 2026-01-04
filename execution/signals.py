import uuid
import hashlib
import json
from dataclasses import asdict, dataclass, field, replace
from datetime import datetime
from typing import Any

from config.constants import CONTRACT_TYPES, SIGNAL_TYPES


@dataclass(frozen=True)
class TradeSignal:
    """
    Base signal structure for trade opportunities.
    """

    signal_type: SIGNAL_TYPES
    contract_type: str  # e.g., 'RISE_FALL', 'TOUCH_NO_TOUCH'
    direction: str | None  # 'CALL', 'PUT', 'TOUCH', 'NO_TOUCH'
    probability: float
    timestamp: datetime
    symbol: str = "R_100" # Default for backward compatibility
    metadata: dict[str, Any] = field(default_factory=dict)
    signal_id: str = field(default="")

    def __post_init__(self):
        """Generate signal ID if not provided."""
        if not self.signal_id:
            # We use object.__setattr__ because the class is frozen
            object.__setattr__(self, "signal_id", self.generate_id())

    def generate_id(self) -> str:
        """
        Generate a deterministic ID based on signal parameters.
        
        This enables robust idempotency across system restarts and network
        instability. The ID is stable for identical trade opportunities.
        """
        # Create a stable representation of the signal content
        content = {
            "type": self.signal_type.value if hasattr(self.signal_type, "value") else str(self.signal_type),
            "contract": str(self.contract_type),
            "dir": str(self.direction),
            "ts": self.timestamp.isoformat(),
            # We exclude metadata["stake"] from ID as it's injected later
            # and we want the same underlying signal to have the same ID
            "meta_reduced": {k: v for k, v in self.metadata.items() if k != "stake"}
        }
        content_str = json.dumps(content, sort_keys=True)
        return hashlib.sha256(content_str.encode()).hexdigest()[:16]

    def with_metadata(self, **updates: Any) -> "TradeSignal":
        """
        Create a new instance with updated metadata.
        
        Preserves immutability by returning a copy.
        """
        new_metadata = {**self.metadata, **updates}
        return replace(self, metadata=new_metadata)

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary for JSON export."""
        d = asdict(self)
        d["timestamp"] = self.timestamp.isoformat()
        return d

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "TradeSignal":
        """Deserialize from dictionary."""
        d_copy = d.copy()
        if isinstance(d["timestamp"], str):
             d_copy["timestamp"] = datetime.fromisoformat(d["timestamp"])
        # If it's already datetime, leave it
        return cls(**d_copy)


@dataclass(frozen=True)
class ShadowTrade(TradeSignal):
    """
    Extended signal for paper trading with outcome tracking.
    """

    trade_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    entry_price: float | None = None
    exit_price: float | None = None
    outcome: bool | None = None  # True = win, False = loss
    pnl: float | None = None

    def __post_init__(self):
        """Shadow signals use a distinct ID namespace or same depending on need."""
        if not self.signal_id:
            # We use object.__setattr__ because the class is frozen
            object.__setattr__(self, "signal_id", self.generate_id())

    def with_outcome(
        self,
        outcome: bool,
        exit_price: float,
        stake: float,
        payout: float = 0.95,  # Payout ratio example
    ) -> "ShadowTrade":
        """Update trade with final outcome (returns NEW instance)."""
        if outcome:
            pnl = stake * payout
        else:
            pnl = -stake
            
        return replace(
            self,
            outcome=outcome,
            exit_price=exit_price,
            pnl=pnl
        )

    def to_record(self) -> dict[str, Any]:
        """Convert to flat record for logging/training."""
        base = self.to_dict()
        base.update(
            {
                "trade_id": self.trade_id,
                "entry_price": self.entry_price,
                "exit_price": self.exit_price,
                "outcome": self.outcome,
                "pnl": self.pnl,
            }
        )
        return base
