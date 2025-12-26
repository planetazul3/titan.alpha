import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime
from typing import Any

from config.constants import CONTRACT_TYPES, SIGNAL_TYPES


@dataclass
class TradeSignal:
    """
    Base signal structure for trade opportunities.
    """

    signal_type: SIGNAL_TYPES
    contract_type: CONTRACT_TYPES
    direction: str | None  # 'CALL', 'PUT'
    probability: float
    timestamp: datetime
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary for JSON export."""
        d = asdict(self)
        d["timestamp"] = self.timestamp.isoformat()
        return d

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "TradeSignal":
        """Deserialize from dictionary."""
        d_copy = d.copy()
        d_copy["timestamp"] = datetime.fromisoformat(d["timestamp"])
        return cls(**d_copy)


@dataclass
class ShadowTrade(TradeSignal):
    """
    Extended signal for paper trading with outcome tracking.
    """

    trade_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    entry_price: float | None = None
    exit_price: float | None = None
    outcome: bool | None = None  # True = win, False = loss
    pnl: float | None = None

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

    def update_outcome(
        self,
        outcome: bool,
        exit_price: float,
        stake: float,
        payout: float = 0.95,  # Payout ratio example
    ) -> None:
        """Update trade with final outcome."""
        self.outcome = outcome
        self.exit_price = exit_price
        if outcome:
            self.pnl = stake * payout
        else:
            self.pnl = -stake
