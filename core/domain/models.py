"""
Domain models for trading.
"""
from typing import Literal, Optional, List
from datetime import datetime
from uuid import uuid4, UUID
from pydantic import Field, field_validator

from core.domain.base import AggregateRoot, DomainEntity, now_utc

ContractType = Literal["CALL", "PUT", "TOUCH", "NO_TOUCH", "ONETOUCH", "CALLE", "PUTE", "DIGITMATCH", "DIGITDIFF", "ASIANU", "ASIAND"]
OrderStatus = Literal["PENDING", "SUBMITTED", "FILLED", "REJECTED", "CANCELLED", "EXPIRED"]
TradeStatus = Literal["OPEN", "CLOSED", "WIN", "LOSS"]

class Order(AggregateRoot):
    """
    Represents an order to be executed by the broker.
    """
    order_id: UUID = Field(default_factory=uuid4)
    symbol: str
    contract_type: ContractType
    stake: float = Field(..., gt=0)
    duration: int = Field(..., gt=0)
    duration_unit: Literal["t", "s", "m", "h", "d"] = "m"
    barrier: Optional[float] = None
    created_at: datetime = Field(default_factory=now_utc)
    status: OrderStatus = "PENDING"
    external_id: Optional[str] = None # Broker's order/contract ID

    error_message: Optional[str] = None

    @property
    def is_terminal(self) -> bool:
        return self.status in ("FILLED", "REJECTED", "CANCELLED", "EXPIRED")

class Trade(AggregateRoot):
    """
    Represents a realized trade (position).
    """
    trade_id: UUID = Field(default_factory=uuid4)
    order_id: UUID
    contract_id: str # External ID from broker
    symbol: str
    contract_type: ContractType
    entry_time: datetime
    exit_time: Optional[datetime] = None

    entry_price: Optional[float] = None
    exit_price: Optional[float] = None

    stake: float
    payout: Optional[float] = None # Total payout (stake + profit)
    profit: Optional[float] = None # Net profit/loss

    status: TradeStatus = "OPEN"

    metadata: dict = Field(default_factory=dict)

    def close(self, exit_time: datetime, payout: float, profit: float, exit_price: Optional[float] = None):
        """Closes the trade."""
        object.__setattr__(self, 'exit_time', exit_time)
        object.__setattr__(self, 'payout', payout)
        object.__setattr__(self, 'profit', profit)
        object.__setattr__(self, 'exit_price', exit_price)
        object.__setattr__(self, 'status', "WIN" if profit > 0 else "LOSS")

class Position(DomainEntity):
    """
    Represents current market exposure.
    (Simplified for binary options where positions are distinct contracts)
    """
    symbol: str
    contract_id: str
    stake: float
    contract_type: ContractType
    entry_time: datetime
