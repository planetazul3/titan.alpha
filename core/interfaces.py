"""
Core interfaces for the trading system.
"""
from typing import Protocol, List, Optional, Any
from abc import abstractmethod

from core.domain.entities import Order, Trade, Position

class IExecutor(Protocol):
    """Interface for trade execution."""
    async def execute_order(self, order: Order) -> Trade:
        ...

    async def get_positions(self) -> List[Position]:
        ...

class IStrategy(Protocol):
    """Interface for trading strategies."""
    async def analyze(self, market_data: Any) -> Optional[Order]:
        ...

class IDataProvider(Protocol):
    """Interface for data ingestion."""
    async def get_latest_data(self, symbol: str) -> Any:
        ...

class IRiskController(Protocol):
    """Interface for risk controller."""
    def can_place_order(self, order: Order) -> tuple[bool, str]:
        ...

    def record_trade(self, trade: Trade):
        ...
