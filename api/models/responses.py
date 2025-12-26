"""
Pydantic response models for Dashboard API.

These models define the structure of API responses.
"""

from datetime import datetime
from typing import Optional, Dict, Any
from pydantic import BaseModel, Field


class ShadowTradeStats(BaseModel):
    """Shadow trade aggregate statistics."""
    total: int = Field(description="Total shadow trades")
    resolved: int = Field(description="Resolved (with outcome) trades")
    unresolved: int = Field(description="Pending trades")
    wins: int = Field(description="Winning trades")
    losses: int = Field(description="Losing trades")
    win_rate: float = Field(description="Win rate (0-1)")
    simulated_pnl: float = Field(description="Simulated P&L in dollars")
    roi: float = Field(description="Return on investment percentage")


class SafetyState(BaseModel):
    """Safety state metrics."""
    trades_attempted: float = 0
    trades_executed: float = 0
    trades_blocked_rate_limit: float = 0
    trades_blocked_kill_switch: float = 0
    consecutive_failures: float = 0
    daily_pnl: float = 0


class ContractTypeStats(BaseModel):
    """Stats by contract type."""
    total: int
    wins: int
    win_rate: float


class TradingMetrics(BaseModel):
    """Complete trading metrics response."""
    shadow_trades: ShadowTradeStats
    safety_state: Dict[str, float] = Field(default_factory=dict)
    system_status: str = "unknown"
    timestamp: Optional[str] = None


class ShadowTrade(BaseModel):
    """Single shadow trade record."""
    trade_id: str
    timestamp: str
    contract_type: str
    direction: str
    probability: float
    entry_price: float
    exit_price: Optional[float] = None
    outcome: Optional[int] = None  # null=pending, 0=loss, 1=win
    reconstruction_error: float
    regime_state: str


class ShadowTradeList(BaseModel):
    """List of shadow trades."""
    trades: list[ShadowTrade]
    count: int


class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    timestamp: str


class WebSocketMessage(BaseModel):
    """WebSocket message format."""
    type: str  # 'metrics', 'heartbeat', 'error'
    timestamp: str
    data: Dict[str, Any]
