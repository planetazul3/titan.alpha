"""
Structured Logging for Shadow Trade Lifecycle.

R06: Add structured logging events for shadow trade tracking.
Enables debugging of individual trade flows and system-wide analytics.
"""

import logging
import json
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from enum import Enum
from typing import Any


logger = logging.getLogger(__name__)


class ShadowTradeStage(str, Enum):
    """Stages in the shadow trade lifecycle."""
    CREATED = "created"
    STORED = "stored"
    PENDING_RESOLUTION = "pending_resolution"
    RESOLVED = "resolved"
    STALE = "stale"
    TRAINING_USED = "training_used"


@dataclass
class ShadowTradeEvent:
    """Structured event for shadow trade lifecycle tracking."""
    trade_id: str
    stage: ShadowTradeStage
    timestamp: datetime
    contract_type: str | None = None
    direction: str | None = None
    probability: float | None = None
    outcome: bool | None = None
    exit_price: float | None = None
    metadata: dict[str, Any] | None = None
    
    def to_dict(self) -> dict[str, Any]:
        """Convert to JSON-serializable dictionary."""
        data = asdict(self)
        data["stage"] = self.stage.value
        data["timestamp"] = self.timestamp.isoformat()
        if self.metadata:
            data["metadata"] = self.metadata
        return data
    
    def to_json(self) -> str:
        """Convert to JSON string."""
        return json.dumps(self.to_dict())


class ShadowTradeLogger:
    """
    Structured logger for shadow trade lifecycle events.
    
    Provides consistent event logging format for debugging and analytics.
    """
    
    def __init__(self, logger_name: str = "shadow.lifecycle"):
        """Initialize with a specific logger."""
        self._logger = logging.getLogger(logger_name)
    
    def log_created(
        self,
        trade_id: str,
        contract_type: str,
        direction: str,
        probability: float,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Log shadow trade creation."""
        event = ShadowTradeEvent(
            trade_id=trade_id,
            stage=ShadowTradeStage.CREATED,
            timestamp=datetime.now(timezone.utc),
            contract_type=contract_type,
            direction=direction,
            probability=probability,
            metadata=metadata,
        )
        self._logger.info(f"SHADOW_TRADE_EVENT: {event.to_json()}")
    
    def log_stored(self, trade_id: str) -> None:
        """Log shadow trade stored to database."""
        event = ShadowTradeEvent(
            trade_id=trade_id,
            stage=ShadowTradeStage.STORED,
            timestamp=datetime.now(timezone.utc),
        )
        self._logger.info(f"SHADOW_TRADE_EVENT: {event.to_json()}")
    
    def log_resolved(
        self,
        trade_id: str,
        outcome: bool,
        exit_price: float,
    ) -> None:
        """Log shadow trade resolution."""
        event = ShadowTradeEvent(
            trade_id=trade_id,
            stage=ShadowTradeStage.RESOLVED,
            timestamp=datetime.now(timezone.utc),
            outcome=outcome,
            exit_price=exit_price,
        )
        self._logger.info(f"SHADOW_TRADE_EVENT: {event.to_json()}")
    
    def log_stale(self, trade_id: str, reason: str | None = None) -> None:
        """Log shadow trade marked as stale."""
        event = ShadowTradeEvent(
            trade_id=trade_id,
            stage=ShadowTradeStage.STALE,
            timestamp=datetime.now(timezone.utc),
            metadata={"reason": reason} if reason else None,
        )
        self._logger.warning(f"SHADOW_TRADE_EVENT: {event.to_json()}")
    
    def log_training_used(
        self,
        trade_id: str,
        batch_id: str | None = None,
    ) -> None:
        """Log shadow trade used in training."""
        event = ShadowTradeEvent(
            trade_id=trade_id,
            stage=ShadowTradeStage.TRAINING_USED,
            timestamp=datetime.now(timezone.utc),
            metadata={"batch_id": batch_id} if batch_id else None,
        )
        self._logger.debug(f"SHADOW_TRADE_EVENT: {event.to_json()}")


# Global instance for convenience
shadow_trade_logger = ShadowTradeLogger()
