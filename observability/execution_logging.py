"""
Execution Logging - Structured observability for trade execution lifecycle.

REC-001: Provides mission-control style visibility into execution successes and failures.
"""

import json
import logging
from datetime import datetime, timezone
from typing import Any, Optional

logger = logging.getLogger("execution.lifecycle")

class ExecutionLogger:
    """
    Handles structured logging for trade execution events.
    """
    
    def log_event(
        self, 
        event_type: str, 
        signal_id: str, 
        success: bool, 
        details: Optional[dict[str, Any]] = None,
        error: Optional[str] = None
    ):
        """Log a structured execution event."""
        event = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "event_type": event_type,
            "signal_id": signal_id,
            "success": success,
            "error": error,
            "details": details or {}
        }
        logger.info(f"EXECUTION_EVENT: {json.dumps(event)}")

    def log_trade_attempt(self, signal_id: str, contract_type: str, direction: str, stake: float):
        self.log_event(
            "attempt", 
            signal_id, 
            True, 
            {"contract_type": contract_type, "direction": direction, "stake": stake}
        )

    def log_trade_success(self, signal_id: str, contract_id: int, entry_price: float):
        self.log_event(
            "success", 
            signal_id, 
            True, 
            {"contract_id": contract_id, "entry_price": entry_price}
        )

    def log_trade_failure(self, signal_id: str, error: str, details: Optional[dict] = None):
        self.log_event(
            "failure", 
            signal_id, 
            False, 
            error=error,
            details=details
        )

execution_logger = ExecutionLogger()
