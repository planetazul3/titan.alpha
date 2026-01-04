
"""
Alerting System for x.titan Trading System.

Centralizes alert management, suppression, and routing to ensuring critical
issues are surfaced without flooding logs or communication channels.
"""

import logging
import threading
import time
from abc import ABC, abstractmethod
from collections import defaultdict
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)


class AlertLevel(Enum):
    """
    Severity levels for alerts.
    """
    INFO = auto()      # Informational events (e.g., daily summary)
    WARNING = auto()   # Potential issues (e.g., reconstruction error high)
    CRITICAL = auto()  # System failures (e.g., circuit breaker, kill switch)


@dataclass
class Alert:
    """
    Structured alert event.
    """
    name: str
    message: str
    level: AlertLevel
    timestamp: float = field(default_factory=time.time)
    context: Dict[str, str] = field(default_factory=dict)


class AlertChannel(ABC):
    """
    Abstract base class for alert destinations.
    """
    
    @abstractmethod
    def send(self, alert: Alert) -> None:
        """Send the alert to this channel."""
        pass


class LogAlertChannel(AlertChannel):
    """
    Default channel that routes alerts to Python logging.
    """
    
    def send(self, alert: Alert) -> None:
        msg = f"ALERT [{alert.name}]: {alert.message} | Context: {alert.context}"
        if alert.level == AlertLevel.CRITICAL:
            logger.critical(msg)
        elif alert.level == AlertLevel.WARNING:
            logger.warning(msg)
        else:
            logger.info(msg)


class AlertManager:
    """
    Central manager for alert routing and suppression.
    
    Features:
    - Multiple channels (logs, email, slack, etc.)
    - Alert suppression (throttling duplicate alerts)
    """
    
    def __init__(self, suppression_interval_sec: float = 60.0):
        self.channels: List[AlertChannel] = [LogAlertChannel()]
        self._suppression_interval = suppression_interval_sec
        self._last_alert_times: Dict[str, float] = defaultdict(float)
        self._lock = threading.Lock()
        
    def add_channel(self, channel: AlertChannel) -> None:
        """Register a new alert channel."""
        with self._lock:
            self.channels.append(channel)
            
    def trigger(
        self, 
        name: str, 
        message: str, 
        level: AlertLevel = AlertLevel.WARNING,
        context: Optional[Dict[str, str]] = None,
        force: bool = False
    ) -> bool:
        """
        Trigger an alert.
        
        Args:
            name: Unique identifier for alert type (deduplication key)
            message: Human-readable message
            level: Severity level
            context: Key-value pairs for additional context
            force: If True, bypass suppression logic
            
        Returns:
            True if alert was sent, False if suppressed
        """
        now = time.time()
        
        with self._lock:
            if not force:
                last_time = self._last_alert_times.get(name, 0.0)
                if now - last_time < self._suppression_interval:
                    # Suppress execution, but maybe log debug?
                    return False
            
            self._last_alert_times[name] = now
            
        alert = Alert(
            name=name,
            message=message,
            level=level,
            timestamp=now,
            context=context or {}
        )
        
        for channel in self.channels:
            try:
                channel.send(alert)
            except Exception as e:
                logger.error(f"Failed to send alert to channel {channel}: {e}")
                
        return True

    def reset_suppression(self, name: str) -> None:
        """Reset suppression timer for a specific alert."""
        with self._lock:
            if name in self._last_alert_times:
                del self._last_alert_times[name]

# Global singleton for easy access (initialized in live.py)
_global_alert_manager: Optional[AlertManager] = None

def get_alert_manager() -> AlertManager:
    """Get or create default alert manager."""
    global _global_alert_manager
    if _global_alert_manager is None:
        _global_alert_manager = AlertManager()
    return _global_alert_manager
