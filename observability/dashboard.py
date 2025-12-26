"""
Monitoring Dashboard Module.

Provides comprehensive system health monitoring, alerting,
and auto-response capabilities for production trading.

Key components:
- SystemHealthMonitor: Aggregates health from all subsystems
- AlertManager: Handles alert routing and deduplication
- AutoResponder: Automatic response to common issues
- DashboardEndpoint: HTTP-ready health endpoints

ARCHITECTURAL PRINCIPLE:
The dashboard provides a single pane of glass for system health.
All components report their status here, and the dashboard
can trigger automatic responses (e.g., pause trading) based
on configured rules.

Example:
    >>> from observability.dashboard import SystemHealthMonitor
    >>> monitor = SystemHealthMonitor()
    >>> health = monitor.get_system_health()
    >>> if not health.is_healthy:
    ...     for alert in health.active_alerts:
    ...         handle_alert(alert)
"""

import logging
import time
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from enum import Enum
from typing import Any, Callable, cast

logger = logging.getLogger(__name__)


# R05: Rate Limiter for Dashboard API
class RateLimiter:
    """
    R05 Fix: Simple sliding window rate limiter.
    
    Tracks requests per IP address within a sliding window
    and blocks requests that exceed the limit.
    
    Attributes:
        requests_per_minute: Maximum requests per minute per IP
        window_seconds: Sliding window size in seconds
    """
    
    def __init__(self, requests_per_minute: int = 60, window_seconds: int = 60):
        """
        Initialize rate limiter.
        
        Args:
            requests_per_minute: Max requests per minute per IP
            window_seconds: Window size in seconds
        """
        self.limit = requests_per_minute
        self.window = window_seconds
        self._requests: dict[str, list[float]] = {}
    
    def is_allowed(self, client_ip: str) -> bool:
        """
        Check if request is allowed for the client IP.
        
        Args:
            client_ip: Client IP address
            
        Returns:
            True if allowed, False if rate limited
        """
        now = time.monotonic()
        
        # Initialize or clean old entries
        if client_ip not in self._requests:
            self._requests[client_ip] = []
        
        # Remove old entries outside window
        self._requests[client_ip] = [
            ts for ts in self._requests[client_ip] 
            if now - ts < self.window
        ]
        
        # Check limit
        if len(self._requests[client_ip]) >= self.limit:
            logger.warning(f"Rate limit exceeded for {client_ip}")
            return False
        
        # Record request
        self._requests[client_ip].append(now)
        return True
    
    def get_remaining(self, client_ip: str) -> int:
        """Get remaining requests for client IP."""
        if client_ip not in self._requests:
            return self.limit
        
        now = time.monotonic()
        valid = [ts for ts in self._requests[client_ip] if now - ts < self.window]
        return max(0, self.limit - len(valid))
    
    def cleanup(self) -> int:
        """Remove stale entries. Returns number cleaned."""
        now = time.monotonic()
        cleaned = 0
        empty_keys = []
        
        for ip, timestamps in self._requests.items():
            new_list = [ts for ts in timestamps if now - ts < self.window]
            cleaned += len(timestamps) - len(new_list)
            if not new_list:
                empty_keys.append(ip)
            else:
                self._requests[ip] = new_list
        
        for key in empty_keys:
            del self._requests[key]
        
        return cleaned


# Global rate limiter instance for dashboard API
dashboard_rate_limiter = RateLimiter(requests_per_minute=60)


class AlertSeverity(Enum):
    """Alert severity levels."""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class AlertStatus(Enum):
    """Alert status."""
    ACTIVE = "active"
    ACKNOWLEDGED = "acknowledged"
    RESOLVED = "resolved"


@dataclass
class Alert:
    """
    System alert.
    
    Attributes:
        alert_id: Unique identifier
        severity: Alert severity
        source: Component that raised the alert
        message: Human-readable message
        status: Current status
        created_at: When alert was created
        acknowledged_at: When alert was acknowledged
        resolved_at: When alert was resolved
        metadata: Additional context
    """
    alert_id: str
    severity: AlertSeverity
    source: str
    message: str
    status: AlertStatus = AlertStatus.ACTIVE
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    acknowledged_at: datetime | None = None
    resolved_at: datetime | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "alert_id": self.alert_id,
            "severity": self.severity.value,
            "source": self.source,
            "message": self.message,
            "status": self.status.value,
            "created_at": self.created_at.isoformat(),
            "acknowledged_at": self.acknowledged_at.isoformat() if self.acknowledged_at else None,
            "resolved_at": self.resolved_at.isoformat() if self.resolved_at else None,
            "metadata": self.metadata,
        }


@dataclass
class ComponentHealth:
    """
    Health status of a single component.
    
    Attributes:
        name: Component name
        healthy: Is component healthy
        status: Status message
        last_check: When last checked
        metrics: Component-specific metrics
    """
    name: str
    healthy: bool
    status: str
    last_check: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    metrics: dict[str, float] = field(default_factory=dict)
    
    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "name": self.name,
            "healthy": self.healthy,
            "status": self.status,
            "last_check": self.last_check.isoformat(),
            "metrics": self.metrics,
        }


@dataclass
class SystemHealth:
    """
    Overall system health status.
    
    Attributes:
        is_healthy: Overall health status
        components: Health of each component
        active_alerts: Currently active alerts
        timestamp: When health was checked
    """
    is_healthy: bool
    components: dict[str, ComponentHealth]
    active_alerts: list[Alert]
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    
    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "is_healthy": self.is_healthy,
            "components": {k: v.to_dict() for k, v in self.components.items()},
            "active_alerts": [a.to_dict() for a in self.active_alerts],
            "timestamp": self.timestamp.isoformat(),
        }


class AlertManager:
    """
    Manages alerts with deduplication and routing.
    
    Features:
    - Alert deduplication by source+message
    - Automatic expiration of old alerts
    - Alert acknowledgment and resolution
    - Alert history
    """
    
    def __init__(
        self,
        dedup_window_seconds: int = 300,
        max_active_alerts: int = 100,
    ):
        """
        Initialize alert manager.
        
        Args:
            dedup_window_seconds: Window for deduplication
            max_active_alerts: Maximum active alerts to track
        """
        self.dedup_window = timedelta(seconds=dedup_window_seconds)
        self.max_active = max_active_alerts
        
        self._alerts: dict[str, Alert] = {}
        self._alert_counter = 0
        self._history: deque = deque(maxlen=1000)
        self._handlers: dict[AlertSeverity, list[Callable]] = {
            s: [] for s in AlertSeverity
        }
    
    def raise_alert(
        self,
        severity: AlertSeverity,
        source: str,
        message: str,
        metadata: dict[str, Any] | None = None,
    ) -> Alert | None:
        """
        Raise a new alert.
        
        Args:
            severity: Alert severity
            source: Component raising the alert
            message: Alert message
            metadata: Additional context
        
        Returns:
            Alert if created, None if deduplicated
        """
        # Check for deduplication
        dedup_key = f"{source}:{message}"
        now = datetime.now(timezone.utc)
        
        for alert in self._alerts.values():
            if alert.status == AlertStatus.ACTIVE:
                if f"{alert.source}:{alert.message}" == dedup_key:
                    if now - alert.created_at < self.dedup_window:
                        return None  # Deduplicated
        
        # Create new alert
        self._alert_counter += 1
        alert = Alert(
            alert_id=f"alert_{self._alert_counter}",
            severity=severity,
            source=source,
            message=message,
            metadata=metadata or {},
        )
        
        self._alerts[alert.alert_id] = alert
        
        # Trigger handlers
        for handler in self._handlers.get(severity, []):
            try:
                handler(alert)
            except Exception as e:
                logger.error(f"Alert handler failed: {e}")
        
        logger.warning(
            f"Alert raised: [{severity.value}] {source}: {message}"
        )
        
        return alert
    
    def acknowledge(self, alert_id: str) -> bool:
        """Acknowledge an alert."""
        if alert_id in self._alerts:
            alert = self._alerts[alert_id]
            alert.status = AlertStatus.ACKNOWLEDGED
            alert.acknowledged_at = datetime.now(timezone.utc)
            return True
        return False
    
    def resolve(self, alert_id: str) -> bool:
        """Resolve an alert."""
        if alert_id in self._alerts:
            alert = self._alerts[alert_id]
            alert.status = AlertStatus.RESOLVED
            alert.resolved_at = datetime.now(timezone.utc)
            self._history.append(alert)
            del self._alerts[alert_id]
            return True
        return False
    
    def resolve_by_source(self, source: str) -> int:
        """Resolve all alerts from a source."""
        resolved = 0
        for alert_id in list(self._alerts.keys()):
            if self._alerts[alert_id].source == source:
                self.resolve(alert_id)
                resolved += 1
        return resolved
    
    def get_active_alerts(
        self,
        severity: AlertSeverity | None = None,
    ) -> list[Alert]:
        """Get active alerts, optionally filtered by severity."""
        alerts = [
            a for a in self._alerts.values()
            if a.status == AlertStatus.ACTIVE
        ]
        if severity:
            alerts = [a for a in alerts if a.severity == severity]
        return sorted(alerts, key=lambda a: a.created_at, reverse=True)
    
    def register_handler(
        self,
        severity: AlertSeverity,
        handler: Callable[[Alert], Any],
    ) -> None:
        """Register alert handler for severity level."""
        self._handlers[severity].append(handler)


class AutoResponder:
    """
    Automatic response to system issues.
    
    Implements configurable rules that trigger actions
    based on alert patterns.
    
    Example rules:
    - Pause trading on CRITICAL alerts
    - Reduce risk on ERROR alerts
    - Send notification on WARNING
    """
    
    def __init__(
        self,
        pause_on_critical: bool = True,
        reduce_risk_on_error: bool = True,
    ):
        """
        Initialize auto-responder.
        
        Args:
            pause_on_critical: Pause trading on critical alerts
            reduce_risk_on_error: Reduce risk on error alerts
        """
        self.pause_on_critical = pause_on_critical
        self.reduce_risk_on_error = reduce_risk_on_error
        
        self._actions_taken: deque = deque(maxlen=100)
        self._pause_callback: Callable | None = None
        self._risk_callback: Callable[[float], None] | None = None
    
    def set_pause_callback(self, callback: Callable) -> None:
        """Set callback to pause trading."""
        self._pause_callback = callback
    
    def set_risk_callback(self, callback: Callable[[float], None]) -> None:
        """Set callback to adjust risk (takes scale factor)."""
        self._risk_callback = callback
    
    def handle_alert(self, alert: Alert) -> dict[str, Any] | None:
        """
        Handle alert with automatic response.
        
        Args:
            alert: Alert to handle
        
        Returns:
            Action taken, if any
        """
        action = None
        
        if alert.severity == AlertSeverity.CRITICAL and self.pause_on_critical:
            if self._pause_callback:
                self._pause_callback()
            action = {
                "type": "pause_trading",
                "reason": alert.message,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }
            logger.warning(f"Auto-response: Pausing trading due to {alert.message}")
        
        elif alert.severity == AlertSeverity.ERROR and self.reduce_risk_on_error:
            if self._risk_callback:
                self._risk_callback(0.5)  # Reduce to 50%
            action = cast(dict[str, Any], {
                "type": "reduce_risk",
                "scale": 0.5,
                "reason": alert.message,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            })
            logger.warning(f"Auto-response: Reducing risk due to {alert.message}")
        
        if action:
            self._actions_taken.append(action)
        
        return action
    
    def get_recent_actions(self, limit: int = 10) -> list[dict]:
        """Get recent auto-response actions."""
        return list(self._actions_taken)[-limit:]


class SystemHealthMonitor:
    """
    Central system health monitoring.
    
    Aggregates health from all subsystems and provides
    a unified view for dashboards and alerting.
    
    Example:
        >>> monitor = SystemHealthMonitor()
        >>> monitor.register_component("model", model_health_checker)
        >>> monitor.register_component("executor", executor_health_checker)
        >>> health = monitor.get_system_health()
    """
    
    def __init__(
        self,
        check_interval_seconds: int = 60,
    ):
        """
        Initialize system health monitor.
        
        Args:
            check_interval_seconds: Interval between automatic checks
        """
        self.check_interval = timedelta(seconds=check_interval_seconds)
        
        self._components: dict[str, Callable[[], ComponentHealth]] = {}
        self._last_check: datetime | None = None
        self._cached_health: SystemHealth | None = None
        
        self.alerts = AlertManager()
        self.auto_responder = AutoResponder()
        
        # Register auto-responder as alert handler
        for severity in [AlertSeverity.CRITICAL, AlertSeverity.ERROR]:
            self.alerts.register_handler(severity, self.auto_responder.handle_alert)
        
        logger.info("SystemHealthMonitor initialized")
    
    def register_component(
        self,
        name: str,
        health_checker: Callable[[], ComponentHealth],
    ) -> None:
        """
        Register a component for monitoring.
        
        Args:
            name: Component name
            health_checker: Function that returns component health
        """
        self._components[name] = health_checker
        logger.info(f"Registered health checker: {name}")
    
    def check_component(self, name: str) -> ComponentHealth | None:
        """Check health of a specific component."""
        if name not in self._components:
            return None
        
        try:
            return self._components[name]()
        except Exception as e:
            logger.error(f"Health check failed for {name}: {e}")
            return ComponentHealth(
                name=name,
                healthy=False,
                status=f"Health check error: {e}",
            )
    
    def get_system_health(self, force_refresh: bool = False) -> SystemHealth:
        """
        Get overall system health.
        
        Args:
            force_refresh: Force fresh check even if cached
        
        Returns:
            SystemHealth with all component statuses
        """
        now = datetime.now(timezone.utc)
        
        # Use cached if recent
        if not force_refresh and self._cached_health and self._last_check:
            if now - self._last_check < self.check_interval:
                return self._cached_health
        
        # Check all components
        components = {}
        for name in self._components:
            comp_health = self.check_component(name)
            if comp_health:
                components[name] = comp_health
                
                # Raise alerts for unhealthy components
                if not comp_health.healthy:
                    self.alerts.raise_alert(
                        severity=AlertSeverity.ERROR,
                        source=name,
                        message=comp_health.status,
                        metadata=comp_health.metrics,
                    )
        
        # Calculate overall health
        is_healthy = all(c.healthy for c in components.values())
        
        # Get active alerts
        active_alerts = self.alerts.get_active_alerts()
        
        # If any critical alerts, system is unhealthy
        if any(a.severity == AlertSeverity.CRITICAL for a in active_alerts):
            is_healthy = False
        
        health = SystemHealth(
            is_healthy=is_healthy,
            components=components,
            active_alerts=active_alerts,
        )
        
        self._cached_health = health
        self._last_check = now
        
        return health
    
    def get_dashboard_data(self) -> dict[str, Any]:
        """
        Get data for dashboard display.
        
        Returns:
            Dictionary with all dashboard data
        """
        health = self.get_system_health()
        
        return {
            "status": "healthy" if health.is_healthy else "unhealthy",
            "timestamp": health.timestamp.isoformat(),
            "components": {
                name: {
                    "healthy": comp.healthy,
                    "status": comp.status,
                    "last_check": comp.last_check.isoformat(),
                }
                for name, comp in health.components.items()
            },
            "alerts": {
                "active_count": len(health.active_alerts),
                "by_severity": {
                    s.value: len([a for a in health.active_alerts if a.severity == s])
                    for s in AlertSeverity
                },
                "recent": [a.to_dict() for a in health.active_alerts[:5]],
            },
            "auto_responses": self.auto_responder.get_recent_actions(5),
        }
    
    def health_check_endpoint(self) -> dict[str, Any]:
        """
        R07 Fix: Minimal health check endpoint for orchestration systems.
        
        Returns a lightweight response suitable for load balancers,
        Kubernetes probes, or monitoring systems.
        
        Returns:
            Dictionary with status, healthy boolean, and timestamp
        """
        health = self.get_system_health()
        
        return {
            "status": "ok" if health.is_healthy else "degraded",
            "healthy": health.is_healthy,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "components_healthy": sum(1 for c in health.components.values() if c.healthy),
            "components_total": len(health.components),
            "active_alerts": len(health.active_alerts),
        }


def create_model_health_checker(model_monitor: Any) -> Callable[[], ComponentHealth]:
    """Factory for model health checker."""
    def checker() -> ComponentHealth:
        if model_monitor is None:
            return ComponentHealth(
                name="model",
                healthy=True,
                status="Not monitored",
            )
        
        result = model_monitor.check_health()
        return ComponentHealth(
            name="model",
            healthy=result.is_healthy,
            status=result.recommendation,
            metrics=result.metrics,
        )
    return checker


def create_executor_health_checker(executor: Any) -> Callable[[], ComponentHealth]:
    """Factory for executor health checker."""
    def checker() -> ComponentHealth:
        if executor is None:
            return ComponentHealth(
                name="executor",
                healthy=True,
                status="Not connected",
            )
        
        # Check if executor is operational
        try:
            is_healthy = not getattr(executor, "_kill_switch_active", False)
            status = "Operational" if is_healthy else "Kill switch active"
            
            return ComponentHealth(
                name="executor",
                healthy=is_healthy,
                status=status,
            )
        except Exception as e:
            return ComponentHealth(
                name="executor",
                healthy=False,
                status=str(e),
            )
    return checker
