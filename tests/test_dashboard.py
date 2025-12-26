"""
Unit tests for monitoring dashboard.
"""

from datetime import datetime, timezone

import pytest

from observability.dashboard import (
    Alert,
    AlertManager,
    AlertSeverity,
    AlertStatus,
    AutoResponder,
    ComponentHealth,
    SystemHealth,
    SystemHealthMonitor,
)


class TestAlert:
    """Tests for Alert dataclass."""

    def test_creation(self):
        """Test alert creation."""
        alert = Alert(
            alert_id="test_1",
            severity=AlertSeverity.WARNING,
            source="test",
            message="Test message",
        )
        
        assert alert.status == AlertStatus.ACTIVE
        assert alert.acknowledged_at is None

    def test_to_dict(self):
        """Test serialization."""
        alert = Alert(
            alert_id="test_1",
            severity=AlertSeverity.ERROR,
            source="model",
            message="Model degraded",
        )
        
        d = alert.to_dict()
        
        assert d["alert_id"] == "test_1"
        assert d["severity"] == "error"
        assert d["status"] == "active"


class TestComponentHealth:
    """Tests for ComponentHealth."""

    def test_to_dict(self):
        """Test serialization."""
        health = ComponentHealth(
            name="executor",
            healthy=True,
            status="Operational",
            metrics={"latency_ms": 10.5},
        )
        
        d = health.to_dict()
        
        assert d["name"] == "executor"
        assert d["healthy"] is True


class TestAlertManager:
    """Tests for AlertManager."""

    def test_raise_alert(self):
        """Test raising alert."""
        manager = AlertManager()
        
        alert = manager.raise_alert(
            AlertSeverity.WARNING,
            "test",
            "Test warning",
        )
        
        assert alert is not None
        assert alert.severity == AlertSeverity.WARNING

    def test_deduplication(self):
        """Test alert deduplication."""
        manager = AlertManager(dedup_window_seconds=60)
        
        alert1 = manager.raise_alert(AlertSeverity.WARNING, "test", "Same message")
        alert2 = manager.raise_alert(AlertSeverity.WARNING, "test", "Same message")
        
        assert alert1 is not None
        assert alert2 is None  # Deduplicated

    def test_acknowledge(self):
        """Test alert acknowledgment."""
        manager = AlertManager()
        
        alert = manager.raise_alert(AlertSeverity.ERROR, "test", "Error")
        result = manager.acknowledge(alert.alert_id)
        
        assert result is True
        assert alert.status == AlertStatus.ACKNOWLEDGED

    def test_resolve(self):
        """Test alert resolution."""
        manager = AlertManager()
        
        alert = manager.raise_alert(AlertSeverity.ERROR, "test", "Error")
        result = manager.resolve(alert.alert_id)
        
        assert result is True
        assert len(manager.get_active_alerts()) == 0

    def test_get_active_alerts(self):
        """Test getting active alerts."""
        manager = AlertManager()
        
        manager.raise_alert(AlertSeverity.INFO, "a", "Info")
        manager.raise_alert(AlertSeverity.WARNING, "b", "Warn")
        manager.raise_alert(AlertSeverity.ERROR, "c", "Error")
        
        all_active = manager.get_active_alerts()
        warnings = manager.get_active_alerts(AlertSeverity.WARNING)
        
        assert len(all_active) == 3
        assert len(warnings) == 1

    def test_register_handler(self):
        """Test alert handler registration."""
        manager = AlertManager()
        handled = []
        
        manager.register_handler(
            AlertSeverity.CRITICAL,
            lambda a: handled.append(a),
        )
        
        manager.raise_alert(AlertSeverity.CRITICAL, "test", "Critical!")
        
        assert len(handled) == 1


class TestAutoResponder:
    """Tests for AutoResponder."""

    def test_handle_critical(self):
        """Test auto-response to critical alert."""
        responder = AutoResponder(pause_on_critical=True)
        paused = []
        
        responder.set_pause_callback(lambda: paused.append(True))
        
        alert = Alert(
            alert_id="1",
            severity=AlertSeverity.CRITICAL,
            source="test",
            message="Critical error",
        )
        
        action = responder.handle_alert(alert)
        
        assert action is not None
        assert action["type"] == "pause_trading"
        assert len(paused) == 1

    def test_handle_error(self):
        """Test auto-response to error alert."""
        responder = AutoResponder(reduce_risk_on_error=True)
        risk_scale = []
        
        responder.set_risk_callback(lambda s: risk_scale.append(s))
        
        alert = Alert(
            alert_id="1",
            severity=AlertSeverity.ERROR,
            source="test",
            message="Error occurred",
        )
        
        action = responder.handle_alert(alert)
        
        assert action is not None
        assert action["type"] == "reduce_risk"
        assert risk_scale == [0.5]

    def test_get_recent_actions(self):
        """Test getting recent actions."""
        responder = AutoResponder()
        
        responder.handle_alert(Alert(
            alert_id="1",
            severity=AlertSeverity.CRITICAL,
            source="test",
            message="Critical",
        ))
        
        actions = responder.get_recent_actions()
        
        assert len(actions) == 1


class TestSystemHealthMonitor:
    """Tests for SystemHealthMonitor."""

    def test_initialization(self):
        """Test initialization."""
        monitor = SystemHealthMonitor()
        
        assert monitor.alerts is not None
        assert monitor.auto_responder is not None

    def test_register_component(self):
        """Test component registration."""
        monitor = SystemHealthMonitor()
        
        def checker():
            return ComponentHealth("test", True, "OK")
        
        monitor.register_component("test", checker)
        
        health = monitor.check_component("test")
        assert health.healthy is True

    def test_get_system_health(self):
        """Test system health aggregation."""
        monitor = SystemHealthMonitor()
        
        monitor.register_component(
            "healthy",
            lambda: ComponentHealth("healthy", True, "OK"),
        )
        monitor.register_component(
            "unhealthy",
            lambda: ComponentHealth("unhealthy", False, "Error"),
        )
        
        health = monitor.get_system_health()
        
        assert health.is_healthy is False  # One unhealthy component
        assert len(health.components) == 2

    def test_get_dashboard_data(self):
        """Test dashboard data generation."""
        monitor = SystemHealthMonitor()
        
        monitor.register_component(
            "test",
            lambda: ComponentHealth("test", True, "Operational"),
        )
        
        data = monitor.get_dashboard_data()
        
        assert "status" in data
        assert "components" in data
        assert "alerts" in data

    def test_unhealthy_component_raises_alert(self):
        """Test unhealthy component auto-raises alert."""
        monitor = SystemHealthMonitor()
        
        monitor.register_component(
            "failing",
            lambda: ComponentHealth("failing", False, "Connection lost"),
        )
        
        monitor.get_system_health(force_refresh=True)
        
        alerts = monitor.alerts.get_active_alerts()
        assert len(alerts) > 0
        assert any("failing" in a.source for a in alerts)


class TestSystemHealth:
    """Tests for SystemHealth."""

    def test_to_dict(self):
        """Test serialization."""
        health = SystemHealth(
            is_healthy=True,
            components={"a": ComponentHealth("a", True, "OK")},
            active_alerts=[],
        )
        
        d = health.to_dict()
        
        assert d["is_healthy"] is True
        assert "a" in d["components"]
