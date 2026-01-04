
import pytest
from unittest.mock import MagicMock
import time
from observability.alerting import AlertManager, AlertLevel, AlertChannel, Alert

class MockChannel(AlertChannel):
    def __init__(self):
        self.alerts = []
        
    def send(self, alert: Alert) -> None:
        self.alerts.append(alert)

class TestAlertManager:
    def test_alert_triggering(self):
        """Test basic alert triggering."""
        manager = AlertManager()
        channel = MockChannel()
        manager.add_channel(channel)
        
        manager.trigger("test_alert", "Something happened", AlertLevel.WARNING)
        
        assert len(channel.alerts) == 1
        assert channel.alerts[0].name == "test_alert"
        assert channel.alerts[0].level == AlertLevel.WARNING
        assert channel.alerts[0].message == "Something happened"

    def test_suppression(self):
        """Test that duplicate alerts are suppressed."""
        manager = AlertManager(suppression_interval_sec=1.0)
        channel = MockChannel()
        manager.add_channel(channel)
        
        # First alert - should pass
        triggered1 = manager.trigger("duplicate_alert", "First", AlertLevel.INFO)
        assert triggered1 is True
        assert len(channel.alerts) == 1
        
        # Immediate duplicate - should be suppressed
        triggered2 = manager.trigger("duplicate_alert", "Second", AlertLevel.INFO)
        assert triggered2 is False
        assert len(channel.alerts) == 1
        
        # After delay - should pass
        time.sleep(1.1)
        triggered3 = manager.trigger("duplicate_alert", "Third", AlertLevel.INFO)
        assert triggered3 is True
        assert len(channel.alerts) == 2
        
    def test_force_override(self):
        """Test that force=True overrides suppression."""
        manager = AlertManager(suppression_interval_sec=60.0)
        channel = MockChannel()
        manager.add_channel(channel)
        
        manager.trigger("forced_alert", "First", AlertLevel.CRITICAL)
        
        # Force triggered duplicate
        triggered = manager.trigger("forced_alert", "Second", AlertLevel.CRITICAL, force=True)
        assert triggered is True
        assert len(channel.alerts) == 2

    def test_alert_level_filtering(self):
        """Test that levels are correctly propagated."""
        # Note: Filtering usually happens in the Channel, but Manager passes it through.
        manager = AlertManager()
        channel = MockChannel()
        manager.add_channel(channel)
        
        manager.trigger("info", "Info", AlertLevel.INFO)
        manager.trigger("crit", "Crit", AlertLevel.CRITICAL)
        
        assert channel.alerts[0].level == AlertLevel.INFO
        assert channel.alerts[1].level == AlertLevel.CRITICAL
