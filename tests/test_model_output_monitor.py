import pytest
from observability.model_output_monitor import ModelOutputMonitor

def test_monitor_stats():
    monitor = ModelOutputMonitor(window_size=10)
    for i in range(10):
        monitor.record(float(i)/10.0)
        
    stats = monitor.get_statistics()
    assert stats["count"] == 10
    assert 0.0 <= stats["min"] <= 1.0
    assert 0.0 <= stats["max"] <= 1.0

def test_monitor_detects_stuck_model():
    monitor = ModelOutputMonitor(window_size=10)
    for _ in range(20):
        monitor.record(0.5) # Stuck at 0.5
        
    anomalies = monitor.check_anomalies()
    assert any("stuck" in a.lower() for a in anomalies)

def test_monitor_detects_extreme_polarity():
    monitor = ModelOutputMonitor(window_size=10)
    for _ in range(20):
        monitor.record(0.999) 
        
    anomalies = monitor.check_anomalies()
    assert any("high probabilities" in a for a in anomalies)

def test_rolling_window_size():
    monitor = ModelOutputMonitor(window_size=5)
    for i in range(10):
        monitor.record(float(i))
        
    stats = monitor.get_statistics()
    assert stats["count"] == 5
    assert stats["max"] == 9.0
    assert stats["min"] == 5.0
