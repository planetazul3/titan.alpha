
import pytest
from datetime import datetime
from execution.metrics import TradingMetrics

def test_metrics_calculation_basic():
    trades = [
        {"outcome": "win", "profit_loss": 9.0, "exit_time": datetime.now()},
        {"outcome": "loss", "profit_loss": -10.0, "exit_time": datetime.now()},
        {"outcome": "win", "profit_loss": 9.0, "exit_time": datetime.now()},
    ]
    
    metrics = TradingMetrics.calculate(trades, initial_balance=100.0)
    
    assert metrics.total_trades == 3
    assert metrics.winning_trades == 2
    assert metrics.losing_trades == 1
    assert metrics.win_rate == pytest.approx(66.666, 0.01)
    assert metrics.net_profit == 8.0 # 9 - 10 + 9
    assert metrics.final_balance == 108.0
    assert metrics.initial_balance == 100.0

def test_metrics_empty():
    metrics = TradingMetrics.calculate([], initial_balance=100.0)
    assert metrics.total_trades == 0
    assert metrics.net_profit == 0.0
    assert metrics.initial_balance == 100.0
    assert metrics.final_balance == 100.0

def test_drawdown_calculation():
    trades = [
        {"outcome": "win", "profit_loss": 10.0, "exit_time": datetime.now()}, # Bal 110 (Peak 110)
        {"outcome": "loss", "profit_loss": -20.0, "exit_time": datetime.now()}, # Bal 90 (Peak 110, DD 20)
        {"outcome": "loss", "profit_loss": -10.0, "exit_time": datetime.now()}, # Bal 80 (Peak 110, DD 30)
        {"outcome": "win", "profit_loss": 50.0, "exit_time": datetime.now()}, # Bal 130 (Peak 130)
    ]
    
    metrics = TradingMetrics.calculate(trades, initial_balance=100.0)
    # Max DD was at Balance 80 from Peak 110.
    # DD amount = 30.
    # DD pct = 30 / 110 = 27.27%
    
    assert metrics.max_drawdown == 30.0
    assert metrics.max_drawdown_pct == pytest.approx(27.27, 0.01)
    assert metrics.final_balance == 130.0
