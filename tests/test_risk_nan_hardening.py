import pytest
import numpy as np
from execution.adaptive_risk import AdaptiveRiskManager

def test_risk_manager_nan_resilience():
    """Verify that NaN PnL doesn't break the risk manager."""
    risk_mgr = AdaptiveRiskManager(base_daily_limit=100.0)
    
    # Record some good trades
    risk_mgr.record_trade(10.0, 1010.0)
    risk_mgr.record_trade(20.0, 1030.0)
    
    # Record a NaN trade (simulates API error or calculation failure)
    risk_mgr.record_trade(float('nan'), 1030.0)
    
    stats = risk_mgr.get_statistics()
    print(f"Stats after NaN: {stats}")
    
    # If not resilient, sharpe_ratio or daily_pnl will be NaN
    assert not np.isnan(stats['daily_pnl']), "daily_pnl should not be NaN"
    assert not np.isnan(stats['sharpe_ratio']), "sharpe_ratio should not be NaN"
    
    # Adjusted limits should still be reasonable
    limits = risk_mgr.get_adjusted_limits()
    assert not np.isnan(limits.daily_loss_limit), "limits should not be NaN"

if __name__ == "__main__":
    test_risk_manager_nan_resilience()
