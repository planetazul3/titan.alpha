import pytest
import math
from execution.position_sizer import KellyPositionSizer, PositionSizeResult

@pytest.fixture
def kelly_sizer():
    return KellyPositionSizer(base_stake=10.0)

def test_kelly_hardens_nan_probability(kelly_sizer):
    # Pass NaN probability
    res = kelly_sizer.compute_stake(probability=float('nan'))
    
    # Should default to 0.0 prob -> Kelly <= 0 -> stake 0
    assert res.stake == 0.0
    assert "Negative edge" in res.reason or "Below minimum" in res.reason

def test_kelly_hardens_inf_balance(kelly_sizer):
    # Pass Inf balance
    res = kelly_sizer.compute_stake(probability=0.7, account_balance=float('inf'))
    
    # Should treat balance as 0.0 (default for bad input) -> stake 0
    # Or handled gracefully
    assert res.stake == 0.0
    assert "Insufficient" in res.reason or "Below minimum" in res.reason

def test_kelly_hardens_nan_drawdown(kelly_sizer):
    # Pass NaN drawdown
    res = kelly_sizer.compute_stake(probability=0.7, current_drawdown=float('nan'))
    
    # NaN drawdown -> 0.0 -> Normal sizing
    # 0.7 prob, 0.9 payout -> Kelly = (0.7*0.9 - 0.3)/0.9 = (0.63 - 0.3)/0.9 = 0.33/0.9 = 0.366
    # 0.5 safety -> 0.183
    # Base 10 -> Stake ~ 18.33
    assert res.stake > 0.0
    assert res.drawdown_multiplier == 1.0 # default behavior

def test_kelly_normal_operation(kelly_sizer):
    res = kelly_sizer.compute_stake(probability=0.7, current_drawdown=0.0)
    assert res.stake > 0.0
