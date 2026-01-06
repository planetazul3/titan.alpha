import pytest
import math
from execution.position_sizer import KellyPositionSizer
from utils.numerical_validation import validate_probability, validate_stake_amount

def test_validate_probability():
    assert validate_probability(0.5) == 0.5
    assert validate_probability(1.5) == 1.0
    assert validate_probability(-0.1) == 0.0
    assert validate_probability(float('nan')) == 0.0
    assert validate_probability(float('inf')) == 0.0

def test_validate_stake():
    assert validate_stake_amount(10.0) == 10.0
    assert validate_stake_amount(-5.0) == 0.0
    assert validate_stake_amount(2000.0, max_stake=1000.0) == 1000.0
    assert validate_stake_amount(float('nan')) == 0.0

def test_kelly_sizer_safety():
    sizer = KellyPositionSizer(base_stake=10.0, max_stake=100.0)
    
    # NaN probability
    res = sizer.compute_stake(probability=float('nan'))
    assert res.stake == 0.0
    
    # Inf probability
    res = sizer.compute_stake(probability=float('inf'))
    assert res.stake == 0.0 # clamped to 0.0 if logic treats 0 prob as no trade
    
    # Negative Probability (should be clamped 0)
    res = sizer.compute_stake(probability=-0.5)
    assert res.stake == 0.0
    
    # Excessive stake check
    # High prob + High payout (fake) -> huge kelly
    # But sizer caps at max_stake
    res = sizer.compute_stake(probability=0.99, payout_ratio=100.0, account_balance=1000000)
    assert res.stake <= 100.0
