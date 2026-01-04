import pytest
import numpy as np
from execution.regime.algorithms.volatility import VolatilityRegimeDetector
from execution.regime.types import VolatilityRegime

def test_h4_warmup_veto():
    """
    Verify H4 Warmup Veto:
    If buffer length < min_warmup, VolatilityRegimeDetector should return HIGH (Veto).
    """
    min_warmup = 120
    detector = VolatilityRegimeDetector(min_warmup=min_warmup)
    
    # 1. Short buffer (Insufficient Data)
    short_prices = np.random.rand(50) + 100
    regime, _ = detector.detect(short_prices)
    assert regime == VolatilityRegime.HIGH, "Should veto (HIGH) when history < min_warmup"

    # 2. Borderline Short
    borderline_prices = np.random.rand(119) + 100
    regime, _ = detector.detect(borderline_prices)
    assert regime == VolatilityRegime.HIGH, "Should veto (HIGH) when history 119 < 120"
    
    # 3. Sufficient Data
    long_prices = np.random.rand(150) + 100
    # Add minimal variation to avoid NaN/Inf but verify it returns standard regime (not purely veto logic)
    # The random data implies high volatility probably, or random. 
    # We just check it executes logic.
    regime, score = detector.detect(long_prices)
    # It could be LOW/MEDIUM/HIGH based on data, but let's ensure it doesn't crash
    # and "detect" ran (so returns valid enum)
    assert isinstance(regime, VolatilityRegime)
    assert 0 <= score <= 100
