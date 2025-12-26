
import pytest
from hypothesis import given, strategies as st
from datetime import datetime
import numpy as np

from execution.shadow_resolution import ShadowTradeResolver
from execution.shadow_store import ShadowTradeRecord, ShadowTradeStore
from config.constants import CONTRACT_TYPES

# Mock Store
class MockStore:
    def query(self, *args, **kwargs): return []
    def update_outcome(self, *args, **kwargs): pass
    def mark_stale(self, *args, **kwargs): pass

@st.composite
def price_paths(draw, start_price=100.0, length=10):
    """Generate a random price path."""
    changes = draw(st.lists(st.floats(min_value=-1.0, max_value=1.0), min_size=length, max_size=length))
    prices = [start_price]
    for change in changes:
        prices.append(prices[-1] + change)
    return prices

class TestShadowResolutionProperties:
    
    @given(st.floats(min_value=50.0, max_value=150.0), 
           st.floats(min_value=0.1, max_value=5.0),
           price_paths(start_price=100.0, length=20))
    def test_touch_no_touch_exclusivity(self, entry_price, barrier_offset, prices):
        """
        Property: For any price path, TOUCH and NO_TOUCH outcomes must be logical opposites
        (ignoring the None case for invalid data).
        
        Note: This tests the _determine_outcome LOGIC directly.
        """
        # Create a mock trade
        record = ShadowTradeRecord.create(
            contract_type=CONTRACT_TYPES.TOUCH_NO_TOUCH,
            direction="TOUCH",
            probability=0.8,
            entry_price=entry_price,
            reconstruction_error=0.0,
            regime_state="TRUSTED",
            tick_window=np.array([]),
            candle_window=np.array([]),
            barrier_level=barrier_offset,
        )
        
        # We simulate the resolution context by manually constructing "candles" from the price path
        # For simplicity, we'll treat the path as a series of high/lows
        highs = np.array(prices)
        lows = np.array(prices) # Simplification: H=L=Price
        
        # Populate resolution context manually
        context = []
        for p in prices:
            context.append([p, p, p]) # H, L, C
        
        record.resolution_context = context
        
        resolver = ShadowTradeResolver(MockStore())
        
        # Test TOUCH
        record_touch = record # direction is TOUCH
        outcome_touch = resolver._determine_outcome(
            record_touch, 
            exit_price=prices[-1], 
            high_price=prices[-1], # Current candle
            low_price=prices[-1]
        )
        
        # Test NO_TOUCH
        record_no_touch = ShadowTradeRecord.create(
            contract_type=CONTRACT_TYPES.TOUCH_NO_TOUCH,
            direction="NO_TOUCH",
            probability=0.8,
            entry_price=entry_price,
            reconstruction_error=0.0,
            regime_state="TRUSTED",
            tick_window=np.array([]),
            candle_window=np.array([]),
            barrier_level=barrier_offset,
        )
        record_no_touch.resolution_context = context
        
        outcome_no_touch = resolver._determine_outcome(
            record_no_touch, 
            exit_price=prices[-1], 
            high_price=prices[-1],
            low_price=prices[-1]
        )
        
        # Invariants
        if outcome_touch is not None and outcome_no_touch is not None:
            assert outcome_touch != outcome_no_touch, \
                f"TOUCH and NO_TOUCH cannot have same outcome for path {prices}"

    @given(st.floats(min_value=50.0, max_value=150.0), price_paths(length=5))
    def test_rise_fall_monotonicity(self, entry_price, prices):
        """
        Property: CALL should win if exit > entry, PUT should win if exit < entry.
        """
        exit_price = prices[-1]
        
        # Use exact floating point comparison logic from implementation
        win_call = exit_price > entry_price
        win_put = exit_price < entry_price
        
        record_call = ShadowTradeRecord.create(
            contract_type=CONTRACT_TYPES.RISE_FALL,
            direction="CALL",
            probability=0.8,
            entry_price=entry_price,
            reconstruction_error=0.0,
            regime_state="TRUSTED",
            tick_window=np.array([]),
            candle_window=np.array([])
        )
        
        resolver = ShadowTradeResolver(MockStore())
        outcome = resolver._determine_outcome(record_call, exit_price)
        
        assert outcome == win_call

