
import pytest
import numpy as np
from datetime import datetime, timezone
from execution.shadow_store import ShadowTradeRecord
from execution.outcome_resolver import OutcomeResolver, ResolutionConfig

class TestBarrierResolution:
    def test_resolve_touch_with_specific_barrier(self):
        """
        Verify that resolution uses the specific barrier_level from the trade record
        instead of the default percentage from config.
        """
        # Setup
        entry_price = 100.0
        # Default barrier is 0.5% = 0.5
        config = ResolutionConfig(touch_barrier_percent=0.005) 
        resolver = OutcomeResolver(config)
        
        # Scenario: Price moves 0.8 units away.
        # This is > 0.5 (default) but < 1.0 (specific barrier).
        # So "NO_TOUCH" should WIN if we respect the specific barrier (1.0).
        # It would LOSE if we used the default (0.5).
        
        ticks = np.array([100.0, 100.2, 100.8, 100.5])
        
        # Case 1: Trade with specific barrier_level = 1.0
        # TIMING: The record uses datetime.now() by default. We need to align it with our mock ticks.
        # Mock ticks are at [1, 2, 3, 4].
        base_ts = datetime.fromtimestamp(1.0, tz=timezone.utc)
        
        trade_with_barrier = ShadowTradeRecord.create(
            contract_type="TOUCH_NO_TOUCH",
            direction="NO_TOUCH",
            probability=0.8,
            entry_price=entry_price,
            reconstruction_error=0.1,
            regime_state="TRUSTED",
            tick_window=np.array([]),
            candle_window=np.array([]),
            barrier_level=1.0  # Explicit barrier
        )
        # Manually align timestamp
        from dataclasses import replace
        trade_with_barrier = replace(trade_with_barrier, timestamp=base_ts)
        
        # Should be a WIN (NO_TOUCH) because max deviation 0.8 < 1.0
        outcome, _ = resolver._resolve_single(trade_with_barrier, ticks, np.array([1, 2, 3, 4]))
        assert outcome is True, "Trade should be WIN (NO_TOUCH) with barrier=1.0"

    def test_resolve_touch_fallback_to_config(self):
        """Verify fallback to config percentage when barrier_level is None."""
        entry_price = 100.0
        config = ResolutionConfig(touch_barrier_percent=0.005) # 0.5
        resolver = OutcomeResolver(config)
        
        # Deviation 0.8 > 0.5
        ticks = np.array([100.0, 100.8])
        
        # Deviation 0.8 > 0.5
        ticks = np.array([100.0, 100.8])
        base_ts = datetime.fromtimestamp(1.0, tz=timezone.utc)
        
        trade_no_barrier = ShadowTradeRecord.create(
            contract_type="TOUCH_NO_TOUCH",
            direction="NO_TOUCH",
            probability=0.8,
            entry_price=entry_price,
            reconstruction_error=0.1,
            regime_state="TRUSTED",
            tick_window=np.array([]),
            candle_window=np.array([]),
            barrier_level=None
        )
        from dataclasses import replace
        trade_no_barrier = replace(trade_no_barrier, timestamp=base_ts)
        
        # Should be a LOSS (NO_TOUCH failed) because 0.8 > 0.5
        outcome, _ = resolver._resolve_single(trade_no_barrier, ticks, np.array([1, 2]))
        assert outcome is False, "Trade should be LOSS (TOUCHED) with default barrier=0.5"

if __name__ == "__main__":
    pytest.main([__file__])
