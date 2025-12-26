
import threading
import time
import pytest
from execution.position_sizer import CompoundingPositionSizer

class TestSizerConcurrency:

    def test_compounding_thread_safety(self):
        """Test that CompoundingPositionSizer is thread-safe under concurrent access."""
        # Set max wins higher than total trades to prevent resetting during test
        sizer = CompoundingPositionSizer(base_stake=10.0, max_consecutive_wins=2000)
        
        # Shared state to track expected value
        # We will simulate N wins concurrently
        n_threads = 20
        trades_per_thread = 50
        
        exceptions = []
        
        def worker():
            try:
                for _ in range(trades_per_thread):
                    # Simulate read
                    _ = sizer.compute_stake(probability=0.8)
                    # Simulate write (win)
                    # We pass fixed P&L 
                    sizer.record_outcome(pnl=10.0, won=True)
                    time.sleep(0.0001) # Small delay to force context switches
            except Exception as e:
                exceptions.append(e)

        threads = [threading.Thread(target=worker) for _ in range(n_threads)]
        
        for t in threads:
            t.start()
            
        for t in threads:
            t.join()
            
        assert not exceptions, f"Exceptions occurred: {exceptions}"
        
        # Expected streak should be n_threads * trades_per_thread
        expected_streak = n_threads * trades_per_thread
        assert sizer.get_current_streak() == expected_streak, \
            f"Streak mismatch: expected {expected_streak}, got {sizer.get_current_streak()}"
