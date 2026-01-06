
import asyncio
import unittest
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from datetime import datetime, timezone
from data.buffer import MarketDataBuffer
from data.events import CandleEvent
from execution.synchronizer import StartupSynchronizer

class TestStartupSynchronizer(unittest.TestCase):
    def setUp(self):
        self.buffer = MarketDataBuffer(tick_length=20, candle_length=10)
        self.synchronizer = StartupSynchronizer(self.buffer)

    def test_tick_buffering_logic(self):
        """Verify buffering sequence and atomic switch."""
        # 1. Simulate Live Stream (Buffering)
        live_ticks = [10.0, 11.0, 12.0]
        for t in live_ticks:
            buffered = self.synchronizer.handle_tick(t)
            self.assertTrue(buffered)
            
        # Verify nothing in main buffer yet
        self.assertEqual(len(self.buffer.get_ticks()), 0)
        
        # 2. Simulate History Fetch
        history_ticks = [1.0, 2.0, 3.0]
        history_candles = []
        
        # 3. Finalize
        self.synchronizer.finalize_startup(history_ticks, history_candles)
        
        # Verify Content: History + Live
        expected = [1.0, 2.0, 3.0, 10.0, 11.0, 12.0]
        actual = list(self.buffer.get_ticks())
        self.assertEqual(actual, expected)
        
        # 4. Verify Switch to Live
        self.assertTrue(self.synchronizer.is_live())
        
        # Next tick should NOT be buffered inside synchronizer (returns False)
        # It's up to the caller to append to buffer
        next_tick = 13.0
        buffered = self.synchronizer.handle_tick(next_tick)
        self.assertFalse(buffered)

    def test_candle_buffering(self):
        """Verify candle buffering."""
        now = datetime.now(timezone.utc)
        
        # Live Candles
        c1 = CandleEvent("S", 100, 105, 95, 101, 10, now, {})
        buffered = self.synchronizer.handle_candle(c1)
        self.assertTrue(buffered)
        
        # History
        hist_candles = [{
            "symbol": "S", "epoch": now.timestamp() - 60, 
            "open": 90, "high": 95, "low": 85, "close": 92
        }]
        
        self.synchronizer.finalize_startup([], hist_candles)
        
        # Verify Buffer
        candles = self.buffer.get_candles_array(include_forming=False)
        # Should have history + live
        # Note: update_candle might merge if timestamps align, but here they differ by 60s
        self.assertEqual(len(candles), 2)
        # Check close prices
        self.assertEqual(candles[0][3], 92.0)
        self.assertEqual(candles[1][3], 101.0)

if __name__ == "__main__":
    unittest.main()
