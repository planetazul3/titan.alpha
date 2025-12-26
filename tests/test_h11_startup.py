import asyncio
import unittest
from datetime import datetime, timezone
from data.buffer import MarketDataBuffer
from data.events import CandleEvent

class TestStartupSequence(unittest.IsolatedAsyncioTestCase):
    async def test_tick_buffering_sequence(self):
        """Verify the subscribe-then-fetch pattern for ticks."""
        buffer = MarketDataBuffer(tick_length=20, candle_length=10)
        startup_buffer_ticks = []
        startup_complete = asyncio.Event()
        
        # Simulate Live Stream Source
        # Emits ticks 0, 1, 2, 3, 4...
        async def mock_tick_stream():
            for i in range(10):
                tick_price = float(i)
                if not startup_complete.is_set():
                    startup_buffer_ticks.append(tick_price)
                else:
                    buffer.append_tick(tick_price)
                await asyncio.sleep(0.01)

        # Start stream task (it will buffer initially)
        stream_task = asyncio.create_task(mock_tick_stream())
        
        # Simulate History Fetch Latency (Network delay)
        # Should allow some ticks to accumulate in buffer
        await asyncio.sleep(0.035) 
        
        # Mock History Data (ticks -5 to -1)
        history_ticks = [-5.0, -4.0, -3.0, -2.0, -1.0]
        
        # "Load" History
        buffer.append_ticks(history_ticks)
        
        # FLUSH Startup Buffer
        # Copy buffered ticks to main buffer
        for t in startup_buffer_ticks:
            buffer.append_tick(t)
        startup_buffer_ticks.clear()
        
        # Enable Live Mode
        startup_complete.set()
        
        # Wait for stream to finish
        await stream_task
        
        # Verify Buffer Contents
        # Should contain: History + StartupBuffered + Live
        # History: -5, -4, -3, -2, -1
        # Stream: 0, 1, 2, 3, 4, 5, 6, 7, 8, 9
        expected = [-5.0, -4.0, -3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]
        actual = list(buffer.get_ticks())
        
        # Verify integrity (no gaps, no disorder)
        self.assertEqual(actual, expected)
        # Verify we actually tested buffering
        # If startup_buffer wasn't used, logic is trivial. 
        # But we can't easily check startup_buffer size here as it's cleared.
        # But the order proves flush happened after history load.

    async def test_candle_deduplication(self):
        """Verify candle buffering with deduplication for overlapping candles."""
        buffer = MarketDataBuffer(tick_length=20, candle_length=10)
        startup_buffer_candles = []
        startup_complete = asyncio.Event()
        
        # Current time
        now = datetime.now(timezone.utc)
        ts = now.timestamp()
        
        # Mock Candle Stream
        # Emits:
        # 1. Forming candle (Update 1)
        # 2. Forming candle (Update 2) -> This corresponds to the LAST candle in history
        # 3. New candle (Closed)
        async def mock_candle_stream():
            events = [
                # Candle at TS (Forming) - Update 1
                CandleEvent(symbol="TEST", open=100, high=105, low=100, close=102, volume=0, timestamp=now, metadata={"epoch": ts}),
                # Candle at TS (Forming) - Update 2 (Final close for this period)
                CandleEvent(symbol="TEST", open=100, high=105, low=99, close=104, volume=0, timestamp=now, metadata={"epoch": ts}),
                # Next Candle (New Period)
                CandleEvent(symbol="TEST", open=104, high=106, low=104, close=105, volume=0, timestamp=datetime.fromtimestamp(ts + 60, timezone.utc), metadata={"epoch": ts+60}),
            ]
            
            for event in events:
                if not startup_complete.is_set():
                    startup_buffer_candles.append(event)
                else:
                    buffer.update_candle(event)
                await asyncio.sleep(0.01)

        stream_task = asyncio.create_task(mock_candle_stream())
        
        await asyncio.sleep(0.015)
        
        # Mock History
        # History INCLUDES the candle at TS (but maybe a slightly earlier version, or the final one)
        # Let's say history gives us the candle at TS as "closed" (or forming).
        # In Deriv API, history gives OHLC. 
        # If we fetch history, we get the last N candles.
        # Startup buffer captured updates for the Nth candle.
        
        # Case: History returns candle at TS.
        # Buffer sees: History[TS].
        # Startup Buffer sees: Stream[TS, TS, TS+60].
        # Flush should update History[TS] with Stream[TS] (if meaningful) or just handle it.
        # MarketDataBuffer.update_candle checks timestamp.
        # If abs(ts - last_ts) < 1.0, it updates in place.
        
        hist_candle = [100.0, 105.0, 100.0, 101.0, 0.0, ts] # Slightly different close (101)
        prev_candle = [90.0, 95.0, 90.0, 92.0, 0.0, ts - 60]
        
        buffer.preload_candles([prev_candle, hist_candle])
        
        # buffer candles: [Prev, Hist(Close=101)]
        
        # Flush
        for c in startup_buffer_candles:
            buffer.update_candle(c)
        startup_buffer_candles.clear()
        startup_complete.set()
        
        await stream_task
        
        # Verify
        candles = buffer.get_candles_array(include_forming=True)
        # Should have:
        # 1. Prev Candle (TS-60)
        # 2. Current Candle (TS) -> Updated to final close 104
        # 3. Next Candle (TS+60) -> 105
        
        self.assertEqual(len(candles), 3)
        self.assertEqual(candles[0][5], ts - 60)
        self.assertEqual(candles[1][5], ts)
        self.assertEqual(candles[1][3], 104.0) # Should be updated from 101 to 104
        self.assertEqual(candles[2][5], ts + 60)

if __name__ == "__main__":
    unittest.main()
