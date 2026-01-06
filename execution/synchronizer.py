"""
Startup Synchronization Manager.

Handles the critical transition from "Buffering" to "Live" mode during system startup.
Ensures no race conditions between:
1. Historical data loading (main thread/coroutine)
2. Live data arrival (background tasks)
3. Buffer population

This module fixes Audit Issue #2 (Startup Fragility) and enables fixing Issue #1 (Control Flow).
"""

import asyncio
import logging
from typing import Any, List, Optional
from dataclasses import dataclass, field
from data.buffer import MarketDataBuffer
from data.events import CandleEvent

logger = logging.getLogger(__name__)

class StartupSynchronizer:
    """
    Manages the startup synchronization phase.
    
    State Machine:
    1. INITIALIZING (Default): Live events are buffered in internal lists.
    2. HISTORY_LOADED: History has been pushed to the main buffer.
    3. FLUSHING: Buffered live events are being replayed.
    4. LIVE: Direct passthrough to the main buffer.
    """
    
    def __init__(self, buffer: MarketDataBuffer):
        self.buffer = buffer
        self._buffering_active = True
        self._live_active = False
        
        # Internal buffers for data arriving during history fetch
        self._buffered_ticks: List[float] = []
        self._buffered_candles: List[CandleEvent] = []
        
        # Metrics
        self.stats = {
            "buffered_ticks_count": 0,
            "buffered_candles_count": 0,
            "history_ticks_count": 0,
            "history_candles_count": 0
        }

    def handle_tick(self, price: float) -> bool:
        """
        Handle a live tick event.
        
        Returns:
            bool: True if event was buffered (stop processing), 
                  False if event should be processed normally (live mode).
        """
        if self._buffering_active:
            self._buffered_ticks.append(price)
            self.stats["buffered_ticks_count"] += 1
            return True
        return False

    def handle_candle(self, candle: CandleEvent) -> bool:
        """
        Handle a live candle event.
        
        Returns:
            bool: True if event was buffered (stop processing), 
                  False if event should be processed normally (live mode).
        """
        if self._buffering_active:
            self._buffered_candles.append(candle)
            self.stats["buffered_candles_count"] += 1
            return True
        return False

    def is_live(self) -> bool:
        """Check if completely in live mode."""
        return self._live_active

    def finalize_startup(self, history_ticks: List[float], history_candles: List[dict]):
        """
        Execute the atomic transition from Buffering to Live.
        
        Args:
            history_ticks: List of historical tick prices
            history_candles: List of historical candle dicts (from API)
        """
        logger.info("[SYNC] Starting atomic startup synchronization...")
        
        # 1. Populate History (Synchronous Logic)
        self.stats["history_ticks_count"] = len(history_ticks)
        self.stats["history_candles_count"] = len(history_candles)
        
        for price in history_ticks:
            self.buffer.append_tick(price)
            
        # Convert and append historical candles
        from datetime import datetime, timezone
        for c in history_candles:
            try:
                # Handle both dicts and objects if necessary, assuming dict from API
                ts = c.get("epoch")
                if ts:
                    timestamp = datetime.fromtimestamp(ts, tz=timezone.utc)
                else:
                    # Fallback or error
                    timestamp = datetime.now(timezone.utc)
                    
                ce = CandleEvent(
                    symbol=c.get("symbol", "UNKNOWN"),
                    open=float(c["open"]),
                    high=float(c["high"]),
                    low=float(c["low"]),
                    close=float(c["close"]),
                    volume=0.0,
                    timestamp=timestamp,
                    metadata={"source": "history"}
                )
                self.buffer.update_candle(ce)
            except Exception as e:
                logger.error(f"[SYNC] Failed to parse historical candle: {e}")

        logger.info(f"[SYNC] History populated. Buffered {len(self._buffered_ticks)} ticks, {len(self._buffered_candles)} candles during fetch.")

        # 2. Atomic Switch
        # We capture the current buffer state and disable buffering.
        # Since this method runs in the main coroutine, and handle_tick/handle_candle
        # run in the same event loop (single threaded), this switch is safe 
        # as long as we don't await in the critical section.
        
        captured_ticks = list(self._buffered_ticks)
        captured_candles = list(self._buffered_candles)
        
        # CLEAR internal buffers to release memory and reset state
        self._buffered_ticks = []
        self._buffered_candles = []
        
        # DISABLE buffering flag - next events will return False in handle_*
        self._buffering_active = False
        self._live_active = True
        
        # 3. Replay Buffered Events
        logger.info("[SYNC] Replaying buffered live events...")
        
        for t in captured_ticks:
            self.buffer.append_tick(t)
            
        for c in captured_candles:
            self.buffer.update_candle(c)
            
        logger.info("[SYNC] Synchronization complete. System is LIVE.")
