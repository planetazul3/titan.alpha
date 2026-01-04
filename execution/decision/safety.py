import logging
import time
from typing import Optional

from execution.safety_store import SQLiteSafetyStateStore
from utils.numerical_validation import ensure_finite

logger = logging.getLogger(__name__)

class SafetyStateSynchronizer:
    """
    Synchronizes safety state (P&L) from the safety store.
    """
    def __init__(self, safety_store: Optional[SQLiteSafetyStateStore]):
        self.safety_store = safety_store
        self._last_safety_sync: float = 0.0
        self._safety_sync_interval: float = 5.0  # seconds
        self._current_daily_pnl: float = 0.0

    async def sync(self, force: bool = False) -> float:
        """
        Sync current P&L from SafetyStateStore.
        Returns the current daily P&L.
        """
        if not self.safety_store:
            return 0.0
            
        now = time.monotonic()
        
        if not force and (now - self._last_safety_sync) < self._safety_sync_interval:
            return self._current_daily_pnl
            
        try:
            _, daily_pnl = await self.safety_store.get_daily_stats_async()
            self._current_daily_pnl = ensure_finite(
                daily_pnl, 
                "SafetyStateSynchronizer.sync", 
                default=0.0
            ) 
            self._last_safety_sync = now
        except Exception as e:
            logger.error(f"Failed to sync safety state: {e}")
            
        return self._current_daily_pnl

    def get_current_pnl(self) -> float:
        return self._current_daily_pnl
