from typing import Any

class DecisionMetrics:
    """
    Tracks statistics for the Decision Engine.
    """
    def __init__(self):
        self._stats: dict[str, int] = {
            "processed": 0,
            "real": 0,
            "shadow": 0,
            "ignored": 0,
            "regime_vetoed": 0,
            "regime_caution": 0,
        }

    def increment(self, metric: str, count: int = 1) -> None:
        if metric in self._stats:
            self._stats[metric] += count

    def get_mutable_stats(self) -> dict[str, int]:
        """Return reference to internal stats dict for legacy compatibility.
        
        Use increment() for normal operations. This method exists only for
        compatibility with process_signals_batch which mutates stats directly.
        """
        return self._stats

    def get_statistics(self) -> dict[str, Any]:
        return self._stats.copy()
