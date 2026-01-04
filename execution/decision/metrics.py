from typing import Any

class DecisionMetrics:
    """
    Tracks statistics for the Decision Engine.
    """
    def __init__(self):
        self._stats = {
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

    def get_statistics(self) -> dict[str, Any]:
        return self._stats.copy()
