"""
Windowed Percentile Tracker for normalizing streaming metrics.
"""

from collections import deque
import numpy as np

class WindowedPercentileTracker:
    """
    Tracks a sliding window of values to compute dynamic percentiles.
    
    Used for normalizing unbounded metrics like reconstruction error,
    making thresholds robust to regime shifts and scale changes.
    """
    def __init__(self, window_size: int = 1000):
        self.window_size = window_size
        self.history = deque(maxlen=window_size)
        self.sorted_window: list[float] = []
        self._dirty = False

    def update(self, value: float) -> float:
        """
        Add new value and return its percentile rank in the current window.
        
        Args:
            value: The new metric value.
            
        Returns:
            Percentile rank (0.0 to 100.0). Returns 50.0 if window is empty.
        """
        if not np.isfinite(value):
            return 100.0 # Treat NaN/Inf as extreme anomaly
            
        self.history.append(value)
        self._dirty = True
        
        return self.get_percentile_rank(value)

    def get_percentile_rank(self, value: float) -> float:
        """Calculate percentile rank of a value against the history."""
        if not self.history:
            return 50.0
            
        if self._dirty:
            self.sorted_window = sorted(self.history)
            self._dirty = False
            
        N = len(self.sorted_window)
        if N <= 1:
            return 50.0
            
        # Optimization
        if value > self.sorted_window[-1]:
            return 100.0
        if value < self.sorted_window[0]:
            return 0.0
            
        import bisect
        # bisect_left gives count of elements strictly less than value
        count_less = bisect.bisect_left(self.sorted_window, value)
        
        # We want the percentile rank.
        # Use simple rank: (count_less / (N - 1)) * 100
        # If N=2, [0, 10]. Checking 0: 0/1 = 0%. Checking 10: 1/1 = 100%.
        # If N=3, [0, 5, 10]. Checking 5: 1/2 = 50%.
        return (count_less / (N - 1)) * 100.0
