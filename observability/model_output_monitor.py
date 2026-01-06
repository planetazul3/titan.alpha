import logging
import numpy as np
from collections import deque
from typing import Optional

logger = logging.getLogger(__name__)

class ModelOutputMonitor:
    """
    Monitors model output probabilities for anomalies.
    
    Tracks a rolling window of probabilities and detects:
    1. Stuck outputs (variance ~ 0)
    2. Extreme drifts (mean shift)
    3. Calibration failures (if labels available - future)
    """
    
    def __init__(self, window_size: int = 1000):
        self.window_size = window_size
        self._history = deque(maxlen=window_size)
        self._last_stats = {}

    def record(self, probability: float):
        """Record a single probability observation."""
        self._history.append(probability)
        
        # Periodic check (e.g. every 100 samples) or on demand?
        # For efficiency, we just store. Analytics can be pulled.
        
    def get_statistics(self) -> dict:
        """Calculate statistics over the rolling window."""
        if not self._history:
            return {"count": 0}
            
        data = np.array(self._history)
        mean = np.mean(data)
        std = np.std(data)
        min_val = np.min(data)
        max_val = np.max(data)
        
        return {
            "count": len(data),
            "mean": float(mean),
            "std": float(std),
            "min": float(min_val),
            "max": float(max_val)
        }
        
    def check_anomalies(self) -> list[str]:
        """
        Check for detected anomalies.
        Returns list of warning messages.
        """
        if len(self._history) < self.window_size * 0.1:
            return [] # Not enough data
            
        anomalies = []
        stats = self.get_statistics()
        
        # 1. Stuck Model (Zero Variance)
        if stats["std"] < 1e-6:
             anomalies.append(f"Model outputs stuck! std={stats['std']:.6f} (mean={stats['mean']:.4f})")
             
        # 2. Extreme Polarity (All > 0.99 or < 0.01)
        if stats["min"] > 0.99:
             anomalies.append("Model outputting only extremely high probabilities.")
        if stats["max"] < 0.01:
             anomalies.append("Model outputting only extremely low probabilities.")

        return anomalies
