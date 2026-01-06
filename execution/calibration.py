
import logging
import numpy as np
from typing import Literal, Optional, Any
from datetime import datetime
import threading

from config.settings import ProbabilityCalibrationConfig

logger = logging.getLogger(__name__)

class ProbabilityCalibrator:
    """
    Calibrates raw model probabilities to match realized win rates.
    
    Uses Isotonic Regression (non-parametric) to map model outputs to empirical probabilities.
    Ideally, if the model predicts 0.7, we should win 70% of the time.
    
    If raw model is overconfident (e.g. predicts 0.9 but wins 0.6), this calibrator
    will map 0.9 -> 0.6, preventing Kelly criterion from over-betting.
    """
    
    def __init__(self, config: ProbabilityCalibrationConfig):
        self.config = config
        self._regressor: Any = None
        self._last_update = datetime.min
        self._lock = threading.Lock()
        self._is_fitted = False
        
        # Fallback for simple binning if sklearn not available or for cold start
        self._bins = np.linspace(0, 1, 11) # 0.0, 0.1, ... 1.0
        self._bin_wins = np.zeros(10)
        self._bin_counts = np.zeros(10)
        

            
    def update(self, probabilities: list[float], outcomes: list[bool]) -> None:
        """
        Retrain the calibrator with new historical data.
        
        Args:
            probabilities: List of raw model predictions (0..1)
            outcomes: List of actual trade results (True=Win, False=Loss)
        """
        if not self.config.enabled:
            return
            
        if len(probabilities) < self.config.min_samples:
            logger.debug(f"Insufficient samples for calibration: {len(probabilities)} < {self.config.min_samples}")
            return

        with self._lock:
            try:
                # Lazy import to avoid hard dependency if not used
                from sklearn.isotonic import IsotonicRegression
                
                # Convert outcomes to 0/1 integers
                y = np.array([1 if x else 0 for x in outcomes])
                X = np.array(probabilities)
                
                # Fit Isotonic Regression
                # y_min=0, y_max=1 enforces validity
                iso = IsotonicRegression(y_min=0, y_max=1, out_of_bounds='clip')
                iso.fit(X, y)
                
                self._regressor = iso
                self._is_fitted = True
                self._last_update = datetime.now()
                
                logger.info(f"Probability Calibrator updated with {len(probabilities)} samples.")
                
            except ImportError:
                logger.warning("sklearn not found. Using simple binning calibration.")
                # Simple Binning Implementation
                # 1. Reset bins
                self._bin_wins.fill(0)
                self._bin_counts.fill(0)
                
                # 2. Populate bins
                for p, outcome in zip(probabilities, outcomes):
                    # Bin index 0..9
                    idx = min(9, int(p * 10))
                    self._bin_counts[idx] += 1
                    if outcome:
                        self._bin_wins[idx] += 1
                        
                self._is_fitted = True
                self._last_update = datetime.now()
                self._regressor = "BINNING"
                
            except Exception as e:
                logger.error(f"Failed to update calibrator: {e}")


    def calibrate(self, raw_prob: float) -> float:
        """
        Apply calibration to a raw probability score.
        """
        if not self.config.enabled:
            return raw_prob
            
        if not self._is_fitted:
            return raw_prob
            
        try:
            if self._regressor == "BINNING":
                # Linear interpolation between bin centers could be better,
                # but simple lookup is safer for fallback.
                idx = min(9, int(raw_prob * 10))
                count = self._bin_counts[idx]
                if count < 5:
                    # Not enough data for this bin, trust input (soft calibration)
                    # or bias slightly towards 0.5?
                    # Let's weighted average with raw prob
                    return raw_prob
                
                win_rate = self._bin_wins[idx] / count
                return float(win_rate)
            else:
                # Scikit-learn path
                calibrated = self._regressor.predict([raw_prob])[0]
                return float(max(0.0, min(1.0, calibrated)))

        except Exception as e:
            logger.warning(f"Calibration failed, using raw: {e}")
            return raw_prob

    def get_status(self) -> dict:
        return {
            "enabled": self.config.enabled,
            "fitted": self._is_fitted,
            "last_update": self._last_update.isoformat() if self._last_update != datetime.min else None,
            "method": self.config.method
        }
