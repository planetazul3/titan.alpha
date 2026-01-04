import logging
from datetime import datetime, timezone
from typing import Any

from config.settings import Settings
from execution.signals import TradeSignal
from execution.calibration import ProbabilityCalibrator
from execution.filters import filter_signals

logger = logging.getLogger(__name__)

class SignalProcessor:
    """
    Handles probability calibration and signal filtering.
    """
    def __init__(self, settings: Settings, calibrator: ProbabilityCalibrator):
        self.settings = settings
        self.calibrator = calibrator

    def process(
        self, 
        probs: dict[str, float], 
        timestamp: datetime | None = None
    ) -> list[TradeSignal]:
        """
        Calibrate probabilities and filter into signals.
        """
        if timestamp is None:
            timestamp = datetime.now(timezone.utc)
            
        # R03: Calibrate Probabilities
        calibrated_probs = {}
        for contract, raw_prob in probs.items():
            calibrated_probs[contract] = self.calibrator.calibrate(raw_prob)

        # R02: Filter probabilities into signals
        all_signals = filter_signals(calibrated_probs, self.settings, timestamp)
        
        return all_signals
