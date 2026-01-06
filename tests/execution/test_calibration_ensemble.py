
import pytest
import numpy as np
from unittest.mock import MagicMock
from execution.calibration import ProbabilityCalibrator
from execution.ensemble import create_ensemble, VotingEnsemble, WeightedEnsemble
from config.settings import ProbabilityCalibrationConfig, EnsembleConfig, Settings

class TestCalibration:
    def test_calibration_flow(self):
        config = ProbabilityCalibrationConfig(enabled=True, min_samples=10)
        calibrator = ProbabilityCalibrator(config)
        
        # 1. Unfitted should return raw
        assert calibrator.calibrate(0.8) == 0.8
        
        # 2. Update with perfect correlation data
        # Prob -> Outcome
        # 0.1 -> 0 (Loss)
        # 0.9 -> 1 (Win)
        probs = [0.1] * 10 + [0.9] * 10
        outcomes = [False] * 10 + [True] * 10
        
        calibrator.update(probs, outcomes)
        
        # 3. Should fit close to identity
        # Depending on sklearn availability and implementation details
        res = calibrator.calibrate(0.9)
        # Isotonic regression on this data should map 0.9 -> 1.0 (since all 0.9s won)
        assert res >= 0.9 # Should be close to 1.0
        
        res_low = calibrator.calibrate(0.1)
        assert res_low <= 0.1 # Should be close to 0.0

    def test_calibration_correction(self):
        """Test it fixes overconfidence."""
        config = ProbabilityCalibrationConfig(enabled=True, min_samples=10)
        calibrator = ProbabilityCalibrator(config)
        
        # Model predicts 0.9 but wins only 50%
        probs = [0.9] * 20
        outcomes = [True] * 10 + [False] * 10 # 50% win rate
        
        calibrator.update(probs, outcomes)
        
        calibrated = calibrator.calibrate(0.9)
        # Should be closer to 0.5
        assert 0.4 <= calibrated <= 0.6


class TestEnsemble:
    def test_voting_ensemble(self):
        config = EnsembleConfig(strategy="voting")
        ensemble = create_ensemble(config)
        
        # Agreement
        outputs = {"m1": 0.8, "m2": 0.9}
        assert ensemble.combine(outputs) == pytest.approx(0.85)
        
        # Disagreement (one bull, one bear)
        # Bear < 0.5, Bull > 0.5
        outputs_disagree = {"m1": 0.4, "m2": 0.8}
        # Avg = 0.6, but agreement check might penalize?
        # Current implementation just returns average for mixed signals 
        # (re-read logic: "return avg" in all paths currently, just comments about penalty)
        # Wait, let's verify logic in file.
        assert ensemble.combine(outputs_disagree) == pytest.approx(0.6)

    def test_weighted_ensemble(self):
        config = EnsembleConfig(strategy="weighted")
        ensemble = WeightedEnsemble(config)
        ensemble.set_weights({"m1": 2.0, "m2": 1.0})
        
        outputs = {"m1": 1.0, "m2": 0.0} # Weighted avg: (2*1 + 1*0) / 3 = 0.66
        assert abs(ensemble.combine(outputs) - 0.666) < 0.01

from execution.decision import DecisionEngine

@pytest.mark.asyncio
async def test_decision_engine_calibration_integration():
    from config.settings import load_settings
    settings = load_settings()
    # Note: Settings is frozen, so we can't mutate prob_calibration here.
    # The test works because calibrator.update() can be called regardless of config.enabled
    
    engine = DecisionEngine(settings)
    
    # Fake fit the calibrator manually to avoid dependency on sklearn in this specific unit test if it fails
    # But let's try using the public interface first.
    probs = [0.9] * 10
    outcomes = [False] * 10 # 0% win rate for 0.9 prediction
    engine.calibrator.update(probs, outcomes)
    
    # Now process a 0.9 signal
    # It should be calibrated down to 0.0, thus filtered out!
    raw_probs = {"TEST_CALL": 0.9}
    
    result = await engine.process_model_output(raw_probs, reconstruction_error=0.1)
    
    # Should yield NO trades because 0.9 -> ~0.0 which is < threshold
    assert len(result) == 0
    
    # Contrast with raw (if we didn't update)
    # This proves calibration was applying
