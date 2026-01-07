import pytest
import torch
from execution.decision import DecisionEngine
from config.settings import load_settings

@pytest.fixture
def settings():
    """Create test settings with warmup disabled and kill switch off for boundary testing."""
    base_settings = load_settings()
    # Use model_copy for frozen model - need to create new nested objects
    updated_data_shapes = base_settings.data_shapes.model_copy(update={"warmup_steps": 0})
    # Disable kill switch for tests that verify signal generation
    updated_execution_safety = base_settings.execution_safety.model_copy(update={"kill_switch_enabled": False})
    return base_settings.model_copy(update={
        "data_shapes": updated_data_shapes,
        "execution_safety": updated_execution_safety,
    })

@pytest.fixture
def decision_engine(settings):
    return DecisionEngine(settings)

@pytest.mark.asyncio
async def test_decision_boundaries_binary_rise(decision_engine):
    """Test decision boundaries for Rise/Fall contracts."""
    # Scenario 1: Just below threshold
    # Default confidence_threshold_high is usually 0.80 based on .env
    probs = {"rise_fall_prob": 0.79}
    signals = await decision_engine.process_model_output(probs, reconstruction_error=0.1)
    assert len(signals) == 0
    
    # Scenario 2: Above threshold
    probs = {"rise_fall_prob": 0.81}
    signals = await decision_engine.process_model_output(probs, reconstruction_error=0.1)
    assert len(signals) == 1
    assert signals[0].direction == "CALL"
    assert signals[0].probability == 0.81

@pytest.mark.asyncio
async def test_decision_boundaries_veto(decision_engine):
    """Test that vetos correctly block decisions even if threshold is met."""
    probs = {"rise_fall_prob": 0.90}
    
    # Normal case (low reconstruction error)
    signals = await decision_engine.process_model_output(probs, reconstruction_error=0.1)
    assert len(signals) == 1
    
    # Veto case (high reconstruction error)
    # Default veto threshold is 0.3 or similar
    signals = await decision_engine.process_model_output(probs, reconstruction_error=2.0)
    assert len(signals) == 0

@pytest.mark.asyncio
async def test_extreme_probabilities(decision_engine):
    """Test engine behavior with extreme or degenerate probabilities."""
    # Prob 0.5 is ambiguous for Rise/Fall (CALL vs PUT)
    probs = {"rise_fall_prob": 0.5}
    signals = await decision_engine.process_model_output(probs, reconstruction_error=0.1)
    # Based on filters.py: if prob_call >= prob_put -> CALL
    # Prob Call 0.5, Prob Put 0.5 -> Should result in IGNORE if threshold is 0.8
    assert len(signals) == 0
    
    # High certainty
    probs = {"rise_fall_prob": 1.0}
    signals = await decision_engine.process_model_output(probs, reconstruction_error=0.1)
    assert len(signals) == 1
    assert signals[0].direction == "CALL"
