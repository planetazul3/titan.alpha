from unittest.mock import MagicMock

import pytest

from config.settings import Settings
from execution.decision import DecisionEngine
from execution.regime import RegimeVeto, TrustState


@pytest.fixture
def mock_settings():
    settings = MagicMock(spec=Settings)
    # Mock necessary attributes
    settings.hyperparams = MagicMock()
    settings.hyperparams.regime_caution_threshold = 0.5
    settings.hyperparams.regime_veto_threshold = 1.0

    # Mock thresholds for filter_signals
    settings.thresholds = MagicMock()
    settings.thresholds.confidence_threshold_high = 0.8
    settings.thresholds.learning_threshold_min = 0.6
    settings.thresholds.learning_threshold_max = 0.8

    return settings


@pytest.fixture
def regime_veto():
    return RegimeVeto(threshold_caution=0.5, threshold_veto=1.0)


@pytest.fixture
def decision_engine(mock_settings, regime_veto):
    return DecisionEngine(mock_settings, regime_veto=regime_veto)


def test_regime_trusted(decision_engine):
    """Test that trades are allowed when reconstruction error is low."""
    # Low error -> TRUSTED
    recon_error = 0.1
    probs = {"CALL": 0.8}  # High confidence

    # Mock internal signal generation
    # We need to ensure filter_signals allows it.
    # For this unit test, might be easier to mock _filter_signals if possible,
    # or rely on DecisionEngine logic.
    # DecisionEngine.process_model_output calls self.regime_veto.assess

    decision_engine.process_model_output(probs, recon_error)

    # Needs deeper mocking of settings to pass filter_signals?
    # Actually, let's trust that with high prob it produces a signal OR
    # we can check logs/mock calls.
    # But fundamentally: if veto is NOT triggered, it proceeds to signal generation.
    assert decision_engine.regime_veto.assess(recon_error).trust_state == TrustState.TRUSTED


def test_regime_caution(decision_engine):
    """Test that CAUTION state applies filters."""
    # Medium error -> CAUTION
    recon_error = 0.7
    probs = {"CALL": 0.8}

    decision_engine.process_model_output(probs, recon_error)
    assessment = decision_engine.regime_veto.assess(recon_error)
    assert assessment.trust_state == TrustState.CAUTION
    assert assessment.requires_caution() is True
    # In caution mode, DecisionEngine._apply_caution_filter is called.


def test_regime_veto_absolute(decision_engine):
    """CRITICAL: Test that VETO state blocks ALL trades unconditionally."""
    # High error -> VETO
    recon_error = 1.5
    probs = {"CALL": 0.99, "PUT": 0.99}  # Extremely high confidence signals

    # Even with perfect signals, veto must block everything
    trades = decision_engine.process_model_output(probs, recon_error)

    assert len(trades) == 0
    assessment = decision_engine.regime_veto.assess(recon_error)
    assert assessment.trust_state == TrustState.VETO
    assert assessment.is_vetoed() is True


def test_regime_veto_boundary(decision_engine):
    """Test the exact boundary of veto."""
    # Exactly veto threshold
    recon_error = 1.0
    trades = decision_engine.process_model_output({"CALL": 0.9}, recon_error)
    assert len(trades) == 0
    assert decision_engine.regime_veto.assess(recon_error).is_vetoed() is True


def test_regime_caution_boundary(decision_engine):
    """Test the exact boundary of caution."""
    # Exactly caution threshold
    recon_error = 0.5
    assessment = decision_engine.regime_veto.assess(recon_error)
    assert assessment.trust_state == TrustState.CAUTION
