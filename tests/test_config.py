"""
Unit tests for config module.

Tests configuration loading, validation, and constants.
"""

import pytest

from config.constants import (
    CONTRACT_TYPES,
    DEFAULT_SEED,
    MAX_SEQUENCE_LENGTH,
    MIN_SEQUENCE_LENGTH,
    SIGNAL_TYPES,
)
from config.settings import DataShapes, ModelHyperparams, Thresholds


class TestConstants:
    """Tests for config constants."""

    def test_contract_types_exist(self):
        """Verify all contract types are defined."""
        assert hasattr(CONTRACT_TYPES, "RISE_FALL")
        assert hasattr(CONTRACT_TYPES, "TOUCH_NO_TOUCH")
        assert hasattr(CONTRACT_TYPES, "STAYS_BETWEEN")

    def test_signal_types_exist(self):
        """Verify all signal types are defined."""
        assert hasattr(SIGNAL_TYPES, "REAL_TRADE")
        assert hasattr(SIGNAL_TYPES, "SHADOW_TRADE")
        assert hasattr(SIGNAL_TYPES, "IGNORE")

    def test_sequence_length_valid(self):
        """Verify sequence length constraints are valid."""
        assert MIN_SEQUENCE_LENGTH > 0
        assert MAX_SEQUENCE_LENGTH > MIN_SEQUENCE_LENGTH
        assert DEFAULT_SEED >= 0


class TestThresholds:
    """Tests for threshold validation."""

    def test_valid_thresholds(self):
        """Valid thresholds should not raise."""
        thresholds = Thresholds(
            confidence_threshold_high=0.75, learning_threshold_min=0.40, learning_threshold_max=0.60
        )
        assert thresholds.confidence_threshold_high == 0.75

    def test_invalid_threshold_order(self):
        """Thresholds in wrong order should raise ValueError."""
        # Update regex to match Pydantic's actual output or be more generic
        with pytest.raises(ValueError):
            Thresholds(
                confidence_threshold_high=0.50,  # Lower than learning_max
                learning_threshold_min=0.40,
                learning_threshold_max=0.60,
            )

    def test_threshold_boundary_values(self):
        """Test boundary conditions for thresholds."""
        # Exactly at boundaries should work
        thresholds = Thresholds(
            confidence_threshold_high=1.0, learning_threshold_min=0.0, learning_threshold_max=0.5
        )
        assert thresholds is not None


class TestModelHyperparams:
    """Tests for model hyperparameter validation."""

    def test_valid_hyperparams(self):
        """Valid hyperparams should not raise."""
        hp = ModelHyperparams(
            learning_rate=0.001,
            batch_size=32,
            lstm_hidden_size=64,
            cnn_filters=32,
            latent_dim=16,
            dropout_rate=0.1,
        )
        assert hp.learning_rate == 0.001

    def test_negative_learning_rate_raises(self):
        """Negative learning rate should raise error."""
        with pytest.raises(ValueError):
            ModelHyperparams(
                learning_rate=-0.001,
                batch_size=32,
                lstm_hidden_size=64,
                cnn_filters=32,
                latent_dim=16,
            )

    def test_invalid_dropout_raises(self):
        """Dropout outside [0, 1) should raise error."""
        with pytest.raises(ValueError):
            ModelHyperparams(
                learning_rate=0.001,
                batch_size=32,
                lstm_hidden_size=64,
                cnn_filters=32,
                latent_dim=16,
                dropout_rate=1.5,
            )


class TestDataShapes:
    """Tests for data shape validation."""

    def test_valid_shapes(self):
        """Valid data shapes should not raise."""
        shapes = DataShapes(sequence_length_ticks=100, sequence_length_candles=50)
        assert shapes.sequence_length_ticks == 100

    def test_below_minimum_raises(self):
        """Sequence length below MIN should raise."""
        with pytest.raises(ValueError):
            DataShapes(
                sequence_length_ticks=5,  # Below MIN_SEQUENCE_LENGTH
                sequence_length_candles=50,
            )
