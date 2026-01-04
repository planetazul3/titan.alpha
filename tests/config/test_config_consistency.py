
import pytest
import os
from unittest.mock import patch, MagicMock
from pydantic import ValidationError

# We import Settings inside tests to allow Env updates to take effect (if using pydantic_settings with reload)
# But ConfigDict frozen=True means we can't mutate.
# So we instantiate Settings(arg=...) for testing validators.

from config.settings import Settings, Thresholds, ShadowTradeConfig, DataShapes
from config.constants import MIN_SEQUENCE_LENGTH

class TestConfigConsistency:
    
    def test_settings_load_defaults(self):
        """Verify settings load with reasonable defaults."""
        # We must provide required fields that don't have defaults
        kwargs = {
            "trading": {"symbol": "R_100", "stake_amount": 10},
            "thresholds": {"confidence_threshold_high": 0.8, "learning_threshold_min": 0.4, "learning_threshold_max": 0.6},
            "hyperparams": {"learning_rate": 0.001, "batch_size": 32, "lstm_hidden_size": 16, "cnn_filters": 8, "latent_dim": 4},
            "data_shapes": {"sequence_length_ticks": 60, "sequence_length_candles": 60},
            "deriv_api_token": "test_token"
        }
        
        settings = Settings(_env_file=None, **kwargs) 
        
        # Now verify DEFAULTS for other fields
        # Note: BaseSettings usually looks at os.environ.
        # We assume defaults in class definition are valid.
        assert settings.trading.timeframe == "1m" # Default from Trading model
        assert settings.execution_safety.max_trades_per_minute_per_symbol <= settings.execution_safety.max_trades_per_minute

    def test_threshold_logic_validation(self):
        """Verify threshold logic validator."""
        # Valid case
        Thresholds(
            confidence_threshold_high=0.8,
            learning_threshold_min=0.4,
            learning_threshold_max=0.6
        )
        
        # Invalid: max > high
        with pytest.raises(ValidationError):
            Thresholds(
                confidence_threshold_high=0.7,
                learning_threshold_min=0.4,
                learning_threshold_max=0.8 # Invalid > 0.7
            )
            
        # Invalid: min > max
        with pytest.raises(ValidationError):
            Thresholds(
                confidence_threshold_high=0.9,
                learning_threshold_min=0.6,
                learning_threshold_max=0.5 # Invalid < min
            )

    def test_data_shapes_constants_consistency(self):
        """Verify data shapes respect constants."""
        # MIN_SEQUENCE_LENGTH is imported from constants
        
        # Valid
        DataShapes(
            sequence_length_ticks=MIN_SEQUENCE_LENGTH,
            sequence_length_candles=MIN_SEQUENCE_LENGTH
        )
        
        # Invalid (too short)
        with pytest.raises(ValidationError):
            DataShapes(
                sequence_length_ticks=MIN_SEQUENCE_LENGTH - 1,
                sequence_length_candles=MIN_SEQUENCE_LENGTH
            )

    def test_shadow_tracking_consistency_warning(self, caplog):
        """Verify warning if shadow tracking threshold > learning min."""
        # If we only track shadows > 0.6, but learning zone starts at 0.4,
        # signals between 0.4 and 0.6 are "learning" but not tracked?
        # Actually logic says: learning zone is min..max.
        # If shadow tracking min > learning min, we lose some learning signals.
        
        kwargs = {
            "trading": {"symbol": "R_100", "stake_amount": 10},
            "environment": "development",
            "deriv_api_token": "test_token",
            "thresholds": {
                "confidence_threshold_high": 0.8,
                "learning_threshold_min": 0.4,
                "learning_threshold_max": 0.6
            },
            "shadow_trade": {
                "min_probability_track": 0.5 # > 0.4, should warn
            },
            "hyperparams": {
                "learning_rate": 0.001, "batch_size": 32, "lstm_hidden_size": 64, 
                "cnn_filters": 32, "latent_dim": 16
            },
            "data_shapes": {
                "sequence_length_ticks": 60, "sequence_length_candles": 60
            }
        }
        
        with patch.dict(os.environ, {}, clear=True):
             settings = Settings(**kwargs)
             
        # Check logs for warning
        assert "Configuration Warning" in caplog.text
        assert "shadow_trade.min_probability_track" in caplog.text

    def test_security_production_token_in_test_env(self):
        """Verify CRITICAL security check: no prod tokens in test env."""
        pass 
        # Actually this is hard to test safely without mocking the validator explicitly 
        # or passing a dummy 'prod' token.
        # Let's try passing a dummy token that looks like strict prod token if we know the format?
        # The validator checks: "prod" in token_val.lower()
        
        kwargs = {
            "trading": {"symbol": "R_100", "stake_amount": 10},
            "environment": "development", # TEST MODE
            "deriv_api_token": "my_production_token_123", # Contains 'production'
            "thresholds": {
                "confidence_threshold_high": 0.8, "learning_threshold_min": 0.4, "learning_threshold_max": 0.6
            },
            "hyperparams": {
                "learning_rate": 0.001, "batch_size": 32, "lstm_hidden_size": 64, 
                "cnn_filters": 32, "latent_dim": 16
            },
            "data_shapes": {
                "sequence_length_ticks": 60, "sequence_length_candles": 60
            }
        }

        with pytest.raises(RuntimeError, match="SECURITY EXCEPTION"):
             Settings(**kwargs)

