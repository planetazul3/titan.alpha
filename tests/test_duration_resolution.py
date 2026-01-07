import pytest
from execution.contract_params import ContractParameterService
from config.constants import CONTRACT_TYPES
from config.settings import Settings, ContractConfig, Trading, Thresholds, ModelHyperparams, ExecutionSafety

def create_mock_settings(timeframe="5m", rise_fall_duration=1, rise_fall_unit="m"):
    return Settings.model_construct(
        contracts=ContractConfig(
            duration_rise_fall=rise_fall_duration,
            duration_unit_rise_fall=rise_fall_unit
        ),
        trading=Trading(symbol="R_100", stake_amount=10, timeframe=timeframe),
        thresholds=Thresholds(
            confidence_threshold_high=0.8,
            learning_threshold_min=0.4,
            learning_threshold_max=0.6
        ),
        hyperparams=ModelHyperparams(
            learning_rate=0.001,
            batch_size=32,
            lstm_hidden_size=64,
            cnn_filters=32,
            latent_dim=16
        ),
        execution_safety=ExecutionSafety(),
        environment="test"
    )

def test_resolve_duration_basic():
    settings = create_mock_settings(timeframe="1m", rise_fall_duration=3, rise_fall_unit="m")
    service = ContractParameterService(settings)
    
    dur, unit = service.resolve_duration(CONTRACT_TYPES.RISE_FALL)
    assert dur == 3
    assert unit == "m"

def test_resolve_duration_units():
    # Test ticks
    settings = create_mock_settings(timeframe="1m", rise_fall_duration=5, rise_fall_unit="t")
    service = ContractParameterService(settings)
    
    dur, unit = service.resolve_duration(CONTRACT_TYPES.RISE_FALL)
    assert dur == 5
    assert unit == "t"
    
    # Test hours
    settings = create_mock_settings(timeframe="1h", rise_fall_duration=2, rise_fall_unit="h")
    service = ContractParameterService(settings)
    
    dur, unit = service.resolve_duration(CONTRACT_TYPES.RISE_FALL)
    assert dur == 2
    assert unit == "h"

def test_timeframe_parsing():
    settings = create_mock_settings()
    service = ContractParameterService(settings)
    
    assert service._parse_timeframe("5m") == (5, "m")
    assert service._parse_timeframe("1h") == (1, "h")
    assert service._parse_timeframe("10m") == (10, "m")

def test_consistency_check_logging(caplog):
    # Timeframe 5m, Duration 1m -> Mismatch
    settings = create_mock_settings(timeframe="5m", rise_fall_duration=1, rise_fall_unit="m")
    service = ContractParameterService(settings)
    
    with caplog.at_level("DEBUG"):
        service.resolve_duration(CONTRACT_TYPES.RISE_FALL)
        
    assert "Duration mismatch" in caplog.text
    assert "Config=1m" in caplog.text
    assert "Timeframe=5m" in caplog.text
