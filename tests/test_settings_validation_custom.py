import pytest
from config.settings import Settings, Trading, ExecutionSafety, Thresholds, ModelHyperparams, DataShapes

def test_settings_validation_stake():
    # Valid config
    t = Trading(symbol="R_100", stake_amount=10.0)
    es = ExecutionSafety(max_stake_per_trade=20.0)
    
    # We need dummy values for other required fields
    th = Thresholds(confidence_threshold_high=0.8, learning_threshold_min=0.4, learning_threshold_max=0.6)
    hp = ModelHyperparams(learning_rate=0.001, batch_size=32, lstm_hidden_size=64, cnn_filters=32, latent_dim=16)
    ds = DataShapes(sequence_length_ticks=100, sequence_length_candles=50)
    
    # This should pass
    s = Settings(
        trading=t, 
        execution_safety=es, 
        thresholds=th, 
        hyperparams=hp, 
        data_shapes=ds,
        _env_file=None
    )
    
    # Invalid config
    t_inv = Trading(symbol="R_100", stake_amount=30.0)
    
    with pytest.raises(ValueError, match="exceeds maximum allowed stake"):
        Settings(
            trading=t_inv, 
            execution_safety=es, 
            thresholds=th, 
            hyperparams=hp, 
            data_shapes=ds,
            _env_file=None
        )
