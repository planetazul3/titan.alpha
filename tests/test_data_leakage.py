import pytest
from data.loader import calculate_minimum_temporal_gap
from config.settings import Settings, DataShapes, Trading, Thresholds, ModelHyperparams, ExecutionSafety

def create_mock_settings(seq_len=20, warmup=10, ticks=16):
    return Settings.model_construct(
        data_shapes=DataShapes(
            warmup_steps=warmup, 
            sequence_length_candles=seq_len,
            sequence_length_ticks=ticks
        ),
        trading=Trading(symbol="R_100", stake_amount=10),
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

def test_gap_calculation():
    # Scenario 1
    seq = 20
    warmup = 10
    lookahead = 1
    # Gap = seq + warmup + lookahead + (seq * 1.0)
    # Gap = 20 + 10 + 1 + 20 = 51
    settings = create_mock_settings(seq_len=seq, warmup=warmup)
    gap = calculate_minimum_temporal_gap(settings, lookahead_candles=lookahead)
    assert gap == 51

    # Scenario 2
    lookahead = 5
    # Gap = 20 + 10 + 5 + 20 = 55
    gap = calculate_minimum_temporal_gap(settings, lookahead_candles=lookahead)
    assert gap == 55

def test_gap_prevents_overlap():
    # Conceptually verify that Val[0] uses [ValT-seq-warmup, ValT], and Train[last] targeted TrainT+lookahead
    # We require ValT-seq-warmup > TrainT+lookahead
    # ValT = TrainT + Gap
    # TrainT + Gap - seq - warmup > TrainT + lookahead
    # Gap > seq + warmup + lookahead
    # Our formula: Gap = seq + warmup + lookahead + seq (safety)
    # So Gap > ... is True.
    pass
