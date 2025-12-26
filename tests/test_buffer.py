import pytest

from data.buffer import CandleData, MarketDataBuffer


@pytest.fixture
def buffer():
    return MarketDataBuffer(tick_length=10, candle_length=5)


def test_buffer_initialization(buffer):
    assert not buffer.is_ready()
    assert buffer.tick_count() == 0
    assert buffer.candle_count() == 0


def test_tick_buffering(buffer):
    for i in range(15):
        buffer.append_tick(float(i))

    # Should cap at tick_length=10
    assert buffer.tick_count() == 10
    ticks = buffer.get_ticks_array()
    assert len(ticks) == 10
    assert ticks[-1] == 14.0
    assert ticks[0] == 5.0


def test_candle_update_vs_append(buffer):
    # 1. New candle
    c1 = CandleData(1, 2, 0.5, 1.5, 100, timestamp=1000.0)
    is_new = buffer.update_candle(c1)
    assert is_new is True
    assert buffer.candle_count() == 1

    # 2. Update same candle (timestamp matches)
    c1_update = CandleData(1, 2.5, 0.5, 2.0, 150, timestamp=1000.0)
    is_new = buffer.update_candle(c1_update)
    assert is_new is False
    assert buffer.candle_count() == 1
    # Verify update happened
    arr = buffer.get_candles_array()
    assert arr[0][3] == 2.0  # Close price updated

    # 3. New candle (timestamp diff > 1s)
    c2 = CandleData(2, 3, 1.5, 2.5, 50, timestamp=1060.0)
    is_new = buffer.update_candle(c2)
    assert is_new is True
    assert buffer.candle_count() == 2
    arr = buffer.get_candles_array()
    assert arr[1][5] == 1060.0


def test_candle_update_drift_tolerance(buffer):
    """Test slight timestamp drift handling."""
    buffer.update_candle(CandleData(1, 1, 1, 1, 1, 1000.0))

    # Drift 0.5s -> should consider same candle
    is_new = buffer.update_candle(CandleData(1, 1, 1, 2, 1, 1000.5))
    assert is_new is False

    # Drift 1.5s -> should consider new candle
    is_new = buffer.update_candle(CandleData(2, 2, 2, 2, 2, 1001.5))
    assert is_new is True


def test_is_ready(buffer):
    # Fill ticks
    for i in range(10):
        buffer.append_tick(1.0)
    assert not buffer.is_ready()  # Need candles

    # Fill candles
    for i in range(5):
        buffer.update_candle(CandleData(1, 1, 1, 1, 1, 1000.0 + i * 60))

    assert buffer.is_ready()


def test_clear(buffer):
    buffer.append_tick(1.0)
    buffer.update_candle(CandleData(1, 1, 1, 1, 1, 1000.0))
    buffer.clear()
    assert buffer.tick_count() == 0
    assert buffer.candle_count() == 0
    assert not buffer.is_ready()


def test_include_forming_parameter():
    """Test that include_forming parameter correctly filters forming candles."""
    # Buffer with capacity for 3 closed candles
    buffer = MarketDataBuffer(tick_length=5, candle_length=3)
    
    # Add 3 candles (will become closed candles)
    for i in range(3):
        buffer.update_candle(CandleData(1, 2, 0.5, 1.5, 100, 1000.0 + i * 60))
    
    # Add a 4th candle (the "forming" candle)
    buffer.update_candle(CandleData(2, 3, 1.5, 2.5, 50, 1000.0 + 3 * 60))
    
    # Buffer should now have 4 candles (3 closed + 1 forming)
    assert buffer.candle_count() == 4
    
    # Without include_forming (default) - should return 3 closed candles
    closed_candles = buffer.get_candles_array(include_forming=False)
    assert len(closed_candles) == 3
    # Last returned candle should be the 3rd one, not the 4th
    assert closed_candles[-1][5] == 1000.0 + 2 * 60  # timestamp of 3rd candle
    
    # With include_forming=True - should return all 4 candles
    all_candles = buffer.get_candles_array(include_forming=True)
    assert len(all_candles) == 4
    assert all_candles[-1][5] == 1000.0 + 3 * 60  # timestamp of 4th candle


def test_buffer_capacity_with_forming():
    """Test that buffer correctly handles capacity with forming candle."""
    buffer = MarketDataBuffer(tick_length=5, candle_length=3)
    
    # Fill buffer beyond capacity (5 candles for a 3-candle buffer)
    for i in range(5):
        buffer.update_candle(CandleData(i, i + 1, i - 0.5, i + 0.5, 100, 1000.0 + i * 60))
    
    # Buffer maxlen is 3+1=4, so only last 4 candles remain
    assert buffer.candle_count() == 4
    
    # get_candles_array(include_forming=False) should return 3 (not 4)
    closed = buffer.get_candles_array(include_forming=False)
    assert len(closed) == 3

