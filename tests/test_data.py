"""
Unit tests for data preprocessing module.

Tests normalizers, preprocessors, and data loading.
"""

from unittest.mock import MagicMock

import numpy as np
import pytest


class TestNormalizers:
    """Tests for data normalization functions."""

    def test_log_returns_basic(self):
        """Test basic log returns calculation."""
        from data.normalizers import log_returns

        prices = np.array([100.0, 101.0, 102.0, 101.0], dtype=np.float64)
        returns = log_returns(prices)

        assert returns.shape == prices.shape
        assert returns[0] == 0.0  # First element should be 0
        assert np.isfinite(returns).all()

    def test_log_returns_insufficient_data(self):
        """Log returns with < 2 points returns zeros."""
        from data.normalizers import log_returns

        prices = np.array([100.0])
        returns = log_returns(prices)

        assert len(returns) == 1
        assert returns[0] == 0.0

    def test_log_returns_invalid_type_raises(self):
        """Should raise TypeError for non-numpy input."""
        from data.normalizers import log_returns

        with pytest.raises(TypeError, match="must be np.ndarray"):
            log_returns([100, 101, 102])  # List instead of array

    def test_log_returns_negative_prices_raises(self):
        """Should raise ValueError for non-positive prices."""
        from data.normalizers import log_returns

        prices = np.array([100.0, -50.0, 102.0])
        with pytest.raises(ValueError, match="must be positive"):
            log_returns(prices)

    def test_z_score_normalize_basic(self):
        """Test basic z-score normalization."""
        from data.normalizers import z_score_normalize

        values = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        normalized = z_score_normalize(values)

        # Mean should be close to 0, std close to 1
        assert np.abs(np.mean(normalized)) < 0.01
        assert normalized.dtype == np.float32

    def test_z_score_empty_array(self):
        """Empty array should return empty array."""
        from data.normalizers import z_score_normalize

        normalized = z_score_normalize(np.array([]))
        assert len(normalized) == 0

    def test_min_max_normalize_basic(self):
        """Test min-max normalization to [0, 1]."""
        from data.normalizers import min_max_normalize

        values = np.array([10.0, 20.0, 30.0, 40.0, 50.0])
        normalized = min_max_normalize(values)

        assert np.min(normalized) == pytest.approx(0.0)
        assert np.max(normalized) == pytest.approx(1.0)

    def test_min_max_constant_array_returns_zeros(self):
        """Constant array should return zeros."""
        from data.normalizers import min_max_normalize

        values = np.array([5.0, 5.0, 5.0, 5.0])
        normalized = min_max_normalize(values)

        assert np.all(normalized == 0.0)

    def test_robust_scale_basic(self):
        """Test robust scaling."""
        from data.normalizers import robust_scale

        values = np.array([1.0, 2.0, 3.0, 4.0, 100.0])  # Has outlier
        scaled = robust_scale(values)

        # Outlier should not dominate the result
        assert np.abs(scaled[-1]) < 100  # Much smaller than naive scaling


class TestPreprocessors:
    """Tests for data preprocessors."""

    @pytest.fixture
    def mock_settings(self):
        """Create mock settings for preprocessors."""
        settings = MagicMock()
        settings.data_shapes.sequence_length_ticks = 100
        settings.data_shapes.sequence_length_candles = 50
        return settings

    def test_tick_preprocessor_output_shape(self, mock_settings):
        """Test tick preprocessor output dimensions."""
        from data.processor import TickPreprocessor

        pp = TickPreprocessor(mock_settings)
        ticks = np.random.rand(150).astype(np.float32) * 100 + 1  # Positive prices

        result = pp.process(ticks)

        assert result.shape == (100,)
        assert result.dtype == np.float32

    def test_tick_preprocessor_padding(self, mock_settings):
        """Test tick preprocessor pads short sequences."""
        from data.processor import TickPreprocessor

        pp = TickPreprocessor(mock_settings)
        ticks = np.random.rand(50).astype(np.float32) * 100 + 1

        result = pp.process(ticks)

        assert result.shape == (100,)
        # First part should be zeros (padding)
        assert np.sum(result[:50] == 0) > 0

    def test_candle_preprocessor_output_shape(self, mock_settings):
        """Test candle preprocessor output dimensions."""
        from data.processor import CandlePreprocessor

        pp = CandlePreprocessor(mock_settings)
        # OHLCVT format
        candles = np.random.rand(100, 6).astype(np.float32) * 100 + 1

        result = pp.process(candles)

        assert result.shape == (50, 10)  # 10 features
        assert result.dtype == np.float32

    def test_candle_preprocessor_invalid_shape_raises(self, mock_settings):
        """Invalid candle shape should raise ValueError."""
        from data.processor import CandlePreprocessor

        pp = CandlePreprocessor(mock_settings)
        candles = np.random.rand(100, 4)  # Wrong number of columns

        with pytest.raises(ValueError, match="must have 6 columns"):
            pp.process(candles)

    def test_volatility_extractor_output_shape(self):
        """Test volatility metrics extractor output."""
        from data.processor import VolatilityMetricsExtractor

        extractor = VolatilityMetricsExtractor()
        candles = np.random.rand(100, 6).astype(np.float32) * 100 + 1

        result = extractor.extract(candles)

        assert result.shape == (4,)
        assert result.dtype == np.float32
        assert np.isfinite(result).all()


class TestFeatureBuilder:
    """Tests for canonical FeatureBuilder."""

    @pytest.fixture
    def mock_settings(self):
        """Create mock settings for FeatureBuilder."""
        settings = MagicMock()
        settings.data_shapes.sequence_length_ticks = 100
        settings.data_shapes.sequence_length_candles = 50
        return settings

    def test_feature_builder_initialization(self, mock_settings):
        """FeatureBuilder should initialize with schema."""
        from data.features import FeatureBuilder

        builder = FeatureBuilder(mock_settings)

        assert builder.schema.tick_length == 100
        assert builder.schema.candle_length == 50
        assert builder.get_schema_version() == "1.1"

    def test_feature_builder_build_output_shapes(self, mock_settings):
        """FeatureBuilder.build should produce correct output shapes."""
        from data.features import FeatureBuilder

        builder = FeatureBuilder(mock_settings)

        # Create test data
        ticks = np.random.rand(150).astype(np.float32) * 100 + 1
        candles = np.random.rand(100, 6).astype(np.float32) * 100 + 1

        features = builder.build(ticks=ticks, candles=candles)

        assert "ticks" in features
        assert "candles" in features
        assert "vol_metrics" in features

        assert features["ticks"].shape == (100,)
        assert features["candles"].shape == (50, 10)
        assert features["vol_metrics"].shape == (4,)

    def test_feature_builder_schema_validation(self, mock_settings):
        """FeatureBuilder should validate output shapes."""
        from data.features import FeatureBuilder

        builder = FeatureBuilder(mock_settings)
        schema = builder.get_schema()

        # Valid shapes should pass
        valid_ticks = np.zeros((100,), dtype=np.float32)
        schema.validate_ticks(valid_ticks)  # Should not raise

        # Invalid shapes should raise
        with pytest.raises(ValueError, match="shape mismatch"):
            schema.validate_ticks(np.zeros((50,), dtype=np.float32))
