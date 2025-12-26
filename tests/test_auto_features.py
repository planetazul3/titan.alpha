"""
Unit tests for automated feature engineering.
"""

import numpy as np
import pytest

from data.auto_features import (
    AutoFeatureGenerator,
    FeatureCandidate,
    FourierExtractor,
    MutualInformationScorer,
    WaveletDecomposer,
)


class TestWaveletDecomposer:
    """Tests for WaveletDecomposer."""

    def test_haar_transform(self):
        """Test single level Haar transform."""
        decomposer = WaveletDecomposer()
        signal = np.array([1.0, 2.0, 3.0, 4.0])
        
        approx, detail = decomposer._haar_transform(signal)
        
        assert len(approx) == 2
        assert len(detail) == 2

    def test_decompose_levels(self):
        """Test multi-level decomposition."""
        decomposer = WaveletDecomposer(max_level=3)
        signal = np.random.randn(64)
        
        coeffs = decomposer.decompose(signal)
        
        assert "detail_L1" in coeffs
        assert "detail_L2" in coeffs
        assert "detail_L3" in coeffs
        assert "approx_final" in coeffs

    def test_generate_features(self):
        """Test feature generation."""
        decomposer = WaveletDecomposer(max_level=2)
        signal = np.random.randn(32)
        
        features = decomposer.generate_features(signal)
        
        assert len(features) > 0
        assert all(isinstance(f, FeatureCandidate) for f in features)
        assert all(f.category == "wavelet" for f in features)


class TestFourierExtractor:
    """Tests for FourierExtractor."""

    def test_transform(self):
        """Test FFT computation."""
        extractor = FourierExtractor(n_components=5)
        signal = np.sin(np.linspace(0, 4 * np.pi, 100))
        
        result = extractor.transform(signal)
        
        assert "power_spectrum" in result
        assert "dominant_frequency" in result
        assert "total_power" in result

    def test_generate_features(self):
        """Test Fourier feature generation."""
        extractor = FourierExtractor(n_components=5)
        signal = np.random.randn(50)
        
        features = extractor.generate_features(signal)
        
        assert len(features) > 0
        assert all(f.category == "fourier" for f in features)

    def test_short_signal(self):
        """Test handling of short signals."""
        extractor = FourierExtractor()
        signal = np.array([1.0, 2.0])
        
        result = extractor.transform(signal)
        
        assert result == {}


class TestMutualInformationScorer:
    """Tests for MutualInformationScorer."""

    def test_perfect_correlation(self):
        """Test MI for perfectly correlated variables."""
        scorer = MutualInformationScorer(n_bins=5)
        
        feature = np.arange(100).astype(float)
        target = np.arange(100).astype(float)
        
        mi = scorer.compute(feature, target)
        
        # Should be high for perfect correlation
        assert mi > 1.0

    def test_independent_variables(self):
        """Test MI for independent variables."""
        scorer = MutualInformationScorer(n_bins=5)
        
        np.random.seed(42)
        feature = np.random.randn(100)
        target = np.random.randn(100)
        
        mi = scorer.compute(feature, target)
        
        # Should be low for independent variables
        assert mi < 0.5

    def test_score_features(self):
        """Test batch feature scoring."""
        scorer = MutualInformationScorer()
        
        feature1 = FeatureCandidate("f1", np.arange(50).astype(float))
        feature2 = FeatureCandidate("f2", np.random.randn(50))
        targets = np.arange(50)
        
        scored = scorer.score_features([feature1, feature2], targets)
        
        assert len(scored) == 2
        assert scored[0].mi_score > scored[1].mi_score  # f1 should be better


class TestAutoFeatureGenerator:
    """Tests for AutoFeatureGenerator."""

    def test_initialization(self):
        """Test initialization."""
        generator = AutoFeatureGenerator()
        
        assert generator.wavelet is not None
        assert generator.fourier is not None
        assert generator.scorer is not None

    def test_generate_technical(self):
        """Test technical feature generation."""
        generator = AutoFeatureGenerator()
        prices = np.cumsum(np.random.randn(100)) + 100
        
        features = generator.generate_technical(prices)
        
        assert len(features) > 0
        assert any("return" in f.name for f in features)
        assert any("volatility" in f.name for f in features)

    def test_generate_all(self):
        """Test full feature generation."""
        generator = AutoFeatureGenerator()
        prices = np.cumsum(np.random.randn(100)) + 100
        
        features = generator.generate(prices)
        
        categories = {f.category for f in features}
        assert "technical" in categories
        assert "wavelet" in categories
        assert "fourier" in categories

    def test_select_top_k(self):
        """Test top-k selection."""
        generator = AutoFeatureGenerator()
        prices = np.cumsum(np.random.randn(100)) + 100
        targets = np.random.randint(0, 2, 100)
        
        features = generator.generate(prices)
        selected = generator.select_top_k(features, targets, k=5)
        
        assert len(selected) == 5
        # Should be sorted by MI score
        for i in range(len(selected) - 1):
            assert selected[i].mi_score >= selected[i+1].mi_score

    def test_forward_selection(self):
        """Test forward selection with mRMR."""
        generator = AutoFeatureGenerator()
        
        # Create correlated features to test redundancy handling
        # F1: High correlation with target (Relevance=High)
        # F2: High correlation with target (Relevance=High), but copy of F1 (Redundancy=High)
        # F3: Moderate correlation with target (Relevance=Med), independent of F1 (Redundancy=Low)
        
        n_samples = 100
        np.random.seed(42)
        
        target = np.random.randint(0, 2, n_samples)
        
        # F1 aligns well with target
        f1_vals = target.astype(float) + np.random.normal(0, 0.1, n_samples)
        f1 = FeatureCandidate("F1", f1_vals,category="test")
        
        # F2 is almost identical to F1 (Redundant)
        f2_vals = f1_vals + np.random.normal(0, 0.01, n_samples) 
        f2 = FeatureCandidate("F2", f2_vals, category="test")
        
        # F3 is independent noise + some target signal
        f3_vals = np.random.normal(0, 1, n_samples) + target.astype(float) * 0.5
        f3 = FeatureCandidate("F3", f3_vals, category="test")
        
        features = [f1, f2, f3]
        
        selected = generator.forward_selection(features, target, max_features=2)
        
        # Should select at least 2 features
        # Ideally F1 and F3 (since F2 is redundant with F1)
        assert len(selected) >= 2
        
        names = [f.name for f in selected]
        assert "F1" in names  # Best individual
        assert "F3" in names  # Best complement
        # F2 might be skipped because F3 adds more new info (score = relevance - redundancy)


    def test_get_feature_summary(self):
        """Test feature summary."""
        generator = AutoFeatureGenerator()
        prices = np.cumsum(np.random.randn(100)) + 100
        
        features = generator.generate(prices)
        summary = generator.get_feature_summary(features)
        
        assert "total" in summary
        assert "by_category" in summary


class TestFeatureCandidate:
    """Tests for FeatureCandidate."""

    def test_creation(self):
        """Test candidate creation."""
        candidate = FeatureCandidate(
            name="test_feature",
            values=np.array([1.0, 2.0, 3.0]),
            category="technical",
        )
        
        assert candidate.name == "test_feature"
        assert candidate.mi_score == 0.0
