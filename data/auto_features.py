"""
Automated Feature Engineering Module.

Implements automatic generation and selection of trading features
using signal processing and information-theoretic methods.

Key capabilities:
- Wavelet decomposition for multi-scale analysis
- Fourier transform for frequency components
- Mutual information for feature scoring
- Forward selection for optimal feature subset

ARCHITECTURAL PRINCIPLE:
Feature engineering should be systematic and data-driven.
Instead of manual feature creation, we generate candidates
automatically and select based on predictive power.

Example:
    >>> from data.auto_features import AutoFeatureGenerator
    >>> generator = AutoFeatureGenerator()
    >>> features = generator.generate(price_series)
    >>> best_features = generator.select_top_k(features, targets, k=20)
"""

import logging
from datetime import datetime, timezone
from dataclasses import dataclass, field
from typing import Any, cast

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class FeatureCandidate:
    """
    A candidate feature with metadata.
    
    Attributes:
        name: Feature name
        values: Feature values array
        mi_score: Mutual information score (if computed)
        category: Feature category (wavelet, fourier, technical, etc.)
    """
    name: str
    values: np.ndarray
    mi_score: float = 0.0
    category: str = "unknown"


class WaveletDecomposer:
    """
    Wavelet decomposition for multi-scale time series analysis.
    
    Uses Haar wavelet for simplicity and interpretability.
    Decomposes signal into approximation and detail coefficients
    at multiple scales.
    """
    
    def __init__(self, max_level: int = 4):
        """
        Initialize decomposer.
        
        Args:
            max_level: Maximum decomposition level
        """
        self.max_level = max_level
    
    def _haar_transform(self, signal: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        Single-level Haar wavelet transform.
        
        Returns:
            Tuple of (approximation, detail) coefficients
        """
        n = len(signal)
        if n < 2:
            return signal, np.array([])
        
        # Truncate to even length
        n = n - (n % 2)
        signal = signal[:n]
        
        # Haar transform
        sqrt2 = np.sqrt(2)
        approx = (signal[0::2] + signal[1::2]) / sqrt2
        detail = (signal[0::2] - signal[1::2]) / sqrt2
        
        return approx, detail
    
    def decompose(self, signal: np.ndarray) -> dict[str, np.ndarray]:
        """
        Multi-level wavelet decomposition.
        
        Args:
            signal: Input time series
        
        Returns:
            Dictionary with approximation and detail coefficients per level
        """
        coefficients = {}
        current = signal.copy()
        
        for level in range(1, self.max_level + 1):
            if len(current) < 2:
                break
            
            approx, detail = self._haar_transform(current)
            coefficients[f"detail_L{level}"] = detail
            current = approx
        
        coefficients["approx_final"] = current
        
        return coefficients
    
    def generate_features(self, signal: np.ndarray) -> list[FeatureCandidate]:
        """
        Generate wavelet-based features.
        
        Args:
            signal: Input time series
        
        Returns:
            List of feature candidates
        """
        coeffs = self.decompose(signal)
        features: list[FeatureCandidate] = []
        
        for name, values in coeffs.items():
            if len(values) == 0:
                continue
            
            # Statistical features from coefficients
            features.extend([
                FeatureCandidate(
                    name=f"wavelet_{name}_mean",
                    values=np.array([np.mean(values)]),
                    category="wavelet",
                ),
                FeatureCandidate(
                    name=f"wavelet_{name}_std",
                    values=np.array([np.std(values)]),
                    category="wavelet",
                ),
                FeatureCandidate(
                    name=f"wavelet_{name}_energy",
                    values=np.array([np.sum(values ** 2)]),
                    category="wavelet",
                ),
            ])
        
        return features


class FourierExtractor:
    """
    Fourier transform for frequency domain features.
    
    Extracts dominant frequencies and power spectrum features
    from time series data.
    """
    
    def __init__(self, n_components: int = 10):
        """
        Initialize extractor.
        
        Args:
            n_components: Number of frequency components to extract
        """
        self.n_components = n_components
    
    def transform(self, signal: np.ndarray) -> dict[str, np.ndarray]:
        """
        Compute FFT and extract features.
        
        Args:
            signal: Input time series
        
        Returns:
            Dictionary with frequency features
        """
        n = len(signal)
        if n < 4:
            return {}
        
        # Compute FFT
        fft_vals = np.fft.fft(signal)
        freqs = np.fft.fftfreq(n)
        
        # Power spectrum (positive frequencies only)
        power = np.abs(fft_vals[:n//2]) ** 2
        pos_freqs = freqs[:n//2]
        
        # Extract top components
        top_indices = np.argsort(power)[-self.n_components:]
        
        return {
            "power_spectrum": power,
            "frequencies": pos_freqs,
            "top_powers": power[top_indices],
            "top_frequencies": pos_freqs[top_indices],
            "total_power": np.sum(power),
            "dominant_frequency": pos_freqs[np.argmax(power)] if len(power) > 0 else 0,
        }
    
    def generate_features(self, signal: np.ndarray) -> list[FeatureCandidate]:
        """
        Generate Fourier-based features.
        
        Args:
            signal: Input time series
        
        Returns:
            List of feature candidates
        """
        fft_data = self.transform(signal)
        
        if not fft_data:
            return []
        
        features = [
            FeatureCandidate(
                name="fourier_total_power",
                values=np.array([fft_data["total_power"]]),
                category="fourier",
            ),
            FeatureCandidate(
                name="fourier_dominant_freq",
                values=np.array([fft_data["dominant_frequency"]]),
                category="fourier",
            ),
        ]
        
        # Add top frequency powers
        for i, power in enumerate(fft_data["top_powers"]):
            features.append(FeatureCandidate(
                name=f"fourier_power_{i}",
                values=np.array([power]),
                category="fourier",
            ))
        
        return features


class MutualInformationScorer:
    """
    Score features using mutual information.
    
    MI measures how much knowing a feature reduces uncertainty
    about the target variable.
    """
    
    def __init__(self, n_bins: int = 10):
        """
        Initialize scorer.
        
        Args:
            n_bins: Number of bins for discretization
        """
        self.n_bins = n_bins
    
    def _discretize(self, values: np.ndarray) -> np.ndarray:
        """Discretize continuous values into bins."""
        if len(values) == 0:
            return np.array([])
        
        # Handle constant values
        if np.std(values) == 0:
            return np.zeros_like(values, dtype=int)
        
        # Percentile-based binning for robustness
        percentiles = np.linspace(0, 100, self.n_bins + 1)
        bin_edges = np.percentile(values, percentiles)
        return cast(np.ndarray, np.digitize(values, bin_edges[:-1]) - 1)
    
    def _entropy(self, labels: np.ndarray) -> float:
        """Compute Shannon entropy."""
        if len(labels) == 0:
            return 0.0
        
        _, counts = np.unique(labels, return_counts=True)
        probs = counts / len(labels)
        return cast(float, -np.sum(probs * np.log2(probs + 1e-10)))
    
    def compute(self, feature: np.ndarray, target: np.ndarray) -> float:
        """
        Compute mutual information between feature and target.
        
        Args:
            feature: Feature values
            target: Target values
        
        Returns:
            Mutual information score
        """
        if len(feature) != len(target) or len(feature) < 10:
            return 0.0
        
        # Discretize
        feat_bins = self._discretize(feature)
        tgt_bins = self._discretize(target.astype(float))
        
        # Compute MI = H(target) - H(target | feature)
        h_target = self._entropy(tgt_bins)
        
        # Conditional entropy
        h_cond = 0.0
        for bin_val in np.unique(feat_bins):
            mask = feat_bins == bin_val
            if mask.sum() > 0:
                p_bin = mask.sum() / len(feat_bins)
                h_cond += p_bin * self._entropy(tgt_bins[mask])
        
        return max(0, h_target - h_cond)
    
    def score_features(
        self,
        features: list[FeatureCandidate],
        targets: np.ndarray,
    ) -> list[FeatureCandidate]:
        """
        Score all features by mutual information.
        
        Args:
            features: List of feature candidates
            targets: Target values
        
        Returns:
            Features with MI scores set
        """
        for feature in features:
            # Expand feature if scalar
            if len(feature.values) == 1:
                expanded = np.full(len(targets), feature.values[0])
            elif len(feature.values) == len(targets):
                expanded = feature.values
            else:
                expanded = np.interp(
                    np.linspace(0, 1, len(targets)),
                    np.linspace(0, 1, len(feature.values)),
                    feature.values,
                )
            
            feature.mi_score = self.compute(expanded, targets)
        
        return features


class AutoFeatureGenerator:
    """
    Automated feature generation and selection pipeline.
    
    Combines multiple feature extraction methods and
    selects the best features using mutual information.
    
    Example:
        >>> generator = AutoFeatureGenerator()
        >>> features = generator.generate(prices)
        >>> selected = generator.select_top_k(features, targets, k=10)
    """
    
    def __init__(
        self,
        wavelet_levels: int = 4,
        fourier_components: int = 10,
        mi_bins: int = 10,
    ):
        """
        Initialize generator.
        
        Args:
            wavelet_levels: Max wavelet decomposition levels
            fourier_components: Number of Fourier components
            mi_bins: Bins for mutual information computation
        """
        self.wavelet = WaveletDecomposer(max_level=wavelet_levels)
        self.fourier = FourierExtractor(n_components=fourier_components)
        self.scorer = MutualInformationScorer(n_bins=mi_bins)
        
        logger.info(
            f"AutoFeatureGenerator initialized: "
            f"wavelet_levels={wavelet_levels}, "
            f"fourier_components={fourier_components}"
        )
    
    def generate_technical(self, prices: np.ndarray) -> list[FeatureCandidate]:
        """
        Generate technical indicator features.
        
        Args:
            prices: Price series
        
        Returns:
            List of technical features
        """
        features: list[FeatureCandidate] = []
        n = len(prices)
        
        if n < 20:
            return features
        
        # Returns at various horizons
        for period in [1, 5, 10, 20]:
            if n > period:
                returns = np.diff(prices, period) / prices[:-period]
                features.append(FeatureCandidate(
                    name=f"return_{period}",
                    values=returns,
                    category="technical",
                ))
        
        # Rolling volatility
        for window in [5, 10, 20]:
            if n > window:
                vol = np.array([
                    np.std(prices[max(0, i-window):i])
                    for i in range(window, n)
                ])
                features.append(FeatureCandidate(
                    name=f"volatility_{window}",
                    values=vol,
                    category="technical",
                ))
        
        # Moving average ratios
        for fast, slow in [(5, 20), (10, 30)]:
            if n > slow:
                ma_fast = np.convolve(prices, np.ones(fast)/fast, 'valid')
                ma_slow = np.convolve(prices, np.ones(slow)/slow, 'valid')
                min_len = min(len(ma_fast), len(ma_slow))
                ratio = ma_fast[-min_len:] / (ma_slow[-min_len:] + 1e-8)
                features.append(FeatureCandidate(
                    name=f"ma_ratio_{fast}_{slow}",
                    values=ratio,
                    category="technical",
                ))
        
        return features
    
    def generate(self, prices: np.ndarray) -> list[FeatureCandidate]:
        """
        Generate all feature candidates.
        
        Args:
            prices: Price series
        
        Returns:
            List of all feature candidates
        """
        all_features = []
        
        # Technical features
        all_features.extend(self.generate_technical(prices))
        
        # Wavelet features
        all_features.extend(self.wavelet.generate_features(prices))
        
        # Fourier features
        all_features.extend(self.fourier.generate_features(prices))
        
        logger.info(f"Generated {len(all_features)} feature candidates")
        return all_features
    
    def select_top_k(
        self,
        features: list[FeatureCandidate],
        targets: np.ndarray,
        k: int = 20,
    ) -> list[FeatureCandidate]:
        """
        Select top-k features by mutual information.
        
        Args:
            features: All feature candidates
            targets: Target values
            k: Number of features to select
        
        Returns:
            Top-k features sorted by MI score
        """
        scored = self.scorer.score_features(features, targets)
        sorted_features = sorted(scored, key=lambda f: f.mi_score, reverse=True)
        
        selected = sorted_features[:k]
        
        logger.info(
            f"Selected top {k} features. "
            f"Best: {selected[0].name} (MI={selected[0].mi_score:.4f})"
        )
        
        return selected
    
    def forward_selection(
        self,
        features: list[FeatureCandidate],
        targets: np.ndarray,
        max_features: int = 10,
        min_improvement: float = 0.01,
    ) -> list[FeatureCandidate]:
        """
        Greedy forward feature selection.
        
        Iteratively adds features that provide the most
        additional information given already selected features.
        
        Args:
            features: All feature candidates
            targets: Target values
            max_features: Maximum features to select
            min_improvement: Minimum MI improvement to continue
        
        Returns:
            Selected features
        """
        if not features:
            return []
        
        # Score all features against target (Relevance)
        scored = self.scorer.score_features(features.copy(), targets)
        
        selected: list[FeatureCandidate] = []
        remaining = list(scored)
        
        while len(selected) < max_features and remaining:
            best_feature = None
            best_score = -float('inf')
            
            for feature in remaining:
                # Relevance: I(X; Y)
                relevance = feature.mi_score
                
                # Redundancy: average(I(X; S))
                redundancy = 0.0
                if selected:
                    redundancy_sum = 0.0
                    for s in selected:
                        mi = self.scorer.compute(feature.values, s.values)
                        redundancy_sum += mi
                    redundancy = redundancy_sum / len(selected)
                
                # mRMR score: Relevance - Redundancy
                score = relevance - redundancy
                
                # Keep track of best
                if score > best_score:
                    best_score = score
                    best_feature = feature
            
            # Stop if score is too low (heuristic for "no good features left")
            # Note: mRMR score can be negative if redundancy > relevance
            if best_score < min_improvement and selected:
                 # Ensure we don't stop prematurely if it's just the first feature
                 # But usually mRMR continues as long as we have features
                 # Adjust: logic to stop if marginal gain is very low? 
                 # For mRMR, we usually just fill to K, or stop if score drops drastically.
                 # Let's keep a small threshold but allow negative scores if relevance is high.
                 # A safe bet is to stop if relevance is very low, regardless of redundancy.
                 pass

            # Improved stopping criteria: 
            # If the best feature has very low relevance, stop.
            if best_feature is None or best_feature.mi_score < min_improvement:
                break
                
            selected.append(best_feature)
            remaining.remove(best_feature)
        
        logger.info(f"mRMR selection chose {len(selected)} features")
        return selected
    
    def get_feature_summary(self, features: list[FeatureCandidate]) -> dict[str, Any]:
        """Get summary of features by category."""
        summary: dict[str, Any] = {
            "total": len(features),
            "by_category": {},
        }
        
        for feat in features:
            cat = feat.category
            if cat not in summary["by_category"]:
                summary["by_category"][cat] = {"count": 0, "avg_mi": 0.0}
            summary["by_category"][cat]["count"] += 1
            summary["by_category"][cat]["avg_mi"] += feat.mi_score
        
        for cat in summary["by_category"]:
            count = summary["by_category"][cat]["count"]
            if count > 0:
                summary["by_category"][cat]["avg_mi"] /= count
        
        return summary
