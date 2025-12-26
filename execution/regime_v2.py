"""
Hierarchical Regime Detection Module.

Provides multi-level market regime detection for smarter trading decisions.
Instead of a single reconstruction error threshold, this module detects
regimes at multiple time scales:

1. **Macro Level**: Bull/Bear/Sideways market trend (long lookback)
2. **Meso Level**: Volatility regime (Low/Medium/High)
3. **Micro Level**: Microstructure regime (Trending/Mean-reverting via Hurst)

References:
- Hidden Markov Models in Finance (Nystrup et al., 2015)
- Changepoint Detection Survey (Aminikhanghahi & Cook, 2017)
- Hurst exponent for market regime (Mandelbrot, 1971)

ARCHITECTURAL PRINCIPLE:
The hierarchical regime feeds into the decision engine via soft gating.
Each regime level contributes to a trust score that modulates trading
behavior without hard cutoffs.

Example:
    >>> from execution.regime_v2 import HierarchicalRegimeDetector
    >>> detector = HierarchicalRegimeDetector()
    >>> assessment = detector.assess(prices=price_history, volatility=vol_metrics)
    >>> if assessment.trust_score < 0.5:
    ...     # Reduce position sizing
"""

import logging
from dataclasses import dataclass
from enum import Enum
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


class MacroRegime(Enum):
    """Macro-level market trend regime."""
    BULL = "bull"       # Uptrending market
    BEAR = "bear"       # Downtrending market
    SIDEWAYS = "sideways"  # Range-bound market


class VolatilityRegime(Enum):
    """Meso-level volatility regime."""
    LOW = "low"         # Quiet market
    MEDIUM = "medium"   # Normal volatility
    HIGH = "high"       # Elevated volatility


class MicroRegime(Enum):
    """Micro-level market microstructure."""
    TRENDING = "trending"       # Momentum-driven (Hurst > 0.5)
    RANDOM = "random"          # Random walk (Hurst ≈ 0.5)
    MEAN_REVERTING = "mean_reverting"  # Anti-persistent (Hurst < 0.5)


@dataclass
class HierarchicalRegimeAssessment:
    """
    Complete hierarchical regime assessment.
    
    Attributes:
        macro: Macro-level regime (Bull/Bear/Sideways)
        volatility: Meso-level volatility regime (Low/Medium/High)
        micro: Micro-level microstructure (Trending/Random/Mean-reverting)
        trust_score: Overall regime trust score (0 to 1, higher is safer)
        reconstruction_error: Original reconstruction error (for compatibility)
        details: Additional diagnostic information
    """
    macro: MacroRegime
    volatility: VolatilityRegime
    micro: MicroRegime
    trust_score: float
    reconstruction_error: float
    details: dict[str, Any]
    
    def is_vetoed(self) -> bool:
        """Check if regime warrants trade veto (trust_score < 0.3)."""
        return self.trust_score < 0.3
    
    def requires_caution(self) -> bool:
        """Check if regime requires cautious trading (0.3 <= trust_score < 0.6)."""
        return 0.3 <= self.trust_score < 0.6
    
    def is_favorable(self) -> bool:
        """Check if regime is favorable for aggressive trading (trust_score >= 0.8)."""
        return self.trust_score >= 0.8
    
    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for logging/serialization."""
        return {
            "macro": self.macro.value,
            "volatility": self.volatility.value,
            "micro": self.micro.value,
            "trust_score": self.trust_score,
            "reconstruction_error": self.reconstruction_error,
            "is_vetoed": self.is_vetoed(),
            "requires_caution": self.requires_caution(),
            **self.details,
        }


class HurstExponentEstimator:
    """
    Estimate the Hurst exponent for regime detection.
    
    The Hurst exponent H indicates market behavior:
    - H > 0.5: Trending (persistent) - momentum strategies work
    - H ≈ 0.5: Random walk - no predictable pattern
    - H < 0.5: Mean-reverting (anti-persistent) - mean reversion strategies work
    
    Uses R/S (Rescaled Range) analysis for estimation.
    """
    
    def __init__(self, min_window: int = 8, max_window: int | None = None):
        """
        Initialize Hurst estimator.
        
        Args:
            min_window: Minimum window size for R/S calculation
            max_window: Maximum window size (defaults to half of data length)
        """
        self.min_window = min_window
        self.max_window = max_window
    
    def estimate(self, prices: np.ndarray) -> float:
        """
        Estimate Hurst exponent from price series.
        
        Args:
            prices: Price series (1D array)
        
        Returns:
            Hurst exponent (0 to 1)
        """
        if len(prices) < self.min_window * 2:
            logger.warning(f"Insufficient data ({len(prices)}) for Hurst estimation")
            return 0.5  # Default to random walk
        
        # Calculate log returns
        returns = np.diff(np.log(prices + 1e-10))
        
        if len(returns) < self.min_window:
            return 0.5
        
        max_window = self.max_window or len(returns) // 2
        
        # R/S analysis at multiple window sizes
        window_sizes = []
        rs_values = []
        
        for window in range(self.min_window, min(max_window + 1, len(returns) + 1)):
            n_windows = len(returns) // window
            if n_windows < 1:
                continue
            
            rs_list = []
            for i in range(n_windows):
                segment = returns[i * window:(i + 1) * window]
                
                # Mean-adjusted cumulative deviation
                mean_return = np.mean(segment)
                cumdev = np.cumsum(segment - mean_return)
                
                # Range
                r = np.max(cumdev) - np.min(cumdev)
                
                # Standard deviation
                s = np.std(segment, ddof=1)
                
                if s > 0:
                    rs_list.append(r / s)
            
            if rs_list:
                window_sizes.append(window)
                rs_values.append(np.mean(rs_list))
        
        if len(window_sizes) < 2:
            return 0.5
        
        # Linear regression in log-log space: log(R/S) ~ H * log(n)
        log_n = np.log(window_sizes)
        log_rs = np.log(rs_values)
        
        # Simple least squares
        n = len(log_n)
        sum_x = np.sum(log_n)
        sum_y = np.sum(log_rs)
        sum_xy = np.sum(log_n * log_rs)
        sum_xx = np.sum(log_n ** 2)
        
        denominator = n * sum_xx - sum_x ** 2
        if abs(denominator) < 1e-10:
            return 0.5
        
        hurst = (n * sum_xy - sum_x * sum_y) / denominator
        
        # Clamp to valid range
        return float(np.clip(hurst, 0.0, 1.0))


class VolatilityRegimeDetector:
    """
    Detect volatility regime using rolling volatility and percentiles.
    
    Classifies market into Low/Medium/High volatility based on
    where current volatility falls in historical distribution.
    """
    
    def __init__(
        self,
        low_percentile: float = 30,
        high_percentile: float = 70,
        lookback: int = 100,
    ):
        """
        Initialize volatility regime detector.
        
        Args:
            low_percentile: Percentile below which volatility is "low"
            high_percentile: Percentile above which volatility is "high"
            lookback: Number of periods for rolling calculations
        """
        self.low_percentile = low_percentile
        self.high_percentile = high_percentile
        self.lookback = lookback
    
    def detect(self, prices: np.ndarray) -> tuple[VolatilityRegime, float]:
        """
        Detect volatility regime from price series.
        
        Args:
            prices: Price series
        
        Returns:
            Tuple of (regime, current_volatility_percentile)
        """
        if len(prices) < 20:
            return VolatilityRegime.MEDIUM, 50.0
        
        # Calculate rolling volatility
        returns = np.diff(np.log(prices + 1e-10))
        
        if len(returns) < self.lookback:
            lookback = len(returns)
        else:
            lookback = self.lookback
        
        # Rolling std with window
        window = min(20, lookback // 2)
        if window < 5:
            return VolatilityRegime.MEDIUM, 50.0
        
        rolling_vol = np.array([
            np.std(returns[max(0, i - window):i + 1])
            for i in range(len(returns))
        ])
        
        current_vol = rolling_vol[-1]
        historical_vol = rolling_vol[-lookback:]
        
        # Calculate percentile
        percentile = 100 * np.sum(historical_vol < current_vol) / len(historical_vol)
        
        if percentile < self.low_percentile:
            regime = VolatilityRegime.LOW
        elif percentile > self.high_percentile:
            regime = VolatilityRegime.HIGH
        else:
            regime = VolatilityRegime.MEDIUM
        
        return regime, float(percentile)


class TrendDetector:
    """
    Detect macro trend regime using moving average crossovers.
    
    Simple but effective: compare current price to short-term
    and long-term moving averages.
    """
    
    def __init__(self, short_window: int = 20, long_window: int = 50):
        """
        Initialize trend detector.
        
        Args:
            short_window: Short-term MA period
            long_window: Long-term MA period
        """
        self.short_window = short_window
        self.long_window = long_window
    
    def detect(self, prices: np.ndarray) -> tuple[MacroRegime, float]:
        """
        Detect macro regime from price series.
        
        Args:
            prices: Price series
        
        Returns:
            Tuple of (regime, trend_strength)
        """
        if len(prices) < self.long_window:
            return MacroRegime.SIDEWAYS, 0.0
        
        # Calculate MAs
        short_ma = np.mean(prices[-self.short_window:])
        long_ma = np.mean(prices[-self.long_window:])
        current_price = prices[-1]
        
        # Calculate trend strength as % deviation
        if long_ma > 0:
            trend_strength = (short_ma - long_ma) / long_ma
        else:
            trend_strength = 0.0
        
        # Thresholds for regime classification
        bull_threshold = 0.01   # 1% above
        bear_threshold = -0.01  # 1% below
        
        if trend_strength > bull_threshold and current_price > short_ma:
            regime = MacroRegime.BULL
        elif trend_strength < bear_threshold and current_price < short_ma:
            regime = MacroRegime.BEAR
        else:
            regime = MacroRegime.SIDEWAYS
        
        return regime, float(trend_strength)


class HierarchicalRegimeDetector:
    """
    Multi-level regime detector combining macro, volatility, and micro signals.
    
    Produces a comprehensive regime assessment with trust score for
    decision-making. Compatible with existing RegimeVeto interface via
    the is_vetoed() and requires_caution() methods.
    
    Trust Score Calculation:
    - Starts at 1.0 (full trust)
    - Reduced by high volatility
    - Reduced by adverse macro conditions
    - Reduced by random walk micro-regime (unpredictable)
    - Reduced by high reconstruction error
    
    Example:
        >>> detector = HierarchicalRegimeDetector()
        >>> assessment = detector.assess(prices, reconstruction_error=0.1)
        >>> print(f"Trust: {assessment.trust_score:.2f}, Regime: {assessment.macro.value}")
    """
    
    def __init__(
        self,
        trend_short_window: int = 20,
        trend_long_window: int = 50,
        vol_low_percentile: float = 30,
        vol_high_percentile: float = 70,
        hurst_min_window: int = 8,
        reconstruction_weight: float = 0.3,
    ):
        """
        Initialize hierarchical regime detector.
        
        Args:
            trend_short_window: Short MA for trend detection
            trend_long_window: Long MA for trend detection
            vol_low_percentile: Low volatility threshold percentile
            vol_high_percentile: High volatility threshold percentile
            hurst_min_window: Minimum window for Hurst estimation
            reconstruction_weight: Weight for reconstruction error in trust
        """
        self.trend_detector = TrendDetector(trend_short_window, trend_long_window)
        self.vol_detector = VolatilityRegimeDetector(
            vol_low_percentile, vol_high_percentile
        )
        self.hurst_estimator = HurstExponentEstimator(hurst_min_window)
        self.reconstruction_weight = reconstruction_weight
        
        logger.info(
            f"HierarchicalRegimeDetector initialized: "
            f"trend_windows=({trend_short_window},{trend_long_window}), "
            f"vol_percentiles=({vol_low_percentile},{vol_high_percentile})"
        )
    
    def assess(
        self,
        prices: np.ndarray,
        reconstruction_error: float = 0.0,
    ) -> HierarchicalRegimeAssessment:
        """
        Perform full hierarchical regime assessment.
        
        Args:
            prices: Price history (1D array, minimum 50 points recommended)
            reconstruction_error: Volatility expert reconstruction error
        
        Returns:
            HierarchicalRegimeAssessment with trust score and regime details
        """
        # 1. Macro regime (trend)
        macro, trend_strength = self.trend_detector.detect(prices)
        
        # 2. Volatility regime
        volatility, vol_percentile = self.vol_detector.detect(prices)
        
        # 3. Micro regime (Hurst)
        hurst = self.hurst_estimator.estimate(prices)
        if hurst > 0.55:
            micro = MicroRegime.TRENDING
        elif hurst < 0.45:
            micro = MicroRegime.MEAN_REVERTING
        else:
            micro = MicroRegime.RANDOM
        
        # 4. Calculate trust score
        trust_score = self._calculate_trust_score(
            macro, volatility, micro,
            trend_strength, vol_percentile, hurst,
            reconstruction_error
        )
        
        details = {
            "trend_strength": trend_strength,
            "vol_percentile": vol_percentile,
            "hurst_exponent": hurst,
        }
        
        assessment = HierarchicalRegimeAssessment(
            macro=macro,
            volatility=volatility,
            micro=micro,
            trust_score=trust_score,
            reconstruction_error=reconstruction_error,
            details=details,
        )
        
        logger.debug(
            f"Regime assessment: {macro.value}/{volatility.value}/{micro.value}, "
            f"trust={trust_score:.3f}, hurst={hurst:.3f}"
        )
        
        return assessment
    
    def _calculate_trust_score(
        self,
        macro: MacroRegime,
        volatility: VolatilityRegime,
        micro: MicroRegime,
        trend_strength: float,
        vol_percentile: float,
        hurst: float,
        reconstruction_error: float,
    ) -> float:
        """
        Calculate overall trust score from regime signals.
        
        Combines multiple signals into a single trust metric.
        Higher score = safer to trade aggressively.
        """
        trust = 1.0
        
        # Volatility penalty (high vol = lower trust)
        if volatility == VolatilityRegime.HIGH:
            trust *= 0.6
        elif volatility == VolatilityRegime.MEDIUM:
            trust *= 0.85
        # LOW volatility is best for predictability
        
        # Macro regime adjustment
        # Sideways markets are harder to predict
        if macro == MacroRegime.SIDEWAYS:
            trust *= 0.8
        # Strong trends (bull or bear) are easier
        elif abs(trend_strength) > 0.02:
            trust *= 1.0  # Bonus for clear trend
        
        # Micro regime (Hurst)
        # Random walk is worst (H ≈ 0.5)
        # Trending or mean-reverting are better
        if micro == MicroRegime.RANDOM:
            trust *= 0.7
        else:
            trust *= 0.9  # Trending or mean-reverting
        
        # Reconstruction error penalty
        # Maps reconstruction error to trust reduction
        # High error = model is confused about current regime
        recon_penalty = 1.0 - min(1.0, reconstruction_error * 2) * self.reconstruction_weight
        trust *= recon_penalty
        
        return float(np.clip(trust, 0.0, 1.0))
    
    def assess_from_reconstruction_error(
        self,
        reconstruction_error: float,
        prices: np.ndarray | None = None,
    ) -> HierarchicalRegimeAssessment:
        """
        Simplified assessment when only reconstruction error is available.
        
        Useful for backward compatibility with existing RegimeVeto users.
        Falls back to default regimes when no price data is provided.
        """
        if prices is not None and len(prices) >= 50:
            return self.assess(prices, reconstruction_error)
        
        # Fallback: use reconstruction error only
        if reconstruction_error >= 0.3:
            volatility = VolatilityRegime.HIGH
            trust = 0.2
        elif reconstruction_error >= 0.1:
            volatility = VolatilityRegime.MEDIUM
            trust = 0.6
        else:
            volatility = VolatilityRegime.LOW
            trust = 0.9
        
        return HierarchicalRegimeAssessment(
            macro=MacroRegime.SIDEWAYS,
            volatility=volatility,
            micro=MicroRegime.RANDOM,
            trust_score=trust,
            reconstruction_error=reconstruction_error,
            details={"fallback_mode": True},
        )
