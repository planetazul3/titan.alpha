"""
Integration utilities for autonomous trading brain components.

Provides factory functions and helpers to wire together the
enhanced trading components:

- KellyPositionSizer with SafeTradeExecutor
- HierarchicalRegimeDetector with DecisionEngine
- Combined regime-aware position sizing

Example:
    >>> from execution.integration import create_enhanced_executor, create_enhanced_engine
    >>> executor = create_enhanced_executor(raw_executor, settings)
    >>> engine = create_enhanced_engine(settings, shadow_store)
"""

import logging
from pathlib import Path
from typing import Any

import numpy as np

from config.settings import Settings
from execution.position_sizer import KellyPositionSizer, PositionSizeResult
from execution.regime import RegimeAssessment, RegimeAssessmentProtocol, RegimeVeto
from execution.common.types import TrustState
from execution.regime_v2 import (
    HierarchicalRegimeAssessment,
    HierarchicalRegimeDetector,
)
from execution.signals import TradeSignal


logger = logging.getLogger(__name__)


class RegimeAwarePositionSizer:
    """
    Position sizer that integrates regime assessment for dynamic sizing.
    
    Combines Kelly Criterion with hierarchical regime detection:
    - Trust score modulates position size
    - High volatility regime → smaller positions
    - Favorable regime → allows larger positions
    
    Example:
        >>> sizer = RegimeAwarePositionSizer(base_stake=1.0)
        >>> stake = sizer.compute_stake_for_signal(
        ...     signal=trade_signal,
        ...     regime_assessment=regime,
        ...     current_drawdown=0.05
        ... )
    """
    
    def __init__(
        self,
        base_stake: float = 1.0,
        safety_factor: float = 0.5,
        max_stake: float = 10.0,
        min_stake: float = 0.35,
        payout_ratio: float = 0.95,
    ):
        """
        Initialize regime-aware position sizer.
        
        Args:
            base_stake: Base stake for Kelly calculation
            safety_factor: Fractional Kelly (0.5 = half-Kelly)
            max_stake: Maximum allowed stake
            min_stake: Minimum viable stake
            payout_ratio: Binary option payout ratio
        """
        self.kelly_sizer = KellyPositionSizer(
            base_stake=base_stake,
            safety_factor=safety_factor,
            max_stake=max_stake,
            min_stake=min_stake,
        )
        self.payout_ratio = payout_ratio
        
        logger.info(f"RegimeAwarePositionSizer initialized with base=${base_stake}")
    
    def compute_stake_for_signal(
        self,
        signal: TradeSignal,
        regime_assessment: HierarchicalRegimeAssessment | None = None,
        current_drawdown: float = 0.0,
        account_balance: float | None = None,
    ) -> PositionSizeResult:
        """
        Compute stake using Kelly + regime information.
        
        Args:
            signal: Trade signal with probability
            regime_assessment: Optional hierarchical regime assessment
            current_drawdown: Current account drawdown (0-1)
            account_balance: Optional balance for dynamic sizing
        
        Returns:
            PositionSizeResult with computed stake and diagnostics
        """
        probability = signal.probability
        
        # Determine volatility regime string
        volatility_regime = "normal"
        model_confidence = 1.0
        
        if regime_assessment:
            # Map volatility regime to string
            volatility_regime = regime_assessment.volatility.value  # "low", "medium", "high"
            
            # Use trust score as confidence multiplier
            model_confidence = regime_assessment.trust_score
        
        result = self.kelly_sizer.compute_stake(
            probability=probability,
            payout_ratio=self.payout_ratio,
            model_confidence=model_confidence,
            current_drawdown=current_drawdown,
            volatility_regime=volatility_regime,
            account_balance=account_balance,
        )
        
        return result
    
    def create_stake_resolver(
        self,
        get_regime_assessment: Any = None,
        get_drawdown: Any = None,
    ):
        """
        Create a stake resolver function for SafeTradeExecutor.
        
        Args:
            get_regime_assessment: Callable returning HierarchicalRegimeAssessment
            get_drawdown: Callable returning current drawdown
        
        Returns:
            Callable[TradeSignal] -> float suitable for SafeTradeExecutor
        """
        def stake_resolver(signal: TradeSignal) -> float:
            regime = get_regime_assessment() if get_regime_assessment else None
            drawdown = get_drawdown() if get_drawdown else 0.0
            
            result = self.compute_stake_for_signal(
                signal=signal,
                regime_assessment=regime,
                current_drawdown=drawdown,
            )
            return result.stake
        
        return stake_resolver


class EnhancedRegimeVeto(RegimeVeto):
    """
    Adapter that makes HierarchicalRegimeDetector compatible with RegimeVeto interface.
    
    This allows the DecisionEngine to use hierarchical regime detection
    without code changes, by adapting the interface.
    
    The adapter caches recent price data for the hierarchical assessment.
    """
    
    def __init__(
        self,
        hierarchical_detector: HierarchicalRegimeDetector | None = None,
        fallback_veto: RegimeVeto | None = None,
    ):
        """
        Initialize enhanced regime veto.
        
        Args:
            hierarchical_detector: HierarchicalRegimeDetector to use
            fallback_veto: Traditional RegimeVeto for fallback
        """
        self.hierarchical = hierarchical_detector or HierarchicalRegimeDetector()
        self.fallback = fallback_veto or RegimeVeto()
        self._price_cache: np.ndarray | None = None
        
        # Expose thresholds for compatibility
        self.threshold_caution = self.fallback.threshold_caution
        self.threshold_veto = self.fallback.threshold_veto
    
    def update_prices(self, prices: np.ndarray) -> None:
        """
        Update the cached price history for hierarchical assessment.
        
        Should be called before assess() with recent close prices.
        """
        self._price_cache = prices
    
    def assess(self, reconstruction_error: float | Any) -> RegimeAssessmentProtocol:
        # Check for Tensor
        if hasattr(reconstruction_error, "item"):
             reconstruction_error = reconstruction_error.item()
         
        """
        Assess market regime using hierarchical detection.
        
        Falls back to simple reconstruction error if no price data available.
        
        Args:
            reconstruction_error: Volatility expert reconstruction error
        
        Returns:
            HierarchicalRegimeAssessment (compatible with RegimeAssessment interface)
        """
        if self._price_cache is not None and len(self._price_cache) >= 50:
            hier_assessment = self.hierarchical.assess(
                prices=self._price_cache,
                reconstruction_error=reconstruction_error,
            )
        else:
            # Fallback: use reconstruction error only
            hier_assessment = self.hierarchical.assess_from_reconstruction_error(
                reconstruction_error=reconstruction_error,
                prices=self._price_cache,
            )
            
        return hier_assessment

    
    def get_hierarchical_assessment(
        self,
        prices: np.ndarray,
        reconstruction_error: float,
    ) -> HierarchicalRegimeAssessment:
        """
        Get full hierarchical assessment with provided prices.
        
        Use this when you have price data available at call time.
        """
        return self.hierarchical.assess(prices, reconstruction_error)


def create_enhanced_executor(
    raw_executor: Any,
    settings: Settings,
    position_sizer: RegimeAwarePositionSizer | None = None,
    state_file: Path | None = None,
) -> Any:
    """
    Create a SafeTradeExecutor with enhanced position sizing.
    
    Args:
        raw_executor: Underlying trade executor (e.g., DerivTradeExecutor)
        settings: Application settings
        position_sizer: Optional regime-aware position sizer
        state_file: Optional path for safety state persistence
    
    Returns:
        SafeTradeExecutor with Kelly-based position sizing
    """
    from execution.safety import ExecutionSafetyConfig, SafeTradeExecutor
    
    # Create safety config from settings
    config = ExecutionSafetyConfig(
        max_trades_per_minute=getattr(settings.execution_safety, 'max_trades_per_minute', 5),
        max_trades_per_minute_per_symbol=getattr(settings.execution_safety, 'max_trades_per_minute_per_symbol', 3),
        max_daily_loss=getattr(settings.execution_safety, 'max_daily_loss', 50.0),
        max_stake_per_trade=getattr(settings.execution_safety, 'max_stake_per_trade', 10.0),
        max_consecutive_failures=getattr(settings.execution_safety, 'max_consecutive_failures', 10),
    )
    
    # Create stake resolver if position sizer provided
    stake_resolver = None
    if position_sizer:
        stake_resolver = position_sizer.create_stake_resolver()
    
    return SafeTradeExecutor(
        executor=raw_executor,
        config=config,
        stake_resolver=stake_resolver,
        state_file=state_file,
    )


def create_enhanced_engine(
    settings: Settings,
    shadow_store: Any = None,
    model_version: str = "unknown",
    use_hierarchical_regime: bool = True,
) -> Any:
    """
    Create a DecisionEngine with enhanced regime detection.
    
    Args:
        settings: Application settings
        shadow_store: ShadowTradeStore for trade logging
        model_version: Model version string
        use_hierarchical_regime: Whether to use hierarchical regime detection
    
    Returns:
        DecisionEngine with optional hierarchical regime
    """
    from execution.decision import DecisionEngine
    
    if use_hierarchical_regime:
        regime_veto: RegimeVeto = EnhancedRegimeVeto()
    else:
        regime_veto = RegimeVeto(
            threshold_caution=0.1,
            threshold_veto=0.3,
        )
    
    return DecisionEngine(
        settings=settings,
        regime_veto=regime_veto,
        shadow_store=shadow_store,
        model_version=model_version,
    )
