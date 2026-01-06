import asyncio
import logging
from datetime import datetime, timezone
from typing import Any

import numpy as np

from config.settings import Settings
from execution.contract_params import ContractDurationResolver
from execution.decision_logic import process_signals_batch
from execution.policy import ExecutionPolicy, SafetyProfile, VetoPrecedence
from execution.regime import RegimeAssessmentProtocol
# Use Protocol for type hinting to break dependency cycle risk
from execution.regime.types import RegimeVetoProtocol
from execution.safety_store import SQLiteSafetyStateStore
from execution.shadow_store import ShadowTradeStore
from execution.sqlite_shadow_store import SQLiteShadowStore
from execution.signals import TradeSignal
from execution.calibration import ProbabilityCalibrator
from execution.ensemble import create_ensemble, EnsembleStrategy

from .shadow import ShadowDispatcher
from .metrics import DecisionMetrics
from .safety import SafetyStateSynchronizer
from .processor import SignalProcessor

try:
    from opentelemetry import trace
    TRACING_ENABLED = True
except ImportError:
    TRACING_ENABLED = False

logger = logging.getLogger(__name__)


class DecisionEngine:
    """
    Central coordinator for trading decisions with ABSOLUTE regime veto authority.

    Uses ExecutionPolicy to enforce strict veto hierarchy.
    Delegates logic to dedicated modules for Micro-Modularity (C-004):
    - SignalProcessor: Calibration and Filtering
    - SafetyStateSynchronizer: DB State Sync
    - DecisionMetrics: Telemetry
    - ShadowDispatcher: Paper Trading
    
    Decision Hierarchy (STRICTLY ENFORCED):
    1. **Regime Veto** (Absolute Authority): If volatility anomaly > threshold, BLOCK ALL TRADES.
    2. **Execution Policy** (Safety Wrapper): Kill Switch, Circuit Breaker, P&L Caps.
    3. **Regime Caution**: If anomaly > caution_threshold, strict filtering (demote weak signals to shadow).
    4. **Confidence Filtering**: Standard probability threshold check.
    5. **Shadow Trade Logging**: All valid signals are logged for counterfactual analysis.
    """

    def __init__(
        self,
        settings: Settings,
        regime_veto: RegimeVetoProtocol | None = None,
        shadow_store: ShadowTradeStore | SQLiteShadowStore | None = None,
        safety_store: SQLiteSafetyStateStore | None = None,
        policy: ExecutionPolicy | None = None,
        model_version: str = "unknown",
        execution_mode: str = "REAL",
    ):
        self.settings = settings
        self.execution_mode = execution_mode
        self.model_version = model_version
        
        # Micro-Module: Safety Sync
        self.safety_sync = SafetyStateSynchronizer(safety_store)
        
        # Micro-Module: Metrics
        self.metrics = DecisionMetrics()
        
        # IMPORTANT-002: Link regime thresholds to settings
        if regime_veto is None:
            # Lazy import concrete implementation to avoid circular dependency
            from execution.regime import RegimeVeto
            caution = getattr(settings.hyperparams, "regime_caution_threshold", 0.1)
            veto = getattr(settings.hyperparams, "regime_veto_threshold", 0.3)
            self.regime_veto: RegimeVetoProtocol = RegimeVeto(threshold_caution=caution, threshold_veto=veto)
        else:
            self.regime_veto = regime_veto

        self.duration_resolver = ContractDurationResolver(settings)
        
        # AUDIT-FIX: Pass configurable circuit breaker timeout from settings
        self.policy = policy or ExecutionPolicy(
            circuit_breaker_reset_minutes=settings.execution_safety.circuit_breaker_reset_minutes
        )
        
        self._last_reconstruction_error = 0.0
        
        # IMPORTANT-004: Apply safety profile with dynamic providers
        # We bind to safety_sync.get_current_pnl() instead of local variable
        SafetyProfile.apply(
            policy=self.policy, 
            settings=settings,
            pnl_provider=self.safety_sync.get_current_pnl,
            calibration_provider=lambda: self._last_reconstruction_error
        )
        
        # Register Regime Veto (L4) into the policy
        self.policy.register_veto(
            level=VetoPrecedence.REGIME,
            check_fn=lambda reconstruction_error=0.0, **kwargs: self.regime_veto.assess(reconstruction_error).is_vetoed(),
            reason=lambda: self._generate_regime_veto_reason(self._last_reconstruction_error),
            use_context=True
        )

        

        
        if TRACING_ENABLED:
            self.tracer: Any = trace.get_tracer(__name__)
        else:
            self.tracer = None
            
        # R03: Initialize Calibration and Ensemble
        self.calibrator = ProbabilityCalibrator(settings.prob_calibration)
        self.ensemble = create_ensemble(settings.ensemble)
        
        # Micro-Module: Processor
        self.processor = SignalProcessor(settings, self.calibrator)
        
        # Initialize Shadow Dispatcher
        self.shadow_dispatcher = ShadowDispatcher(
            settings=settings,
            shadow_store=shadow_store,
            model_version=model_version,
            execution_mode=execution_mode
        )
        
        # Safely get store path name for logging
        shadow_name = "None"
        if shadow_store and hasattr(shadow_store, "_store_path"):
            shadow_name = str(shadow_store._store_path.name)
        elif shadow_store:
             shadow_name = "Available"
             
        safety_name = "None"
        if safety_store and hasattr(safety_store, "db_path"):
            safety_name = str(safety_store.db_path.name)
        
        logger.info(
            f"DecisionEngine initialized with regime thresholds: "
            f"CAUTION={self.regime_veto.threshold_caution:.3f}, "
            f"VETO={self.regime_veto.threshold_veto:.3f}, "
            f"shadow_store={shadow_name}, safety_store={safety_name}"
        )

        # C-003: Warmup Tracking
        self._warmup_candles_observed: int = 0
        self._warmup_threshold: int = settings.data_shapes.warmup_steps
        self._warmup_complete: bool = False
        logger.info(f"Warmup configured: needs {self._warmup_threshold} candles before trading.")

    async def process_model_output(
        self,
        probs: dict[str, float],
        reconstruction_error: float,
        timestamp: datetime | None = None,
        market_data: dict[str, Any] | None = None,
        entry_price: float | None = None,
    ) -> list[TradeSignal]:
        """
        Main entry point for decision making with ABSOLUTE regime veto authority.
        """
        if timestamp is None:
            timestamp = datetime.now(timezone.utc)

        # C-003: Warmup Validation
        # Increment observed count (assuming 1 call = 1 new candle processed in this context)
        # Note: If called per-tick, this logic needs adjustment. Assuming called per-candle here.
        if not self._warmup_complete:
            self._warmup_candles_observed += 1
            if self._warmup_candles_observed < self._warmup_threshold:
                if self._warmup_candles_observed % 10 == 0:
                    logger.debug(f"Warmup: {self._warmup_candles_observed}/{self._warmup_threshold} candles")
                return []
            else:
                 self._warmup_complete = True
                 logger.info("Warmup Period Complete: System ready for trading")

        # 0. Sync Safety State
        await self.safety_sync.sync()

        try:
            reconstruction_error = float(reconstruction_error)
        except (TypeError, ValueError):
            pass
            
        self._last_reconstruction_error = reconstruction_error
        
        span = None
        if self.tracer:
            span = self.tracer.start_span("decision_engine.process_model_output")
            span.set_attribute("model_version", self.model_version)
            span.set_attribute("reconstruction_error", reconstruction_error)

        try:
            # Check execution policy (Strict Enforcement)
            # CRITICAL-001: Use async check to avoid blocking loop on DB calls
            policy_veto = await self.policy.async_check_vetoes(reconstruction_error=reconstruction_error)
            if policy_veto:
                if span:
                    span.set_attribute("veto.type", "policy")
                    span.set_attribute("veto.reason", str(policy_veto))
                logger.warning(f"TRADE BLOCKED BY EXECUTION POLICY: {policy_veto}")
                self.metrics.increment("processed")
                if policy_veto.level == VetoPrecedence.REGIME:
                        self.metrics.increment("regime_vetoed")
                else:
                        self.metrics.increment("ignored")
                return []

            regime_assessment = self.regime_veto.assess(reconstruction_error)
            if span:
                span.set_attribute("regime_state", self.get_regime_state_string(regime_assessment))

            # Delegate to SignalProcessor (R02/R03)
            all_signals = self.processor.process(probs, timestamp)
            
            if span:
                span.set_attribute("signals.filtered_count", len(all_signals))

            # Dispatch shadow trades (Background)
            self.shadow_dispatcher.dispatch(
                signals=all_signals,
                reconstruction_error=reconstruction_error,
                regime_assessment=regime_assessment,
                entry_price=entry_price or 0.0,
                market_data=market_data
            )

            # Delegate to core logic
            # Note: process_signals_batch still expects a 'stats' dict. 
            # We pass metrics.get_statistics() ? No, it likely iterates the dict.
            # Ideally we refactor process_signals_batch too, but for now we might need to pass the dict ref?
            # Or wrapping metrics to look like dict?
            # Let's peek at process_signals_batch. It probably updates 'stats'.
            # "stats=self._stats" was passed.
            # We can expose the internal dict for legacy compatibility or refactor the callee.
            # For this step, we'll expose access to the internal dict via a property or just pass metrics._stats.
            
            real_trades, shadow_trades = process_signals_batch(
                all_signals=all_signals,
                regime_assessment=regime_assessment,
                settings=self.settings,
                reconstruction_error=reconstruction_error,
                stats=self.metrics._stats, # Direct access for compatibility
                tracer=span
            )
            
            # Dispatch demoted shadow trades
            # R04 Fix: Removed redundant shadow dispatch.
            # all_signals are already dispatched above.

            if span:
                span.set_attribute("signals.real_count", len(real_trades))
            
            return real_trades

        finally:
            if span:
                span.end()

    async def process_with_context(
        self,
        probs: dict[str, float],
        reconstruction_error: float,
        tick_window: np.ndarray,
        candle_window: np.ndarray,
        entry_price: float,
        timestamp: datetime | None = None,
        market_data: dict[str, Any] | None = None,
    ) -> list[TradeSignal]:
        """
        Process model output with full market context for shadow trade capture.
        """
        if timestamp is None:
            timestamp = datetime.now(timezone.utc)

        assessment = self.regime_veto.assess(reconstruction_error)
        
        span = None
        if self.tracer:
            span = self.tracer.start_span("decision_engine.process_with_context")
            span.set_attribute("model_version", self.model_version)
            span.set_attribute("reconstruction_error", reconstruction_error)

        try:
            # Delegate to SignalProcessor
            all_signals = self.processor.process(probs, timestamp)
            
            if span:
                span.set_attribute("signals_count", len(all_signals))

            # Dispatch shadow trades with context
            self.shadow_dispatcher.dispatch(
                signals=all_signals,
                reconstruction_error=reconstruction_error,
                regime_assessment=assessment,
                entry_price=entry_price,
                market_data=market_data,
                tick_window=tick_window,
                candle_window=candle_window
            )

            # Delegate to core logic
            real_trades, shadow_trades = process_signals_batch(
                all_signals=all_signals,
                regime_assessment=assessment,
                settings=self.settings,
                reconstruction_error=reconstruction_error,
                stats=self.metrics._stats, # Compat
                tracer=span
            )

            # R04 Fix: Removed redundant shadow dispatch. 
            # all_signals are already dispatched above. 
            # shadow_trades are just a subset (demoted signals).

            if span:
                span.set_attribute("real_trades_count", len(real_trades))
                
            return real_trades

        finally:
            if span:
                span.end()

    def get_regime_assessment(self, reconstruction_error: float) -> RegimeAssessmentProtocol:
        """Get regime assessment without processing."""
        return self.regime_veto.assess(reconstruction_error)

    def get_statistics(self) -> dict[str, Any]:
        """Get decision statistics including regime veto counts."""
        return self.metrics.get_statistics()

    def get_regime_state_string(self, assessment: RegimeAssessmentProtocol) -> str:
        """Extract regime state string."""
        if hasattr(assessment, "trust_state") and hasattr(assessment.trust_state, "value"):
            return str(assessment.trust_state.value)
        if assessment.is_vetoed():
            return "veto"
        elif assessment.requires_caution():
            return "caution"
        else:
            return "trusted"

    async def sync_safety_state(self, force: bool = False) -> None:
        """Sync current P&L from SafetyStateStore."""
        await self.safety_sync.sync(force)

    async def shutdown(self, timeout: float = 5.0) -> None:
        """Gracefully shutdown the engine, flushing pending tasks."""
        await self.shadow_dispatcher.shutdown(timeout=timeout)

    def _generate_regime_veto_reason(self, error: float) -> str:
        """Generate detailed reason for regime veto."""
        assessment = self.regime_veto.assess(error)
        details = ""
        # Check for hierarchical details
        if hasattr(assessment, "micro") and hasattr(assessment.micro, "value"):
             details += f" Micro={assessment.micro.value}"
        if hasattr(assessment, "volatility") and hasattr(assessment.volatility, "value"):
             details += f" Vol={assessment.volatility.value}"
        
        return f"Market anomaly detected (Regime Veto L4). Error: {error:.3f}{details}"
