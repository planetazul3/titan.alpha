import asyncio
import logging
from datetime import datetime, timezone
from typing import Any

import numpy as np

from config.settings import Settings
from execution.contract_params import ContractDurationResolver
from execution.decision_logic import process_signals_batch
from execution.filters import filter_signals
from execution.policy import ExecutionPolicy, SafetyProfile, VetoPrecedence
from execution.regime import RegimeAssessmentProtocol, RegimeVeto
from execution.safety_store import SQLiteSafetyStateStore
from execution.shadow_ops import fire_shadow_trade_task
from execution.shadow_store import ShadowTradeStore
from execution.sqlite_shadow_store import SQLiteShadowStore
from execution.signals import TradeSignal

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
    Delegates logic to `decision_logic` and `shadow_ops` for modularity.
    
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
        regime_veto: RegimeVeto | None = None,
        shadow_store: ShadowTradeStore | SQLiteShadowStore | None = None,
        safety_store: SQLiteSafetyStateStore | None = None,
        policy: ExecutionPolicy | None = None,
        model_version: str = "unknown",
        execution_mode: str = "REAL",
    ):
        self.settings = settings
        self.execution_mode = execution_mode
        
        # IMPORTANT-002: Link regime thresholds to settings
        if regime_veto is None:
            caution = getattr(settings.hyperparams, "regime_caution_threshold", 0.1)
            veto = getattr(settings.hyperparams, "regime_veto_threshold", 0.3)
            self.regime_veto = RegimeVeto(threshold_caution=caution, threshold_veto=veto)
        else:
            self.regime_veto = regime_veto

        self.shadow_store = shadow_store
        self.safety_store = safety_store
        self.duration_resolver = ContractDurationResolver(settings)
        self.model_version = model_version
        
        # AUDIT-FIX: Pass configurable circuit breaker timeout from settings
        self.policy = policy or ExecutionPolicy(
            circuit_breaker_reset_minutes=settings.execution_safety.circuit_breaker_reset_minutes
        )
        
        # State trackers for policy providers
        self._current_daily_pnl = 0.0
        self._last_reconstruction_error = 0.0
        
        # IMPORTANT-004: Apply safety profile with dynamic providers
        SafetyProfile.apply(
            policy=self.policy, 
            settings=settings,
            pnl_provider=lambda: self._current_daily_pnl,
            calibration_provider=lambda: self._last_reconstruction_error
        )
        
        # Register Regime Veto (L4) into the policy
        self.policy.register_veto(
            level=VetoPrecedence.REGIME,
            check_fn=lambda reconstruction_error=0.0, **kwargs: self.regime_veto.assess(reconstruction_error).is_vetoed(),
            reason=lambda: f"Market anomaly detected (Regime Veto L4). Error: {self._last_reconstruction_error:.3f}",
            use_context=True
        )
        
        if TRACING_ENABLED:
            self.tracer: Any = trace.get_tracer(__name__)
        else:
            self.tracer = None
            
        self._stats = {
            "processed": 0,
            "real": 0,
            "shadow": 0,
            "ignored": 0,
            "regime_vetoed": 0,
            "regime_caution": 0,
        }
        
        # Track pending background tasks for graceful shutdown
        self._pending_shadow_tasks: set[asyncio.Task] = set()
        
        # PERF: Throttle safety state sync
        self._last_safety_sync: float = 0.0
        self._safety_sync_interval: float = 5.0  # seconds
        
        logger.info(
            f"DecisionEngine initialized with regime thresholds: "
            f"CAUTION={self.regime_veto.threshold_caution:.3f}, "
            f"VETO={self.regime_veto.threshold_veto:.3f}"
            + (f", shadow_store={shadow_store._store_path.name}" if shadow_store else "")
            + (f", safety_store={safety_store.db_path.name}" if safety_store else "")
        )

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

        # 0. Sync Safety State if possible
        if self.safety_store:
            await self.sync_safety_state()

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
            policy_veto = self.policy.check_vetoes(reconstruction_error=reconstruction_error)
            if policy_veto:
                if span:
                    span.set_attribute("veto.type", "policy")
                    span.set_attribute("veto.reason", str(policy_veto))
                logger.warning(f"TRADE BLOCKED BY EXECUTION POLICY: {policy_veto}")
                self._stats["processed"] += 1
                if policy_veto.level == VetoPrecedence.REGIME:
                        self._stats["regime_vetoed"] += 1
                else:
                        self._stats["ignored"] += 1
                return []

            regime_assessment = self.regime_veto.assess(reconstruction_error)
            if span:
                span.set_attribute("regime_state", self._get_regime_state_string(regime_assessment))

            # R02: Filter probabilities into signals
            all_signals = filter_signals(probs, self.settings, timestamp)
            if span:
                span.set_attribute("signals.filtered_count", len(all_signals))

            # Store shadow trades in background
            if self.shadow_store:
                for sig in all_signals:
                    fire_shadow_trade_task(
                        pending_tasks=self._pending_shadow_tasks,
                        store=self.shadow_store,
                        settings=self.settings,
                        resolver=self.duration_resolver,
                        model_version=self.model_version,
                        execution_mode=self.execution_mode,
                        signal=sig,
                        reconstruction_error=reconstruction_error,
                        regime_state=self._get_regime_state_string(regime_assessment),
                        entry_price=entry_price or 0.0,
                        regime_vetoed=regime_assessment.is_vetoed(),
                        metadata=market_data,
                    )

            # Delegate to core logic
            real_trades, shadow_trades = process_signals_batch(
                all_signals=all_signals,
                regime_assessment=regime_assessment,
                settings=self.settings,
                reconstruction_error=reconstruction_error,
                stats=self._stats,
                tracer=span
            )
            
            # Fire tasks for demoted shadow trades
            if self.shadow_store:
                 for st in shadow_trades:
                     fire_shadow_trade_task(
                        pending_tasks=self._pending_shadow_tasks,
                        store=self.shadow_store,
                        settings=self.settings,
                        resolver=self.duration_resolver,
                        model_version=self.model_version,
                        execution_mode=self.execution_mode,
                        signal=st, # ShadowTrade matches TradeSignal protocol for storage
                        reconstruction_error=reconstruction_error,
                        regime_state=self._get_regime_state_string(regime_assessment),
                        entry_price=entry_price or 0.0,
                        regime_vetoed=regime_assessment.is_vetoed(),
                        metadata=market_data,
                    )

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
        regime_state = self._get_regime_state_string(assessment)
        
        span = None
        if self.tracer:
            span = self.tracer.start_span("decision_engine.process_with_context")
            span.set_attribute("model_version", self.model_version)
            span.set_attribute("reconstruction_error", reconstruction_error)

        try:
            # I02 Fix: Get all signals ONCE and reuse
            all_signals = filter_signals(probs, self.settings, timestamp)
            if span:
                span.set_attribute("signals_count", len(all_signals))

            # Store shadow trades with full context
            if self.shadow_store:
                for sig in all_signals:
                    fire_shadow_trade_task(
                        pending_tasks=self._pending_shadow_tasks,
                        store=self.shadow_store,
                        settings=self.settings,
                        resolver=self.duration_resolver,
                        model_version=self.model_version,
                        execution_mode=self.execution_mode,
                        signal=sig,
                        reconstruction_error=reconstruction_error,
                        regime_state=regime_state,
                        entry_price=entry_price,
                        tick_window=tick_window,
                        candle_window=candle_window,
                        regime_vetoed=assessment.is_vetoed(),
                        metadata=market_data,
                    )

            # Delegate to core logic
            real_trades, shadow_trades = process_signals_batch(
                all_signals=all_signals,
                regime_assessment=assessment,
                settings=self.settings,
                reconstruction_error=reconstruction_error,
                stats=self._stats,
                tracer=span
            )

             # Fire tasks for demoted shadow trades
            if self.shadow_store:
                 for st in shadow_trades:
                     fire_shadow_trade_task(
                        pending_tasks=self._pending_shadow_tasks,
                        store=self.shadow_store,
                        settings=self.settings,
                        resolver=self.duration_resolver,
                        model_version=self.model_version,
                        execution_mode=self.execution_mode,
                        signal=st,
                        reconstruction_error=reconstruction_error,
                        regime_state=regime_state,
                        entry_price=entry_price,
                        tick_window=tick_window,
                        candle_window=candle_window,
                        regime_vetoed=assessment.is_vetoed(),
                        metadata=market_data,
                    )

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
        return self._stats.copy()

    def _get_regime_state_string(self, assessment: RegimeAssessmentProtocol) -> str:
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
        if not self.safety_store:
            return
            
        import time
        now = time.monotonic()
        
        if not force and (now - self._last_safety_sync) < self._safety_sync_interval:
            return
            
        try:
            _, daily_pnl = await self.safety_store.get_daily_stats_async()
            from utils.numerical_validation import ensure_finite
            self._current_daily_pnl = ensure_finite(
                daily_pnl, 
                "DecisionEngine.sync_pnl", 
                default=0.0
            ) 
            self._last_safety_sync = now
        except Exception as e:
            logger.error(f"Failed to sync safety state: {e}")

    async def shutdown(self, timeout: float = 5.0) -> None:
        """Gracefully shutdown the engine, flushing pending tasks."""
        if not self._pending_shadow_tasks:
            return

        logger.info(f"DecisionEngine: Waiting for {len(self._pending_shadow_tasks)} shadow tasks to flush...")
        _, pending = await asyncio.wait(self._pending_shadow_tasks, timeout=timeout)
        
        if pending:
            logger.warning(f"DecisionEngine shutdown timed out with {len(pending)} tasks remaining.")
            for task in pending:
                task.cancel()
        else:
            logger.info("DecisionEngine: All shadow tasks flushed successfully.")
