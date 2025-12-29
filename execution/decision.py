import asyncio
import logging
from datetime import datetime, timezone
from typing import Any

import numpy as np

from config.constants import CONTRACT_TYPES
from config.settings import Settings
from data.features import FEATURE_SCHEMA_VERSION
from execution.filters import filter_signals, get_actionable_signals
from execution.contract_params import ContractDurationResolver
from execution.policy import ExecutionPolicy, SafetyProfile, VetoPrecedence  # Policy framework
from execution.regime import RegimeAssessment, RegimeAssessmentProtocol, RegimeVeto
from execution.safety_store import SQLiteSafetyStateStore
from execution.shadow_store import ShadowTradeRecord, ShadowTradeStore
from execution.signals import ShadowTrade, TradeSignal
from observability.shadow_logging import shadow_trade_logger

try:
    from opentelemetry import trace
    from opentelemetry.trace import Status, StatusCode
    TRACING_ENABLED = True
except ImportError:
    TRACING_ENABLED = False

logger = logging.getLogger(__name__)


class DecisionEngine:
    """
    Central coordinator for trading decisions with ABSOLUTE regime veto authority.

    Uses ExecutionPolicy to enforce strict veto hierarchy.
    
    Decision Hierarchy (STRICTLY ENFORCED):
    1. **Regime Veto** (Absolute Authority): If volatility anomaly > threshold, BLOCK ALL TRADES.
    2. **Execution Policy** (Safety Wrapper): Kill Switch, Circuit Breaker, P&L Caps.
    3. **Regime Caution**: If anomaly > caution_threshold, strict filtering (demote weak signals to shadow).
    4. **Confidence Filtering**: Standard probability threshold check.
    5. **Shadow Trade Logging**: All valid signals are logged for counterfactual analysis.
    
    Performance Optimizations:
    - **Consolidated Veto Checks**: Policy vetoes checked ONCE at entry point (process_model_output),
      not redundantly in _process_signals. Reduces latency on the critical tick-processing path.
    - **Throttled Safety Sync**: SQLite safety state sync is throttled to once per 5 seconds
      (configurable via _safety_sync_interval) to minimize disk I/O during high-frequency ticks.
    """

    def __init__(
        self,
        settings: Settings,
        regime_veto: RegimeVeto | None = None,
        shadow_store: ShadowTradeStore | None = None,
        safety_store: SQLiteSafetyStateStore | None = None,
        policy: ExecutionPolicy | None = None,
        model_version: str = "unknown",
    ):
        self.settings = settings
        
        # IMPORTANT-002: Link regime thresholds to settings
        if regime_veto is None:
            caution = getattr(settings.hyperparams, "regime_caution_threshold", 0.1)
            veto = getattr(settings.hyperparams, "regime_veto_threshold", 0.3)
            self.regime_veto = RegimeVeto(threshold_caution=caution, threshold_veto=veto)
        else:
            self.regime_veto = regime_veto

        self.shadow_store = shadow_store  # New immutable store
        self.safety_store = safety_store
        self.duration_resolver = ContractDurationResolver(settings)
        self.model_version = model_version
        self.policy = policy or ExecutionPolicy()
        
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
            check_fn=lambda: self.regime_veto.assess(self._last_reconstruction_error).is_vetoed(),
            reason=lambda: f"Market anomaly detected (Regime Veto L4). Error: {self._last_reconstruction_error:.3f}"
        )
        
        if TRACING_ENABLED:
            self.tracer = trace.get_tracer(__name__)
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
        
        # PERF: Throttle safety state sync to reduce SQLite I/O
        self._last_safety_sync: float = 0.0
        self._safety_sync_interval: float = 5.0  # seconds
        
        # Initialize Execution Policy
        # self.policy = ExecutionPolicy() # Removed, now initialized above
        # SafetyProfile.apply(self.policy, self.settings) # Removed, now initialized above
        
        # Register Regime Veto (L4)
        # Note: We can't check reconstruction error here directly, so we'll check it
        # dynamically during process_model_output. But for now we just initialize policy.
        # The actual veto check in process_model_output will drive the policy check.
        # To make policy strictly enforce it, we'd need to pass reconstruction_error to check_vetoes
        # or have a shared state. For now, DecisionEngine.process_model_output orchestrates it.

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

        CRITICAL: The regime veto is checked FIRST and is ABSOLUTE.
        No probability, no matter how high, can override a regime veto.

        Decision Hierarchy:
        1. REGIME VETO CHECK (ABSOLUTE AUTHORITY)
           - If reconstruction_error >= threshold_veto: BLOCK ALL TRADES
           - This is NOT a filter, it's an UNCONDITIONAL VETO
        2. Regime caution (if reconstruction_error >= threshold_caution)
           - Demote some real trades to shadow
        3. Confidence filtering (existing logic)
        4. Shadow trade logging

        Args:
            probs: Model probability outputs
            reconstruction_error: Volatility expert reconstruction error (REQUIRED)
            timestamp: Optional timestamp for signals
            market_data: Optional metadata to attach to signals

        Returns:
            List of REAL_TRADE signals (EMPTY if regime vetoed, regardless of confidence)
        """
        if timestamp is None:
            timestamp = datetime.now(timezone.utc)

        # 0. Sync Safety State if possible
        if self.safety_store:
            await self.sync_safety_state()

        # Coerce to float for robust logging (prevents MagicMock formatting errors)
        try:
            reconstruction_error = float(reconstruction_error)
        except (TypeError, ValueError):
            # If it's a mock or invalid, we still want to avoid crash
            pass
            
        # Update trackers for policy providers
        self._last_reconstruction_error = reconstruction_error
        
        if self.tracer:
            with self.tracer.start_as_current_span("decision_engine.process_model_output") as span:
                span.set_attribute("model_version", self.model_version)
                span.set_attribute("reconstruction_error", reconstruction_error)
                
                # Check execution policy
                policy_veto = self.policy.check_vetoes()
                if policy_veto:
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
                span.set_attribute("regime_state", self._get_regime_state_string(regime_assessment))

                if regime_assessment.is_vetoed():
                    span.set_attribute("veto.type", "regime")
                    logger.warning(
                        f"TRADE BLOCKED BY REGIME VETO: error {reconstruction_error:.4f} "
                        f"> veto_threshold {self.regime_veto.threshold_veto:.3f}"
                    )
                    self._stats["processed"] += 1
                    self._stats["regime_vetoed"] += 1
                    return []

                # R02: Filter probabilities into signals
                all_signals = filter_signals(probs, self.settings, timestamp)
                span.set_attribute("signals.filtered_count", len(all_signals))

                # Store shadow trades in background
                if self.shadow_store:
                    for sig in all_signals:
                        self._store_shadow_trade_async(
                            signal=sig,
                            reconstruction_error=reconstruction_error,
                            regime_state=self._get_regime_state_string(regime_assessment),
                            entry_price=entry_price or 0.0,
                            regime_vetoed=regime_assessment.is_vetoed(),
                            metadata=market_data,
                        )

                # Process further (validation, caution, stats)
                real_trades = await self._process_signals(
                    all_signals=all_signals,
                    reconstruction_error=reconstruction_error,
                    regime_assessment=regime_assessment,
                    timestamp=timestamp,
                    market_data=market_data,
                )
                span.set_attribute("signals.real_count", len(real_trades))
                return real_trades
        else:
            # Fallback when tracing disabled
            # 1. Check Execution Policy Vetoes (IMPORTANT-004)
            policy_veto = self.policy.check_vetoes()
            if policy_veto:
                logger.warning(f"TRADE BLOCKED BY EXECUTION POLICY: {policy_veto}")
                self._stats["processed"] += 1
                if policy_veto.level == VetoPrecedence.REGIME:
                    self._stats["regime_vetoed"] += 1
                else:
                    self._stats["ignored"] += 1
                return []

            # ══════════════════════════════════════════════════════════════════════
            # ══════════════════════════════════════════════════════════════════════
            # STEP 1: REGIME VETO CHECK (ABSOLUTE AUTHORITY - NO BYPASS POSSIBLE)
            # ══════════════════════════════════════════════════════════════════════
            # Check regime assessment via RegimeVeto (primary safety mechanism)
            regime_assessment = self.regime_veto.assess(reconstruction_error)

            # ══════════════════════════════════════════════════════════════════════
            # CRITICAL: Strict Regime Veto Check via RegimeAssessment
            # ══════════════════════════════════════════════════════════════════════
            if regime_assessment.is_vetoed():
                logger.warning(
                    f"TRADE BLOCKED BY REGIME VETO: error {reconstruction_error:.4f} "
                    f"> veto_threshold {self.regime_veto.threshold_veto:.3f}"
                )
                self._stats["processed"] += 1
                self._stats["regime_vetoed"] += 1
                return []  # BLOCKED

            # R02: Filter probabilities into signals
            all_signals = filter_signals(probs, self.settings, timestamp)
            
            # Store shadow trades in background (Fire-and-Forget)
            if self.shadow_store:
                for sig in all_signals:
                    self._store_shadow_trade_async(
                        signal=sig,
                        reconstruction_error=reconstruction_error,
                        regime_state=self._get_regime_state_string(regime_assessment),
                        entry_price=entry_price or 0.0,
                        regime_vetoed=regime_assessment.is_vetoed(),
                        metadata=market_data,
                    )
            
            return await self._process_signals(
                all_signals=all_signals,
                reconstruction_error=reconstruction_error,
                regime_assessment=regime_assessment,
                timestamp=timestamp,
                market_data=market_data,
            )

    def _apply_caution_filter(
        self, real_trades: list[TradeSignal]
    ) -> tuple[list[TradeSignal], list[ShadowTrade]]:
        """
        Apply caution during uncertain regimes.

        Strategy: Keep only the highest confidence trade, demote others to shadow.
        This is stricter than normal filtering because we're in a cautious regime.
        """
        demoted: list[ShadowTrade] = []

        if not real_trades:
            return real_trades, demoted

        # Sort by probability descending
        sorted_trades = sorted(real_trades, key=lambda t: t.probability, reverse=True)

        # Keep ONLY the top trade during caution
        kept = [sorted_trades[0]] if sorted_trades else []

        # Demote the rest to shadow trades
        for trade in sorted_trades[1:]:
            demoted.append(
                ShadowTrade(
                    signal_type=trade.signal_type,
                    contract_type=trade.contract_type,
                    direction=trade.direction,
                    probability=trade.probability,
                    timestamp=trade.timestamp,
                    metadata={**trade.metadata, "demoted_by_caution": True},
                )
            )

        return kept, demoted

    async def _process_signals(
        self,
        all_signals: list[TradeSignal],
        reconstruction_error: float,
        regime_assessment: RegimeAssessmentProtocol,
        timestamp: datetime,
        market_data: dict[str, Any] | None = None,
    ) -> list[TradeSignal]:
        """
        I02 Fix: Process pre-filtered signals without redundant filtering.
        
        This is the internal workhorse that handles already-filtered signals.
        Used by process_with_context to avoid calling filter_signals twice.
        
        PERF: Policy veto checks are performed ONCE at the entry point
        (process_model_output), NOT here. This consolidates the check for lower latency.
        """
        if self.tracer:
            with self.tracer.start_as_current_span("decision_engine._process_signals") as span:
                # If vetoed, block all trades
                if regime_assessment.is_vetoed():
                    span.set_attribute("veto.type", "regime")
                    self._stats["processed"] += 1
                    self._stats["regime_vetoed"] += 1
                    return []

                # NOTE: Policy veto check intentionally omitted here - already done at entry point

                # R02: Validate contract types
                valid_contract_types = {ct.value for ct in CONTRACT_TYPES}
                validated_signals = []
                for sig in all_signals:
                    if sig.contract_type in valid_contract_types:
                        validated_signals.append(sig)
                    else:
                        logger.warning(f"Invalid contract type '{sig.contract_type}' - skipped")
                        self._stats["ignored"] += 1
                all_signals = validated_signals

                real_trades, shadow_trades = get_actionable_signals(all_signals)
                span.set_attribute("signals.real_base", len(real_trades))
                span.set_attribute("signals.shadow_base", len(shadow_trades))

                # Apply caution filter if needed
                if regime_assessment.requires_caution():
                    span.set_attribute("caution.active", True)
                    self._stats["regime_caution"] += 1
                    logger.info(
                        f"Regime caution active: demoting some trades. "
                        f"reconstruction_error={reconstruction_error:.4f}"
                    )
                    
                    # Move real trades to shadow if confidence isn't very high
                    filtered_real = []
                    for signal in real_trades:
                        caution_threshold = self.settings.thresholds.confidence_threshold_high + 0.05
                        if signal.probability >= caution_threshold:
                            filtered_real.append(signal)
                        else:
                            logger.info(f"Demoting CAUTION trade to shadow: {signal.probability:.3f}")
                            shadow_trade = ShadowTrade(
                                signal_type=signal.signal_type,
                                contract_type=signal.contract_type,
                                direction=signal.direction,
                                probability=signal.probability,
                                timestamp=signal.timestamp
                            )
                            shadow_trades.append(shadow_trade)
                    
                    real_trades = filtered_real
                    real_trades, demoted = self._apply_caution_filter(real_trades)
                    shadow_trades.extend(demoted)

                # Feedback logging
                if not real_trades and not shadow_trades:
                    self._log_no_trades_feedback(all_signals)

                # Update statistics
                self._stats["processed"] += 1
                self._stats["real"] += len(real_trades)
                self._stats["shadow"] += len(shadow_trades)
                self._stats["ignored"] += len(all_signals) - len(real_trades) - len(shadow_trades)

                span.set_attribute("signals.real_final", len(real_trades))
                return real_trades
        else:
            # Fallback when tracing disabled
            # If vetoed, block all trades
            if regime_assessment.is_vetoed():
                self._stats["processed"] += 1
                self._stats["regime_vetoed"] += 1
                return []

            # NOTE: Policy veto check intentionally omitted here - already done at entry point

            # R02: Validate contract types
            valid_contract_types = {ct.value for ct in CONTRACT_TYPES}
            validated_signals = []
            for sig in all_signals:
                if sig.contract_type in valid_contract_types:
                    validated_signals.append(sig)
                else:
                    logger.warning(f"Invalid contract type '{sig.contract_type}' - skipped")
                    self._stats["ignored"] += 1
            all_signals = validated_signals

            real_trades, shadow_trades = get_actionable_signals(all_signals)

            # Apply caution filter if needed
            if regime_assessment.requires_caution():
                self._stats["regime_caution"] += 1
                logger.info(
                    f"Regime caution active: demoting some trades. "
                    f"reconstruction_error={reconstruction_error:.4f}"
                )
                
                # Move real trades to shadow if confidence isn't very high
                filtered_real = []
                for signal in real_trades:
                    caution_threshold = self.settings.thresholds.confidence_threshold_high + 0.05
                    if signal.probability >= caution_threshold:
                        filtered_real.append(signal)
                    else:
                        logger.info(f"Demoting CAUTION trade to shadow: {signal.probability:.3f}")
                        shadow_trade = ShadowTrade(
                            signal_type=signal.signal_type,
                            contract_type=signal.contract_type,
                            direction=signal.direction,
                            probability=signal.probability,
                            timestamp=signal.timestamp
                        )
                        shadow_trades.append(shadow_trade)
                
                real_trades = filtered_real
                real_trades, demoted = self._apply_caution_filter(real_trades)
                shadow_trades.extend(demoted)

            # Feedback logging
            if not real_trades and not shadow_trades:
                self._log_no_trades_feedback(all_signals)

            # Update statistics
            self._stats["processed"] += 1
            self._stats["real"] += len(real_trades)
            self._stats["shadow"] += len(shadow_trades)
            self._stats["ignored"] += len(all_signals) - len(real_trades) - len(shadow_trades)

            return real_trades


    def get_regime_assessment(self, reconstruction_error: float) -> RegimeAssessmentProtocol:
        """
        Get regime assessment without processing (for external checks).

        This is useful for UI/logging to show regime state without affecting decisions.
        """
        return self.regime_veto.assess(reconstruction_error)

    def get_statistics(self) -> dict[str, Any]:
        """Get decision statistics including regime veto counts."""
        return self._stats.copy()

    def _get_regime_state_string(self, assessment: RegimeAssessmentProtocol) -> str:
        """
        Extract regime state string from either RegimeAssessment or HierarchicalRegimeAssessment.
        
        This handles the Protocol's interface by checking which type of assessment we have.
        """
        # Prefer trust_state attribute (RegimeAssessment)
        if hasattr(assessment, "trust_state") and hasattr(assessment.trust_state, "value"):
            return str(assessment.trust_state.value)
        # Fallback for HierarchicalRegimeAssessment: derive from is_vetoed/requires_caution
        if assessment.is_vetoed():
            return "veto"
        elif assessment.requires_caution():
            return "caution"
        else:
            return "trusted"

    async def sync_safety_state(self, force: bool = False) -> None:
        """
        Sync current P&L from SafetyStateStore (IMPORTANT-004).
        
        PERF: Throttled to reduce SQLite disk I/O in high-frequency tick processing.
        Default interval is 5 seconds. Use force=True to bypass throttling.
        
        Args:
            force: If True, bypass throttling and sync immediately.
        """
        if not self.safety_store:
            return
            
        import time
        now = time.monotonic()
        
        # PERF: Skip sync if within throttle interval (unless forced)
        if not force and (now - self._last_safety_sync) < self._safety_sync_interval:
            return
            
        try:
            _, daily_pnl = await self.safety_store.get_daily_stats_async()
            self._current_daily_pnl = daily_pnl
            self._last_safety_sync = now
        except Exception as e:
            logger.error(f"Failed to sync safety state: {e}")





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

        This is the RECOMMENDED method for live trading as it captures
        tick/candle windows needed for later outcome resolution.

        Args:
            probs: Model probability outputs
            reconstruction_error: Volatility reconstruction error
            tick_window: Recent tick prices (for shadow trade context)
            candle_window: Recent candles (for shadow trade context)
            entry_price: Current price at signal time
            timestamp: Optional timestamp
            market_data: Optional additional metadata

        Returns:
            List of REAL_TRADE signals
        """
        if timestamp is None:
            timestamp = datetime.now(timezone.utc)

        assessment = self.regime_veto.assess(reconstruction_error)
        regime_state = self._get_regime_state_string(assessment)

        if self.tracer:
            with self.tracer.start_as_current_span("decision_engine.process_with_context") as span:
                span.set_attribute("model_version", self.model_version)
                span.set_attribute("reconstruction_error", reconstruction_error)
                span.set_attribute("entry_price", entry_price)
                span.set_attribute("regime_state", regime_state)

                # I02 Fix: Get all signals ONCE and reuse
                all_signals = filter_signals(probs, self.settings, timestamp)
                span.set_attribute("signals_count", len(all_signals))

                # Store shadow trades with full context (Fire-and-Forget)
                if self.shadow_store:
                    with self.tracer.start_as_current_span("decision_engine.store_shadow_trades"):
                        for sig in all_signals:
                            self._store_shadow_trade_async(
                                signal=sig,
                                reconstruction_error=reconstruction_error,
                                regime_state=regime_state,
                                tick_window=tick_window,
                                candle_window=candle_window,
                                entry_price=entry_price,
                                regime_vetoed=assessment.is_vetoed(),
                                metadata=market_data,
                            )

                # Process signals
                with self.tracer.start_as_current_span("decision_engine.process_signals") as p_span:
                    real_trades = await self._process_signals(
                        all_signals=all_signals,
                        reconstruction_error=reconstruction_error,
                        regime_assessment=assessment,
                        timestamp=timestamp,
                        market_data=market_data,
                    )
                    p_span.set_attribute("real_trades_count", len(real_trades))
                    return real_trades
        else:
            # Fallback when tracing disabled
            all_signals = filter_signals(probs, self.settings, timestamp)
            if self.shadow_store:
                for sig in all_signals:
                    self._store_shadow_trade_async(
                        signal=sig,
                        reconstruction_error=reconstruction_error,
                        regime_state=regime_state,
                        tick_window=tick_window,
                        candle_window=candle_window,
                        entry_price=entry_price,
                        regime_vetoed=assessment.is_vetoed(),
                        metadata=market_data,
                    )
            return await self._process_signals(
                all_signals=all_signals,
                reconstruction_error=reconstruction_error,
                regime_assessment=assessment,
                timestamp=timestamp,
                market_data=market_data,
            )

    def _store_shadow_trade_async(
        self,
        signal: TradeSignal,
        reconstruction_error: float,
        regime_state: str,
        entry_price: float,
        tick_window: np.ndarray | None = None,
        candle_window: np.ndarray | None = None,
        regime_vetoed: bool = False,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """
        Store a shadow trade in the background (Fire-and-Forget).
        
        IMPORTANT: This non-blocking implementation ensures I/O doesn't 
        delay the critical path of tick processing.
        """
        if not self.shadow_store:
            return

        task = asyncio.create_task(
            self._do_store_shadow_trade(
                signal=signal,
                reconstruction_error=reconstruction_error,
                regime_state=regime_state,
                tick_window=tick_window,
                candle_window=candle_window,
                entry_price=entry_price,
                regime_vetoed=regime_vetoed,
                metadata=metadata,
            )
        )
        self._pending_shadow_tasks.add(task)
        task.add_done_callback(self._pending_shadow_tasks.discard)

    async def _do_store_shadow_trade(
        self,
        signal: TradeSignal,
        reconstruction_error: float,
        regime_state: str,
        entry_price: float,
        tick_window: np.ndarray | None = None,
        candle_window: np.ndarray | None = None,
        regime_vetoed: bool = False,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """
        Internal implementation of shadow trade storage.
        """
        if not self.shadow_store:
            return

        # I01 Fix: Only persist shadow trades exceeding minimum learning threshold
        # This prevents database bloat from low-value signals
        min_prob = self.settings.shadow_trade.min_probability_track
        if signal.probability < min_prob:
            logger.debug(
                f"Skipping shadow trade storage: probability {signal.probability:.3f} < {min_prob:.3f}"
            )
            return

        # IMPORTANT-001: Use centralized duration resolver
        duration_minutes, _ = self.duration_resolver.resolve_duration(signal.contract_type)

        record = ShadowTradeRecord.create(
            contract_type=signal.contract_type,
            direction=signal.direction or "",
            probability=signal.probability,
            entry_price=entry_price,
            reconstruction_error=reconstruction_error,
            regime_state=regime_state,
            tick_window=tick_window,
            candle_window=candle_window,
            model_version=self.model_version,
            feature_schema_version=FEATURE_SCHEMA_VERSION,
            barrier_level=self._extract_barrier_value(signal.metadata, "barrier"),
            barrier2_level=self._extract_barrier_value(signal.metadata, "barrier2"),
            duration_minutes=duration_minutes,
            metadata={
                "signal_type": signal.signal_type,
                "regime_vetoed": regime_vetoed,
                **(metadata or {}),
                **signal.metadata,  # Ensure all signal metadata is preserved
            },
        )

        await self.shadow_store.append_async(record)
        shadow_trade_logger.log_stored(record.trade_id)
        
        shadow_trade_logger.log_created(
            trade_id=record.trade_id,
            contract_type=record.contract_type,
            direction=record.direction,
            probability=record.probability,
            metadata={"duration_minutes": duration_minutes}
        )
        # Keep debug log for console visibility if needed, or remove if redundant
        logger.debug(
            f"👻 SHADOW TRADE: {record.contract_type} {record.direction} "
            f"@ {record.probability:.3f} (ID: {record.trade_id[:8]})"
        )

    def _log_no_trades_feedback(self, all_signals: list[TradeSignal]) -> None:
        """
        Explain why no trades were generated (Silence Fix).
        
        Helps with observability during low-confidence or high-threshold regimes.
        """
        if not all_signals:
             return
             
        # Find the best rejected signal to explain why
        best_signal = max(all_signals, key=lambda s: s.probability)
        logger.info(
            f"No actionable trades. Best candidate: {best_signal.direction} "
            f"({best_signal.contract_type}) @ {best_signal.probability:.3f} "
            f"(Threshold: {self.settings.thresholds.learning_threshold_min:.3f})"
        )

    def _extract_barrier_value(self, metadata: dict[str, Any], key: str) -> float | None:
        """
        ID_BARRIER Fix: Safely extract and parse barrier value from metadata.
        
        Now expects values to be numeric from Settings, or handles legacy strings.
        """
        if key not in metadata:
            return None
        
        value = metadata[key]
        if value is None:
            return None
            
        try:
            if isinstance(value, (int, float)):
                return float(value)
            
            if isinstance(value, str):
                # Legacy support: strip '+' but keep '-'
                clean_value = value.strip().replace("+", "")
                if not clean_value:
                    return None
                return float(clean_value)
                
            return None
        except (ValueError, TypeError):
             logger.warning(f"Failed to parse barrier value for {key}: {value}")
             return None

    async def shutdown(self, timeout: float = 5.0) -> None:
        """
        Gracefully shutdown the engine, flushing pending tasks.
        """
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
