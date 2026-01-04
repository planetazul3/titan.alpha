"""
Core Decision Logic.

Pure functional logic for processing signals and applying caution filters.
Detached from the stateful DecisionEngine to ensure modularity.
"""

import logging
from datetime import datetime
from typing import Any

from config.constants import CONTRACT_TYPES
from config.settings import Settings
from execution.signals import ShadowTrade, TradeSignal
from execution.filters import get_actionable_signals
from execution.regime import RegimeAssessmentProtocol

logger = logging.getLogger(__name__)


def apply_caution_filter(
    real_trades: list[TradeSignal]
) -> tuple[list[TradeSignal], list[ShadowTrade]]:
    """
    Apply caution during uncertain regimes.

    Strategy: Keep only the highest confidence trade, demote others to shadow.
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


def process_signals_batch(
    all_signals: list[TradeSignal],
    regime_assessment: RegimeAssessmentProtocol,
    settings: Settings,
    reconstruction_error: float,
    stats: dict[str, int],
    tracer: Any | None = None
) -> tuple[list[TradeSignal], list[ShadowTrade]]:
    """
    Process a batch of signals with regime context.

    Handles filtering, caution logic, and statistics updates.
    Returns (real_trades, shadow_trades).
    """
    # 1. Check Veto
    if regime_assessment.is_vetoed():
        if tracer:
            tracer.set_attribute("veto.type", "regime")
        stats["processed"] += 1
        stats["regime_vetoed"] += 1
        return [], []

    # 2. Validate contract types
    valid_contract_types = {ct.value for ct in CONTRACT_TYPES}
    validated_signals = []
    for sig in all_signals:
        if sig.contract_type in valid_contract_types:
            validated_signals.append(sig)
        else:
            logger.warning(f"Invalid contract type '{sig.contract_type}' - skipped")
            stats["ignored"] += 1
    
    # 3. Get Actionable Signals (Prob > Threshold)
    real_trades, shadow_trades = get_actionable_signals(validated_signals)
    
    if tracer:
        tracer.set_attribute("signals.real_base", len(real_trades))
        tracer.set_attribute("signals.shadow_base", len(shadow_trades))

    # 4. Apply Caution Logic
    if regime_assessment.requires_caution():
        if tracer:
            tracer.set_attribute("caution.active", True)
        
        stats["regime_caution"] += 1
        logger.info(
            f"Regime caution active: demoting some trades. "
            f"reconstruction_error={reconstruction_error:.4f}"
        )
        
        # Pre-filter real trades requiring higher confidence
        filtered_real = []
        for signal in real_trades:
            caution_threshold = settings.thresholds.confidence_threshold_high + 0.05
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
        
        # Apply strict "Top 1" logic
        real_trades, demoted = apply_caution_filter(real_trades)
        shadow_trades.extend(demoted)

    # 5. Feedback Logging (Silence Fix)
    if not real_trades and not shadow_trades and validated_signals:
        best_signal = max(validated_signals, key=lambda s: s.probability)
        logger.info(
            f"No actionable trades. Best candidate: {best_signal.direction} "
            f"({best_signal.contract_type}) @ {best_signal.probability:.3f} "
            f"(Threshold: {settings.thresholds.learning_threshold_min:.3f})"
        )

    # 6. Update Stats
    stats["processed"] += 1
    stats["real"] += len(real_trades)
    stats["shadow"] += len(shadow_trades)
    stats["ignored"] += len(validated_signals) - len(real_trades) - len(shadow_trades)

    if tracer:
        tracer.set_attribute("signals.real_final", len(real_trades))

    return real_trades, shadow_trades
