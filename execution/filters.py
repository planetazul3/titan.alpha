from datetime import datetime

from config.constants import CONTRACT_TYPES, SIGNAL_TYPES
from config.settings import Settings, Thresholds
from execution.signals import ShadowTrade, TradeSignal


def classify_probability(prob: float, thresholds: Thresholds) -> SIGNAL_TYPES:
    """
    Classify model probability into action category.
    """
    if prob >= thresholds.confidence_threshold_high:
        return SIGNAL_TYPES.REAL_TRADE
    elif thresholds.learning_threshold_min <= prob < thresholds.learning_threshold_max:
        return SIGNAL_TYPES.SHADOW_TRADE
    else:
        return SIGNAL_TYPES.IGNORE


def filter_signals(
    model_outputs: dict[str, float], settings: Settings, timestamp: datetime
) -> list[TradeSignal]:
    """
    Convert raw model probabilities to actionable signals.
    Expects model_outputs keys like 'rise_fall_prob', 'touch_prob', 'range_prob'.
    (Note: ensure caller passes probabilities, not logits)
    """
    signals = []
    thresholds = settings.thresholds

    # Mapping output keys to contract types
    # Assuming 'rise_fall' predicts CALL (Rise) vs PUT (Fall).
    # Usually binary classification: >0.5 Rise, <0.5 Fall?
    # Or is 'rise_fall_logit' outputting prob of RISE?
    # Let's assume prob is for RISE (CALL).
    # If prob < (1-threshold), maybe it is a PUT signal?
    # The Prompt doesn't specify symmetric logic, but usually trading bots do.
    # However, for simplicity and strict adherence to prompt, we assume prob is confidence.
    # But usually models output probabilty of class 1.
    # If prob(Rise) = 0.9 -> REAL TRADE RISE.
    # If prob(Rise) = 0.1 -> prob(Fall) = 0.9 -> REAL TRADE FALL.

    # For this implementation, I will assume the system handles "High Confidence" as > threshold.
    # Dealing with directionality:
    # If rise_fall_prob > threshold -> CALL
    # If rise_fall_prob < (1 - threshold) -> PUT

    # Specific implementation for Rise/Fall
    # IMPORTANT: Only create signal for the HIGHEST PROBABILITY direction
    # If prob(CALL) = 0.84, we trade CALL, not PUT at 0.16
    if "rise_fall_prob" in model_outputs:
        prob_call = model_outputs["rise_fall_prob"]
        prob_put = 1.0 - prob_call
        
        # Trade in the direction with HIGHER probability
        if prob_call >= prob_put:
            # CALL is more likely
            sig_type = classify_probability(prob_call, thresholds)
            signals.append(
                TradeSignal(
                    signal_type=sig_type,
                    contract_type=CONTRACT_TYPES.RISE_FALL,
                    direction="CALL",
                    probability=prob_call,
                    timestamp=timestamp,
                    metadata={"symbol": settings.trading.symbol, "barrier": None}
                )
            )
        else:
            # PUT is more likely
            sig_type = classify_probability(prob_put, thresholds)
            signals.append(
                TradeSignal(
                    signal_type=sig_type,
                    contract_type=CONTRACT_TYPES.RISE_FALL,
                    direction="PUT",
                    probability=prob_put,
                    timestamp=timestamp,
                    metadata={"symbol": settings.trading.symbol, "barrier": None}
                )
            )

    # Touch/No Touch (Assuming prob is for Touch)
    if "touch_prob" in model_outputs:
        prob = model_outputs["touch_prob"]
        sig_type = classify_probability(prob, thresholds)
        signals.append(
            TradeSignal(
                signal_type=sig_type,
                contract_type=CONTRACT_TYPES.TOUCH_NO_TOUCH,
                direction="TOUCH",  # Or similar
                probability=prob,
                timestamp=timestamp,
                metadata={"symbol": settings.trading.symbol}
            )
        )

    # Range/Stays Between
    if "range_prob" in model_outputs:
        prob = model_outputs["range_prob"]
        sig_type = classify_probability(prob, thresholds)
        signals.append(
            TradeSignal(
                signal_type=sig_type,
                contract_type=CONTRACT_TYPES.STAYS_BETWEEN,
                direction="STAYS_BETWEEN",
                probability=prob,
                timestamp=timestamp,
                metadata={"symbol": settings.trading.symbol}
            )
        )

    return signals


def get_actionable_signals(
    signals: list[TradeSignal],
) -> tuple[list[TradeSignal], list[ShadowTrade]]:
    """
    Separate signals into real and shadow trades.
    """
    real_trades = []
    shadow_trades = []

    for sig in signals:
        if sig.signal_type == SIGNAL_TYPES.REAL_TRADE:
            real_trades.append(sig)
        elif sig.signal_type == SIGNAL_TYPES.SHADOW_TRADE:
            # Convert TradeSignal to ShadowTrade
            shadow_trade = ShadowTrade(
                signal_type=sig.signal_type,
                contract_type=sig.contract_type,
                direction=sig.direction,
                probability=sig.probability,
                timestamp=sig.timestamp,
                metadata=sig.metadata,
            )
            shadow_trades.append(shadow_trade)

    return real_trades, shadow_trades
