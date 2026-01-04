# 1. Advanced Position Sizing with Volatility Awareness

Date: 2026-01-04

## Status

Accepted

## Context

The system's original position sizing was either fixed or purely confidence-based, lacking awareness of market volatility. This exposed the portfolio to excessive risk during turbulent market conditions. Additionally, position sizing logic was tightly coupled with signal generation or execution, making it difficult to test or swap strategies.

## Decision

We have decided to decouple position sizing into a dedicated interface (`PositionSizer`) and implement specific strategies:
1.  **Kelly Criterion**: Dynamically sizes bets based on edge (probability vs. payout).
2.  **Target Volatility**: Adjusts position size inversely to realized volatility to maintain constant risk exposure.
3.  **Drawdown Sensitivity**: Reduces exposure when the account is in a drawdown state.

We introduced `execution/position_sizer.py` containing `KellyPositionSizer` and `TargetVolatilitySizer`. The `StrategyAdapter` acts as the bridge, injecting volatility context from `TradeSignal` metadata into the sizer.

## Consequences

**Positive:**
-   Improved risk management: Position sizes automatically shrink during high volatility or drawdown periods.
-   Testability: Sizing logic can be unit-tested independently of market data or models.
-   Flexibility: New sizing strategies can be added by implementing the `PositionSizer` protocol.

**Negative:**
-   Complexity: Requires accurate inputs (volatility, probability) to function correctly. Garbage in -> Garbage out.
-   Configuration: More hyperparameters to tune (Kelly fraction, target vol, drawdown thresholds).
