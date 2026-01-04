# 3. Event-Driven Backtesting Pipeline

Date: 2026-01-04

## Status

Accepted

## Context

Previous backtesting attempts relied on simplified assumptions or partial pipeline execution, leading to significant "simulation gap" where backtest results did not match live performance. Specifically, feature generation lags, model inference times, and complex state management (e.g., regime tracking) were often ignored in vector-based backtests.

## Decision

We have adopted a strict "event-driven" backtesting architecture that replays historical data through the **exact same code paths** as live trading. 

The `BacktestEngine` now:
1.  Feeds historical candles into the real `MarketDataBuffer`.
2.  Triggers the real `FeatureBuilder` and `DerivOmniModel`.
3.  Passes predictions to the real `DecisionEngine` and `StrategyAdapter`.
4.  Executes trades via a `BacktestClient` that simulates network latency and slippage but adheres to the `DerivClient` interface.

## Consequences

**Positive:**
-   Accuracy: "What you test is what you fly." Logic bugs in the live pipeline are caught in backtest.
-   Completeness: Metrics like Drawdown and Sharpe are calculated on realistic execution paths.
-   Confidence: High fidelity simulation builds trust in the system's expected performance.

**Negative:**
-   Speed: Processing every candle through the full PyTorch inference loop is orders of magnitude slower than vectorized pandas backtests.
-   Complexity: Setting up the full stack for backtesting requires more boilerplate (handled by `scripts/backtest.py`).
