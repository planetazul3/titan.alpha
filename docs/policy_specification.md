# Execution Safety Policy Specification

## 1. Scope
This document formalizes the safety rules and risk constraints enforced by the **x.titan** execution engine. These rules are mandatory for all production deployments.

## 2. Hard Constraints (Vetos)

| Rule ID | Name | Trigger Condition | Action |
| :--- | :--- | :--- | :--- |
| **H1** | Daily Loss Limit | Cumulative P&L <= `-MAX_DAILY_LOSS` | Halt trading until reset |
| **H2** | Stake Cap | Stake > `MAX_STAKE` | Reject trade |
| **H3** | Volatility Veto | Market ATR > `MAX_ALLOWED_VOLATILITY` | Veto Signal |
| **H4** | Warmup Veto | Buffer candles < `WARMUP_PERIOD` | Reject Signal |
| **H5** | Regime Veto | Regime == `UNCERTAIN` or `CHAOTIC` | Veto Signal |

## 3. Dynamic Controls

### Adaptive Backoff
In the event of consecutive losses, the system increments `RETRY_DELAY` by:
`DELAY = BASE_DELAY * (2 ^ CONSECUTIVE_LOSSES)`

### Confidence Scaling
The actual stake placed is scaled by the model's confidence:
`EFFECTIVE_STAKE = BASE_STAKE * sigmoid(CONFIDENCE - THRESHOLD)`

## 4. Operational Guardrails

### Shadow Mode Replay
All production signals MUST be mirrored to the `ShadowStore` regardless of execution status. This ensures we have a continuous baseline for "What If" analysis.

### Calibration Check
On startup and every 6 hours, the `VolatilityExpert` must perform a reconstruction test. If reconstruction error > 1.0, the system must transition to **Shadow Only** mode immediately.
