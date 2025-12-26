# x.titan Safety Mechanisms: The Swiss Cheese Model

This document details the multi-layered safety systems protecting the `x.titan` trading engine. A "Swiss Cheese" model is used, where multiple independent layers must all permit a trade for it to be executed.

## Layer 1: Regime Veto (The Absolute)
**Source**: `execution/regime.py`

This is the first and most powerful check. The **Volatility Expert** (Autoencoder) continuously monitors market data.

-   **Mechanism**: It reconstructs current volatility metrics based on training data patterns.
-   **Trigger**: High reconstruction error (> `REGIME_VETO_THRESHOLD`) indicates an **Unknown Regime**.
-   **Action**: **ABSOLUTE BLOCK**. No trade is considered, regardless of confidence.
-   **Philosophy**: "If we don't recognize the market, we don't trade."

## Layer 2: Execution Policy (The Safeguard)
**Source**: `execution/policy.py` & `execution/safety.py`

If the regime is safe, the **SafeTradeExecutor** applies rigid risk rules.

### A. Kill Switch
-   **Config**: `safety.kill_switch_enabled`
-   **Function**: A manual or automated global override to stop all trading immediately.

### B. Circuit Breaker
-   **Config**: `max_consecutive_failures`
-   **Function**: If `N` trades fail (loss) or error out consecutively, the system enters a cooldown state.
-   **Reset**: Requires manual intervention or a long timeout (configurable).

### C. Rate Limiting
-   **Config**: `max_trades_per_minute`
-   **Function**: Prevents API spamming or "machine gun" trading loops.

### D. P&L Cap
-   **Config**: `max_daily_loss` (e.g., -$50)
-   **Function**: If daily realized loss hits this limit, trading stops for the UT day.
-   **State**: Tracked in `data_cache/safety_state.db` (ACID-compliant) to persist across restarts.

## Layer 3: Regime Caution (The Filter)
**Source**: `execution/decision.py`

If the regime is "suspicious" but not fully vetoed (error > `REGIME_CAUTION_THRESHOLD`):
-   **Action**: The system enters **Caution Mode**.
-   **Effect**:
    1.  Only **High Confidence** trades (e.g., > 85%) are allowed.
    2.  Lower confidence trades are demoted to **Shadow Trades** (logged but not executed).

## Layer 4: Compounding Safety (Position Sizing)
**Source**: `execution/position_sizer.py`

Even if a trade is approved, the position size is strictly controlled.
-   **Max Stake Cap**: Hard limit on any single trade amount.
-   **Reset on Loss**: Strategies like Compounding reset to base stake immediately after a loss.
-   **Low Confidence Reset**: If confidence drops, compounding streaks are banked/reset.

## Operational Guide

### Checking Safety State
The system logs all safety blocks to console with `[WARN]` or `[ERROR]`.
State is persisted in `data_cache/safety_state.db`.

### Emergency Stop
To trigger the Kill Switch manually:
1.  Set `KILL_SWITCH_ENABLED=true` in `.env`.
2.  Restart the bot.
