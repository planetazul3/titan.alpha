# DerivOmniModel Runbook

## Overview
This runbook covers normal operations, emergency procedures, and troubleshooting for the DerivOmniModel live trading system.

## 1. Start-Up Procedure
1.  **Activate Environment**:
    ```bash
    source venv/bin/activate
    ```
2.  **Verify State**:
    Ensure `data_cache/` contains `safety_state.db` (if previously run) and `shadow_trades.db`.
3.  **Launch Live Trading**:
    ```bash
    python scripts/live.py --checkpoint best_model
    ```
    *Use `screen` or `tmux` for persistent sessions.*
4.  **Verification**:
    - Look for "CONNECTED!" message.
    - Confirm "Pre-loaded X candles".
    - Check "Reconstruction error < 1.0" in startup logs.

## 2. Monitoring Routine
- **Logs**: Tail `logs/xtitan.log`.
- **Heartbeat (Every 1 min)**:
    - Check `stale_sec < 30` (Data freshness).
    - Check `metrics` (Latency < 200ms).
- **Daily Check**:
    - Verify `daily_pnl` resets at 00:00 UTC.
    - Check `shadow_trades.db` size (ensure it's growing, meaning data is captured).

## 3. Emergency Procedures
### Scenario A: Runaway Losses / Regime Change
**Symptoms**: P&L dropping fast, or large drawdowns.
**Action**:
1.  **Stop Process**: `Ctrl+C` or `kill $(pgrep -f live.py)`.
2.  **Activate Kill Switch** (for restart safety):
    Set `KILL_SWITCH=True` in `.env` or config.
3.  **Investigate**: Check `data_cache/shadow_trades.db` for recent trade rationales.

### Scenario B: API Disconnection
**Symptoms**: "Connection error" in logs, `stale_sec` increasing.
**Action**: System auto-retries with exponential backoff.
- If persistent (> 10 mins): Check internet connection and Deriv API status page.
- Restart script if hung.

### Scenario C: Calibration Drift
**Symptoms**: "Reconstruction error > 1.0" or "Shadow-only mode activated".
**Action**:
1.  System automatically switches to Shadow Mode (no money at risk).
2.  Plan for model retraining using recent data from `ShadowTradeStore`.

## 4. Maintenance
### Database Backup
Weekly backup of safety and shadow stores:
```bash
cp data_cache/shadow_trades.db backups/shadow_trades_$(date +%F).db
sqlite3 data_cache/safety_state.db ".backup 'backups/safety_state_$(date +%F).db'"
```

### Updates
1.  `git pull`
2.  `pip install -r requirements.txt`
3.  Run tests: `pytest tests/`
4.  Restart `live.py`
