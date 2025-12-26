# Deployment Checklist

## Environment & Infrastructure
- [ ] **Python Version**: Ensure Python 3.10+ is installed in the production environment.
- [ ] **Dependencies**: `pip install -r requirements.txt` (ensure `python-deriv-api` is installed).
- [ ] **SQLite Drivers**: Verify SQLite3 library is present (usually built-in, but WAL mode requires decent filesystem support).
- [ ] **Time Sync**: Server clock synchronized via NTP (Critical for accurate timestamping).

## Configuration
- [ ] **.env Secrets**: 
    - `DERIV_APP_ID`: Set to production app ID.
    - `DERIV_API_TOKEN`: Set to live trading token (DANGER).
    - `DERIV_ENV`: Set to `production` (or `real`).
- [ ] **Safety Config**: 
    - `MAX_DAILY_LOSS`: Set to approved risk limit (e.g., $50).
    - `MAX_STAKE`: Set to maximum stake per trade.
    - `KILL_SWITCH`: Ensure it is initially `False`.

## State Initialization
- [ ] **Data Cache**: Ensure `data_cache/` directory exists and is writable.
- [ ] **Safety Store**: Verify `data_cache/safety_state.db` is writable or will be created.
- [ ] **Shadow Store**: Verify `data_cache/shadow_trades.db` is writable.
- [ ] **Checkpoints**: Ensure model checkpoint (e.g., `best_model.pt`) is present in `checkpoints/`.

## Pre-Flight Checks
- [ ] **Dry Run**: Run `python scripts/live.py --test` to verify connection and auth.
- [ ] **Buffer Warmup**: Run normal start and verify "Pre-loaded X candles" in logs.
- [ ] **Calibration Check**: Monitor initial "Reconstruction error" log on startup. If > 1.0, ABORT.

## Monitoring
- [ ] **Logs**: Tail logs at `logs/xtitan.log` or console output.
- [ ] **Heartbeat**: Verify periodic "HEARTBEAT" messages every minute.
- [ ] **Metrics**: (Optional) Connect Prometheus scraper if configured.

## Rollback
- [ ] **Emergency Stop**: Know the command to kill the process (`Ctrl+C` or `kill <pid>`).
- [ ] **Kill Switch**: Use `touch KILL_SWITCH` (if implemented) or manually set config if hot-reloading supported.
