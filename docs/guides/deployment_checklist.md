# Deployment Checklist

## 🛠️ Environment Separation

### 1. Development (Local)
- [ ] **.env**: `DERIV_ENV=development`
- [ ] **Mock API**: Use `DERIV_APP_ID=1001` (test) or similar.
- [ ] **Database**: Local SQLite in `data_cache/dev/`.
- [ ] **Tests**: Run `pytest tests/` before pushing.

### 2. Staging (Shadow Mode)
- [ ] **.env**: `DERIV_ENV=staging`
- [ ] **Real Data/Virtual Account**: Use Deriv virtual account token.
- [ ] **Shadow Mode**: Run with `--shadow` flag.
- [ ] **Observability**: Verify dashboard shows "Staging" tag.

### 3. Production (Live Trading)
- [ ] **.env**: `DERIV_ENV=production`
- [ ] **Real Account**: Use production API token (DANGER).
- [ ] **Safety Config**: 
    - `MAX_DAILY_LOSS`: Set to approved limit.
    - `MAX_STAKE`: Set to maximum allowed per trade.
- [ ] **Firewall**: Ensure only authorized IPs can access the dashboard API.
- [ ] **Backup**: Verify `safety_state.db` is backed up daily.

---

## 🏗️ Infrastructure Pre-Flight
- [ ] **Python Version**: Ensure Python 3.10+ is installed.
- [ ] **Dependencies**: `pip install -r requirements.txt`.
- [ ] **Time Sync**: Server clock synchronized via NTP (Critical).
- [ ] **Disk Space**: At least 5GB free for `data_cache/` and logs.

## 🚀 Pre-Flight Checks
- [ ] **Dry Run**: `python scripts/live.py --test` to verify connection.
- [ ] **Buffer Warmup**: Verify "Pre-loaded X candles" in logs.
- [ ] **Calibration**: Monitor initial "Reconstruction error". If > 1.0, ABORT.

## 🛑 Emergency Procedures
- [ ] **Kill Switch**: `touch KILL_SWITCH` to stop execution immediately.
- [ ] **Manual Stop**: `Ctrl+C` or `kill <pid>`.
- [ ] **Log Access**: `tail -f logs/xtitan.log`.

## 🛡️ Safety Validation (Live)
- [ ] **H1 (Daily Loss)**: Verify trading stops if P&L < Limit.
- [ ] **H3 (Volatility)**: Verify "VETO" log message during high vol periods.
- [ ] **RC-8 (Numeric)**: Verify no `NaN` warnings in logs.
