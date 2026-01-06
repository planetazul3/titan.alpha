# Production Runbook

## 🚨 Emergency Response

### Kill Switch (Immediate Halt)
To immediately stop all trading activity:
```bash
# Method 1: File Trigger
touch KILL_SWITCH

# Method 2: Process Kill
pkill -f "scripts/live.py"
```

### Database Rollback
If the database is corrupted or contains invalid state:
```bash
# 1. Stop the service
pkill -f "scripts/live.py"

# 2. Backup corrupted state
mv data_cache/safety_state.db data_cache/safety_state.db.bak
mv data_cache/shadow_trades.db data_cache/shadow_trades.db.bak

# 3. Restart (Will create fresh DBs)
python scripts/live.py
```

## 🚀 Standard Operations

### Starting Service
```bash
# Activate environment
source venv/bin/activate

# Start with detailed logging
python scripts/live.py > logs/xtitan.log 2>&1 &
```

### Configuration Updates
1.  Edit `.env`.
2.  Restart service: `pkill -f scripts/live.py && python scripts/live.py`.

### Database Maintenance
Prune execution logs older than 30 days:
```bash
python -c "from execution.db_maintenance import prune_all; prune_all(days=30)"
```

## 📊 Monitoring & Alerts

### Log Signatures to Watch
| Severity | Pattern | Action |
| :--- | :--- | :--- |
| **CRITICAL** | `CRITICAL-001` | Database inconsistency. Stop immediately. |
| **CRITICAL** | `RC-8` | Non-finite values. Check feature engineering. |
| **WARNING** | `Regime Veto` | Normal in high volatility. Monitor only. |
| **WARNING** | `Rate Limit` | API throttling. Check `MAX_TRADES_PER_MINUTE`. |

### Health Checks
- **Heartbeat**: Every 60s in logs. If missing > 2min, restart.
- **Latency**: Check `metrics:decision_latency_ms`. Should be < 100ms.
