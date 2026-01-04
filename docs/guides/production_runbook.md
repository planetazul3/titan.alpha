# Production Runbook

## Starting Live Trading

```bash
cd /home/planetazul3/x.titan
source venv/bin/activate
python scripts/live.py
```

## Configuration (.env)

| Setting | Recommended | Description |
|---------|-------------|-------------|
| `THRESHOLDS__CONFIDENCE_THRESHOLD_HIGH` | 0.70 | Real trades above this |
| `EXECUTION_SAFETY__MAX_TRADES_PER_MINUTE` | 2 | Match contract duration |
| `EXECUTION_SAFETY__MAX_DAILY_LOSS` | 50.0 | Stop-loss limit |

## Monitoring

**Console shows:**
- `🧠 Running inference #N... (cooldown: 60.0s)` every minute
- `👻 SHADOW TRADE` for learning-zone signals
- `🎯 Resolved N shadow trade(s)` after 1 minute
- `✅ WIN` or `❌ LOSS` with P&L

**Heartbeat (every 60s):**
```
═══ HEARTBEAT ═══
Ticks: N | Candles: N | Inferences: N
Real Trades: N | Shadow Trades: N
```

## Troubleshooting

| Issue | Cause | Fix |
|-------|-------|-----|
| Trading every 2s | Missing cooldown | Check `/critical-logic` |
| Shadow trades not resolving | Timezone bug | Use `datetime.now(timezone.utc)` |
| "Real trades: 0" | Confidence < threshold | Lower threshold or wait |

## Clearing Data

```bash
# Reset shadow trades
rm -f data_cache/shadow_trades.db*

# Reset safety state (rate limits)
rm -f data_cache/safety_state.db*
```

## Generating Reports

```bash
python scripts/generate_shadow_report.py --days 7 --output reports/weekly.html
```
