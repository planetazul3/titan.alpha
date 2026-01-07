---
description: Critical logic rules for live.py that must not be changed without understanding
---

# Critical Logic - DO NOT CHANGE WITHOUT READING

## 1. Inference Cooldown (60 seconds)
**Location**: `scripts/live.py` in `process_candles()`

```python
inference_cooldown_seconds = 60  # Run inference at most once per minute
```

**Why**: R_100 synthetic index closes "candles" every ~2 seconds (tick-counter timestamps), 
but we want to trade 1-minute contracts. Running inference every 2 seconds would:
- Execute 30 trades/minute instead of 1-2
- Not wait for contract results before next trade

**DO NOT**: Remove the cooldown or reduce it below 30 seconds.

---

## 2. Timezone for Shadow Resolution (UTC)
**Location**: `scripts/live.py` line ~557

```python
current_time=datetime.now(timezone.utc)  # Must be UTC to match trade timestamps
```

**Why**: Shadow trades are stored with UTC timestamps. Using `datetime.now()` (local time)
causes trades to appear "in the future" and never resolve.

**DO NOT**: Change to `datetime.now()` without timezone.

---

## 3. Shadow Resolution Runs on Every Candle Close
**Location**: `scripts/live.py` after inference block

```python
if is_new_candle:  # NOT inside the cooldown block
    resolved_count = resolver.resolve_trades(...)
```

**Why**: Contracts expire after 1 minute regardless of when inference runs.
Resolution must check continuously, not only when inference runs.

**DO NOT**: Move resolution inside the inference cooldown block.

---

## 4. Rate Limit Configuration
**Location**: `.env`

```
EXECUTION_SAFETY__MAX_TRADES_PER_MINUTE=2
```

**Why**: 1-minute contracts should settle before next trade decision.
User can adjust this value but it should match contract duration.

**DO NOT**: Set to 30 or higher without user approval.

---

## 5. Confidence Threshold
**Location**: `.env`

```
THRESHOLDS__CONFIDENCE_THRESHOLD_HIGH=0.70
```

**Why**: User-configured. Model outputs ~52-70% confidence.
Only change if user explicitly requests.

**DO NOT**: Lower threshold without user approval.
