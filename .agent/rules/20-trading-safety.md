---
description: Safety rules for trading system modifications
---

# Trading Safety Rules

**CRITICAL**: Apply when modifying `execution/`, `safety/`, or signal generation.

## The Kill Switch Model (SSOT §4)

Three hard stops. Everything else is an optimization.

### 1. Daily Loss Limit (Hard Stop)
- **Rule**: If Loss > `MAX_DAILY_LOSS`, STOP trading for the day
- **Non-negotiable**: This prevents account drain
- **Location**: Check execution policy/safety store

### 2. Stake Cap
- **Rule**: Never bet more than `MAX_STAKE`
- **Non-negotiable**: Prevents single-trade disasters

### 3. Sanity Checks
- **Stale Data**: Don't trade on data >5 seconds old
- **Broken Data**: Reject signals with NaN values
- **Observable**: Should log rejections clearly

## What is NOT a Safety Requirement

- Regime detection (H5) → Feature, not requirement
- Volatility filters (H3) → Optimization, not requirement
- Complex multi-layer vetoes → Over-engineering

**If it's not a kill switch, it's optional.**

## Hard Rules

- **NEVER** bypass daily loss limit or stake cap
- **NEVER** set `ENVIRONMENT=production` in committed code
- **ALWAYS** test edge cases: empty data, NaN values, rate limits

## Test Before Commit

```bash
pytest tests/test_execution.py tests/test_safety.py -v
```

## When In Doubt

Ask the user. Kill switches are sacred.
