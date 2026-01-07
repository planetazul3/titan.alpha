---
description: Safety rules for trading system modifications
---

# Trading Safety Rules

**CRITICAL**: Apply when modifying `execution/`, `safety.py`, or signal generation.

## Before Modifying Trading Code

1. Read `.agent/workflows/critical-logic.md` FIRST
2. Understand the safety vetoes (H1-H6)
3. Check for existing patterns before adding new ones

## Hard Rules

- **NEVER** set `ENVIRONMENT=production` in committed code
- **NEVER** disable safety vetoes without explicit user approval
- **NEVER** modify cooldowns below 30 seconds
- **ALWAYS** test with edge cases: empty data, NaN values, rate limits

## Test Before Commit

```bash
pytest tests/test_execution.py tests/test_safety.py -v
```

## When In Doubt

Ask the user. Trading safety is not a place for assumptions.
