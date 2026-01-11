---
description: Rules for testing and verification
---

# Verification Phase Rules

**Real-World Validation > Test Coverage**

## Success Criteria (SSOT §1.2)

- **Component**: Runs for 1 hour of live data without crashing
- **System (Live)**: Start with minimum stake, >20 trades, Positive Expectancy (Win Rate >53-55%), Net Profit > $0

## Required Checks

```bash
pytest                 # Core tests pass
ruff check .          # Linting clean
```

Type checking (`mypy`) is optional for rapid prototyping.

## Verification Mindset

Priority order:
1. Does it run live with minimum stake for 1 hour?
2. Are kill switches still functional? (loss limit, stake cap, sanity)
3. Did I break existing working features?
4. Is the change observable in logs?

## Before Committing

- [ ] Component runs without crashing (1hr live test preferred)
- [ ] Safety kill switches intact
- [ ] No regressions in core trading flow
- [ ] Changes logged/observable
