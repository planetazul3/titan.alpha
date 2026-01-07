---
description: Rules for testing and verification
---

# Verification Phase Rules

When verifying changes:

## Required Checks

```bash
pytest                 # All tests pass
ruff check .          # Linting clean
mypy .                # Type checking passes
```

## Verification Mindset

- Did I break any existing tests?
- Did I add tests for new functionality?
- Are edge cases covered?
- Is the change documented?

## Before Committing

- [ ] Tests pass
- [ ] Linting clean
- [ ] Type checking passes
- [ ] No regressions introduced
- [ ] Changes documented
