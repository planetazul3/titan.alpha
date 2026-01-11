---
description: Rules for code implementation
---

# Execution Phase Rules

**Working Prototype > Clean Abstraction**

When implementing code:

1. **Ship Fast**: Get to testable state as quickly as possible
2. **Test with Real Data**: Shadow mode > unit tests for trading logic
3. **Small Commits**: One logical change per commit
4. **Aggressively Delete**: Remove unused abstractions and dead code
5. **Docs if Architecture Changes**: Update SSOT/changelog for structural changes

## Commit Hygiene

```
<type>(<scope>): <brief description>

- Bullet explaining key change
- Another key point
```

## Simplification Rules (SSOT §6.4)

- If a component is too complex, delete it
- If a library helps, add it
- Don't be precious about code if it doesn't contribute to profitability

## Anti-Patterns

- Perfect abstraction before proving the feature works
- Refactoring code that hasn't run in shadow mode
- Adding frameworks when functions would work
