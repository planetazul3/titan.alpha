---
description: Rules for code implementation
---

# Execution Phase Rules

When implementing code:

1. **Follow Patterns**: Match existing code style and architecture
2. **Small Steps**: One logical change per commit
3. **Docs Before Code**: Update SSOT/changelog if architecture changes
4. **Test Continuously**: Run tests after each significant change

## Commit Hygiene

```
<type>(<scope>): <brief description>

- Bullet explaining key change
- Another key point
```

## Anti-Patterns

- Large, untested changes
- Modifying multiple unrelated things
- Ignoring linter/type checker warnings
