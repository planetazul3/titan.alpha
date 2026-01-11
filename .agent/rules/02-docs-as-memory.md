---
description: How to use documentation as persistent memory
alwaysApply: true
---

# Documentation as Memory

You have no persistent memory between sessions. **Documents ARE your memory.**

## Memory Lookup Table

| Question | Read This Document |
|----------|-------------------|
| What is this system? | `docs/reference/ARCHITECTURE_SSOT.md` |
| What changed recently? | `docs/reference/ARCH_CHANGELOG.md` |
| Why was X decided? | `docs/adr/*.md` |
| Trading safety rules? | SSOT §4 (Kill Switches) |
| Critical operational logic? | `.agent/workflows/critical-logic.md` |

## Protocol

**Before changes**: Read ARCH_CHANGELOG.md to understand recent evolution  
**After changes**: Update ARCH_CHANGELOG.md to record your contribution

## Handoff to Next Agent

When completing a task, update docs with:
- Current state (WORKING/BROKEN/IN_PROGRESS)
- What was accomplished
- What failed and why (real data preferred over speculation)
- Recommended next steps
