---
description: Architectural coherence and SSOT directive
alwaysApply: true
---

# Architecture Rules

## The Source of Truth

`docs/reference/ARCHITECTURE_SSOT.md` is the **supreme authority**.

- Any contradiction of SSOT is automatically rejected
- No architectural changes without updating SSOT first
- Record all changes in `docs/reference/ARCH_CHANGELOG.md`

## Before Any Architectural Decision

1. Read `docs/reference/ARCHITECTURE_SSOT.md`
2. Check `docs/adr/` for relevant decisions
3. Research best practices via web search
4. Update SSOT and changelog BEFORE coding

## Safety Vetoes (H1-H6)

These are non-negotiable. See SSOT §7.1 for full definitions.

| ID | Trigger | Action |
|----|---------|--------|
| H1 | Daily loss limit hit | Halt trading |
| H2 | Stake exceeds max | Reject trade |
| H3 | Volatility anomaly | Veto signal |
| H4 | Buffer not warm | Reject signal |
| H5 | Regime uncertain | Veto signal |
| H6 | Stale data | Reject signal |
