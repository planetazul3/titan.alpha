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
3. Research best practices via web search (≥3 sources)
4. Choose the option with highest profitability probability
5. Update SSOT and changelog if architecture changes

## Simplified Safety Model (SSOT §4)

Three hard stops. Everything else is a feature, not a requirement.

| Check | Trigger | Action |
|-------|---------|--------|
| **Daily Loss Limit** | Loss > MAX_DAILY_LOSS | STOP trading for the day |
| **Stake Cap** | Stake > MAX_STAKE | Reject trade |
| **Sanity Checks** | Stale data (>5s) or NaN | Reject signal |

Regime detection, volatility filters → **Optional optimizations**, not safety requirements.

## Optimization Priorities (SSOT §3.5)

When making trade-offs, prioritize in this order:
1. Execution Reliability (must run without crashing)
2. Profitability (Win Rate × Payout)
3. Speed of Development
4. Code Cleanliness
5. Safety (catastrophic loss prevention only)
