# x.titan Workspace Rules

This workspace uses modular agent rules located in `.agent/rules/`.

## Core Rules (Always Applied)

The following rules are automatically applied to all agent interactions in this workspace:

1. **Identity & SDLC Mindset**: `.agent/rules/00-identity.md`
2. **Architecture Coherence**: `.agent/rules/01-architecture.md`
3. **Documentation as Memory**: `.agent/rules/02-docs-as-memory.md`

## Phase-Specific Rules (Model Decision)

These rules are applied contextually based on the current task:

- **Planning**: `.agent/rules/10-planning.md`
- **Execution**: `.agent/rules/11-execution.md`
- **Verification**: `.agent/rules/12-verification.md`
- **Trading Safety**: `.agent/rules/20-trading-safety.md`

## Code Style Rules (Glob)

- **Python Style**: `.agent/rules/30-python-style.md` (applies to `**/*.py`)

## Key Documents

Before any task, orient yourself using these documents:

| Document | Purpose |
|----------|---------|
| `docs/reference/ARCHITECTURE_SSOT.md` | Canonical architecture |
| `docs/reference/ARCH_CHANGELOG.md` | Recent changes |
| `docs/adr/*.md` | Architectural decisions |
| `.agent/workflows/critical-logic.md` | Trading business rules |
