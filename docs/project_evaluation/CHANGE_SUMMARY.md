# CHANGE_SUMMARY.md

## Timeline of Structural Changes

### 1. Architecture Refactor (Recent)
- **Commit**: `ae58535 Refactor architecture: Introduce domain models, centralized risk control, and strict state machine`
- **Impact**: Introduced `core/` directory with domain models and interfaces. This suggests a move towards a more formal DDD (Domain Driven Design) and decoupling of core logic from implementation.
- **New Modules**:
  - `core/interfaces.py`: Formal definitions of system components.
  - `core/domain/base.py`: Base classes for domain objects.
  - `core/domain/models.py`: Domain entity definitions.

### 2. Logging Consolidation
- **Commit**: `a24c421 refactor(logging): Consolidate logging modules into single source`
- **Impact**: Deleted `utils/logging_setup.py`. Logging is likely now centralized in `config/logging_config.py`.

### 3. Risk and Execution Logic Unification
- **Commit**: `14e61e0 refactor(arch): unify veto decisions under ExecutionPolicy layer`
- **Impact**: Moved veto decisions from `DecisionEngine` to `ExecutionPolicy`. This centralizes the "go/no-go" logic.

### 4. Expert Model Optimization
- **Commit**: `a83bf30 refactor(experts): fix duplicate block, optimize fusion, parameterize volatility input`
- **Impact**: Structural changes within the `models/` directory for better efficiency and flexibility.

### 5. Training Pipeline Refactor
- **Commit**: `fc36b01 feat(training): refactor loss weighting and add batch profiler`
- **Impact**: Changes to how multi-task loss is balanced and how training performance is monitored.

## Moved/Renamed Modules Inventory

| Original State (Likely) | Current State | Notes |
|-------------------------|---------------|-------|
| `utils/logging_setup.py`| `config/logging_config.py` | Consolidated |
| `execution/decision.py` (vetos) | `execution/policy.py` (vetos) | Veto logic moved |
| Mixed model logic | `core/domain/` | Moving towards Domain models |
| `HistoricalDataDownloader` | [REMOVED] | Replaced by new ingestion logic |

## New Modules Introduced
- `core/interfaces.py`
- `core/domain/base.py`
- `core/domain/models.py`
- `execution/regime_v2.py` (May have co-existed with `regime.py` or replaced it)
- `execution/sqlite_shadow_store.py` (Persistence for shadow trades)

## Deprecated/Removed Modules
- `utils/logging_setup.py`
- `HistoricalDataDownloader` (and its tests)
- `.agent/prompts/AUDIT_PROMPT.md`
