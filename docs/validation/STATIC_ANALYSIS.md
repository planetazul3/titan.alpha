# STATIC_ANALYSIS.md

## Critical Issues

### 🔴 Duplicate Module Name: `models`
- **Location 1**: `models/` (directory)
- **Location 2**: `core/domain/models.py` (file)
- **Impact**: `mypy` is unable to distinguish between the two, leading to import ambiguity and tool failures. This should be renamed (e.g., `core/domain/entities.py` or similar).

## High Priority Issues

### 🟠 Type Checking Errors (Mypy)
- Mypy failed to run due to the duplicate module naming issue. Once the naming conflict is resolved, a full type check should be performed.

## Medium Priority Issues

### 🟡 Circular Dependencies
- Found in the `data` module:
  - `data.dataset -> data.features -> data.processor -> data -> data.dataset`
  - `data.features -> data.processor -> data -> data.features`
- **Impact**: Can lead to initialization order issues and makes the code harder to reason about and test in isolation.

## Low Priority Issues

### 🟢 Documentation and Style
- [Placeholder for further analysis once more tools are available]

## Summary of Tools Run

| Tool | Status | Result |
|------|--------|--------|
| `mypy` | ❌ FAILED | Blocked by duplicate module `models` |
| `compileall` | ✅ PASSED | No syntax errors detected in the codebase |
| `pylint` | ❌ N/A | Not installed in environment |
| `bandit` | ❌ N/A | Not installed in environment |
| `radon` | ❌ N/A | Not installed in environment |

## Complexity Metrics (Manual Observation)
- `scripts/live.py`: 982 lines (Potential for high complexity)
- `data/ingestion/client.py`: 857 lines
- `execution/decision.py`: 770 lines
- `observability/dashboard.py`: 751 lines

These large files should be audited for high cyclomatic complexity and potential for refactoring into smaller components.
