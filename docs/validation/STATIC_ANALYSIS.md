# STATIC_ANALYSIS.md

## Analysis Summary
- **Mypy**: Identified numerous typing issues, mostly related to library overloads and missing type hints in core modules.
- **Pylint**: 
    - Indentation errors (W0311) in `execution/shadow_resolution.py`.
    - Redefining names from outer scope (W0621).
    - Unused imports and re-imports (W0611, W0404).
    - String formatting inconsistencies.
- **Bandit**: (Pending/Failed to run due to missing tool, but pylint covers some basic security).
- **Radon**: (Pending/Failed to run).

## Findings Categorization

### 🔴 Critical
- **Import Error**: `pre_training_validation` fails to import `models.temporal_v2`. This indicates a potentially broken validation script or a missing module from the refactoring.

### 🟠 High
- **Type Mismatches**: Numerous `mypy` errors in `data/ingestion/historical.py` and `execution/decision.py` regarding Pandas DataFrame handling and return types.
- **Circular Dependency Potential**: Identified through imports (see `DEPENDENCY_MAP.md`).

### 🟡 Medium
- **Code Smells**: Indentation issues in `execution/shadow_resolution.py` could lead to logical errors if not fixed.
- **Shadowing**: Redefining names like `CONTRACT_TYPES` in `execution/shadow_resolution.py`.

### 🟢 Low
- **Style**: Minor naming convention violations and docstring gaps.
