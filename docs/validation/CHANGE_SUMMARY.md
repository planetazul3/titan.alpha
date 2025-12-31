# CHANGE_SUMMARY.md

## Timeline of Recent Structural Changes
- **2025-12-31**: Renamed `core/domain/models.py` to `core/domain/entities.py` to avoid namespace collision. Added `__init__.py` to several directories. Fixed Panda typing issues.
- **2025-12-30**: Resolved all test failures (439 tests passing). Fixed `data/ingestion/versioning.py` DataFrame handling. Refactored `AdaptiveRiskManager`.
- **2025-12-30 (Earlier)**: Major refactor by Jules bot. Introduced `core/domain/` entities and centralized risk control.
- **2025-12-29**: Optimized `ReplayBuffer` stratified sampling. Fixed gradient computation tests.

## Moved/Renamed Modules
- `core/domain/models.py` -> `core/domain/entities.py`
- `execution/regime.py` -> `execution/regime_v2.py` (based on previous exploration/architecture docs)
- `execution/shadow.py` -> `execution/sqlite_shadow_store.py` (likely)

## New Modules Introduced
- `core/domain/entities.py`
- `docs/review101/REPORT.md`
- `observability/performance_tracker.py`
- `tools/validation/` (various scripts)

## Deprecated/Removed Modules
- `conftest.py` (root level, moved or consolidated)
- Legacy evaluation files (cleaned up)
