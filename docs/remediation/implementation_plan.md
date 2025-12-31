# Post-Remediation Validation Plan: x.titan

This plan validates the system integrity after the remediation cycle performed by Google Jules (December 2025). We will re-run the 7-phase validation suite to confirm that critical bugs are fixed and that no new regressions were introduced.

## Proposed Changes

### Phase 1: Codebase Re-Discovery
- **Update**: `FILE_INVENTORY.md` and `CHANGE_SUMMARY.md`.
- **Focus**: Verify removal of redundant modules and new fix commits.

### Phase 2: Functional Re-Validation
- **Static Analysis**: Re-run `mypy`, `pylint`, and `bandit`. Target `shadow_resolution.py` for indentation fixes.
- **Import Check**: Re-run `validate_imports.py` to confirm `pre_training_validation.py` now imports correctly.
- **Integration**: Re-run `live.py --test` and `train.py --test-mode` to verify initialization and path handling fixes.

### Phase 3: Architecture & Data Schema Audit
- **Verification**: Ensure `ARCHITECTURE_CONFORMANCE.md` reflects the current, cleaner state.
- **Data**: Confirm `DATA_SCHEMA_REPORT.md` is current for all SQLite stores.

### Phase 4: Performance & Behavioral Check
- **Benchmark**: Establishing a new `PERFORMANCE_BASELINE.md`.
- **Behavior**: Confirming safety logic still triggers in `BEHAVIORAL_VALIDATION.md`.

### Phase 5: Deep Dive & Reporting
- **Data Flow**: Update `DATA_FLOW_TRACE.md`.
- **Audit**: Finalize `CONFIGURATION_AUDIT.md` and `DEPENDENCY_HEALTH.md`.
- **Master Report**: Generate the updated `VALIDATION_REPORT.md`.

### Phase 6: Automated Testing & Cleanup
- **Coverage**: Refresh `COVERAGE_ANALYSIS.md`.
- **Final Sync**: Update `docs/review101/REPORT.md` and commit all artifacts.

## Verification Plan
- Automated tool runs for static analysis.
- Live system verification using `live.py --test`.
- Successful execution of `tests/smoke_tests.py` (Unified suite).
