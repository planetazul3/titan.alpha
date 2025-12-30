# Remediation Plan: x.titan System Recovery

This plan outlines the step-by-step remediation strategy to address the critical failures, structural defects, and technical debt identified in the `docs/validation/` reports. The goal is to restore the system to a production-ready state, ensuring stability, reliability, and architectural conformance.

## Phase 1: Critical Fixes (Production Blockers)

**Objective**: Resolve issues that prevent the system from starting or performing basic operations. These are the highest priority items.

### 1.1. Fix System Startup (`scripts/live.py`)
- **Issue**: `NameError: name 'model_monitor' is not defined` and potential `system_monitor` scope issues.
- **Action**:
  - Locate the initialization block in `scripts/live.py`.
  - Ensure `model_monitor` and `system_monitor` are correctly instantiated before they are referenced.
  - Verify they are passed to the `TradingEngine` or relevant components as expected.

### 1.2. Resolve Namespace Collision (`models` vs `core/domain/models.py`)
- **Issue**: The `models/` directory (containing neural networks) conflicts with `core/domain/models.py`, causing `mypy` failures and import ambiguity.
- **Action**:
  - Rename `core/domain/models.py` to `core/domain/entities.py`.
  - Update all references in the codebase that import from `core.domain.models` to point to `core.domain.entities`.
  - Run `mypy` to verify the collision is resolved.

### 1.3. Fix Data Ingestion Failure (`scripts/download_data.py`)
- **Issue**: "The truth value of a DataFrame is ambiguous" error prevents saving partitioned data.
- **Action**:
  - Locate the problematic boolean check in `scripts/download_data.py` (likely checking if a dataframe is empty or valid).
  - Replace the ambiguous check (e.g., `if df:`) with an explicit check (e.g., `if not df.empty:`).

### 1.4. Install Missing Dependencies
- **Issue**: `freezegun`, `pylint`, `bandit` are missing but required for tests and validation.
- **Action**:
  - Add `freezegun` to `requirements.txt` (or `requirements-dev.txt` if it exists).
  - Add development tools (`pylint`, `bandit`, `flake8`) to `requirements-dev.txt`.
  - Install these dependencies in the environment.

## Phase 2: Structural Improvements & Technical Debt

**Objective**: Address high-severity architectural violations that hamper maintainability and testing.

### 2.1. Break Circular Dependencies in `data` Package
- **Issue**: Circular imports between `data.dataset`, `data.features`, and `data.processor`.
- **Action**:
  - Analyze the import chains.
  - Refactor shared types or constants into a separate module (e.g., `data.types` or `data.common`) to break the cycle.
  - Ensure `Dataset` depends on `Processor`/`Features` (or vice versa) in a linear fashion, or use dependency injection/local imports where strictly necessary.

### 2.2. Consolidate Regime Logic
- **Issue**: Co-existence of `execution/regime.py` and `execution/regime_v2.py`.
- **Action**:
  - Determine which version is the "canonical" implementation (likely `regime_v2.py` based on usage).
  - If `regime_v2.py` is the target, verify it fully implements the required interface.
  - Replace usage of the old module with the new one.
  - Delete the obsolete file.

### 2.3. Unify Database Storage
- **Issue**: Redundant databases (`shadow_trades.db`, `safety_state.db`) coexist with the new `trading_state.db`.
- **Action**:
  - Verify that `trading_state.db` contains all necessary schemas (shadow trades, safety metrics).
  - Update any remaining code writing to the old databases to use `trading_state.db` (via `SQLiteShadowStore` or similar).
  - Create a migration script (or instructions) to merge existing data if needed.
  - Remove the old database files and references.

## Phase 3: Testing & Validation Enhancements

**Objective**: Restore trust in the system through comprehensive testing and entry point verification.

### 3.1. Implement Smoke Tests for Scripts
- **Issue**: Scripts like `live.py` broke without detection.
- **Action**:
  - Create `tests/smoke_tests.py` (or similar).
  - Add tests that invoke `scripts/live.py`, `scripts/train.py`, etc., with `--help` or a `--dry-run`/`--test` flag to ensure they initialize without crashing.
  - Ensure these tests run in the CI/CD pipeline.

### 3.2. Address `python-deriv-api` Vulnerability
- **Issue**: Reliance on a local/forked `python-deriv-api`.
- **Action**:
  - Document the specific reasons for the fork (custom fixes?).
  - If possible, package the local fork properly or ensure it's explicitly included in `PYTHONPATH` checks.
  - Add a check in `setup.py` or startup scripts to verify the correct version/fork is loaded.

### 3.3. Improve Test Coverage
- **Issue**: Low coverage in `observability` and script entry points.
- **Action**:
  - Add unit tests for `observability` modules (`dashboard.py`, `model_health.py`).
  - Extend integration tests to cover the data flow from `DerivDataset` to `DerivOmniModel` more rigorously.

## Phase 4: Performance & Optimization

**Objective**: Ensure the system runs efficiently and leverages available hardware.

### 4.1. Optimize Device Selection
- **Issue**: System defaults to CPU; P95 latency (488ms) is high for HFT.
- **Action**:
  - Review the device selection logic (`auto` detection).
  - Ensure that if CUDA is available, it is prioritized.
  - Investigate mixed-precision training/inference (`autocast`) to speed up CPU inference if GPU is unavailable.

### 4.2. Finalize Logging Migration
- **Issue**: `utils.logging_setup` is gone, but we must ensure no code attempts to import it.
- **Action**:
  - Grep the codebase for any remaining references to `utils.logging_setup`.
  - Replace them with `config.logging_config`.

## Phase 5: Cleanup & Documentation

**Objective**: Polish the codebase and documentation to reflect the new architecture.

### 5.1. Update Documentation
- **Issue**: Docs might reference old paths or `models/` instead of `core/domain/entities.py`.
- **Action**:
  - Update `ARCHITECTURE.md` (or equivalent) to reflect the `core/` structure.
  - Document the unification of the risk databases.

### 5.2. Remove Dead Code
- **Issue**: `HistoricalDataDownloader` is mentioned as removed, but we should double-check for any other orphaned files.
- **Action**:
  - Scan for unused files identified in `FILE_INVENTORY.md` or `STATIC_ANALYSIS.md`.
  - Delete `tests/test_m13_disk_management.py` if it relies on `freezegun` and we decide not to support it, OR fix the test by adding the dependency (Phase 1). (Prefer fixing).
