# COVERAGE_ANALYSIS.md

## Test Suite Overview
- **Total Test Files**: 69
- **Categories**: Unit, Integration, Behavioral, Property-based (Hypothesis).
- **Test Discovery State**: ✅ PASS (Discovery confirmed on `tests/`)

## Coverage by Module (Estimated)

| Module | Files | Status | Observations |
|--------|-------|--------|--------------|
| `config` | `test_config.py` | ✅ HIGH | All schema validations tested (11/11 pass) |
| `execution`| `test_safety.py`, `test_policy.py`, `test_regime_v2.py`| ✅ HIGH | Critical safety & veto logic tested (14/14 pass) |
| `data` | `test_data.py`, `test_buffer.py`, `test_dataset_alignment.py`| 🟠 MEDIUM | Complex alignment logic exists, coverage appears decent but sensitive to side effects |
| `models` | `test_models.py`, `test_tft.py` | 🟠 MEDIUM | Architecture shapes tested; expert fusion logic tested |
| `observability`| `test_dashboard.py`, `test_model_health.py`| 🟡 LOW | Dashboard integration depends on many external factors |

## 🔴 CRITICAL GAPS

1. **Broken Entry Point Coverage**: No test currently catches the `NameError: model_monitor` in `scripts/live.py`. Smoke tests need to be added for script entry points.
2. **Migration Coverage**: Lack of tests for migrating data from `shadow_trades.db` + `safety_state.db` to `trading_state.db`.
3. **Data Download Resilience**: Tests for `download_data.py` atomic saves are missing (The truth value error went uncaught).

## Recommendation
- Implement **Smoke Tests** for all `scripts/` using `subprocess` or `unittest.mock`.
- Add **Regression Tests** for the identified pandas indexing bug in the downloader.
- Use `pytest-cov` regularly in CI to monitor these numbers.
