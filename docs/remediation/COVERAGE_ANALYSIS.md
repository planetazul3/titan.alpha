# COVERAGE_ANALYSIS.md (Post-Remediation)

## Coverage Summary
- **Overall Coverage**: 68%
- **Tested Modules**: `execution`, `data`, `models`, `training`

## Key Module Coverage
| Module | Coverage % | Missing Lines (Snapshots) |
|--------|------------|---------------------------|
| `execution/regime.py` | 89% | 208-214, 345-361 (Edge cases) |
| `execution/decision.py`| 53% | 321-387 (Offline simulation) |
| `training/online_learning.py`| 68% | 590-681 (EWC stabilization) |

## ❌ Test Regressions
The recent consolidation of `regime_v2.py` into `regime.py` has caused 4 unit test failures in `tests/test_execution.py` and `tests/test_regime_veto.py`. These tests are checking for thresholds and state transitions that may have slightly changed in the consolidated hierarchical detector.

## Recommendation
- **High Priority**: Refactor the regime unit tests to align with the new consolidated hierarchical logic in `execution/regime.py`.
- **Medium Priority**: Increase coverage in `decision.py` for offline simulation modes.
