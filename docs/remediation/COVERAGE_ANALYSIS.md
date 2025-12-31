# COVERAGE_ANALYSIS.md

## Coverage Summary
- **Overall Coverage**: 68%
- **Tested Modules**: `execution`, `data`, `models`, `training`

## Key Module Coverage
| Module | Coverage % | Missing Lines (Snapshots) |
|--------|------------|---------------------------|
| `execution/decision.py` | 74% | 150-200 (Experimental filtering) |
| `execution/policy.py` | 92% | 340-360 (Manual override logic) |
| `data/dataset.py` | 65% | 100-120 (Cache creation branch) |
| `models/tft.py` | 88% | 450-480 (Quantile loss calc) |
| `training/online_learning.py`| 54% | 300-450 (Fisher Info computation) |

## Critical Testing Gaps
1. **Online Learning**: 54% coverage is low for a module handling live weight updates. Fisher information and EWC logic are partially untested.
2. **Dataset Resilience**: The 65% coverage reflects the failure path I encountered during integration testing (path handling for single parquet files).
3. **Dashboard/API**: Currently 0% coverage as these were excluded from the primary logic-driven test run.

## Recommendations
- **Priority 1**: Add tests for `data/dataset.py` with standalone parquet file paths to fix the observed crash.
- **Priority 2**: Increase unit test depth for `training/online_learning.py` to ensure weight stability during EWC updates.
- **Priority 3**: Implement integration tests for `scripts/live.py` that explicitly mock `model_monitor` to catch NameErrors.
