# API_CONTRACT_ANALYSIS.md

## Core Interface Validation

| Component | Function | Signature Match | Notes |
|-----------|----------|-----------------|-------|
| `DerivOmniModel` | `forward()` | ✅ YES | Matches documented tensor input format. |
| `DecisionEngine` | `process_model_output()` | ⚠️ PARTIAL | Now includes `market_data` as optional. |
| `ExecutionPolicy` | `check_vetoes()` | ✅ YES | Strictly follows precedence hierarchy. |
| `Dataset` | `__init__()` | ❌ NO | Broken implementation for parquet files (see Failure #2 in Integration). |

## Breaking Changes Detected
1. **Model Monitoring**: `scripts/live.py` uses `model_monitor` which is undefined, breaking the primary entry point contract.
2. **Dataset Initialization**: `DerivDataset` attempts to create a `.cache` directory within a file path, breaking the data ingestion contract for parquet files.

## New APIs Introduced
- `SafetyProfile.apply()`: Centralized way to apply safety settings to the policy.
- `DecisionEngine.process_with_context()`: RECOMMENDED entry point for shadow capture.

## Backward Compatibility
- Most core neural network interfaces (`models/core.py`) remain compatible, but higher-level orchestration scripts have significant breaking gaps.
