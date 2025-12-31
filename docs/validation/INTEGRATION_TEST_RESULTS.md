# INTEGRATION_TEST_RESULTS.md

## Test Scenarios Status

| Scenario | Status | Error Details |
|----------|--------|---------------|
| **Data Ingestion** (`download_data.py --test`) | ✅ PASS | Downloaded 42,768 ticks, 1,427 candles. Integrity checks passed. |
| **Model Training** (`train.py --test-mode`) | ❌ FAIL | `FileNotFoundError`: Code attempts to create `.cache/` directory inside the parquet file path. |
| **Shadow Trading** (`live.py --test`) | ❌ FAIL | `NameError`: `model_monitor` is not defined in `scripts/live.py:284`. |
| **API Backend** | ⚠️ SKIP | Not tested in this phase. |
| **Dashboard** | ⚠️ SKIP | Not tested in this phase. |

## Detailed Failure Analysis

### 1. `live.py` NameError
- **File**: `scripts/live.py:284`
- **Error**: `name 'model_monitor' is not defined`
- **Impact**: **CRITICAL**. System cannot start the live/shadow trading loop.
- **Root Cause**: Likely a missing initialization of `model_monitor` or a typo in the variable name after refactoring.

### 2. `train.py` FileNotFoundError
- **File**: `data/dataset.py:110` (called from `scripts/train.py`)
- **Error**: `FileNotFoundError: [Errno 2] No such file or directory: 'data_cache/2024-01.parquet/.cache'`
- **Impact**: **HIGH**. Prevents training with existing parquet files unless a specific directory structure is present.
- **Root Cause**: The software assumes the data path is a directory and tries to create a `.cache` subdirectory, but fails when the path is a single parquet file.

## Performance Metrics (Partial)
- **Data Download Speed**: 5466 records/sec
- **Model Load Time**: ~1.5s (from `live.py` log)
- **Resource Usage**: ~15% Memory (RSS), high CPU burst during init.
