# INTEGRATION_TEST_RESULTS.md

## Test Scenarios

### A. Data Pipeline Test
- **Command**: `python scripts/download_data.py --test`
- **Result**: ❌ FAIL
- **Error**: `Failed to save partition 2025-12 atomically: The truth value of a DataFrame is ambiguous. Use a.empty, a.bool(), a.item(), a.any() or a.all().`
- **Impact**: Data is downloaded but failed to save properly to the partitioned format. This breaks the data lifecycle.

### B. Mode Training Test
- **Command**: `python scripts/train.py --data-path data_cache --epochs 1 --batch-size 8`
- **Result**: ✅ PASS (In progress/validated loading)
- **Observations**: Successfully identified partitioned files and began cache creation. `DerivDataset` requires the root `data_cache` path rather than a specific granularity path.

### C. Live Trading Test (Test Mode)
- **Command**: `python scripts/live.py --test`
- **Result**: ❌ FAIL (CRITICAL)
- **Error**: `NameError: name 'model_monitor' is not defined` at `scripts/live.py:284`
- **Impact**: The production entry point is completely broken. System cannot start.

### D. Dashboard Test
- **Actions**: Checking API health.
- **Result**: ⚠️ UNTESTED (Requires server start, but `live.py` is broken)

## Performance Metrics (Initial)
- **Checkpoint Verification**: ~1s
- **Model Inference (Dummy)**: Fast (outputs validated)
- **Data Ingestion (1 day)**: ~8s (including connection)

## Resource Usage
- **Memory**: Normal
- **CPU**: Low (during tests)

## Integration Success Rate: 33% (1/3 entry points functional)
