# CONFIGURATION_AUDIT.md

## Settings Management (`config/settings.py`)
- **Framework**: Pydantic v2 (Type-safe validation).
- **Loading Pattern**: Environment variables with `__` delimiter for nesting.
- **Validation**: Strict validators for thresholds (descending order), hyperparameters (positive values), and sequence lengths.

## 🔴 CRITICAL MISCONFIGURATIONS

1. **`scripts/live.py` - Missing Instance `model_monitor`**
   - **Issue**: Source code references `model_monitor` but it is never instantiated or passed correctly.
   - **Impact**: Immediate crash on startup.

2. **`scripts/live.py` - Missing Instance `system_monitor`**
   - **Issue**: Source code references `system_monitor` (Line 332) but it is not defined in the scope.
   - **Impact**: Crash during component registration.

## 🟠 HIGH SEVERITY OBSERVATIONS

3. **Inconsistent Data Paths**
   - **Issue**: `DerivDataset` requires the root `data_cache` but some scripts might pass subdirectories.
   - **Impact**: `FileNotFoundError` during training or validation if paths aren't precise.

4. **Regime Thresholds in Hyperparameters**
   - **Issue**: `regime_caution_threshold` (0.2) and `regime_veto_threshold` (0.5) are hardcoded in `ModelHyperparams`.
   - **Impact**: These should ideally be in `CalibrationConfig` or a dedicated `RegimeConfig` for better separation of concerns.

## 🟡 MEDIUM SEVERITY OBSERVATIONS

5. **API Credential Exposure**
   - **Issue**: `.env` file contains cleartext tokens (Standard for local dev, but needs caution).
   - **Status**: Using `SecretStr` in Pydantic is a good practice for preventing accidental logging.

6. **Default Device `auto`**
   - **Observation**: Automatically resolves to CPU if CUDA is not available. 
   - **Performance Note**: P95 latency on CPU is ~488ms, which is borderline for 1-minute high-frequency contracts.

## 🟢 SECURITY & BEST PRACTICES

- ✅ **Immutable Settings**: Using `frozen=True` in Pydantic ensures settings aren't modified at runtime.
- ✅ **Schema Versioning**: Mention of `FEATURE_SCHEMA_VERSION` in `DecisionEngine` indicates good version control on data features.
- ✅ **Centralized Logging**: Switching to `config/logging_config.py` is successfully implemented.
