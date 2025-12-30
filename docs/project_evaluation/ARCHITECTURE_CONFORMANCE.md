# ARCHITECTURE_CONFORMANCE.md

## Overall Conformance Score: 75%

The system generally follows the intended multi-expert architecture, but recent refactoring has introduced critical naming collisions and some inconsistencies in module responsibilities.

## 🔴 CRITICAL DEVIATIONS

### 1. Module Name Collision: `models`
- **Original Design**: `models/` directory for neural network architectures.
- **Current Implementation**: A new file `core/domain/models.py` has been added.
- **Impact**: Severe import ambiguity. Tools like `mypy` fail. Python's import system may load the wrong `models` depending on the path.
- **Recommendation**: Rename `core/domain/models.py` to `core/domain/entities.py` or `core/domain/objects.py`.

### 2. Broken Entry Point: `scripts/live.py`
- **Original Design**: Main production loop.
- **Current Implementation**: Fails with `NameError: name 'model_monitor' is not defined`.
- **Impact**: System cannot trade. This is a regression likely introduced during refactoring of observability or model health components.

## 🟠 HIGH SEVERITY DEVIATIONS

### 3. Data Pipeline Atomicity Failure
- **Original Design**: Reliable partitioned data storage.
- **Current Implementation**: `download_data.py` fails to save partitions due to a pandas truth value ambiguity error.
- **Impact**: Impossible to download new training data reliably.

### 4. Circular Dependencies in `data`
- **Original Design**: Linear data pipeline (Ingestion -> Processor -> Dataset).
- **Current Implementation**: `data.dataset` <-> `data.features` <-> `data.processor` circularity.
- **Impact**: Makes testing difficult and can lead to runtime issues.

## 🟡 MEDIUM SEVERITY DEVIATIONS

### 5. Transition to `core/` (Internal Domain)
- **Original Design**: Modules directly under root (`execution`, `models`, `data`).
- **Current Implementation**: Introduction of `core/interfaces.py` and `core/domain/`.
- **Status**: These appear to be "orphaned" or in the process of being adopted. This is a good direction but currently adds noise and potential confusion if not completed.

### 6. Parallel Regime Modules
- **Original Design**: `execution/regime.py`.
- **Current Implementation**: Co-existence of `regime.py` and `regime_v2.py`.
- **Status**: Ambiguity on which one is the "canonical" version, though `DecisionEngine` seems to take `RegimeVeto` which might come from either.

## Comparison Matrix

| Module/Feature | Original Location | Current Location | Status |
|----------------|-------------------|------------------|--------|
| Decision Engine| `execution/decision.py` | `execution/decision.py` | ✅ Match |
| Expert Models | `models/*.py` | `models/*.py` | ✅ Match |
| Data Ingestion | `data/ingestion/` | `data/ingestion/` | ✅ Match |
| Domain Models | [None] | `core/domain/models.py` | ⚠️ Collision |
| Shadow Store | `data_cache/` | `execution/sqlite_shadow_store.py` | ⚠️ Moved from logic to storage |
| Logging Setup | `utils/logging_setup.py` | `config/logging_config.py` | 🔄 Centralized |
