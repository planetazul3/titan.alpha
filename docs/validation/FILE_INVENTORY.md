# FILE_INVENTORY.md (Post-Remediation)

## System Overview
- **Total Python Modules**: 217
- **Critical Entry Points**:
  - `scripts/live.py`: 989 lines (REMEDIATED)
  - `scripts/train.py`: 149 lines
  - `main.py`: 152 lines
  - `api/dashboard_server.py`: 491 lines
  - `pre_training_validation.py`: (REMEDIATED)

## Key Core Modules
- `execution/regime.py`: 549 lines (CONSOLIDATED)
- `execution/decision.py`: 770 lines
- `data/ingestion/client.py`: 857 lines
- `data/ingestion/historical.py`: 856 lines
- `execution/sqlite_shadow_store.py`: 646 lines

## Configuration & Databases
- `trading_state.db`: Unified database for trading state and history (UNIFIED)
- `data_cache/`: Storage for Parquet data and cached features
- `config/settings.py`: Pydantic settings model
- `config/constants.py`: System constants

## Documentation (Review101)
- `docs/review101/REPORT.md`: Master status report
- `docs/validation/`: Detailed audit artifacts
