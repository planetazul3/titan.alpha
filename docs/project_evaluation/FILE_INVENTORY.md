# FILE_INVENTORY.md

- **Total Files**: 767
- **Python Modules**: 218
- **Configuration Files**: 
  - `.env`
  - `.env.example` (in dashboard/)
  - `config/settings.py`: 464 lines
  - `config/constants.py`: 42 lines
  - `config/logging_config.py`: 120 lines
  - `dashboard/package.json`
- **Critical Entry Points**:
  - `scripts/live.py`: 982 lines
  - `scripts/train.py`: 149 lines
  - `main.py`: 152 lines
  - `scripts/backtest_run.py`: 205 lines
  - `scripts/download_data.py`: 273 lines

## Python Modules Inventory

### 1. `execution/`
- `execution/decision.py`: 770 lines
- `execution/policy.py`: 379 lines
- `execution/adaptive_risk.py`: 392 lines
- `execution/regime_v2.py`: 535 lines
- `execution/regime.py`: 383 lines
- `execution/sqlite_shadow_store.py`: 646 lines
- `execution/shadow_store.py`: 482 lines
- `execution/safety.py`: 482 lines
- `execution/position_sizer.py`: 607 lines
- `execution/shadow_resolution.py`: 538 lines
- `execution/pending_trade_store.py`: 389 lines
- `execution/real_trade_tracker.py`: 373 lines
- `execution/backtest.py`: 381 lines
- `execution/outcome_resolver.py`: 320 lines

### 2. `models/`
- `models/core.py`: 165 lines
- `models/tft.py`: 612 lines
- `models/temporal.py`: 100 lines
- `models/spatial.py`: 73 lines
- `models/volatility.py`: 61 lines
- `models/policy.py`: 467 lines
- `models/heads.py`: 52 lines
- `models/blocks.py`: 212 lines
- `models/fusion.py`: 50 lines
- `models/attention.py`: 70 lines

### 3. `data/`
- `data/processor.py`: 416 lines
- `data/dataset.py`: 443 lines
- `data/normalizers.py`: 268 lines
- `data/ingestion/client.py`: 857 lines
- `data/ingestion/historical.py`: 854 lines
- `data/ingestion/integrity.py`: 347 lines

### 4. `training/`
- `training/trainer.py`: 541 lines
- `training/online_learning.py`: 691 lines
- `training/shadow_evaluation.py`: 549 lines
- `training/checkpoint_manifest.py`: 479 lines
- `training/rl_trainer.py`: 368 lines
- `training/metrics.py`: 203 lines

### 5. `core/` (NEW Refactored Core)
- `core/interfaces.py`: 33 lines
- `core/domain/base.py`: 21 lines
- `core/domain/models.py`: 76 lines

### 6. `observability/`
- `observability/dashboard.py`: 751 lines
- `observability/model_health.py`: 541 lines
- `observability/inference_optimizer.py`: 433 lines
- `observability/__init__.py`: 448 lines

### 7. `api/`
- `api/dashboard_server.py`: 491 lines

### 8. `scripts/`
- `scripts/live.py`: 982 lines
- `scripts/generate_shadow_report.py`: 433 lines

## Data and Model Assets

### Databases (`data_cache/`)
- `shadow_trades.db`: Core simulated trade storage
- `safety_state.db`: Risk management state
- `idempotency.db`: Transaction safety
- `pending_trades.db`: Active trade tracking
- `trading_state.db`: General system state

### Parquet Data (`data_cache/R_100/`)
- `candles_60/`: OHLCV data for 2024 and 2025
- `ticks/`: Raw tick data for 2024 and 2025

### Model Weights (`checkpoints/`)
- `best_model.pt` (39MB)
- `final_model.pt` (39MB)
- `online_tuned.pt` (14MB)
- Epoch checkpoints (epoch_5.pt to epoch_30.pt)

## Automated Tools and Shell Scripts
- `setup_fork_migration.sh`
- `scripts/install-talib.sh`
- `scripts/coverage_report.sh`
- `config/jules_setup.sh`
