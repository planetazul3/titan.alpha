# FILE_INVENTORY.md
├── Total Files: 231 (estimated from counts)
├── Python Modules: 168
├── Configuration Files: 7
├── Critical Entry Points:
│   ├── scripts/live.py (982 lines)
│   ├── scripts/train.py (149 lines)
│   ├── main.py (152 lines)
│   ├── scripts/backtest_run.py (205 lines)
│   └── api/dashboard_server.py (491 lines)

## Python File Inventory (by size)
- `scripts/live.py`: 982 lines
- `tools/unify_files.py`: 908 lines
- `data/ingestion/client.py`: 857 lines
- `data/ingestion/historical.py`: 856 lines
- `execution/decision.py`: 770 lines
- `observability/dashboard.py`: 751 lines
- `training/online_learning.py`: 691 lines
- `execution/sqlite_shadow_store.py`: 646 lines
- `models/tft.py`: 612 lines
- `execution/position_sizer.py`: 607 lines

## Configuration Files
- `.env`
- `.env.example`
- `dashboard/.env.example`
- `config/settings.py` (464 lines)
- `config/logging_config.py` (278 lines)
- `config/constants.py` (167 lines)

## Shell Scripts & Automation Tools
- `scripts/install-talib.sh`
- `scripts/coverage_report.sh`
- `config/jules_setup.sh`
- `tools/unify_files.py`
- `tools/verify_checkpoint.py`
- `tools/migrate_shadow_store.py`

## Data Directories & Databases
- `data_cache/`: Primary data store (Parquet, SQLite)
- `logs/`: Application logs
- `checkpoints/`: Model weights

## Frontend Assets (dashboard/)
- `dashboard/package.json`
- `dashboard/tsconfig.json`
- `dashboard/src/` (implied by React structure)
