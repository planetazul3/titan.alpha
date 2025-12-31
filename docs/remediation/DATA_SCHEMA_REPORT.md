# DATA_SCHEMA_REPORT.md

## Database Schema Audit

### 1. `shadow_trades.db`
- **Tale**: `shadow_trades`
- **Columns Added**: `barrier_level`, `barrier2_level`, `duration_minutes`.
- **Integrity**: Includes `reconstruction_error` and `regime_state` for deep traceability.
- **Migration Need**: High. Recent schema changes may require existing data migration.

### 2. `safety_state.db`
- **Table**: `daily_stats`
- **Columns**: `date`, `trade_count`, `daily_pnl`.
- **Table**: `kv_store`
- **Usage**: Handles general persistence for flags and small states.

### 3. `idempotency.db`
- **Usage**: Prevents duplicate trades (Contract ID based).

## Parquet File Format
- Files like `2024-01.parquet` are being used.
- **Issue**: The current code expects these to be directories or handles them incorrectly when they are files (as seen in `train.py` failure).

## Config File Validation
- `config/settings.py` (Pydantic-based): Validates all environment variables.
- `.env.example`: Matches the required structure of `.env`.
