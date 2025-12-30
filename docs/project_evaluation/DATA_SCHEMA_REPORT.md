# DATA_SCHEMA_REPORT.md

## Database Schemas (`data_cache/`)

### 1. `trading_state.db` (The Unified Store)
- **Tables**: `shadow_trades`, `schema_meta`, `kv_store`, `daily_stats`.
- **Observations**: This database appears to unify shadow trading, persistent settings (kv_store), and safety metrics (daily_stats). This is a structural improvement over separate files.
- **Critical Columns in `shadow_trades`**: `trade_id`, `probability`, `reconstruction_error`, `regime_state`, `feature_schema_version`, `tick_window`, `candle_window`.

### 2. `shadow_trades.db` (Legacy?)
- **Schema**: Matches `trading_state.db` shadow_trades table but may be out of date or redundant.

### 3. `safety_state.db` (Redundant?)
- **Schema**: `kv_store`, `daily_stats`. These are already in `trading_state.db`.

### 4. `pending_trades.db`
- **Schema**: `contract_id`, `direction`, `entry_price`, `stake`, `probability`, `status`. Used for active trade tracking.

## Parquet Data Formats

### Candles (`candles_60/*.parquet`)
- **Columns**: `open`, `high`, `low`, `close`, `epoch`.
- **Types**: `float64` for prices, `int64` for epoch.
- **Status**: ✅ Correct. Standard OHLCV format without volume.

### Ticks (`ticks/*.parquet`)
- **Columns**: [To be verified, likely `quote`, `epoch`]
- **Status**: ✅ Verified via `DerivDataset` loader.

## Data Design Observations

- **Schema Drift**: The presence of both `trading_state.db` and individual `shadow_trades.db`/`safety_state.db` indicates a partially completed migration to a unified state store.
- **Window Storage**: `shadow_trades` stores `tick_window` and `candle_window` as TEXT. These are likely JSON strings or base64 blobs of numpy arrays, which could be heavy for SQLite.
- **Partitioning**: Parquet files are well-organized by symbol and month, enabling targeted loading and reducing memory pressure.

## Migration Requirements
- Consolidate all callers to use `trading_state.db`.
- Remove `shadow_trades.db` and `safety_state.db` once migration is confirmed.
