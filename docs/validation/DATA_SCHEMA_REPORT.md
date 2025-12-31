# DATA_SCHEMA_REPORT.md (Post-Remediation)

## Unified Database: `trading_state.db`
Following Jules' remediation, the previous fragmented database structure has been consolidated into a single unified store.

### Tables in `trading_state.db`
- **`shadow_trades`**: Records of synthetic trades with extended resolution context.
- **`safety_state`**: System-wide safety metrics (daily loss, trade frequency).
- **`pending_trades`**: (MIGRATED) Current active trade queue.
- **`daily_stats`**: Aggregated performance metrics.

## Verified Changes
- **Shadow Record v2.0**: Confirmed that `shadow_trades` table now supports the extra resolution columns required for the new hierarchical regime detector.
- **Idempotency**: `idempotency.db` remains a separate lightweight store to ensure high-performance request tracking.

## Recommendation
- Monitor the growth of `trading_state.db`. As it now contains all historical shadow trades and live state, a pruning strategy may be needed in the next iteration.
