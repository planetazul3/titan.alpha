# DATA_FLOW_TRACE.md (Post-Remediation)

## Critical Path: Tick to Trade
1. **Ingestion**: `DerivClient` -> `MarketDataBuffer`.
2. **Features**: `FeatureBuilder` generates Z-Score normalized candles.
3. **Inference**: `DerivOmniModel` performs forward pass (~72ms).
4. **Decision**: `DecisionEngine` evaluates model output + `RegimeVeto`.
5. **Execution**: `SafeTradeExecutor` checks L1-L4 safety before sending to API.
6. **Persistence**: (UPDATED) Every trade and safety state update is committed to the unified `trading_state.db`.

## Key Changes
- **Simplified Storage**: The previous multi-DB flow has been flattened. All critical trading state, shadow records, and safety metrics now converge on a single SQLite connection, reducing the risk of cross-DB reconciliation errors.

## Discovery
- The fix in `scripts/live.py` ensures the `model_monitor` is now correctly receiving telemetry during the "Post-Inference" stage of the data flow.
