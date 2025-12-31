# DATA_FLOW_TRACE.md

## Critical Path Mapping

```mermaid
graph TD
    A[Deriv API Tick] --> B[data/ingestion/client.py: _on_tick]
    B --> C[data/buffer.py: MarketDataBuffer.add_tick]
    C --> D[scripts/live.py: Trading Loop]
    D --> E[data/processor.py: FeatureBuilder.generate_features]
    E --> F[models/core.py: DerivOmniModel.forward]
    F --> G[execution/decision.py: DecisionEngine.process_model_output]
    G --> H[execution/policy.py: ExecutionPolicy.check_vetoes]
    H --> I{All Clear?}
    I -- Yes --> J[execution/executor.py: SafeTradeExecutor.execute_trade]
    I -- No --> K[execution/sqlite_shadow_store.py: SQLiteShadowStore.store_trade]
```

## Step-by-Step Validation

1. **Tick Ingestion**: `DerivClient` correctly handles WebSocket stream and passes data to buffer.
2. **Feature Engineering**: `FeatureBuilder` transforms raw ticks and candles into 3-tuple tensors (ticks, candles, vol_metrics).
3. **Inference**: `DerivOmniModel` fuses temporal (TFT), spatial (CNN), and volatility (VAE) features into contract probabilities.
4. **Decision**: `DecisionEngine` enforces hierarchical vetoes. 
    - **CRITICAL BREAK**: `scripts/live.py` fails to pass `model_monitor` to `run_live_trading`, breaking the chain at the decision stage during live monitoring setup.
5. **Persistence**: `SQLiteShadowStore` captures simulation context. Schema is validated to match current data shapes.

## Side Effects & Logging
- **Observability**: Prometheus metrics are updated at each stage (if enabled).
- **History**: Real trades are tracked in `real_trade_tracker.py` and persisted in `trading_state.db`.
