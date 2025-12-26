# Metrics Proposal: DerivOmniModel Observability

## Goal
Implement structured observability to track system health, model performance, and business outcomes using Prometheus-compatible metrics.

## 1. Business Metrics (Golden Signals)
| Metric Name | Type | Labels | Description |
|---|---|---|---|
| `trading_pnl_daily_usd` | Gauge | `symbol` | Realized Daily P&L (resets 00:00 UTC) |
| `trading_positions_total` | Counter | `symbol`, `status` | Total trades (executed vs blocked) |
| `trading_volume_usd` | Counter | `symbol` | Total volume traded |
| `account_balance_usd` | Gauge | - | Current account balance |

## 2. Model Performance
| Metric Name | Type | Labels | Description |
|---|---|---|---|
| `model_reconstruction_error` | Gauge | `version` | Volatility expert anomaly score |
| `model_inference_latency_seconds` | Histogram | `device` | Time taken for forward pass |
| `regime_veto_state` | Gauge | - | 0=Trusted, 1=Caution, 2=Veto |
| `prediction_confidence` | Histogram | `contract_type` | Model probability distribution |

## 3. System Health
| Metric Name | Type | Labels | Description |
|---|---|---|---|
| `data_freshness_lag_seconds` | Gauge | `stream` | Time since last tick/candle update |
| `api_connection_status` | Gauge | - | 1=Connected, 0=Disconnected |
| `buffer_size` | Gauge | `type` | Current size of tick/candle buffers |
| `safety_circuit_breaker_trips` | Counter | - | Number of circuit breaker activations |

## 4. Implementation Strategy
- **Library**: `prometheus_client`
- **Exposure**: Expose `/metrics` endpoint on localhost:8000 (if running as service) or PushGateway for batch jobs.
- **Integration**:
    - Instantiate `prom_client` in `scripts/live.py`.
    - Update metrics in the main event loop (tick processing).
    - Pass `MetricsCollector` to `DecisionEngine` and `SafeTradeExecutor`.
