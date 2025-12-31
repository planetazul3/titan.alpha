# PERFORMANCE_BASELINE.md

## Execution Metrics

| Metric | Average | P95 | Threshold | Status |
|--------|---------|-----|-----------|--------|
| **Model Inference Latency** | 143.18 ms | 274.50 ms | 100 ms | 🟡 DEGRADED |
| **Database Commit Latency** | 7.59 ms | 12.4 ms | 20 ms | ✅ HEALTHY |
| **Data Processing Throughput** | [ESTIMATED] 5400 records/s | - | 1000 records/s | ✅ HEALTHY |

## Resource Footprint
- **Memory (RSS)**: ~580 MB during live simulation.
- **CPU Load**: High during initialization, stabilizing during steady-state.
- **Disk I/O**: Low, occasional SQLite commits.

## Performance Bottlenecks
- **Model Inference**: 143ms is high for high-frequency trading. Although acceptable for 1-minute timeframes, spike latencies near 300ms (P95) could delay execution during volatile periods.
- **Checkpoint Verification**: Takes ~1s during initialization, delaying startup.
