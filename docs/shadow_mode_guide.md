# Shadow Mode Guide: Risk-Free Verification

**Shadow Mode** allows `x.titan` to run fully operational without risking real capital. It is essential for testing new models, verifying strategies, or analyzing market fit.

## How It Works

In Shadow Mode, the system does everything **except** the final API `buy` call:
1.  Connects to Deriv API.
2.  Streams live ticks/candles.
3.  Runs Inference.
4.  Makes Decisions (including Regime Vetoes).
5.  **LOGS** the decision to the Shadow Store (`data_cache/shadow_trades.db`).

## Activating Shadow Mode

### 1. Global Shadow Mode (Safe Start)
Run the live script with the flag:
```bash
python scripts/live.py --shadow-only
```
*   **Use case**: Testing connection, validating a new model checkpoint.

### 2. Regime-Induced Shadow (Automatic)
The system may *partially* enter shadow mode automatically if:
-   **Regime Caution**: Volatility is high. The system demotes medium-confidence trades to shadow trades.
-   **Calibration Issues**: If `CalibrationMonitor` detects the model is inaccurate, it stops real trading and moves to shadow mode to protect capital while gathering data.

## analyzing Results

Shadow trades are stored in a SQLite database, allowing for powerful analysis.

### Generate Report
Run the report generator to see performance:
```bash
python scripts/generate_shadow_report.py
```
This produces an HTML file (e.g., `reports/shadow_report_20231027.html`) showing:
-   Win Rate
-   Expected Value (EV)
-   Trade Distribution
-   Regime Correlations

### Counterfactual Analysis
Because the Shadow Store saves the **Tick Window** and **Candle Window** for every trade, you can later re-simulate:
*   "What if I used a different confident threshold?"
*   "What if I used a different position sizer?"

## Best Practices

1.  **Always Shadow First**: Run a new model in shadow mode for 24 hours before enabling live trading.
2.  **Monitor Discrepancies**: Compare "Real Trades" vs "Shadow Trades" in logs.
3.  **Database Cleanup**: The shadow store can grow large. Periodically archive `data_cache/shadow_trades.db` if it exceeds 1GB.
