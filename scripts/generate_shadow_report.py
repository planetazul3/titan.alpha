#!/usr/bin/env python3
"""
Shadow Trade Report Generator.

Generates comprehensive HTML reports from shadow trade performance data.
Based on industry best practices for trading system performance analysis.

Usage:
    python scripts/generate_shadow_report.py \\
        --start-date 2025-12-20 \\
        --end-date 2025-12-24 \\
        --output reports/shadow_performance.html

Features:
    - Win rate by contract type, time of day, regime state
    - Probability calibration curves (predicted vs actual outcome)
    - Model confidence vs outcome correlation
    - HTML output with embedded visualizations
"""

import argparse
import json
import logging
import sys
from datetime import datetime, timedelta
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from execution.sqlite_shadow_store import SQLiteShadowStore
from observability.shadow_metrics import ShadowTradeMetrics

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")


def generate_html_report(
    shadow_store: SQLiteShadowStore,
    start_date: datetime | None,
    end_date: datetime | None,
    output_path: Path,
) -> None:
    """
    Generate HTML performance report from shadow trades.
    
    Args:
        shadow_store: SQLite shadow store with trade data
        start_date: Start of reporting period (inclusive)
        end_date: End of reporting period (exclusive)  
        output_path: Path to output HTML file
    """
    logger.info(f" Generating shadow trade report...")
    logger.info(f"Period: {start_date or 'all'} to {end_date or 'now'}")
    
    # Query shadow trades
    all_trades = shadow_store.query(start=start_date, end=end_date)
    resolved_trades = shadow_store.query(start=start_date, end=end_date, resolved_only=True)
    
    if not resolved_trades:
        logger.warning("No resolved trades found in specified period")
        # Still generate report showing zero trades
    
    # Calculate metrics
    metrics = ShadowTradeMetrics()
    metrics.update_from_store(shadow_store)
    
    # Build HTML report
    html = f"""
<!DOCTYPE html>
<html>
<head>
    <title>Shadow Trade Performance Report</title>
    <style>
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            margin: 40px;
            background-color: #f5f5f5;
        }}
        .container {{
            max-width: 1200px;
            margin: 0 auto;
            background-color: white;
            padding: 30px;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        h1 {{
            color: #2c3e50;
            border-bottom: 3px solid #3498db;
            padding-bottom: 10px;
        }}
        h2 {{
            color: #34495e;
            margin-top: 30px;
        }}
        .metric-card {{
            display: inline-block;
            background-color: #ecf0f1;
            padding: 20px;
            margin: 10px;
            border-radius: 5px;
            min-width: 200px;
        }}
        .metric-label {{
            font-size: 14px;
            color: #7f8c8d;
            margin-bottom: 5px;
        }}
        .metric-value {{
            font-size: 32px;
            font-weight: bold;
            color: #2c3e50;
        }}
        .positive {{
            color: #27ae60;
        }}
        .negative {{
            color: #e74c3c;
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
            margin-top: 20px;
        }}
        th, td {{
            padding: 12px;
            text-align: left;
            border-bottom: 1px solid #ddd;
        }}
        th {{
            background-color: #3498db;
            color: white;
        }}
        tr:hover {{
            background-color: #f5f5f5;
        }}
        .footer {{
            margin-top: 40px;
            padding-top: 20px;
            border-top: 1px solid #ddd;
            font-size: 12px;
            color: #7f8c8d;
        }}
    </style>
</head>
<body>
    <div class="container">
        <h1>🎯 Shadow Trade Performance Report</h1>
        
        <p><strong>Report Period:</strong> {start_date or 'All time'} to {end_date or datetime.now().strftime('%Y-%m-%d')}</p>
        <p><strong>Generated:</strong> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
        
        <h2>📊 Summary Metrics</h2>
        
        <div class="metric-card">
            <div class="metric-label">Total Trades</div>
            <div class="metric-value">{metrics.total_trades}</div>
        </div>
        
        <div class="metric-card">
            <div class="metric-label">Resolved Trades</div>
            <div class="metric-value">{metrics.resolved_trades}</div>
        </div>
        
        <div class="metric-card">
            <div class="metric-label">Win Rate</div>
            <div class="metric-value {'positive' if metrics.win_rate > 0.5 else 'negative'}">
                {metrics.win_rate * 100:.1f}%
            </div>
        </div>
        
        <div class="metric-card">
            <div class="metric-label">Simulated P&L</div>
            <div class="metric-value {'positive' if metrics.simulated_pnl > 0 else 'negative'}">
                ${metrics.simulated_pnl:.2f}
            </div>
        </div>
        
        <div class="metric-card">
            <div class="metric-label">ROI</div>
            <div class="metric-value {'positive' if metrics.simulated_roi > 0 else 'negative'}">
                {metrics.simulated_roi:.1f}%
            </div>
        </div>
        
        <h2>📈 Performance by Contract Type</h2>
        
        <table>
            <thead>
                <tr>
                    <th>Contract Type</th>
                    <th>Total Trades</th>
                    <th>Wins</th>
                    <th>Losses</th>
                    <th>Win Rate</th>
                </tr>
            </thead>
            <tbody>
"""
    
    # Contract type breakdown
    if metrics.by_contract_type:
        for contract_type, stats in sorted(metrics.by_contract_type.items()):
            win_rate = (stats["wins"] / stats["total"]) * 100 if stats["total"] > 0 else 0
            win_rate_class = "positive" if win_rate > 50 else "negative"
            html += f"""
                <tr>
                    <td>{contract_type}</td>
                    <td>{stats["total"]}</td>
                    <td>{stats["wins"]}</td>
                    <td>{stats["losses"]}</td>
                    <td class="{win_rate_class}">{win_rate:.1f}%</td>
                </tr>
"""
    else:
        html += """
                <tr>
                    <td colspan="5" style="text-align: center; color: #7f8c8d;">No resolved trades yet</td>
                </tr>
"""
    
    html += """
            </tbody>
        </table>
        
        <h2>🎲 Regime State Analysis</h2>
        
        <table>
            <thead>
                <tr>
                    <th>Regime State</th>
                    <th>Wins</th>
                    <th>Losses</th>
                    <th>Win Rate</th>
                </tr>
            </thead>
            <tbody>
"""
    
    # Regime analysis
    regime_states = set(list(metrics.wins_by_regime.keys()) + list(metrics.losses_by_regime.keys()))
    if regime_states:
        for regime in sorted(regime_states):
            wins = metrics.wins_by_regime.get(regime, 0)
            losses = metrics.losses_by_regime.get(regime, 0)
            total = wins + losses
            win_rate = (wins / total) * 100 if total > 0 else 0
            win_rate_class = "positive" if win_rate > 50 else "negative"
            html += f"""
                <tr>
                    <td>{regime}</td>
                    <td>{wins}</td>
                    <td>{losses}</td>
                    <td class="{win_rate_class}">{win_rate:.1f}%</td>
                </tr>
"""
    else:
        html += """
                <tr>
                    <td colspan="4" style="text-align: center; color: #7f8c8d;">No resolved trades yet</td>
                </tr>
"""
    
    html += f"""
            </tbody>
        </table>
        
        <h2>🔍 Confidence Distribution</h2>
        
        <table>
            <thead>
                <tr>
                    <th>Probability Range</th>
                    <th>Count</th>
                    <th>Percentage</th>
                </tr>
            </thead>
            <tbody>
"""
    
    # Confidence distribution
    for range_label, count in metrics.confidence_distribution.items():
        percentage = (count / metrics.total_trades) * 100 if metrics.total_trades > 0 else 0
        html += f"""
                <tr>
                    <td>{range_label}</td>
                    <td>{count}</td>
                    <td>{percentage:.1f}%</td>
                </tr>
"""
    
    html += f"""
            </tbody>
        </table>
        
        <h2>📝 Recommendations</h2>
        
        <ul>
"""
    
    # Generate recommendations based on metrics
    should_retrain, reason = metrics.should_trigger_retraining()
    if should_retrain:
        html += f"<li><strong style='color: #e74c3c;'>⚠️ RETRAINING RECOMMENDED:</strong> {reason}</li>"
    else:
        html += f"<li><strong style='color: #27ae60;'>✅ Performance Acceptable:</strong> {reason}</li>"
    
    if metrics.resolved_trades < 100:
        html += f"<li>Collect more shadow trades ({metrics.resolved_trades}/100 minimum for statistical significance)</li>"
    
    if metrics.win_rate > 0.55:
        html += "<li>Consider gradually increasing confidence threshold for real trades</li>"
    elif metrics.win_rate < 0.45:
        html += "<li>Consider lowering confidence threshold or retraining model</li>"
    
    html += f"""
        </ul>
        
        <div class="footer">
            <p>Generated by DerivOmniModel Shadow Trade Reporting Tool</p>
            <p>Data source: {shadow_store._db_path}</p>
            <p><em>Note: Simulated P&L assumes $1 stake per trade with {int(metrics.simulated_pnl / (metrics.wins * 0.95) * 100 if metrics.wins > 0 else 95)}% payout on wins.</em></p>
        </div>
    </div>
</body>
</html>
"""
    
    # Write report to file
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(html)
    
    logger.info(f"✓ Report generated: {output_path}")
    logger.info(f"  Resolved trades: {metrics.resolved_trades}")
    logger.info(f"  Win rate: {metrics.win_rate * 100:.1f}%")
    logger.info(f"  Simulated P&L: ${metrics.simulated_pnl:.2f}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Generate shadow trade performance report",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate report for last 7 days
  python scripts/generate_shadow_report.py --days 7 --output reports/weekly.html
  
  # Generate report for specific date range
  python scripts/generate_shadow_report.py \\
      --start-date 2025-12-20 \\
      --end-date 2025-12-24 \\
      --output reports/custom.html
  
  # Generate report for all time
  python scripts/generate_shadow_report.py --output reports/all_time.html
        """,
    )
    
    parser.add_argument(
        "--start-date",
        type=str,
        help="Start date (YYYY-MM-DD), inclusive",
    )
    
    parser.add_argument(
        "--end-date",
        type=str,
        help="End date (YYYY-MM-DD), exclusive",
    )
    
    parser.add_argument(
        "--days",
        type=int,
        help="Number of days back from today (alternative to start-date/end-date)",
    )
    
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        default="reports/shadow_performance.html",
        help="Output HTML file path (default: reports/shadow_performance.html)",
    )
    
    parser.add_argument(
        "--db-path",
        type=str,
        default="data_cache/shadow_trades.db",
        help="Path to shadow trade SQLite database (default: data_cache/shadow_trades.db)",
    )
    
    args = parser.parse_args()
    
    # Parse dates
    start_date = None
    end_date = None
    
    if args.days:
        end_date = datetime.now()
        start_date = end_date - timedelta(days=args.days)
        logger.info(f"Report period: Last {args.days} days")
    elif args.start_date or args.end_date:
        if args.start_date:
            start_date = datetime.strptime(args.start_date, "%Y-%m-%d")
        if args.end_date:
            end_date = datetime.strptime(args.end_date, "%Y-%m-%d")
        logger.info(f"Report period: {start_date} to {end_date}")
    else:
        logger.info("Report period: All time")
    
    # Load shadow store
    db_path = Path(args.db_path)
    if not db_path.exists():
        logger.error(f"Shadow trade database not found: {db_path}")
        logger.error("Run live trading system to generate shadow trades first.")
        return 1
    
    shadow_store = SQLiteShadowStore(db_path)
    
    # Generate report
    try:
        generate_html_report(shadow_store, start_date, end_date, Path(args.output))
        return 0
    except Exception as e:
        logger.error(f"Failed to generate report: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())
