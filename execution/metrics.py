
import numpy as np
import pandas as pd
from typing import Dict, Any, List
from dataclasses import dataclass

@dataclass
class TradeMetrics:
    total_trades: int
    winning_trades: int
    losing_trades: int
    win_rate: float
    gross_profit: float
    gross_loss: float
    net_profit: float
    profit_factor: float
    max_drawdown: float
    max_drawdown_pct: float
    sharpe_ratio: float
    sortino_ratio: float
    expectancy: float
    average_win: float
    average_loss: float
    largest_win: float
    largest_loss: float
    initial_balance: float
    final_balance: float
    return_pct: float

class TradingMetrics:
    """
    Calculates comprehensive trading performance metrics.
    """
    
    @staticmethod
    def calculate(trades: List[Dict[str, Any]], initial_balance: Float = 10000.0) -> TradeMetrics:
        """
        Calculate metrics from a list of trade dictionaries.
        
        Args:
            trades: List of dicts with keys: 'outcome' (win/loss), 'profit_loss' (float), 'exit_time' (datetime)
            initial_balance: Starting account balance
            
        Returns:
            TradeMetrics object
        """
        if not trades:
            return TradeMetrics(
                total_trades=0, winning_trades=0, losing_trades=0, win_rate=0.0,
                gross_profit=0.0, gross_loss=0.0, net_profit=0.0, profit_factor=0.0,
                max_drawdown=0.0, max_drawdown_pct=0.0, sharpe_ratio=0.0, sortino_ratio=0.0,
                expectancy=0.0, average_win=0.0, average_loss=0.0,
                largest_win=0.0, largest_loss=0.0, initial_balance=initial_balance, final_balance=initial_balance, return_pct=0.0
            )

        df = pd.DataFrame(trades)
        
        # Ensure necessary columns
        if 'profit_loss' not in df.columns:
            # Try to infer from result/payout
            # This depends on trade dict structure. Assuming standard 'profit_loss' or 'pnl' is passed.
             raise ValueError("Trades must contain 'profit_loss' key")

        pnl = df['profit_loss'].values
        
        # Basic counts
        total_trades = len(df)
        winning_trades = len(df[df['profit_loss'] > 0])
        losing_trades = len(df[df['profit_loss'] <= 0])
        win_rate = (winning_trades / total_trades) * 100 if total_trades > 0 else 0.0
        
        # Financials
        gross_profit = df[df['profit_loss'] > 0]['profit_loss'].sum()
        gross_loss = abs(df[df['profit_loss'] <= 0]['profit_loss'].sum())
        net_profit = pnl.sum()
        profit_factor = (gross_profit / gross_loss) if gross_loss > 0 else float('inf')
        
        final_balance = initial_balance + net_profit
        return_pct = (net_profit / initial_balance) * 100
        
        # Averages
        average_win = df[df['profit_loss'] > 0]['profit_loss'].mean() if winning_trades > 0 else 0.0
        average_loss = df[df['profit_loss'] <= 0]['profit_loss'].mean() if losing_trades > 0 else 0.0
        largest_win = df['profit_loss'].max()
        largest_loss = df['profit_loss'].min()
        
        expectancy = (win_rate/100 * average_win) + ((1 - win_rate/100) * average_loss)
        
        # Drawdown
        # Construct equity curve
        equity_curve = np.concatenate(([initial_balance], initial_balance + np.cumsum(pnl)))
        peaks = np.maximum.accumulate(equity_curve)
        drawdowns = (peaks - equity_curve) / peaks
        max_drawdown_pct = np.max(drawdowns) * 100
        max_drawdown = np.max(peaks - equity_curve)
        
        # Risk Ratios (Annualized assumption: not really possible without time duration context)
        # We will calculate per-trade Sharpe/Sortino
        # Sharpe = Mean(Returns) / Std(Returns)
        # returns_per_trade = pnl / initial_balance # Simplification
        
        # Better: Sharpe of the equity curve returns
        equity_series = pd.Series(equity_curve)
        returns = equity_series.pct_change().dropna()
        
        if len(returns) > 1 and returns.std() > 0:
            sharpe_ratio = (returns.mean() / returns.std()) * np.sqrt(len(trades)) # annualized based on trade frequency? No, just trade-based sharpe.
            # Usually users want Annualized Sharpe. Let's stick to "Trade Sharpe" for now or assume simple scaling.
            # Standard: Sharpe = mean / std
            sharpe_ratio = returns.mean() / returns.std()
            
            downside_returns = returns[returns < 0]
            if len(downside_returns) > 0 and downside_returns.std() > 0:
                sortino_ratio = returns.mean() / downside_returns.std()
            else:
                sortino_ratio = float('inf') if returns.mean() > 0 else 0.0
        else:
            sharpe_ratio = 0.0
            sortino_ratio = 0.0

        return TradeMetrics(
            total_trades=total_trades,
            winning_trades=winning_trades,
            losing_trades=losing_trades,
            win_rate=win_rate,
            gross_profit=gross_profit,
            gross_loss=gross_loss,
            net_profit=net_profit,
            profit_factor=profit_factor,
            max_drawdown=max_drawdown,
            max_drawdown_pct=max_drawdown_pct,
            sharpe_ratio=sharpe_ratio,
            sortino_ratio=sortino_ratio,
            expectancy=expectancy,
            average_win=average_win,
            average_loss=average_loss,
            largest_win=largest_win,
            largest_loss=largest_loss,
            initial_balance=initial_balance,
            final_balance=final_balance,
            return_pct=return_pct
        )
