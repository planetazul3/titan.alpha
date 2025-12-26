/**
 * Execution statistics component.
 */
import { SafetyState } from '../../types/trading.types';

interface Props {
    safety?: SafetyState;
    loading?: boolean;
}

export function ExecutionStats({ safety, loading }: Props) {
    if (loading || !safety) {
        return (
            <div className="bg-slate-800 rounded-xl border border-slate-700 p-6">
                <h2 className="text-lg font-bold text-white mb-4">Execution Stats</h2>
                <div className="animate-pulse space-y-3">
                    {[1, 2, 3, 4].map(i => (
                        <div key={i} className="h-8 bg-slate-700 rounded"></div>
                    ))}
                </div>
            </div>
        );
    }

    const stats = [
        {
            label: 'Trades Attempted',
            value: safety.trades_attempted || 0,
            color: 'text-slate-300'
        },
        {
            label: 'Trades Executed',
            value: safety.trades_executed || 0,
            color: 'text-emerald-400'
        },
        {
            label: 'Blocked (Rate Limit)',
            value: safety.trades_blocked_rate_limit || 0,
            color: 'text-amber-400'
        },
        {
            label: 'Blocked (Kill Switch)',
            value: safety.trades_blocked_kill_switch || 0,
            color: 'text-red-400'
        },
    ];

    const executionRate = safety.trades_attempted
        ? ((safety.trades_executed || 0) / safety.trades_attempted * 100).toFixed(1)
        : '0.0';

    return (
        <div className="bg-slate-800 rounded-xl border border-slate-700 p-6">
            <div className="flex items-center justify-between mb-4">
                <h2 className="text-lg font-bold text-white">⚡ Execution Stats</h2>
                <span className="text-xs text-slate-400 bg-slate-700 px-2 py-1 rounded">
                    {executionRate}% success
                </span>
            </div>

            <div className="space-y-3">
                {stats.map(stat => (
                    <div key={stat.label} className="flex justify-between items-center">
                        <span className="text-sm text-slate-400">{stat.label}</span>
                        <span className={`font-mono font-medium ${stat.color}`}>
                            {stat.value.toLocaleString()}
                        </span>
                    </div>
                ))}
            </div>

            {(safety.daily_pnl !== undefined && safety.daily_pnl !== 0) && (
                <div className="mt-4 pt-4 border-t border-slate-700">
                    <div className="flex justify-between items-center">
                        <span className="text-sm text-slate-400">Daily P&L</span>
                        <span className={`font-mono font-bold ${safety.daily_pnl >= 0 ? 'text-emerald-400' : 'text-red-400'}`}>
                            ${safety.daily_pnl.toFixed(2)}
                        </span>
                    </div>
                </div>
            )}
        </div>
    );
}
