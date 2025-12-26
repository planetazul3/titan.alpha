/**
 * Shadow trading performance component - MAIN FEATURED CARD.
 */
import { ShadowTradeStats } from '../../types/trading.types';

interface Props {
    stats?: ShadowTradeStats;
    loading?: boolean;
}

export function ShadowPerformance({ stats, loading }: Props) {
    if (loading || !stats) {
        return (
            <div className="bg-slate-800 rounded-xl border border-slate-700 p-6">
                <h2 className="text-xl font-bold text-white mb-4">Shadow Trading Performance</h2>
                <div className="animate-pulse space-y-4">
                    <div className="h-16 bg-slate-700 rounded-lg"></div>
                    <div className="grid grid-cols-3 gap-4">
                        <div className="h-12 bg-slate-700 rounded"></div>
                        <div className="h-12 bg-slate-700 rounded"></div>
                        <div className="h-12 bg-slate-700 rounded"></div>
                    </div>
                </div>
            </div>
        );
    }

    const getWinRateColor = (rate: number) => {
        if (rate >= 0.55) return 'text-emerald-400';
        if (rate >= 0.50) return 'text-amber-400';
        return 'text-red-400';
    };

    const getWinRateBg = (rate: number) => {
        if (rate >= 0.55) return 'bg-emerald-500/10 border-emerald-500/30';
        if (rate >= 0.50) return 'bg-amber-500/10 border-amber-500/30';
        return 'bg-red-500/10 border-red-500/30';
    };

    const getPnLColor = (pnl: number) => {
        return pnl >= 0 ? 'text-emerald-400' : 'text-red-400';
    };

    return (
        <div className="bg-slate-800 rounded-xl border border-slate-700 p-6">
            <div className="flex items-center justify-between mb-6">
                <h2 className="text-xl font-bold text-white">👻 Shadow Trading Performance</h2>
                <span className="text-xs text-slate-500 bg-slate-700 px-2 py-1 rounded">
                    {stats.total.toLocaleString()} total trades
                </span>
            </div>

            {/* Featured Win Rate */}
            <div className={`rounded-xl border p-6 mb-6 ${getWinRateBg(stats.win_rate)}`}>
                <div className="text-center">
                    <div className="text-sm text-slate-400 mb-2">Win Rate</div>
                    <div className={`text-5xl font-mono font-bold ${getWinRateColor(stats.win_rate)}`}>
                        {(stats.win_rate * 100).toFixed(1)}%
                    </div>
                    <div className="text-sm text-slate-500 mt-2">
                        {stats.wins.toLocaleString()} wins / {stats.losses.toLocaleString()} losses
                    </div>
                </div>
            </div>

            {/* Metrics Grid */}
            <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                <div className="bg-slate-700/50 rounded-lg p-4">
                    <div className="text-xs text-slate-400 mb-1">Simulated P&L</div>
                    <div className={`text-2xl font-mono font-bold ${getPnLColor(stats.simulated_pnl)}`}>
                        ${stats.simulated_pnl.toFixed(2)}
                    </div>
                </div>

                <div className="bg-slate-700/50 rounded-lg p-4">
                    <div className="text-xs text-slate-400 mb-1">ROI</div>
                    <div className={`text-2xl font-mono font-bold ${getPnLColor(stats.roi)}`}>
                        {stats.roi.toFixed(1)}%
                    </div>
                </div>

                <div className="bg-slate-700/50 rounded-lg p-4">
                    <div className="text-xs text-slate-400 mb-1">Resolved</div>
                    <div className="text-2xl font-mono font-bold text-cyan-400">
                        {stats.resolved.toLocaleString()}
                    </div>
                </div>

                <div className="bg-slate-700/50 rounded-lg p-4">
                    <div className="text-xs text-slate-400 mb-1">Pending</div>
                    <div className="text-2xl font-mono font-bold text-slate-300">
                        {stats.unresolved.toLocaleString()}
                    </div>
                </div>
            </div>
        </div>
    );
}
