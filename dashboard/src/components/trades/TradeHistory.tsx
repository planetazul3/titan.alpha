/**
 * Trade history table component.
 */
import { useState, useEffect } from 'react';
import { ShadowTrade } from '../../types/trading.types';
import { fetchShadowTrades } from '../../services/api';

interface Props {
    refreshTrigger?: number; // Increment to trigger refresh
}

export function TradeHistory({ refreshTrigger }: Props) {
    const [trades, setTrades] = useState<ShadowTrade[]>([]);
    const [loading, setLoading] = useState(true);
    const [filter, setFilter] = useState<'all' | 'wins' | 'losses' | 'pending'>('all');

    useEffect(() => {
        const load = async () => {
            setLoading(true);
            const data = await fetchShadowTrades(100);
            setTrades(data);
            setLoading(false);
        };
        load();
    }, [refreshTrigger]);

    const filteredTrades = trades.filter(trade => {
        if (filter === 'all') return true;
        if (filter === 'wins') return trade.outcome === 1;
        if (filter === 'losses') return trade.outcome === 0;
        if (filter === 'pending') return trade.outcome === null;
        return true;
    });

    const formatTime = (timestamp: string) => {
        const date = new Date(timestamp);
        return date.toLocaleTimeString('en-US', {
            hour12: false,
            hour: '2-digit',
            minute: '2-digit'
        });
    };

    const getOutcomeBadge = (outcome: number | null) => {
        if (outcome === null) {
            return <span className="text-xs px-2 py-1 rounded bg-slate-600 text-slate-300">Pending</span>;
        }
        if (outcome === 1) {
            return <span className="text-xs px-2 py-1 rounded bg-emerald-500/20 text-emerald-400">Win</span>;
        }
        return <span className="text-xs px-2 py-1 rounded bg-red-500/20 text-red-400">Loss</span>;
    };

    const getRegimeBadge = (regime: string) => {
        const colors: Record<string, string> = {
            trusted: 'bg-emerald-500/20 text-emerald-400',
            caution: 'bg-amber-500/20 text-amber-400',
            veto: 'bg-red-500/20 text-red-400',
        };
        return (
            <span className={`text-xs px-2 py-1 rounded ${colors[regime] || 'bg-slate-600 text-slate-300'}`}>
                {regime}
            </span>
        );
    };

    return (
        <div className="bg-slate-800 rounded-xl border border-slate-700 p-6">
            <div className="flex items-center justify-between mb-4">
                <h2 className="text-lg font-bold text-white">📋 Recent Shadow Trades</h2>

                {/* Filter Buttons */}
                <div className="flex gap-1">
                    {(['all', 'wins', 'losses', 'pending'] as const).map(f => (
                        <button
                            key={f}
                            onClick={() => setFilter(f)}
                            className={`px-3 py-1 text-xs rounded transition-colors ${filter === f
                                    ? 'bg-cyan-500 text-white'
                                    : 'bg-slate-700 text-slate-400 hover:bg-slate-600'
                                }`}
                        >
                            {f.charAt(0).toUpperCase() + f.slice(1)}
                        </button>
                    ))}
                </div>
            </div>

            {loading ? (
                <div className="animate-pulse space-y-2">
                    {[1, 2, 3, 4, 5].map(i => (
                        <div key={i} className="h-12 bg-slate-700 rounded"></div>
                    ))}
                </div>
            ) : (
                <div className="overflow-x-auto">
                    <table className="w-full text-sm">
                        <thead>
                            <tr className="text-slate-400 border-b border-slate-700">
                                <th className="text-left py-3 px-2">Time</th>
                                <th className="text-left py-3 px-2">Contract</th>
                                <th className="text-left py-3 px-2">Direction</th>
                                <th className="text-right py-3 px-2">Probability</th>
                                <th className="text-right py-3 px-2">Entry</th>
                                <th className="text-center py-3 px-2">Regime</th>
                                <th className="text-center py-3 px-2">Outcome</th>
                            </tr>
                        </thead>
                        <tbody>
                            {filteredTrades.slice(0, 20).map(trade => (
                                <tr
                                    key={trade.trade_id}
                                    className="border-b border-slate-700/50 hover:bg-slate-700/30 transition-colors"
                                >
                                    <td className="py-3 px-2 font-mono text-slate-300">
                                        {formatTime(trade.timestamp)}
                                    </td>
                                    <td className="py-3 px-2 text-slate-300">
                                        {trade.contract_type.replace('_', ' ')}
                                    </td>
                                    <td className="py-3 px-2">
                                        <span className={trade.direction === 'CALL' ? 'text-emerald-400' : 'text-red-400'}>
                                            {trade.direction}
                                        </span>
                                    </td>
                                    <td className="py-3 px-2 text-right font-mono text-cyan-400">
                                        {(trade.probability * 100).toFixed(1)}%
                                    </td>
                                    <td className="py-3 px-2 text-right font-mono text-slate-300">
                                        {trade.entry_price.toFixed(2)}
                                    </td>
                                    <td className="py-3 px-2 text-center">
                                        {getRegimeBadge(trade.regime_state)}
                                    </td>
                                    <td className="py-3 px-2 text-center">
                                        {getOutcomeBadge(trade.outcome)}
                                    </td>
                                </tr>
                            ))}
                        </tbody>
                    </table>

                    {filteredTrades.length === 0 && (
                        <div className="text-center py-8 text-slate-500">
                            No trades found
                        </div>
                    )}

                    {filteredTrades.length > 20 && (
                        <div className="text-center py-4 text-slate-500 text-sm">
                            Showing 20 of {filteredTrades.length} trades
                        </div>
                    )}
                </div>
            )}
        </div>
    );
}
