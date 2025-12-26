/**
 * DerivOmniModel Trading Dashboard
 * 
 * Real-time dashboard for monitoring shadow trading performance,
 * execution statistics, and system health.
 */
import { useWebSocket } from './hooks/useWebSocket';
import { ConnectionStatus } from './components/status/ConnectionStatus';
import { ShadowPerformance } from './components/metrics/ShadowPerformance';
import { ExecutionStats } from './components/metrics/ExecutionStats';
import { TradeHistory } from './components/trades/TradeHistory';

function App() {
    const { metrics, connectionStatus, lastUpdate } = useWebSocket();

    return (
        <div className="min-h-screen bg-slate-950 text-white">
            {/* Header */}
            <header className="bg-slate-900 border-b border-slate-800 px-6 py-4">
                <div className="max-w-7xl mx-auto flex justify-between items-center">
                    <div className="flex items-center gap-4">
                        <h1 className="text-xl font-bold">
                            <span className="text-cyan-400">Deriv</span>
                            <span className="text-white">OmniModel</span>
                        </h1>
                        <span className="text-slate-500 text-sm">Trading Dashboard</span>
                    </div>
                    <div className="flex items-center gap-4">
                        <div className="text-sm text-slate-400">
                            {metrics?.system_status === 'connected' && (
                                <span className="flex items-center gap-2">
                                    <span className="text-slate-500">Symbol:</span>
                                    <span className="text-cyan-400 font-mono">R_100</span>
                                </span>
                            )}
                        </div>
                        <ConnectionStatus status={connectionStatus} lastUpdate={lastUpdate} />
                    </div>
                </div>
            </header>

            {/* Main Content */}
            <main className="max-w-7xl mx-auto p-6 space-y-6">
                {/* Top Row: Shadow Performance + Execution Stats */}
                <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
                    {/* Featured: Shadow Performance (2 cols) */}
                    <div className="lg:col-span-2">
                        <ShadowPerformance
                            stats={metrics?.shadow_trades}
                            loading={connectionStatus === 'connecting'}
                        />
                    </div>

                    {/* Execution Stats (1 col) */}
                    <div>
                        <ExecutionStats
                            safety={metrics?.safety_state}
                            loading={connectionStatus === 'connecting'}
                        />
                    </div>
                </div>

                {/* System Status Banner */}
                {connectionStatus === 'disconnected' && (
                    <div className="bg-amber-500/10 border border-amber-500/30 rounded-lg p-4 flex items-center gap-3">
                        <span className="text-amber-400">⚠️</span>
                        <span className="text-amber-300 text-sm">
                            Disconnected from trading system. Attempting to reconnect...
                        </span>
                    </div>
                )}

                {metrics?.system_status === 'no_data' && connectionStatus === 'connected' && (
                    <div className="bg-slate-800 border border-slate-700 rounded-lg p-4 flex items-center gap-3">
                        <span className="text-slate-400">📊</span>
                        <span className="text-slate-400 text-sm">
                            No trading data available. Is the trading system running?
                        </span>
                    </div>
                )}

                {/* Trade History */}
                <TradeHistory />
            </main>

            {/* Footer */}
            <footer className="border-t border-slate-800 px-6 py-4 mt-8">
                <div className="max-w-7xl mx-auto text-center text-slate-500 text-sm">
                    DerivOmniModel Dashboard • Read-Only View •
                    <span className="text-slate-600"> Shadow trades are simulated and do not affect real balance</span>
                </div>
            </footer>
        </div>
    );
}

export default App;
