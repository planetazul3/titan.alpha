/**
 * TypeScript type definitions for trading data.
 */

export interface ShadowTradeStats {
    total: number;
    resolved: number;
    unresolved: number;
    wins: number;
    losses: number;
    win_rate: number;
    simulated_pnl: number;
    roi: number;
}

export interface SafetyState {
    trades_attempted?: number;
    trades_executed?: number;
    trades_blocked_rate_limit?: number;
    trades_blocked_kill_switch?: number;
    daily_pnl?: number;
    consecutive_failures?: number;
}

export interface TradingMetrics {
    shadow_trades: ShadowTradeStats;
    safety_state: SafetyState;
    system_status: 'connected' | 'disconnected' | 'no_data' | 'database_error';
    timestamp?: string;
}

export interface ShadowTrade {
    trade_id: string;
    timestamp: string;
    contract_type: 'RISE_FALL' | 'TOUCH_NO_TOUCH' | 'STAYS_BETWEEN';
    direction: string;
    probability: number;
    entry_price: number;
    exit_price: number | null;
    outcome: number | null; // null=pending, 0=loss, 1=win
    reconstruction_error: number;
    regime_state: 'trusted' | 'caution' | 'veto';
}

export interface WebSocketMessage {
    type: 'metrics' | 'heartbeat' | 'error';
    timestamp: string;
    data: TradingMetrics;
}

export type ConnectionStatus = 'connecting' | 'connected' | 'disconnected';
