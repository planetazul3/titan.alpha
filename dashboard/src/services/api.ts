/**
 * API service for REST endpoints.
 */
import { ShadowTrade, TradingMetrics } from '../types/trading.types';

const API_BASE = import.meta.env.VITE_API_URL || '';

export async function fetchMetrics(): Promise<TradingMetrics | null> {
    try {
        const response = await fetch(`${API_BASE}/api/metrics/current`);
        if (!response.ok) throw new Error(`HTTP ${response.status}`);
        return await response.json();
    } catch (error) {
        console.error('Failed to fetch metrics:', error);
        return null;
    }
}

export async function fetchShadowTrades(limit = 100): Promise<ShadowTrade[]> {
    try {
        const response = await fetch(`${API_BASE}/api/shadow-trades?limit=${limit}`);
        if (!response.ok) throw new Error(`HTTP ${response.status}`);
        const data = await response.json();
        return data.trades || [];
    } catch (error) {
        console.error('Failed to fetch shadow trades:', error);
        return [];
    }
}

export async function fetchContractStats(): Promise<Record<string, { total: number; wins: number; win_rate: number }>> {
    try {
        const response = await fetch(`${API_BASE}/api/shadow-trades/stats`);
        if (!response.ok) throw new Error(`HTTP ${response.status}`);
        const data = await response.json();
        return data.by_contract_type || {};
    } catch (error) {
        console.error('Failed to fetch contract stats:', error);
        return {};
    }
}

export async function checkHealth(): Promise<boolean> {
    try {
        const response = await fetch(`${API_BASE}/api/health`);
        return response.ok;
    } catch {
        return false;
    }
}
