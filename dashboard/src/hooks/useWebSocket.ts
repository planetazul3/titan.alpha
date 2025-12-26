/**
 * WebSocket hook for real-time trading data streaming.
 */
import { useState, useEffect, useCallback, useRef } from 'react';
import { TradingMetrics, WebSocketMessage, ConnectionStatus } from '../types/trading.types';

const WS_URL = import.meta.env.VITE_WS_URL || 'ws://localhost:8000/ws/trading-stream';

export function useWebSocket() {
    const [metrics, setMetrics] = useState<TradingMetrics | null>(null);
    const [connectionStatus, setConnectionStatus] = useState<ConnectionStatus>('connecting');
    const [error, setError] = useState<string | null>(null);
    const [lastUpdate, setLastUpdate] = useState<Date | null>(null);
    const wsRef = useRef<WebSocket | null>(null);
    const reconnectTimeoutRef = useRef<NodeJS.Timeout | null>(null);

    const connect = useCallback(() => {
        // Clear any existing reconnect timeout
        if (reconnectTimeoutRef.current) {
            clearTimeout(reconnectTimeoutRef.current);
        }

        setConnectionStatus('connecting');

        try {
            const ws = new WebSocket(WS_URL);
            wsRef.current = ws;

            ws.onopen = () => {
                console.log('[WebSocket] Connected');
                setConnectionStatus('connected');
                setError(null);
            };

            ws.onmessage = (event) => {
                try {
                    const message: WebSocketMessage = JSON.parse(event.data);
                    if (message.type === 'metrics' && message.data) {
                        setMetrics(message.data);
                        setLastUpdate(new Date());
                    }
                } catch (e) {
                    console.error('[WebSocket] Failed to parse message:', e);
                }
            };

            ws.onerror = (event) => {
                console.error('[WebSocket] Error:', event);
                setError('Connection error');
            };

            ws.onclose = (event) => {
                console.log('[WebSocket] Disconnected:', event.code, event.reason);
                setConnectionStatus('disconnected');
                wsRef.current = null;

                // Attempt reconnection after 5 seconds
                reconnectTimeoutRef.current = setTimeout(() => {
                    console.log('[WebSocket] Attempting reconnection...');
                    connect();
                }, 5000);
            };
        } catch (e) {
            console.error('[WebSocket] Failed to create connection:', e);
            setError('Failed to connect');
            setConnectionStatus('disconnected');

            // Retry after 5 seconds
            reconnectTimeoutRef.current = setTimeout(connect, 5000);
        }
    }, []);

    useEffect(() => {
        connect();

        return () => {
            if (reconnectTimeoutRef.current) {
                clearTimeout(reconnectTimeoutRef.current);
            }
            if (wsRef.current) {
                wsRef.current.close();
            }
        };
    }, [connect]);

    return {
        metrics,
        connectionStatus,
        error,
        lastUpdate,
        reconnect: connect
    };
}
