/**
 * Connection status indicator component.
 */
import { ConnectionStatus as StatusType } from '../../types/trading.types';

interface Props {
    status: StatusType;
    lastUpdate?: Date | null;
}

export function ConnectionStatus({ status, lastUpdate }: Props) {
    const getStatusColor = () => {
        switch (status) {
            case 'connected': return 'bg-emerald-500';
            case 'connecting': return 'bg-amber-500';
            case 'disconnected': return 'bg-red-500';
        }
    };

    const getStatusText = () => {
        switch (status) {
            case 'connected': return 'Connected';
            case 'connecting': return 'Connecting...';
            case 'disconnected': return 'Disconnected';
        }
    };

    const formatTime = (date: Date) => {
        return date.toLocaleTimeString('en-US', {
            hour12: false,
            hour: '2-digit',
            minute: '2-digit',
            second: '2-digit'
        });
    };

    return (
        <div className="flex items-center gap-3 bg-slate-800 px-4 py-2 rounded-lg border border-slate-700">
            <div className="flex items-center gap-2">
                <div
                    className={`w-3 h-3 rounded-full ${getStatusColor()} ${status === 'connected' ? 'animate-pulse' : ''}`}
                />
                <span className="text-sm font-medium text-slate-300">
                    {getStatusText()}
                </span>
            </div>
            {lastUpdate && status === 'connected' && (
                <span className="text-xs text-slate-500">
                    Last update: {formatTime(lastUpdate)}
                </span>
            )}
        </div>
    );
}
