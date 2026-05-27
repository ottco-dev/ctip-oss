'use client';

import { Wifi, WifiOff, Thermometer, Zap } from 'lucide-react';
import { cn, formatVram } from '@/lib/utils';
import { useGpuStatus } from '@/hooks/useGpuStatus';

interface TopBarProps {
  title?: string;
}

export function TopBar({ title }: TopBarProps) {
  const { wsConnected, vramUsedPct, vramUsed, vramTotal, gpuUtilPct, temperature, deviceName } =
    useGpuStatus();

  return (
    <header className="flex items-center justify-between h-14 px-4 bg-surface border-b border-border shrink-0">
      {/* Page title */}
      <div className="flex items-center gap-2">
        {title && (
          <h1 className="text-sm font-semibold text-text-primary">{title}</h1>
        )}
      </div>

      {/* Right side: GPU status + WS indicator */}
      <div className="flex items-center gap-4">
        {/* GPU VRAM */}
        {vramTotal > 0 && (
          <div className="flex items-center gap-2 text-xs text-text-secondary">
            <span className="hidden sm:inline text-text-muted">{deviceName}</span>
            <div className="flex items-center gap-1">
              <span>VRAM</span>
              <span
                className={cn(
                  'font-mono font-medium',
                  vramUsedPct > 85
                    ? 'text-status-error'
                    : vramUsedPct > 65
                    ? 'text-status-warning'
                    : 'text-status-success',
                )}
              >
                {formatVram(vramUsed)}/{formatVram(vramTotal)}
              </span>
              {/* Mini VRAM bar */}
              <div className="w-16 h-1.5 bg-border rounded-full overflow-hidden">
                <div
                  className={cn(
                    'h-full rounded-full transition-all duration-500',
                    vramUsedPct > 85
                      ? 'bg-status-error'
                      : vramUsedPct > 65
                      ? 'bg-status-warning'
                      : 'bg-status-success',
                  )}
                  style={{ width: `${vramUsedPct}%` }}
                />
              </div>
            </div>

            {/* GPU utilization */}
            {gpuUtilPct !== null && (
              <div className="flex items-center gap-1">
                <Zap className="w-3 h-3" />
                <span className="font-mono">{gpuUtilPct}%</span>
              </div>
            )}

            {/* Temperature */}
            {temperature !== null && (
              <div className="flex items-center gap-1">
                <Thermometer className="w-3 h-3" />
                <span
                  className={cn(
                    'font-mono',
                    temperature > 80
                      ? 'text-status-error'
                      : temperature > 70
                      ? 'text-status-warning'
                      : '',
                  )}
                >
                  {temperature}°C
                </span>
              </div>
            )}
          </div>
        )}

        {/* WebSocket status */}
        <div
          className={cn(
            'flex items-center gap-1 text-xs',
            wsConnected ? 'text-status-success' : 'text-text-muted',
          )}
          title={wsConnected ? 'Live data connected' : 'Disconnected — attempting to reconnect'}
        >
          {wsConnected ? (
            <Wifi className="w-3.5 h-3.5" />
          ) : (
            <WifiOff className="w-3.5 h-3.5" />
          )}
          <span className="hidden sm:inline">
            {wsConnected ? 'Live' : 'Offline'}
          </span>
        </div>
      </div>
    </header>
  );
}
