'use client';

import { cn, formatVram } from '@/lib/utils';
import { useGpuStatus } from '@/hooks/useGpuStatus';

interface GpuMonitorProps {
  className?: string;
  compact?: boolean;
}

/**
 * GPU monitor widget with ring gauge and stats.
 * Used on training page and system page.
 */
export function GpuMonitor({ className, compact = false }: GpuMonitorProps) {
  const {
    gpu,
    vramUsedPct,
    vramUsed,
    vramTotal,
    vramFree,
    gpuUtilPct,
    temperature,
    powerDraw,
    deviceName,
    wsConnected,
  } = useGpuStatus();

  const RING_RADIUS = 36;
  const RING_CIRCUMFERENCE = 2 * Math.PI * RING_RADIUS;
  const vramOffset = RING_CIRCUMFERENCE * (1 - vramUsedPct / 100);

  const utilOffset = gpuUtilPct !== null
    ? RING_CIRCUMFERENCE * (1 - gpuUtilPct / 100)
    : RING_CIRCUMFERENCE;

  const vramColor =
    vramUsedPct > 85 ? '#ef4444' : vramUsedPct > 65 ? '#eab308' : '#22c55e';

  if (!gpu?.available) {
    return (
      <div className={cn('card flex items-center justify-center h-32', className)}>
        <div className="text-text-muted text-sm">
          {gpu?.reason ?? 'GPU not available'}
        </div>
      </div>
    );
  }

  return (
    <div className={cn('card', className)}>
      <div className="card-header">GPU — {deviceName}</div>

      <div className={cn('flex items-center gap-6', compact ? 'flex-row' : 'flex-col sm:flex-row')}>
        {/* VRAM Ring */}
        <div className="relative flex-shrink-0">
          <svg width="88" height="88" viewBox="0 0 88 88">
            {/* Track */}
            <circle
              cx="44" cy="44" r={RING_RADIUS}
              className="gpu-ring-track"
            />
            {/* VRAM fill */}
            <circle
              cx="44" cy="44" r={RING_RADIUS}
              className="gpu-ring-fill"
              stroke={vramColor}
              strokeDasharray={RING_CIRCUMFERENCE}
              strokeDashoffset={vramOffset}
              style={{ transform: 'rotate(-90deg)', transformOrigin: '44px 44px' }}
            />
          </svg>
          {/* Center text */}
          <div className="absolute inset-0 flex flex-col items-center justify-center">
            <span className="text-lg font-mono font-bold text-text-primary">
              {Math.round(vramUsedPct)}%
            </span>
            <span className="text-[10px] text-text-muted">VRAM</span>
          </div>
        </div>

        {/* Stats */}
        <div className="flex-1 grid grid-cols-2 gap-3 text-xs">
          <div>
            <div className="text-text-muted">Used</div>
            <div className="font-mono text-text-primary font-medium">
              {formatVram(vramUsed)}
            </div>
          </div>
          <div>
            <div className="text-text-muted">Free</div>
            <div className="font-mono text-text-primary font-medium">
              {formatVram(vramFree)}
            </div>
          </div>
          <div>
            <div className="text-text-muted">Total</div>
            <div className="font-mono text-text-primary font-medium">
              {formatVram(vramTotal)}
            </div>
          </div>
          {gpuUtilPct !== null && (
            <div>
              <div className="text-text-muted">GPU Util</div>
              <div className="font-mono text-text-primary font-medium">
                {gpuUtilPct}%
              </div>
            </div>
          )}
          {temperature !== null && (
            <div>
              <div className="text-text-muted">Temp</div>
              <div
                className={cn(
                  'font-mono font-medium',
                  temperature > 80
                    ? 'text-status-error'
                    : temperature > 70
                    ? 'text-status-warning'
                    : 'text-text-primary',
                )}
              >
                {temperature}°C
              </div>
            </div>
          )}
          {powerDraw !== null && (
            <div>
              <div className="text-text-muted">Power</div>
              <div className="font-mono text-text-primary font-medium">
                {powerDraw.toFixed(0)}W
              </div>
            </div>
          )}
        </div>
      </div>

      {/* VRAM bar */}
      <div className="mt-3 progress-bar">
        <div
          className="progress-fill"
          style={{
            width: `${vramUsedPct}%`,
            backgroundColor: vramColor,
          }}
        />
      </div>
      <div className="flex justify-between text-[10px] text-text-muted mt-1">
        <span>0 GB</span>
        <span>{formatVram(vramTotal)}</span>
      </div>

      {/* Live indicator */}
      <div className="mt-2 flex items-center gap-1.5">
        <span
          className={cn(
            'w-1.5 h-1.5 rounded-full',
            wsConnected ? 'bg-status-success animate-pulse-slow' : 'bg-text-muted',
          )}
        />
        <span className="text-[10px] text-text-muted">
          {wsConnected ? 'Live' : 'Disconnected'}
        </span>
      </div>
    </div>
  );
}
