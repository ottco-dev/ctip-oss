'use client';

import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  Legend,
} from 'recharts';
import { useTrainingStatus } from '@/hooks/useTrainingStatus';
import { cn } from '@/lib/utils';

interface LossChartProps {
  className?: string;
}

/**
 * Live training loss chart using Recharts.
 * Subscribes to WebSocket training metrics.
 */
export function LossChart({ className }: LossChartProps) {
  const { liveMetrics, currentEpoch, isTraining } = useTrainingStatus();

  // Build chart data: one entry per epoch
  const epochMap = new Map<number, Record<string, number>>();

  liveMetrics.forEach((m) => {
    if (!epochMap.has(m.epoch)) {
      epochMap.set(m.epoch, { epoch: m.epoch });
    }
    const entry = epochMap.get(m.epoch)!;

    // Normalize key names for display
    if (m.key === 'train_loss' || m.key.includes('box_loss')) {
      entry['Train Loss'] = m.value;
    } else if (m.key.includes('val_loss') || m.key.includes('val/box')) {
      entry['Val Loss'] = m.value;
    } else if (m.key.includes('mAP50') && !m.key.includes('mAP50-95')) {
      entry['mAP50'] = m.value;
    }
  });

  const data = Array.from(epochMap.entries())
    .sort(([a], [b]) => a - b)
    .map(([, entry]) => entry);

  if (data.length === 0) {
    return (
      <div className={cn('card flex items-center justify-center h-48', className)}>
        <div className="text-text-muted text-sm text-center">
          {isTraining ? (
            <>
              <div className="animate-pulse-slow mb-2">●</div>
              Waiting for first epoch...
            </>
          ) : (
            'No training data. Start a training run to see loss curves.'
          )}
        </div>
      </div>
    );
  }

  const hasMap50 = data.some((d) => 'mAP50' in d);

  return (
    <div className={cn('card', className)}>
      <div className="card-header">
        Training Progress
        {isTraining && (
          <span className="ml-2 text-status-success animate-pulse-slow">● LIVE</span>
        )}
        <span className="ml-auto font-mono text-text-secondary normal-case text-xs">
          Epoch {currentEpoch}
        </span>
      </div>

      <ResponsiveContainer width="100%" height={220}>
        <LineChart data={data} margin={{ top: 5, right: 10, left: -20, bottom: 5 }}>
          <CartesianGrid strokeDasharray="3 3" stroke="var(--border)" />
          <XAxis
            dataKey="epoch"
            tick={{ fill: 'var(--text-secondary)', fontSize: 11 }}
            label={{ value: 'Epoch', position: 'insideBottom', offset: -2, fill: 'var(--text-muted)', fontSize: 11 }}
          />
          <YAxis
            yAxisId="loss"
            tick={{ fill: 'var(--text-secondary)', fontSize: 11 }}
            width={45}
          />
          {hasMap50 && (
            <YAxis
              yAxisId="map"
              orientation="right"
              domain={[0, 1]}
              tick={{ fill: 'var(--text-secondary)', fontSize: 11 }}
              width={40}
            />
          )}
          <Tooltip
            contentStyle={{
              backgroundColor: 'var(--panel)',
              border: '1px solid var(--border)',
              borderRadius: '6px',
              fontSize: '12px',
              color: 'var(--text-primary)',
            }}
            formatter={(value: number, name: string) => [
              name.includes('Loss') ? value.toFixed(4) : value.toFixed(4),
              name,
            ]}
          />
          <Legend
            wrapperStyle={{ fontSize: '12px', color: 'var(--text-secondary)' }}
          />

          <Line
            yAxisId="loss"
            type="monotone"
            dataKey="Train Loss"
            stroke="#3b82f6"
            strokeWidth={1.5}
            dot={false}
            activeDot={{ r: 3 }}
          />
          <Line
            yAxisId="loss"
            type="monotone"
            dataKey="Val Loss"
            stroke="#22d3ee"
            strokeWidth={1.5}
            dot={false}
            strokeDasharray="4 2"
            activeDot={{ r: 3 }}
          />
          {hasMap50 && (
            <Line
              yAxisId="map"
              type="monotone"
              dataKey="mAP50"
              stroke="#22c55e"
              strokeWidth={2}
              dot={false}
              activeDot={{ r: 3 }}
            />
          )}
        </LineChart>
      </ResponsiveContainer>
    </div>
  );
}
