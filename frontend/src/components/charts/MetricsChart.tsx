'use client';

/**
 * MetricsChart — Real-time + historical training metrics visualisation.
 *
 * Combines recharts with the WebSocket training stream AND historical run overlay.
 * Shows:
 *   • Box loss (train + val)
 *   • mAP50 / mAP50-95
 *   • Epoch progress bar
 *   • Optional comparison overlay from a previously completed run
 *
 * Data sources:
 *   - Live: useTrainingStatus() → WS /ws/training
 *   - Historical: GET /training/runs/{run_uuid}/metrics
 */

import React, { useMemo, useState } from 'react';
import { useQuery } from '@tanstack/react-query';
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
import { api } from '@/lib/api';
import { cn } from '@/lib/utils';
import { Activity, TrendingDown, TrendingUp, GitCompare, X } from 'lucide-react';
import type { MetricPoint, RunSummary } from '@/lib/types';

// ─────────────────────────────────────────────────────────────────────────────
// Types
// ─────────────────────────────────────────────────────────────────────────────

type MetricView = 'loss' | 'map' | 'all';

/** Line config for the live (current) run. */
const LIVE_LINES: Record<string, { color: string; dash?: string; label: string }> = {
  'Train Loss': { color: '#60a5fa', label: 'Train box_loss' },
  'Val Loss':   { color: '#f87171', dash: '5 5', label: 'Val box_loss' },
  'mAP50':      { color: '#34d399', label: 'mAP@0.5' },
  'mAP50-95':   { color: '#a78bfa', dash: '3 3', label: 'mAP@0.5:0.95' },
};

/** Line config for the comparison (historical) run — muted palette. */
const CMP_LINES: Record<string, { color: string; dash?: string; label: string }> = {
  'cmp:Train Loss': { color: '#1e40af', dash: '2 4', label: '↩ Train Loss (ref)' },
  'cmp:Val Loss':   { color: '#7f1d1d', dash: '2 4', label: '↩ Val Loss (ref)' },
  'cmp:mAP50':      { color: '#065f46', dash: '2 4', label: '↩ mAP@0.5 (ref)' },
  'cmp:mAP50-95':   { color: '#4c1d95', dash: '2 4', label: '↩ mAP@0.5:0.95 (ref)' },
};

// ─────────────────────────────────────────────────────────────────────────────
// Helpers
// ─────────────────────────────────────────────────────────────────────────────

/** Convert live WS metric stream into per-epoch chart rows. */
function buildLiveData(
  liveMetrics: { epoch: number; key: string; value: number }[],
): Record<string, number>[] {
  const epochMap = new Map<number, Record<string, number>>();

  for (const m of liveMetrics) {
    if (!epochMap.has(m.epoch)) epochMap.set(m.epoch, { epoch: m.epoch });
    const entry = epochMap.get(m.epoch)!;
    const k = m.key.toLowerCase();

    if (k === 'train/box_loss' || k === 'train_loss' || k === 'box_loss') {
      entry['Train Loss'] = m.value;
    } else if (k === 'val/box_loss' || k === 'val_loss' || k === 'val_box_loss') {
      entry['Val Loss'] = m.value;
    } else if ((k.includes('map50') || k === 'metrics/map50') && !k.includes('95')) {
      entry['mAP50'] = m.value;
    } else if (k.includes('map50-95') || k === 'metrics/map50-95') {
      entry['mAP50-95'] = m.value;
    }
  }

  return Array.from(epochMap.entries())
    .sort(([a], [b]) => a - b)
    .map(([, v]) => v);
}

/** Convert historical MetricPoint[] into per-epoch chart rows with `cmp:` prefix keys. */
function buildComparisonData(
  points: MetricPoint[],
): Map<number, Record<string, number>> {
  const epochMap = new Map<number, Record<string, number>>();

  for (const m of points) {
    if (!epochMap.has(m.epoch)) epochMap.set(m.epoch, { epoch: m.epoch });
    const entry = epochMap.get(m.epoch)!;
    const k = m.key.toLowerCase();

    if (k === 'train/box_loss' || k === 'train_loss' || k === 'box_loss') {
      entry['cmp:Train Loss'] = m.value;
    } else if (k === 'val/box_loss' || k === 'val_loss' || k === 'val_box_loss') {
      entry['cmp:Val Loss'] = m.value;
    } else if ((k.includes('map50') || k === 'metrics/map50') && !k.includes('95')) {
      entry['cmp:mAP50'] = m.value;
    } else if (k.includes('map50-95') || k === 'metrics/map50-95') {
      entry['cmp:mAP50-95'] = m.value;
    }
  }

  return epochMap;
}

/**
 * Merge live data + comparison data by epoch.
 * Union of epochs from both sources; missing values left undefined.
 */
function mergeData(
  live: Record<string, number>[],
  cmpMap: Map<number, Record<string, number>>,
): Record<string, number>[] {
  const merged = new Map<number, Record<string, number>>();

  for (const row of live) {
    merged.set(row.epoch, { ...row });
  }

  for (const [epoch, cmpRow] of cmpMap) {
    if (merged.has(epoch)) {
      Object.assign(merged.get(epoch)!, cmpRow);
    } else {
      merged.set(epoch, { epoch, ...cmpRow });
    }
  }

  return Array.from(merged.entries())
    .sort(([a], [b]) => a - b)
    .map(([, v]) => v);
}

// ─────────────────────────────────────────────────────────────────────────────
// Custom tooltip
// ─────────────────────────────────────────────────────────────────────────────

function ChartTooltip({ active, payload, label }: {
  active?: boolean;
  payload?: { name: string; value: number; color: string }[];
  label?: number;
}) {
  if (!active || !payload?.length) return null;
  return (
    <div
      className="rounded-lg px-3 py-2 text-xs space-y-1 min-w-[160px]"
      style={{ background: '#161b22', border: '1px solid #21262d', color: '#e6edf3' }}
    >
      <div className="font-medium mb-1.5 text-white/60">Epoch {label}</div>
      {payload.map((p) => (
        <div key={p.name} className="flex items-center justify-between gap-4">
          <span style={{ color: p.color }}>{p.name}</span>
          <span className="font-mono font-medium">{p.value?.toFixed(4) ?? '—'}</span>
        </div>
      ))}
    </div>
  );
}

// ─────────────────────────────────────────────────────────────────────────────
// Sub-chart — renders live lines + optional comparison lines
// ─────────────────────────────────────────────────────────────────────────────

function SubChart({
  data,
  lines,
  showComparison,
  height = 180,
}: {
  data: Record<string, number>[];
  lines: string[];        // live line keys
  showComparison: boolean;
  height?: number;
}) {
  if (!data.length) return null;

  const cmpKeys = lines.map((l) => `cmp:${l}`);

  return (
    <ResponsiveContainer width="100%" height={height}>
      <LineChart data={data} margin={{ top: 4, right: 8, bottom: 4, left: 0 }}>
        <CartesianGrid strokeDasharray="3 3" stroke="#21262d" vertical={false} />
        <XAxis
          dataKey="epoch"
          tick={{ fill: '#484f58', fontSize: 10 }}
          axisLine={false}
          tickLine={false}
          label={{ value: 'Epoch', position: 'insideBottomRight', offset: 0, fill: '#484f58', fontSize: 9 }}
        />
        <YAxis
          tick={{ fill: '#484f58', fontSize: 10 }}
          axisLine={false}
          tickLine={false}
          width={42}
          tickFormatter={(v: number) => v.toFixed(3)}
        />
        <Tooltip content={<ChartTooltip />} />

        {/* Live run lines */}
        {lines.map((name) => {
          const cfg = LIVE_LINES[name];
          if (!cfg) return null;
          return (
            <Line
              key={name}
              type="monotone"
              dataKey={name}
              name={cfg.label}
              stroke={cfg.color}
              strokeWidth={1.5}
              strokeDasharray={cfg.dash}
              dot={false}
              connectNulls
              activeDot={{ r: 3, fill: cfg.color }}
            />
          );
        })}

        {/* Comparison run lines — only when overlay enabled */}
        {showComparison && cmpKeys.map((name) => {
          const cfg = CMP_LINES[name];
          if (!cfg) return null;
          return (
            <Line
              key={name}
              type="monotone"
              dataKey={name}
              name={cfg.label}
              stroke={cfg.color}
              strokeWidth={1}
              strokeDasharray={cfg.dash}
              dot={false}
              connectNulls
              activeDot={{ r: 2, fill: cfg.color }}
            />
          );
        })}

        <Legend wrapperStyle={{ fontSize: 10, color: '#8b949e', paddingTop: 4 }} />
      </LineChart>
    </ResponsiveContainer>
  );
}

// ─────────────────────────────────────────────────────────────────────────────
// RunSelector — dropdown picker for historical comparison run
// ─────────────────────────────────────────────────────────────────────────────

function RunSelector({
  runs,
  activeRunUuid,
  selected,
  onSelect,
  onClear,
}: {
  runs: RunSummary[];
  activeRunUuid: string | null;
  selected: string | null;
  onSelect: (uuid: string) => void;
  onClear: () => void;
}) {
  const completedRuns = runs.filter(
    (r) => r.status === 'completed' && r.run_uuid !== activeRunUuid,
  );

  if (!completedRuns.length) return null;

  return (
    <div className="flex items-center gap-2">
      <GitCompare className="w-3.5 h-3.5 text-text-muted flex-shrink-0" />
      {selected ? (
        <div className="flex items-center gap-1 bg-surface-tertiary rounded px-2 py-0.5 text-xs">
          <span className="text-text-secondary">vs</span>
          <span className="font-mono text-accent">
            {completedRuns.find((r) => r.run_uuid === selected)?.model_variant ?? '?'}
          </span>
          <span className="font-mono text-text-muted ml-1">
            {selected.slice(0, 6)}…
          </span>
          <button
            onClick={onClear}
            className="ml-1 text-text-muted hover:text-text-primary"
            aria-label="Clear comparison"
          >
            <X className="w-3 h-3" />
          </button>
        </div>
      ) : (
        <select
          className="text-xs bg-transparent border border-border rounded px-1.5 py-0.5 text-text-secondary hover:border-accent cursor-pointer"
          value=""
          onChange={(e) => e.target.value && onSelect(e.target.value)}
        >
          <option value="">Compare with run…</option>
          {completedRuns.map((r) => (
            <option key={r.run_uuid} value={r.run_uuid}>
              {r.model_variant} — {r.run_uuid.slice(0, 8)} (mAP50: {(r.best_map50 * 100).toFixed(1)}%)
            </option>
          ))}
        </select>
      )}
    </div>
  );
}

// ─────────────────────────────────────────────────────────────────────────────
// Main component
// ─────────────────────────────────────────────────────────────────────────────

interface MetricsChartProps {
  /** RunSummary list for the comparison run picker (from training page query). */
  runs?: RunSummary[];
  className?: string;
}

export function MetricsChart({ runs = [], className }: MetricsChartProps) {
  const { liveMetrics, currentEpoch, totalEpochs, bestMap50, isTraining, wsConnected, progressPct, activeRunUuid } =
    useTrainingStatus();

  const [view, setView] = useState<MetricView>('loss');
  const [comparisonUuid, setComparisonUuid] = useState<string | null>(null);

  // Fetch historical metrics for the comparison run
  const { data: cmpMetrics } = useQuery<MetricPoint[]>({
    queryKey: ['run-metrics', comparisonUuid],
    queryFn: () =>
      api.get(`/training/runs/${comparisonUuid}/metrics`).then((r) => r.data),
    enabled: !!comparisonUuid,
    staleTime: 60_000,  // historical data doesn't change — cache 1 min
  });

  const liveData  = useMemo(() => buildLiveData(liveMetrics), [liveMetrics]);
  const cmpMap    = useMemo(
    () => (cmpMetrics ? buildComparisonData(cmpMetrics) : new Map()),
    [cmpMetrics],
  );
  const chartData = useMemo(
    () => (comparisonUuid && cmpMetrics ? mergeData(liveData, cmpMap) : liveData),
    [liveData, cmpMap, comparisonUuid, cmpMetrics],
  );

  const hasLoss = chartData.some((d) => 'Train Loss' in d || 'Val Loss' in d);
  const hasMap  = chartData.some((d) => 'mAP50' in d || 'mAP50-95' in d);
  const hasData = hasLoss || hasMap;
  const showComparison = !!comparisonUuid && !!cmpMetrics;

  // ── Empty state ────────────────────────────────────────────────────────────

  if (!hasData) {
    return (
      <div className={cn('card', className)}>
        <div className="card-header flex items-center gap-2">
          <Activity className="w-4 h-4" />
          Training Metrics
          {isTraining && wsConnected && (
            <span className="ml-2 text-status-success text-xs animate-pulse-slow">● LIVE</span>
          )}
        </div>
        <div className="flex items-center justify-center h-40 text-text-muted text-sm">
          {isTraining ? (
            <div className="text-center">
              <div className="animate-pulse-slow mb-2 text-accent">●</div>
              Waiting for first epoch metrics…
            </div>
          ) : (
            'Start a training run to see metrics.'
          )}
        </div>
      </div>
    );
  }

  // ── Chart view ─────────────────────────────────────────────────────────────

  return (
    <div className={cn('card space-y-3', className)}>
      {/* Header row 1: title + live badge + view toggle + epoch counter */}
      <div className="flex items-center gap-2 flex-wrap">
        <Activity className="w-4 h-4 text-text-muted" />
        <span className="card-header-text">Training Metrics</span>

        {isTraining && wsConnected && (
          <span className="text-status-success text-xs animate-pulse-slow">● LIVE</span>
        )}

        <div className="ml-auto flex items-center gap-2">
          {/* View toggle */}
          <div className="flex rounded-md overflow-hidden border border-border text-xs">
            {(['loss', 'map', 'all'] as MetricView[]).map((v) => (
              <button
                key={v}
                onClick={() => setView(v)}
                className={cn(
                  'px-2.5 py-1 transition-colors',
                  view === v
                    ? 'bg-accent text-white'
                    : 'text-text-secondary hover:text-text-primary',
                )}
              >
                {v === 'loss' ? 'Loss' : v === 'map' ? 'mAP' : 'All'}
              </button>
            ))}
          </div>

          {/* Epoch counter */}
          <span className="text-xs font-mono text-text-secondary">
            {currentEpoch}/{totalEpochs > 0 ? totalEpochs : '?'}
          </span>
        </div>
      </div>

      {/* Header row 2: run comparison selector */}
      {runs.length > 0 && (
        <RunSelector
          runs={runs}
          activeRunUuid={activeRunUuid}
          selected={comparisonUuid}
          onSelect={setComparisonUuid}
          onClear={() => setComparisonUuid(null)}
        />
      )}

      {/* Progress bar */}
      {isTraining && totalEpochs > 0 && (
        <div>
          <div className="progress-bar h-1">
            <div
              className="progress-fill h-1 bg-accent transition-all duration-500"
              style={{ width: `${progressPct}%` }}
            />
          </div>
          <div className="flex justify-between text-[10px] text-text-muted mt-0.5">
            <span>{progressPct.toFixed(1)}%</span>
            <span>{totalEpochs - currentEpoch} epochs remaining</span>
          </div>
        </div>
      )}

      {/* Best mAP badge */}
      {bestMap50 > 0 && (
        <div className="flex items-center gap-2 text-xs">
          <TrendingUp className="w-3.5 h-3.5 text-status-success" />
          <span className="text-text-secondary">Best mAP@0.5:</span>
          <span className="font-mono font-bold text-status-success">
            {(bestMap50 * 100).toFixed(1)}%
          </span>
          {showComparison && (() => {
            const refBest = cmpMetrics
              ? Math.max(...cmpMetrics
                  .filter((m) => m.key.toLowerCase().includes('map50') && !m.key.includes('95'))
                  .map((m) => m.value), 0)
              : 0;
            const delta = bestMap50 - refBest;
            return refBest > 0 ? (
              <span className={cn('font-mono text-xs', delta >= 0 ? 'text-status-success' : 'text-status-error')}>
                ({delta >= 0 ? '+' : ''}{(delta * 100).toFixed(1)}% vs ref)
              </span>
            ) : null;
          })()}
        </div>
      )}

      {/* Charts */}
      {(view === 'loss' || view === 'all') && hasLoss && (
        <div>
          <div className="text-xs text-text-muted mb-1 flex items-center gap-1">
            <TrendingDown className="w-3 h-3" />
            Box Loss
            {showComparison && (
              <span className="ml-1 text-[10px] text-text-muted opacity-60">
                (dashed = reference run)
              </span>
            )}
          </div>
          <SubChart
            data={chartData}
            lines={['Train Loss', 'Val Loss']}
            showComparison={showComparison}
          />
        </div>
      )}

      {(view === 'map' || view === 'all') && hasMap && (
        <div>
          <div className="text-xs text-text-muted mb-1 flex items-center gap-1">
            <TrendingUp className="w-3 h-3" />
            mAP
            {showComparison && (
              <span className="ml-1 text-[10px] text-text-muted opacity-60">
                (dashed = reference run)
              </span>
            )}
          </div>
          <SubChart
            data={chartData}
            lines={['mAP50', 'mAP50-95']}
            showComparison={showComparison}
          />
        </div>
      )}

      {/* Footer */}
      <div className="text-[10px] text-text-muted">
        {chartData.length} epochs · {liveMetrics.length} metric points
        {showComparison && cmpMetrics && (
          <span className="ml-2 text-text-muted/60">
            · {cmpMetrics.length} ref points overlaid
          </span>
        )}
      </div>
    </div>
  );
}
