'use client';

import { useQuery } from '@tanstack/react-query';
import {
  Database,
  Tag,
  Target,
  Eye,
  Cpu,
  Clock,
  TrendingUp,
  AlertCircle,
} from 'lucide-react';
import { api } from '@/lib/api';
import { GpuMonitor } from '@/components/training/GpuMonitor';
import { LossChart } from '@/components/training/LossChart';
import { cn, formatNumber, pct, timeAgo, getStatusBadgeClass } from '@/lib/utils';
import type { RunSummary, DatasetSummary } from '@/lib/types';

// ── KPI CARD ─────────────────────────────────────────────────────

interface KpiCardProps {
  label: string;
  value: string | number;
  sub?: string;
  icon: React.ElementType;
  color?: string;
}

function KpiCard({ label, value, sub, icon: Icon, color = 'text-accent' }: KpiCardProps) {
  return (
    <div className="card flex items-start gap-4">
      <div className={cn('p-2 rounded-md bg-panel', color)}>
        <Icon className="w-5 h-5" />
      </div>
      <div>
        <div className="text-2xl font-mono font-bold text-text-primary">
          {typeof value === 'number' ? formatNumber(value) : value}
        </div>
        <div className="text-xs text-text-muted uppercase tracking-wider mt-0.5">{label}</div>
        {sub && <div className="text-xs text-text-secondary mt-0.5">{sub}</div>}
      </div>
    </div>
  );
}

// ── MATURITY DISTRIBUTION BAR ────────────────────────────────────

interface MaturityBarProps {
  distribution: Record<string, number>;
}

function MaturityBar({ distribution }: MaturityBarProps) {
  const total = Object.values(distribution).reduce((a, b) => a + b, 0);
  if (total === 0) return null;

  const COLORS: Record<string, string> = {
    clear: '#60a5fa',
    cloudy: '#f9fafb',
    amber: '#f59e0b',
    cloudy_amber_mix: '#d97706',
    degraded: '#6b7280',
    unknown: '#4b5563',
  };

  const LABELS: Record<string, string> = {
    clear: 'Clear',
    cloudy: 'Cloudy',
    amber: 'Amber',
    cloudy_amber_mix: 'Mixed',
    degraded: 'Degraded',
    unknown: 'Unknown',
  };

  return (
    <div>
      {/* Stacked bar */}
      <div className="flex h-4 rounded-md overflow-hidden gap-px">
        {Object.entries(distribution).map(([stage, count]) => {
          const frac = count / total;
          if (frac < 0.02) return null;
          return (
            <div
              key={stage}
              style={{ width: pct(frac), backgroundColor: COLORS[stage] ?? '#4b5563' }}
              title={`${LABELS[stage] ?? stage}: ${pct(frac)}`}
            />
          );
        })}
      </div>

      {/* Legend */}
      <div className="flex flex-wrap gap-3 mt-2">
        {Object.entries(distribution).map(([stage, count]) => {
          const frac = count / total;
          if (frac < 0.01) return null;
          return (
            <div key={stage} className="flex items-center gap-1.5 text-xs text-text-secondary">
              <span
                className="w-2.5 h-2.5 rounded-sm"
                style={{ backgroundColor: COLORS[stage] ?? '#4b5563' }}
              />
              {LABELS[stage] ?? stage}: {pct(frac)}
            </div>
          );
        })}
      </div>
    </div>
  );
}

// ── DASHBOARD PAGE ────────────────────────────────────────────────

export default function DashboardPage() {
  const { data: systemInfo } = useQuery({
    queryKey: ['system-info'],
    queryFn: () => api.get('/system/info').then((r) => r.data),
    refetchInterval: 10_000,
  });

  const { data: datasets } = useQuery<DatasetSummary[]>({
    queryKey: ['datasets'],
    queryFn: () => api.get('/datasets').then((r) => r.data),
    refetchInterval: 30_000,
  });

  const { data: recentRuns } = useQuery<RunSummary[]>({
    queryKey: ['recent-runs'],
    queryFn: () => api.get('/training/runs?limit=5').then((r) => r.data),
    refetchInterval: 10_000,
  });

  const totalImages = datasets?.reduce((a, d) => a + d.num_samples, 0) ?? 0;
  const totalAnnotated = datasets?.reduce((a, d) => a + d.num_annotated, 0) ?? 0;
  const totalReviewed = datasets?.reduce((a, d) => a + d.num_reviewed, 0) ?? 0;

  const bestRun = recentRuns
    ?.filter((r) => r.status === 'completed')
    .sort((a, b) => b.best_map50 - a.best_map50)[0];

  const activeRun = recentRuns?.find((r) => r.status === 'running');

  // Mock maturity distribution for demo (real data would come from a session)
  const maturityDistribution = {
    clear: 12,
    cloudy: 45,
    amber: 28,
    cloudy_amber_mix: 10,
    degraded: 5,
  };

  return (
    <div className="space-y-6 max-w-[1400px] mx-auto">
      <div>
        <h1 className="text-xl font-semibold text-text-primary">Dashboard</h1>
        <p className="text-sm text-text-secondary mt-0.5">
          TrichomeLab — Cannabis Trichome Analysis Platform
        </p>
      </div>

      {/* KPI Grid */}
      <div className="grid grid-cols-2 lg:grid-cols-4 gap-4">
        <KpiCard
          label="Total Images"
          value={totalImages}
          sub={`${totalAnnotated} annotated`}
          icon={Database}
          color="text-status-info"
        />
        <KpiCard
          label="Best mAP50"
          value={bestRun ? `${(bestRun.best_map50 * 100).toFixed(1)}%` : '—'}
          sub={bestRun ? `${bestRun.model_variant}` : 'No runs yet'}
          icon={Target}
          color="text-status-success"
        />
        <KpiCard
          label="Pending Review"
          value={0}
          sub="VLM pseudo-labels"
          icon={Eye}
          color="text-status-warning"
        />
        <KpiCard
          label="Reviewed"
          value={totalReviewed}
          sub="Ready for training"
          icon={Tag}
          color="text-accent"
        />
      </div>

      {/* Main grid */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-4">
        {/* Left column: chart + maturity */}
        <div className="lg:col-span-2 space-y-4">
          {/* Training loss chart */}
          <LossChart />

          {/* Maturity distribution */}
          <div className="card">
            <div className="card-header">Last Session — Maturity Distribution</div>
            <MaturityBar distribution={maturityDistribution} />
            <p className="text-[11px] text-text-muted mt-3">
              ⚠️ Visual trichome color does not quantify cannabinoid content.
              Chromatography required for precise measurement.
            </p>
          </div>

          {/* Recent runs */}
          <div className="card">
            <div className="card-header">Recent Training Runs</div>
            {!recentRuns || recentRuns.length === 0 ? (
              <div className="text-sm text-text-muted py-4 text-center">
                No training runs yet. Start one in the Training page.
              </div>
            ) : (
              <div className="space-y-2">
                {recentRuns.map((run) => (
                  <div
                    key={run.run_uuid}
                    className="flex items-center justify-between py-2 border-b border-border last:border-0"
                  >
                    <div className="flex items-center gap-3">
                      <span className={cn('badge', getStatusBadgeClass(run.status))}>
                        {run.status}
                      </span>
                      <span className="text-sm text-text-primary font-mono">
                        {run.model_variant}
                      </span>
                    </div>
                    <div className="flex items-center gap-4 text-xs text-text-secondary">
                      {run.status === 'completed' && (
                        <span className="font-mono text-status-success">
                          mAP50 {(run.best_map50 * 100).toFixed(1)}%
                        </span>
                      )}
                      {run.started_at && (
                        <span className="text-text-muted">{timeAgo(run.started_at)}</span>
                      )}
                    </div>
                  </div>
                ))}
              </div>
            )}
          </div>
        </div>

        {/* Right column: GPU + system */}
        <div className="space-y-4">
          <GpuMonitor />

          {/* Active training banner */}
          {activeRun && (
            <div className="card border-status-info/30 bg-status-info/5">
              <div className="flex items-center gap-2 mb-2">
                <Cpu className="w-4 h-4 text-status-info animate-pulse-slow" />
                <span className="text-sm font-medium text-status-info">Training Active</span>
              </div>
              <div className="text-xs text-text-secondary space-y-1">
                <div>Model: <span className="text-text-primary">{activeRun.model_variant}</span></div>
                <div>
                  Progress:{' '}
                  <span className="text-text-primary font-mono">
                    {activeRun.best_epoch}/{activeRun.total_epochs} epochs
                  </span>
                </div>
                {activeRun.best_map50 > 0 && (
                  <div>
                    Best mAP50:{' '}
                    <span className="text-status-success font-mono">
                      {(activeRun.best_map50 * 100).toFixed(1)}%
                    </span>
                  </div>
                )}
              </div>
              {/* Progress bar */}
              <div className="progress-bar mt-3">
                <div
                  className="progress-fill"
                  style={{
                    width: activeRun.total_epochs > 0
                      ? `${(activeRun.best_epoch / activeRun.total_epochs) * 100}%`
                      : '0%',
                  }}
                />
              </div>
            </div>
          )}

          {/* Scientific caveat */}
          <div className="scientific-caveat">
            <AlertCircle className="w-4 h-4 shrink-0 mt-0.5" />
            <div>
              <strong className="block mb-0.5">Scientific Note</strong>
              Trichome color analysis cannot determine THC/CBD content.
              Chromatography (GC-MS, HPLC) required for quantification.
              All VLM labels require human review before training use.
            </div>
          </div>

          {/* Dataset summary */}
          <div className="card">
            <div className="card-header">Datasets</div>
            {!datasets || datasets.length === 0 ? (
              <div className="text-sm text-text-muted">No datasets yet.</div>
            ) : (
              <div className="space-y-2">
                {datasets.slice(0, 4).map((ds) => (
                  <div
                    key={ds.id}
                    className="flex items-center justify-between text-xs"
                  >
                    <span className="text-text-primary truncate max-w-[120px]">{ds.name}</span>
                    <span className="text-text-muted font-mono">{formatNumber(ds.num_samples)}</span>
                  </div>
                ))}
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}
