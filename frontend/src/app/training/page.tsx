'use client';

import { useState, useEffect, Suspense } from 'react';
import { useRouter, useSearchParams } from 'next/navigation';
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query';
import {
  Play, Square, ChevronDown, ChevronUp, RotateCcw,
  Settings, Activity, Sliders, Cpu, FlaskConical,
  Plus, RefreshCw, Loader2, Trash2, ArchiveRestore, Archive,
  AlertTriangle, X,
} from 'lucide-react';
import { api } from '@/lib/api';
import { GpuMonitor } from '@/components/training/GpuMonitor';
import { MetricsChart } from '@/components/charts/MetricsChart';
import { LabelStudioDatasetPicker } from '@/components/training/LabelStudioDatasetPicker';
import { TrainingLog } from '@/components/training/TrainingLog';
import { useTrainingStore } from '@/store/trainingStore';
import { useTrainingStatus } from '@/hooks/useTrainingStatus';
import { cn, getStatusBadgeClass, timeAgo, formatDuration } from '@/lib/utils';
import type { TrainingStartRequest, RunSummary, PrepareDatasetResponse } from '@/lib/types';

// ── Constants ──────────────────────────────────────────────────────────────

const MODEL_OPTIONS = [
  { value: 'yolo11n', label: 'YOLOv11 Nano  (~0.6 GB VRAM)', note: 'fastest' },
  { value: 'yolo11s', label: 'YOLOv11 Small (~1.2 GB VRAM)', note: 'recommended' },
  { value: 'yolo11m', label: 'YOLOv11 Medium (~2.5 GB VRAM)', note: 'best quality' },
];

const DEFAULT_FORM: TrainingStartRequest = {
  experiment_name: 'trichome-detection',
  model_variant: 'yolo11s',
  data_yaml: '/data/datasets/trichome/data.yaml',
  epochs: 150,
  batch_size: 4,
  imgsz: 1280,
  amp: true,
  seed: 42,
  notes: '',
  // LR schedule
  lr0: 0.01,
  lrf: 0.01,
  warmup_epochs: 3.0,
  cos_lr: true,
  // Regularisation
  weight_decay: 0.0005,
  momentum: 0.937,
  // Early stopping
  patience: 50,
  // Augmentation
  augment: true,
  mosaic: 1.0,
  close_mosaic: 10,
  hsv_h: 0.015,
  hsv_s: 0.7,
  hsv_v: 0.4,
  degrees: 0.0,
  scale: 0.5,
  flipud: 0.0,
  fliplr: 0.5,
};

// ── Training helper components ─────────────────────────────────────────────

function Toggle({
  value,
  onChange,
  label,
  hint,
}: {
  value: boolean;
  onChange: (v: boolean) => void;
  label: string;
  hint?: string;
}) {
  return (
    <div className="flex items-center justify-between">
      <div>
        <div className="text-sm text-text-primary">{label}</div>
        {hint && <div className="text-xs text-text-muted">{hint}</div>}
      </div>
      <button
        onClick={() => onChange(!value)}
        className={cn(
          'w-10 h-5 rounded-full transition-colors relative flex-shrink-0',
          value ? 'bg-accent' : 'bg-border',
        )}
        aria-pressed={value}
      >
        <span
          className={cn(
            'absolute top-0.5 left-0.5 w-4 h-4 bg-white rounded-full transition-transform',
            value && 'translate-x-5',
          )}
        />
      </button>
    </div>
  );
}

function RangeInput({
  label,
  hint,
  value,
  min,
  max,
  step,
  onChange,
}: {
  label: string;
  hint?: string;
  value: number;
  min: number;
  max: number;
  step: number;
  onChange: (v: number) => void;
}) {
  return (
    <div>
      <div className="flex items-center justify-between mb-1">
        <label className="text-xs text-text-secondary">{label}</label>
        <span className="font-mono text-xs text-accent">{value}</span>
      </div>
      {hint && <div className="text-xs text-text-muted mb-1.5">{hint}</div>}
      <input
        type="range"
        className="w-full accent-accent h-1.5 cursor-pointer"
        min={min}
        max={max}
        step={step}
        value={value}
        onChange={(e) => onChange(parseFloat(e.target.value))}
      />
      <div className="flex justify-between text-[10px] text-text-muted mt-0.5">
        <span>{min}</span>
        <span>{max}</span>
      </div>
    </div>
  );
}

function NumberInput({
  label,
  hint,
  value,
  min,
  max,
  step,
  onChange,
  mono,
}: {
  label: string;
  hint?: string;
  value: number;
  min?: number;
  max?: number;
  step?: number;
  onChange: (v: number) => void;
  mono?: boolean;
}) {
  return (
    <div>
      <label className="block text-xs text-text-secondary mb-1.5">{label}</label>
      {hint && <div className="text-xs text-text-muted mb-1">{hint}</div>}
      <input
        className={cn('input', mono && 'font-mono text-xs')}
        type="number"
        value={value}
        min={min}
        max={max}
        step={step}
        onChange={(e) => onChange(parseFloat(e.target.value))}
      />
    </div>
  );
}

function Section({
  title,
  icon: Icon,
  defaultOpen = false,
  children,
}: {
  title: string;
  icon: React.ElementType;
  defaultOpen?: boolean;
  children: React.ReactNode;
}) {
  const [open, setOpen] = useState(defaultOpen);
  return (
    <div className="border border-border rounded-lg overflow-hidden">
      <button
        className="w-full flex items-center justify-between px-4 py-2.5 bg-surface-secondary hover:bg-surface-tertiary transition-colors"
        onClick={() => setOpen((o) => !o)}
      >
        <div className="flex items-center gap-2">
          <Icon className="w-4 h-4 text-text-secondary" />
          <span className="text-sm font-medium text-text-primary">{title}</span>
        </div>
        {open ? (
          <ChevronUp className="w-4 h-4 text-text-muted" />
        ) : (
          <ChevronDown className="w-4 h-4 text-text-muted" />
        )}
      </button>
      {open && <div className="p-4 space-y-4 border-t border-border">{children}</div>}
    </div>
  );
}

// ── Runs tab ───────────────────────────────────────────────────────────────

function RunsTab() {
  const qc = useQueryClient();
  const { setActiveRun, clearActiveRun } = useTrainingStore();
  const { wsConnected } = useTrainingStatus();

  const [form, setForm] = useState<TrainingStartRequest>(DEFAULT_FORM);
  const set = <K extends keyof TrainingStartRequest>(key: K, val: TrainingStartRequest[K]) =>
    setForm((f) => ({ ...f, [key]: val }));

  const handleDatasetReady = (yamlPath: string, _info: PrepareDatasetResponse) => {
    set('data_yaml', yamlPath);
  };

  const { data: runs } = useQuery<RunSummary[]>({
    queryKey: ['training-runs'],
    queryFn: () => api.get('/training/runs?limit=20').then((r) => r.data),
    refetchInterval: 5000,
  });

  const startMutation = useMutation({
    mutationFn: (req: TrainingStartRequest) =>
      api.post('/training/start', req).then((r) => r.data),
    onSuccess: (data) => {
      setActiveRun(data.run_uuid, form.epochs);
      qc.invalidateQueries({ queryKey: ['training-runs'] });
    },
  });

  const stopMutation = useMutation({
    mutationFn: (runUuid: string) =>
      api.post(`/training/stop/${runUuid}`).then((r) => r.data),
    onSuccess: () => {
      clearActiveRun();
      qc.invalidateQueries({ queryKey: ['training-runs'] });
    },
  });

  const activeRun = runs?.find((r) => r.status === 'running');
  const isRunning = !!activeRun;

  const handleStart = () => {
    if (!form.data_yaml) {
      alert('Please specify the dataset YAML path.');
      return;
    }
    startMutation.mutate(form);
  };

  const handleReset = () => setForm(DEFAULT_FORM);

  return (
    <div className="space-y-6">
      {/* Active run banner */}
      {isRunning && activeRun && (
        <div className="card border-status-info/30 bg-status-info/5">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-3">
              <span className="w-2 h-2 bg-status-info rounded-full animate-pulse-slow" />
              <div>
                <div className="text-sm font-medium text-text-primary">
                  Training in progress — {activeRun.model_variant}
                </div>
                <div className="text-xs text-text-muted">
                  Run: <span className="font-mono">{activeRun.run_uuid.slice(0, 8)}…</span>
                </div>
              </div>
            </div>
            <button
              className="btn-danger flex items-center gap-2"
              onClick={() => stopMutation.mutate(activeRun.run_uuid)}
              disabled={stopMutation.isPending}
            >
              <Square className="w-4 h-4" />
              Stop Training
            </button>
          </div>
        </div>
      )}

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Config panel */}
        <div className="lg:col-span-1 space-y-3">
          <div className="card">
            <div className="card-header flex items-center justify-between">
              <span>Training Configuration</span>
              <button
                className="text-xs text-text-muted hover:text-text-secondary flex items-center gap-1"
                onClick={handleReset}
                title="Reset to defaults"
              >
                <RotateCcw className="w-3 h-3" />
                Reset
              </button>
            </div>

            <div className="space-y-4">
              <div>
                <label className="block text-xs text-text-secondary mb-1.5">Experiment Name</label>
                <input
                  className="input"
                  value={form.experiment_name}
                  onChange={(e) => set('experiment_name', e.target.value)}
                  placeholder="trichome-detection"
                />
              </div>

              <div>
                <label className="block text-xs text-text-secondary mb-1.5">Model</label>
                <select
                  className="input"
                  value={form.model_variant}
                  onChange={(e) => set('model_variant', e.target.value)}
                >
                  {MODEL_OPTIONS.map((opt) => (
                    <option key={opt.value} value={opt.value}>
                      {opt.label}
                    </option>
                  ))}
                </select>
              </div>

              <div className="space-y-2">
                <LabelStudioDatasetPicker onDatasetReady={handleDatasetReady} />
                <div>
                  <label className="block text-xs text-text-secondary mb-1.5">
                    Dataset YAML Path
                    <span className="ml-1 text-text-muted">(auto-filled from Label Studio or enter manually)</span>
                  </label>
                  <input
                    className="input font-mono text-xs"
                    value={form.data_yaml}
                    onChange={(e) => set('data_yaml', e.target.value)}
                    placeholder="/data/datasets/ctip_combined/dataset.yaml"
                  />
                </div>
              </div>

              <div className="grid grid-cols-2 gap-3">
                <NumberInput
                  label="Epochs"
                  value={form.epochs}
                  min={1}
                  max={500}
                  step={1}
                  onChange={(v) => set('epochs', v)}
                />
                <NumberInput
                  label="Batch Size"
                  value={form.batch_size}
                  min={1}
                  max={32}
                  step={1}
                  onChange={(v) => set('batch_size', v)}
                />
              </div>

              <div>
                <label className="block text-xs text-text-secondary mb-1.5">Image Size</label>
                <select
                  className="input"
                  value={form.imgsz}
                  onChange={(e) => set('imgsz', parseInt(e.target.value))}
                >
                  <option value={640}>640 px</option>
                  <option value={960}>960 px</option>
                  <option value={1280}>1280 px (recommended)</option>
                </select>
              </div>

              <Toggle
                value={form.amp}
                onChange={(v) => set('amp', v)}
                label="Mixed Precision (FP16)"
                hint="Recommended for RTX 4060"
              />

              <div className="code-block">
                Effective batch: {form.batch_size} × 4 accum ={' '}
                <span className="text-accent font-bold">{form.batch_size * 4}</span>
              </div>
            </div>
          </div>

          <Section title="Learning Rate Schedule" icon={Activity} defaultOpen={false}>
            <NumberInput
              label="Initial LR (lr0)"
              value={form.lr0}
              min={0.0001}
              max={0.1}
              step={0.001}
              onChange={(v) => set('lr0', v)}
              mono
            />
            <NumberInput
              label="Final LR fraction (lrf)"
              hint="Final LR = lr0 × lrf. Lower → more decay."
              value={form.lrf}
              min={0.001}
              max={1.0}
              step={0.001}
              onChange={(v) => set('lrf', v)}
              mono
            />
            <RangeInput
              label="Warmup Epochs"
              hint="Linear warmup before full LR"
              value={form.warmup_epochs}
              min={0}
              max={10}
              step={0.5}
              onChange={(v) => set('warmup_epochs', v)}
            />
            <Toggle
              value={form.cos_lr}
              onChange={(v) => set('cos_lr', v)}
              label="Cosine LR Schedule"
              hint="Smoother decay. Recommended for runs > 100 epochs."
            />
          </Section>

          <Section title="Regularisation & Optimiser" icon={Settings} defaultOpen={false}>
            <NumberInput
              label="Weight Decay"
              value={form.weight_decay}
              min={0}
              max={0.1}
              step={0.0001}
              onChange={(v) => set('weight_decay', v)}
              mono
            />
            <NumberInput
              label="Momentum (β₁)"
              value={form.momentum}
              min={0.0}
              max={1.0}
              step={0.001}
              onChange={(v) => set('momentum', v)}
              mono
            />
            <NumberInput
              label="Early-Stop Patience"
              hint="Epochs without mAP50 improvement before stopping"
              value={form.patience}
              min={1}
              max={500}
              step={1}
              onChange={(v) => set('patience', v)}
            />
          </Section>

          <Section title="Augmentation" icon={Sliders} defaultOpen={false}>
            <Toggle
              value={form.augment}
              onChange={(v) => set('augment', v)}
              label="Enable Augmentation"
              hint="Disable to train on clean crops only"
            />

            {form.augment && (
              <>
                <RangeInput
                  label="Mosaic probability"
                  hint="Combines 4 images into one tile"
                  value={form.mosaic}
                  min={0}
                  max={1}
                  step={0.05}
                  onChange={(v) => set('mosaic', v)}
                />
                <NumberInput
                  label="Close mosaic (final N epochs)"
                  hint="Stabilises training near convergence"
                  value={form.close_mosaic}
                  min={0}
                  max={50}
                  step={1}
                  onChange={(v) => set('close_mosaic', v)}
                />
                <RangeInput
                  label="Scale augmentation"
                  hint="Random resize ±fraction"
                  value={form.scale}
                  min={0}
                  max={0.9}
                  step={0.05}
                  onChange={(v) => set('scale', v)}
                />
                <RangeInput
                  label="Rotation (degrees)"
                  hint="Trichomes appear at all angles"
                  value={form.degrees}
                  min={0}
                  max={180}
                  step={5}
                  onChange={(v) => set('degrees', v)}
                />
                <div className="grid grid-cols-2 gap-3">
                  <RangeInput
                    label="Flip horizontal"
                    value={form.fliplr}
                    min={0}
                    max={1}
                    step={0.1}
                    onChange={(v) => set('fliplr', v)}
                  />
                  <RangeInput
                    label="Flip vertical"
                    value={form.flipud}
                    min={0}
                    max={1}
                    step={0.1}
                    onChange={(v) => set('flipud', v)}
                  />
                </div>
                <div>
                  <label className="block text-xs text-text-secondary mb-2">
                    HSV Augmentation
                  </label>
                  <div className="space-y-2">
                    <RangeInput
                      label="Hue ±"
                      value={form.hsv_h}
                      min={0}
                      max={0.1}
                      step={0.005}
                      onChange={(v) => set('hsv_h', v)}
                    />
                    <RangeInput
                      label="Saturation ±"
                      value={form.hsv_s}
                      min={0}
                      max={1}
                      step={0.05}
                      onChange={(v) => set('hsv_s', v)}
                    />
                    <RangeInput
                      label="Value (brightness) ±"
                      value={form.hsv_v}
                      min={0}
                      max={1}
                      step={0.05}
                      onChange={(v) => set('hsv_v', v)}
                    />
                  </div>
                </div>
              </>
            )}
          </Section>

          <button
            className="btn-primary w-full flex items-center justify-center gap-2"
            onClick={handleStart}
            disabled={isRunning || startMutation.isPending}
          >
            <Play className="w-4 h-4" />
            {isRunning ? 'Training in progress…' : 'Start Training'}
          </button>

          {startMutation.isError && (
            <div className="text-xs text-status-error bg-status-error/10 rounded p-2">
              {String(startMutation.error)}
            </div>
          )}
        </div>

        {/* Charts + GPU + Log */}
        <div className="lg:col-span-2 space-y-4">
          <MetricsChart runs={runs ?? []} />
          <TrainingLog wsConnected={wsConnected} />
          <GpuMonitor />
        </div>
      </div>

      {/* Run history */}
      <div className="card">
        <div className="card-header">Run History</div>
        {!runs || runs.length === 0 ? (
          <div className="text-sm text-text-muted py-4 text-center">No runs yet.</div>
        ) : (
          <div className="overflow-x-auto">
            <table className="w-full text-sm">
              <thead>
                <tr className="border-b border-border">
                  <th className="table-header text-left py-2 pr-4">Run ID</th>
                  <th className="table-header text-left py-2 pr-4">Model</th>
                  <th className="table-header text-left py-2 pr-4">Status</th>
                  <th className="table-header text-right py-2 pr-4">mAP50</th>
                  <th className="table-header text-right py-2 pr-4">Precision</th>
                  <th className="table-header text-right py-2 pr-4">Recall</th>
                  <th className="table-header text-right py-2 pr-4">Epochs</th>
                  <th className="table-header text-right py-2">Duration</th>
                </tr>
              </thead>
              <tbody>
                {runs.map((run) => {
                  const isActive = run.status === 'running';
                  const progressPct = run.total_epochs > 0
                    ? Math.round((run.best_epoch / run.total_epochs) * 100)
                    : 0;
                  return (
                    <tr key={run.run_uuid} className={cn('table-row', isActive && 'bg-status-info/5')}>
                      <td className="py-2 pr-4">
                        <div className="flex items-center gap-1.5">
                          {isActive && (
                            <span className="w-1.5 h-1.5 rounded-full bg-status-info animate-pulse-slow flex-shrink-0" />
                          )}
                          <span className="font-mono text-xs text-text-muted">
                            {run.run_uuid.slice(0, 8)}…
                          </span>
                        </div>
                      </td>
                      <td className="py-2 pr-4 font-mono text-xs">{run.model_variant}</td>
                      <td className="py-2 pr-4">
                        <span className={cn('badge', getStatusBadgeClass(run.status))}>
                          {run.status}
                        </span>
                      </td>
                      <td className="py-2 pr-4 text-right font-mono">
                        {run.best_map50 > 0 ? (
                          <span className="text-status-success">
                            {(run.best_map50 * 100).toFixed(1)}%
                          </span>
                        ) : (
                          '—'
                        )}
                      </td>
                      <td className="py-2 pr-4 text-right font-mono text-xs">
                        {run.best_precision > 0 ? `${(run.best_precision * 100).toFixed(1)}%` : '—'}
                      </td>
                      <td className="py-2 pr-4 text-right font-mono text-xs">
                        {run.best_recall > 0 ? `${(run.best_recall * 100).toFixed(1)}%` : '—'}
                      </td>
                      <td className="py-2 pr-4 text-right font-mono text-xs">
                        {isActive ? (
                          <div className="flex flex-col items-end gap-0.5">
                            <span>{run.best_epoch}/{run.total_epochs}</span>
                            <div className="w-16 h-1 bg-border rounded-full overflow-hidden">
                              <div
                                className="h-full bg-status-info rounded-full transition-all"
                                style={{ width: `${progressPct}%` }}
                              />
                            </div>
                          </div>
                        ) : (
                          <span>{run.best_epoch}/{run.total_epochs}</span>
                        )}
                      </td>
                      <td className="py-2 text-right text-xs text-text-muted">
                        {formatDuration(run.duration_s)}
                      </td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
        )}
      </div>
    </div>
  );
}

// ── Experiments tab types ──────────────────────────────────────────────────

interface Experiment {
  id: number;
  name: string;
  description?: string;
  tags?: string[];
  is_archived?: boolean;
  archived?: boolean;
  run_count?: number;
  best_map50?: number | null;
  best_run_id?: number | null;
  created_at?: number | string;
  updated_at?: number | string;
  status?: string;
}

interface CreateExperimentRequest {
  name: string;
  description?: string;
  tags?: string[];
}

// ── Create experiment modal ────────────────────────────────────────────────

function CreateModal({ onClose }: { onClose: () => void }) {
  const queryClient = useQueryClient();
  const [form, setForm] = useState<CreateExperimentRequest>({
    name: '',
    description: '',
    tags: [],
  });
  const [tagInput, setTagInput] = useState('');

  const createMutation = useMutation({
    mutationFn: () => api.post('/experiments', form).then((r) => r.data),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['experiments'] });
      onClose();
    },
  });

  const addTag = () => {
    const t = tagInput.trim();
    if (t && !form.tags?.includes(t)) {
      setForm({ ...form, tags: [...(form.tags ?? []), t] });
      setTagInput('');
    }
  };

  const removeTag = (tag: string) => {
    setForm({ ...form, tags: form.tags?.filter((t) => t !== tag) });
  };

  return (
    <div
      className="fixed inset-0 flex items-center justify-center z-50 p-4"
      style={{ background: 'rgba(0,0,0,0.7)' }}
    >
      <div
        className="w-full max-w-md rounded-2xl p-6 space-y-5"
        style={{ background: '#0d1117', border: '1px solid #21262d' }}
      >
        <div className="flex items-center justify-between">
          <h2 className="text-lg font-bold text-white">New Experiment</h2>
          <button onClick={onClose} className="p-1 rounded transition-colors" style={{ color: '#484f58' }}>
            <X className="w-5 h-5" />
          </button>
        </div>

        <div className="space-y-4">
          <div>
            <label className="text-xs mb-1.5 block" style={{ color: '#8b949e' }}>Name *</label>
            <input
              className="w-full px-3 py-2 text-sm rounded-lg focus:outline-none"
              style={{ background: '#161b22', border: '1px solid #21262d', color: '#e6edf3' }}
              placeholder="e.g. yolo11s-baseline-v1"
              value={form.name}
              onChange={(e) => setForm({ ...form, name: e.target.value })}
            />
          </div>

          <div>
            <label className="text-xs mb-1.5 block" style={{ color: '#8b949e' }}>Description</label>
            <textarea
              className="w-full px-3 py-2 text-sm rounded-lg focus:outline-none resize-none"
              style={{ background: '#161b22', border: '1px solid #21262d', color: '#e6edf3' }}
              rows={3}
              placeholder="Experiment description…"
              value={form.description}
              onChange={(e) => setForm({ ...form, description: e.target.value })}
            />
          </div>

          <div>
            <label className="text-xs mb-1.5 block" style={{ color: '#8b949e' }}>Tags</label>
            <div className="flex gap-2">
              <input
                className="flex-1 px-3 py-2 text-sm rounded-lg focus:outline-none"
                style={{ background: '#161b22', border: '1px solid #21262d', color: '#e6edf3' }}
                placeholder="Add tag…"
                value={tagInput}
                onChange={(e) => setTagInput(e.target.value)}
                onKeyDown={(e) => e.key === 'Enter' && (e.preventDefault(), addTag())}
              />
              <button
                onClick={addTag}
                className="px-3 py-2 rounded-lg text-sm transition-colors"
                style={{ background: '#161b22', border: '1px solid #21262d', color: '#8b949e' }}
              >
                Add
              </button>
            </div>
            {(form.tags?.length ?? 0) > 0 && (
              <div className="flex flex-wrap gap-1.5 mt-2">
                {form.tags!.map((tag) => (
                  <span
                    key={tag}
                    className="flex items-center gap-1 text-xs px-2 py-0.5 rounded-full"
                    style={{ background: 'rgba(168,85,247,0.2)', color: '#c084fc' }}
                  >
                    {tag}
                    <button onClick={() => removeTag(tag)} className="opacity-70 hover:opacity-100">
                      <X className="w-3 h-3" />
                    </button>
                  </span>
                ))}
              </div>
            )}
          </div>
        </div>

        <div className="flex gap-3">
          <button
            onClick={onClose}
            className="flex-1 py-2 text-sm rounded-lg transition-colors"
            style={{ background: '#161b22', border: '1px solid #21262d', color: '#8b949e' }}
          >
            Cancel
          </button>
          <button
            onClick={() => createMutation.mutate()}
            disabled={createMutation.isPending || !form.name.trim()}
            className="flex-1 py-2 bg-blue-600 hover:bg-blue-500 disabled:opacity-50 text-white text-sm font-medium rounded-lg transition-colors"
          >
            {createMutation.isPending ? (
              <span className="flex items-center justify-center gap-2">
                <Loader2 className="w-4 h-4 animate-spin" />
                Creating…
              </span>
            ) : 'Create Experiment'}
          </button>
        </div>

        {createMutation.isError && (
          <p className="text-xs text-red-400">
            {(createMutation.error as Error)?.message ?? 'Failed to create experiment'}
          </p>
        )}
      </div>
    </div>
  );
}

// ── Experiment card ────────────────────────────────────────────────────────

function ExperimentCard({ experiment }: { experiment: Experiment }) {
  const queryClient = useQueryClient();
  const isArchived = experiment.is_archived ?? experiment.archived ?? false;

  const archiveMutation = useMutation({
    mutationFn: () =>
      api.put(`/experiments/${experiment.id}`, { is_archived: !isArchived }).then((r) => r.data),
    onSuccess: () => queryClient.invalidateQueries({ queryKey: ['experiments'] }),
  });

  const deleteMutation = useMutation({
    mutationFn: () => api.delete(`/experiments/${experiment.id}`).then((r) => r.data),
    onSuccess: () => queryClient.invalidateQueries({ queryKey: ['experiments'] }),
  });

  const createdAt =
    experiment.created_at !== undefined
      ? typeof experiment.created_at === 'number'
        ? timeAgo(experiment.created_at)
        : timeAgo(new Date(experiment.created_at as string).getTime() / 1000)
      : null;

  const runCount = experiment.run_count ?? 0;
  const bestMap50 = experiment.best_map50;

  return (
    <div
      className={cn('rounded-xl p-4 space-y-3 transition-all')}
      style={{
        background: isArchived ? 'rgba(13,17,23,0.6)' : '#0d1117',
        border: isArchived ? '1px solid rgba(33,38,45,0.5)' : '1px solid #21262d',
        opacity: isArchived ? 0.7 : 1,
      }}
    >
      <div className="flex items-start justify-between gap-3">
        <div className="flex-1 min-w-0">
          <div className="flex items-center gap-2 flex-wrap">
            <h3 className="text-sm font-semibold text-white truncate">{experiment.name}</h3>
            {isArchived && (
              <span
                className="text-[10px] px-1.5 py-0.5 rounded font-medium"
                style={{ background: 'rgba(107,114,128,0.2)', color: '#9ca3af' }}
              >
                archived
              </span>
            )}
            {experiment.status && experiment.status !== 'active' && !isArchived && (
              <span
                className="text-[10px] px-1.5 py-0.5 rounded font-medium capitalize"
                style={{
                  background:
                    experiment.status === 'running'
                      ? 'rgba(59,130,246,0.2)'
                      : 'rgba(34,197,94,0.2)',
                  color:
                    experiment.status === 'running' ? '#60a5fa' : '#22c55e',
                }}
              >
                {experiment.status}
              </span>
            )}
          </div>
          {experiment.description && (
            <p className="text-xs mt-0.5 line-clamp-2" style={{ color: '#484f58' }}>
              {experiment.description}
            </p>
          )}
          {(experiment.tags?.length ?? 0) > 0 && (
            <div className="flex flex-wrap gap-1 mt-1.5">
              {experiment.tags!.map((tag) => (
                <span
                  key={tag}
                  className="text-[10px] px-1.5 py-0.5 rounded-full"
                  style={{ background: 'rgba(168,85,247,0.15)', color: '#c084fc' }}
                >
                  {tag}
                </span>
              ))}
            </div>
          )}
        </div>

        <div className="flex items-center gap-1 flex-shrink-0">
          <button
            onClick={() => archiveMutation.mutate()}
            disabled={archiveMutation.isPending}
            className="p-1.5 rounded transition-colors disabled:opacity-50"
            style={{ color: '#484f58' }}
            title={isArchived ? 'Restore' : 'Archive'}
          >
            {archiveMutation.isPending ? (
              <Loader2 className="w-4 h-4 animate-spin" />
            ) : isArchived ? (
              <ArchiveRestore className="w-4 h-4 hover:text-blue-400" />
            ) : (
              <Archive className="w-4 h-4" />
            )}
          </button>
          <button
            onClick={() => {
              if (confirm(`Delete experiment "${experiment.name}"? This cannot be undone.`)) {
                deleteMutation.mutate();
              }
            }}
            disabled={deleteMutation.isPending}
            className="p-1.5 rounded transition-colors disabled:opacity-50 hover:text-red-400"
            style={{ color: '#484f58' }}
            title="Delete"
          >
            {deleteMutation.isPending ? (
              <Loader2 className="w-4 h-4 animate-spin" />
            ) : (
              <Trash2 className="w-4 h-4" />
            )}
          </button>
        </div>
      </div>

      {/* Stats row */}
      <div className="flex items-center gap-4 text-xs">
        <div className="flex items-center gap-1.5">
          <span style={{ color: '#484f58' }}>Runs:</span>
          <span className="font-mono font-bold text-white">{runCount}</span>
        </div>
        {bestMap50 !== null && bestMap50 !== undefined && bestMap50 > 0 && (
          <div className="flex items-center gap-1.5">
            <span style={{ color: '#484f58' }}>Best mAP50:</span>
            <span className="font-mono font-bold text-green-400">
              {(bestMap50 * 100).toFixed(1)}%
            </span>
          </div>
        )}
        {createdAt && (
          <span className="ml-auto" style={{ color: '#484f58' }}>{createdAt}</span>
        )}
      </div>

      {/* mAP50 bar */}
      {bestMap50 !== null && bestMap50 !== undefined && bestMap50 > 0 && (
        <div
          className="h-1 rounded-full overflow-hidden"
          style={{ background: '#21262d' }}
        >
          <div
            className="h-full rounded-full transition-all"
            style={{
              width: `${bestMap50 * 100}%`,
              background: bestMap50 >= 0.8 ? '#22c55e' : bestMap50 >= 0.6 ? '#eab308' : '#3b82f6',
            }}
          />
        </div>
      )}
    </div>
  );
}

// ── Experiments tab ────────────────────────────────────────────────────────

function ExperimentsTab() {
  const [showCreate, setShowCreate] = useState(false);
  const [showArchived, setShowArchived] = useState(false);

  const {
    data,
    isLoading,
    isError,
    error,
    refetch,
  } = useQuery({
    queryKey: ['experiments'],
    queryFn: () => api.get('/experiments').then((r) => r.data),
    staleTime: 30_000,
    refetchInterval: 15_000,
  });

  const experiments: Experiment[] = Array.isArray(data)
    ? data
    : data?.experiments ?? data?.items ?? [];

  const active = experiments.filter((e) => !(e.is_archived ?? e.archived));
  const archived = experiments.filter((e) => e.is_archived ?? e.archived);
  const displayed = showArchived ? experiments : active;

  const totalRuns = experiments.reduce((sum, e) => sum + (e.run_count ?? 0), 0);
  const bestMap = experiments.reduce((best, e) => {
    const m = e.best_map50 ?? 0;
    return m > best ? m : best;
  }, 0);

  return (
    <div className="space-y-4">
      {/* Toolbar */}
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-4 text-xs">
          {experiments.length > 0 && (
            <>
              {[
                { label: 'Experiments', value: active.length, color: '#e6edf3' },
                { label: 'Total Runs', value: totalRuns, color: '#8b949e' },
                {
                  label: 'Best mAP50',
                  value: bestMap > 0 ? `${(bestMap * 100).toFixed(1)}%` : '—',
                  color: '#22c55e',
                },
                { label: 'Archived', value: archived.length, color: '#484f58' },
              ].map(({ label, value, color }) => (
                <div key={label} className="flex items-center gap-1.5">
                  <span style={{ color: '#484f58' }}>{label}:</span>
                  <span className="font-bold font-mono" style={{ color }}>{value}</span>
                </div>
              ))}
            </>
          )}
        </div>
        <div className="flex items-center gap-2">
          <button
            onClick={() => setShowArchived((v) => !v)}
            className="px-2.5 py-1 text-xs rounded-lg transition-colors"
            style={{
              background: showArchived ? 'rgba(107,114,128,0.2)' : 'transparent',
              border: '1px solid #21262d',
              color: showArchived ? '#9ca3af' : '#484f58',
            }}
          >
            {showArchived ? `Hide archived (${archived.length})` : `Show archived (${archived.length})`}
          </button>
          <button
            onClick={() => refetch()}
            className="p-1.5 rounded transition-colors"
            style={{ color: '#484f58' }}
            title="Refresh"
          >
            <RefreshCw className="w-4 h-4" />
          </button>
          <button
            onClick={() => setShowCreate(true)}
            className="flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-sm font-medium bg-blue-600 hover:bg-blue-500 text-white transition-colors"
          >
            <Plus className="w-4 h-4" />
            New Experiment
          </button>
        </div>
      </div>

      {/* Loading */}
      {isLoading && (
        <div className="flex items-center justify-center py-16">
          <Loader2 className="w-6 h-6 text-blue-400 animate-spin" />
        </div>
      )}

      {/* Error */}
      {isError && (
        <div
          className="flex items-start gap-3 px-4 py-3 rounded-lg"
          style={{ background: 'rgba(239,68,68,0.1)', border: '1px solid rgba(239,68,68,0.2)' }}
        >
          <AlertTriangle className="w-4 h-4 text-red-400 mt-0.5" />
          <div>
            <p className="text-sm font-medium text-red-400">Failed to load experiments</p>
            <p className="text-xs mt-0.5" style={{ color: 'rgba(252,165,165,0.7)' }}>
              {(error as Error)?.message ?? 'Unknown error'}
            </p>
          </div>
        </div>
      )}

      {/* Empty state */}
      {!isLoading && !isError && displayed.length === 0 && (
        <div className="text-center py-20" style={{ color: '#484f58' }}>
          <FlaskConical className="w-12 h-12 mx-auto mb-4 opacity-30" />
          <p className="text-base font-medium">
            {showArchived ? 'No archived experiments' : 'No experiments yet'}
          </p>
          <p className="text-sm mt-1">
            {!showArchived && 'Click "New Experiment" to create your first experiment'}
          </p>
        </div>
      )}

      {/* Grid */}
      {!isLoading && !isError && displayed.length > 0 && (
        <div className="grid grid-cols-1 md:grid-cols-2 xl:grid-cols-3 gap-4">
          {displayed.map((exp) => (
            <ExperimentCard key={exp.id} experiment={exp} />
          ))}
        </div>
      )}

      {showCreate && <CreateModal onClose={() => setShowCreate(false)} />}
    </div>
  );
}

// ── Distributed training types ─────────────────────────────────────────────

interface DistributedStatus {
  gpu_count: number;
  nccl_available: boolean;
  gloo_available: boolean;
  optimal_world_size: number;
  current_job: string | null;
}

interface DistributedStartRequest {
  data_yaml: string;
  epochs: number;
  world_size: number;
  backend: 'nccl' | 'gloo';
  gradient_accumulation_steps: number;
  mixed_precision: 'fp16' | 'bf16' | 'no';
  sync_batchnorm: boolean;
  gradient_checkpointing: boolean;
}

interface DistributedStartResponse {
  task_id: string;
  world_size: number;
  backend: string;
}

interface DistributedJobStatus {
  task_id: string;
  status: string;
  frames_processed?: number;
  error?: string;
}

// ── Distributed tab ────────────────────────────────────────────────────────

function DistributedTab() {
  const qc = useQueryClient();

  const [form, setForm] = useState<DistributedStartRequest>({
    data_yaml: '/data/datasets/trichome/data.yaml',
    epochs: 150,
    world_size: -1,
    backend: 'nccl',
    gradient_accumulation_steps: 4,
    mixed_precision: 'fp16',
    sync_batchnorm: true,
    gradient_checkpointing: false,
  });

  const setField = <K extends keyof DistributedStartRequest>(
    key: K,
    val: DistributedStartRequest[K],
  ) => setForm((f) => ({ ...f, [key]: val }));

  const [activeTaskId, setActiveTaskId] = useState<string | null>(null);

  // GPU status — auto-refresh every 10s
  const { data: gpuStatus } = useQuery<DistributedStatus>({
    queryKey: ['distributed-status'],
    queryFn: () => api.get('/training/distributed/status').then((r) => r.data),
    refetchInterval: 10_000,
  });

  // Sync side-effects from GPU status (RQ v5: onSuccess removed, use useEffect)
  useEffect(() => {
    if (!gpuStatus) return;
    // Auto-select backend based on availability
    setForm((f) => ({
      ...f,
      backend: gpuStatus.nccl_available ? 'nccl' : 'gloo',
    }));
    // Track existing job from server if none tracked locally
    if (gpuStatus.current_job && !activeTaskId) {
      setActiveTaskId(gpuStatus.current_job);
    }
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [gpuStatus]);

  // Active job polling — every 2s when a job is running
  const { data: jobStatus } = useQuery<DistributedJobStatus>({
    queryKey: ['distributed-job', activeTaskId],
    queryFn: () =>
      api.get(`/training/distributed/jobs/${activeTaskId}`).then((r) => r.data),
    enabled: !!activeTaskId,
    refetchInterval: (query) => {
      const d = query.state.data;
      if (!d) return 2_000;
      return d.status === 'running' ? 2_000 : false;
    },
  });

  const startMutation = useMutation<DistributedStartResponse, Error, DistributedStartRequest>({
    mutationFn: (req) =>
      api.post('/training/distributed/start', req).then((r) => r.data),
    onSuccess: (data) => {
      setActiveTaskId(data.task_id);
      qc.invalidateQueries({ queryKey: ['distributed-status'] });
    },
  });

  const stopMutation = useMutation<{ stopped: boolean }, Error, string>({
    mutationFn: (taskId) =>
      api.post(`/training/distributed/stop/${taskId}`).then((r) => r.data),
    onSuccess: () => {
      setActiveTaskId(null);
      qc.invalidateQueries({ queryKey: ['distributed-status'] });
      qc.removeQueries({ queryKey: ['distributed-job'] });
    },
  });

  const isJobRunning =
    !!activeTaskId && jobStatus?.status === 'running';

  const handleStart = () => {
    if (!form.data_yaml) {
      alert('Please specify the dataset YAML path.');
      return;
    }
    startMutation.mutate(form);
  };

  const gpuCount = gpuStatus?.gpu_count ?? 0;
  const optimalWorldSize = gpuStatus?.optimal_world_size ?? 1;

  // Status badge helper
  const jobStatusBadge = () => {
    if (!jobStatus) return null;
    const { status } = jobStatus;
    if (status === 'running') {
      return (
        <span className="inline-flex items-center gap-1.5 text-xs font-medium text-blue-400">
          <span className="w-2 h-2 rounded-full bg-blue-400 animate-pulse" />
          Running
        </span>
      );
    }
    if (status === 'complete' || status === 'completed') {
      return (
        <span className="inline-flex items-center gap-1.5 text-xs font-medium text-green-400">
          <span className="w-2 h-2 rounded-full bg-green-400" />
          Complete
        </span>
      );
    }
    if (status === 'error' || status === 'failed') {
      return (
        <span className="inline-flex items-center gap-1.5 text-xs font-medium text-red-400">
          <span className="w-2 h-2 rounded-full bg-red-400" />
          Error
        </span>
      );
    }
    return (
      <span className="inline-flex items-center gap-1.5 text-xs font-medium text-text-secondary capitalize">
        {status}
      </span>
    );
  };

  return (
    <div className="space-y-6">
      {/* GPU status card */}
      <div className="card">
        <div className="card-header flex items-center gap-2">
          <Cpu className="w-4 h-4 text-text-secondary" />
          GPU Environment
        </div>
        <div className="grid grid-cols-1 sm:grid-cols-3 gap-4 mt-2">
          {/* GPU count */}
          <div className="flex items-center gap-3">
            <Cpu className="w-5 h-5 text-text-muted flex-shrink-0" />
            <div>
              <div className="text-sm font-medium text-text-primary">
                {gpuCount > 0 ? `${gpuCount} GPU${gpuCount > 1 ? 's' : ''} detected` : 'No CUDA GPUs'}
              </div>
              <div className="text-xs text-text-muted">CUDA devices</div>
            </div>
          </div>

          {/* Backend availability */}
          <div className="space-y-1">
            <div className="text-xs text-text-muted mb-1">Backend availability</div>
            <div className="flex items-center gap-2 text-xs">
              {gpuStatus?.nccl_available ? (
                <span className="text-green-400 font-medium">✓ NCCL</span>
              ) : (
                <span className="text-red-400 font-medium">✗ NCCL</span>
              )}
              <span className="text-green-400 font-medium">✓ Gloo</span>
            </div>
          </div>

          {/* Recommended world size */}
          <div>
            <div className="text-xs text-text-muted mb-1">Recommendation</div>
            <div className="text-sm text-text-primary">
              {gpuCount > 0
                ? `Recommended: ${optimalWorldSize} GPU${optimalWorldSize > 1 ? 's' : ''} for this VRAM budget`
                : 'No GPUs available for DDP'}
            </div>
          </div>
        </div>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Config panel */}
        <div className="lg:col-span-1 space-y-3">
          <Section title="Dataset" icon={Settings} defaultOpen>
            <div>
              <label className="block text-xs text-text-secondary mb-1.5">
                Dataset YAML Path
              </label>
              <input
                className="input font-mono text-xs"
                value={form.data_yaml}
                onChange={(e) => setField('data_yaml', e.target.value)}
                placeholder="/data/datasets/trichome/data.yaml"
              />
            </div>
          </Section>

          <Section title="Parallelism" icon={Activity} defaultOpen>
            <NumberInput
              label="World Size"
              hint={`-1 = auto (detected: ${gpuCount} GPU${gpuCount !== 1 ? 's' : ''})`}
              value={form.world_size}
              min={-1}
              max={8}
              step={1}
              onChange={(v) => setField('world_size', v)}
              mono
            />
            <div>
              <label className="block text-xs text-text-secondary mb-1.5">Backend</label>
              <select
                className="input"
                value={form.backend}
                onChange={(e) =>
                  setField('backend', e.target.value as 'nccl' | 'gloo')
                }
              >
                <option value="nccl">
                  NCCL{gpuStatus?.nccl_available ? ' (available)' : ' (unavailable)'}
                </option>
                <option value="gloo">Gloo (always available)</option>
              </select>
            </div>
            <NumberInput
              label="Gradient Accumulation Steps"
              value={form.gradient_accumulation_steps}
              min={1}
              max={16}
              step={1}
              onChange={(v) => setField('gradient_accumulation_steps', v)}
            />
          </Section>

          <Section title="Precision & Memory" icon={Sliders} defaultOpen>
            <div>
              <label className="block text-xs text-text-secondary mb-1.5">
                Mixed Precision
              </label>
              <select
                className="input"
                value={form.mixed_precision}
                onChange={(e) =>
                  setField('mixed_precision', e.target.value as 'fp16' | 'bf16' | 'no')
                }
              >
                <option value="fp16">fp16 — recommended for RTX 4060</option>
                <option value="bf16">bf16 — Ampere+ only</option>
                <option value="no">no — full fp32</option>
              </select>
            </div>
            <Toggle
              value={form.sync_batchnorm}
              onChange={(v) => setField('sync_batchnorm', v)}
              label="Sync BatchNorm"
              hint="Convert BN → SyncBN (required for DDP)"
            />
            <Toggle
              value={form.gradient_checkpointing}
              onChange={(v) => setField('gradient_checkpointing', v)}
              label="Gradient Checkpointing"
              hint="Trade compute for VRAM"
            />
          </Section>

          <Section title="Training" icon={Activity} defaultOpen>
            <NumberInput
              label="Epochs"
              value={form.epochs}
              min={1}
              max={1000}
              step={1}
              onChange={(v) => setField('epochs', v)}
            />
          </Section>

          <button
            className="btn-primary w-full flex items-center justify-center gap-2"
            onClick={handleStart}
            disabled={isJobRunning || startMutation.isPending}
          >
            <Play className="w-4 h-4" />
            {isJobRunning ? 'Job running…' : 'Launch Distributed Training'}
          </button>

          {startMutation.isError && (
            <div className="text-xs text-status-error bg-status-error/10 rounded p-2">
              {(startMutation.error as Error)?.message ?? String(startMutation.error)}
            </div>
          )}
        </div>

        {/* Right panel */}
        <div className="lg:col-span-2 space-y-4">
          {/* Active job panel */}
          {activeTaskId && (
            <div
              className={cn(
                'card',
                jobStatus?.status === 'running' && 'border-blue-500/30 bg-blue-500/5',
                (jobStatus?.status === 'complete' || jobStatus?.status === 'completed') &&
                  'border-green-500/30 bg-green-500/5',
                (jobStatus?.status === 'error' || jobStatus?.status === 'failed') &&
                  'border-red-500/30 bg-red-500/5',
              )}
            >
              <div className="card-header flex items-center justify-between">
                <span>Active Distributed Job</span>
                {jobStatusBadge()}
              </div>

              <div className="space-y-3 mt-2">
                <div className="flex items-center gap-2">
                  <span className="text-xs text-text-muted">Task ID:</span>
                  <span className="font-mono text-xs text-text-primary">
                    {activeTaskId}
                  </span>
                </div>

                {jobStatus?.frames_processed !== undefined && (
                  <div className="flex items-center gap-2">
                    <span className="text-xs text-text-muted">Frames processed:</span>
                    <span className="font-mono text-xs text-text-primary">
                      {jobStatus.frames_processed.toLocaleString()}
                    </span>
                  </div>
                )}

                {(jobStatus?.status === 'error' || jobStatus?.status === 'failed') &&
                  jobStatus.error && (
                    <div className="text-xs text-red-400 bg-red-500/10 rounded p-2">
                      {jobStatus.error}
                    </div>
                  )}

                {isJobRunning && (
                  <button
                    className="btn-danger flex items-center gap-2"
                    onClick={() => stopMutation.mutate(activeTaskId)}
                    disabled={stopMutation.isPending}
                  >
                    {stopMutation.isPending ? (
                      <Loader2 className="w-4 h-4 animate-spin" />
                    ) : (
                      <Square className="w-4 h-4" />
                    )}
                    Stop Job
                  </button>
                )}

                {!isJobRunning &&
                  jobStatus &&
                  jobStatus.status !== 'running' && (
                    <button
                      className="text-xs text-text-muted hover:text-text-secondary underline"
                      onClick={() => {
                        setActiveTaskId(null);
                        qc.removeQueries({ queryKey: ['distributed-job'] });
                      }}
                    >
                      Dismiss
                    </button>
                  )}
              </div>
            </div>
          )}

          {/* Idle state */}
          {!activeTaskId && (
            <div className="card flex flex-col items-center justify-center py-16 text-center">
              <Cpu className="w-12 h-12 text-text-muted opacity-30 mb-4" />
              <p className="text-sm font-medium text-text-secondary">
                No distributed job running
              </p>
              <p className="text-xs text-text-muted mt-1">
                Configure the options on the left and click &ldquo;Launch Distributed Training&rdquo;
              </p>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}

// ── Tab-aware inner component (reads searchParams) ─────────────────────────

type TabId = 'runs' | 'experiments' | 'distributed';

const TABS: { id: TabId; label: string; icon: React.ElementType }[] = [
  { id: 'runs', label: 'Runs', icon: Activity },
  { id: 'experiments', label: 'Experiments', icon: FlaskConical },
  { id: 'distributed', label: 'Distributed', icon: Cpu },
];

function TrainingPageInner() {
  const router = useRouter();
  const searchParams = useSearchParams();

  const rawTab = searchParams.get('tab');
  const activeTab: TabId =
    rawTab === 'experiments'
      ? 'experiments'
      : rawTab === 'distributed'
      ? 'distributed'
      : 'runs';

  const setTab = (tab: TabId) => {
    router.replace(`/training?tab=${tab}`);
  };

  return (
    <div className="space-y-6 max-w-[1400px] mx-auto">
      {/* Page header */}
      <div className="flex items-center gap-3">
        <Cpu className="w-5 h-5 text-text-secondary" />
        <div>
          <h1 className="text-xl font-semibold text-[#e6edf3]">Training</h1>
          <p className="text-sm text-text-secondary mt-0.5">
            Manage training runs and experiments
          </p>
        </div>
      </div>

      {/* Tab bar */}
      <div className="flex border-b border-[#21262d]">
        {TABS.map(({ id, label, icon: Icon }) => (
          <button
            key={id}
            onClick={() => setTab(id)}
            className={cn(
              'flex items-center gap-2 px-4 py-2.5 text-sm font-medium transition-colors -mb-px',
              activeTab === id
                ? 'border-b-2 border-blue-400 text-[#e6edf3]'
                : 'border-b-2 border-transparent text-text-secondary hover:text-[#e6edf3]',
            )}
          >
            <Icon className="w-4 h-4" />
            {label}
          </button>
        ))}
      </div>

      {/* Tab content */}
      {activeTab === 'runs' && <RunsTab />}
      {activeTab === 'experiments' && <ExperimentsTab />}
      {activeTab === 'distributed' && <DistributedTab />}
    </div>
  );
}

// ── Default export (Suspense boundary for useSearchParams) ─────────────────

export default function TrainingPage() {
  return (
    <Suspense fallback={
      <div className="flex items-center justify-center py-16">
        <Loader2 className="w-6 h-6 text-blue-400 animate-spin" />
      </div>
    }>
      <TrainingPageInner />
    </Suspense>
  );
}
