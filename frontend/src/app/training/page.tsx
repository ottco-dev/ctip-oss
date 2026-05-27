'use client';

import { useState } from 'react';
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query';
import {
  Play, Square, ChevronDown, ChevronUp, RotateCcw,
  Settings, Activity, Sliders,
} from 'lucide-react';
import { api } from '@/lib/api';
import { GpuMonitor } from '@/components/training/GpuMonitor';
import { MetricsChart } from '@/components/charts/MetricsChart';
import { useTrainingStore } from '@/store/trainingStore';
import { cn, getStatusBadgeClass, timeAgo, formatDuration } from '@/lib/utils';
import type { TrainingStartRequest, RunSummary } from '@/lib/types';

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

// ── Helper components ──────────────────────────────────────────────────────

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

// ── Collapsible section ────────────────────────────────────────────────────

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

// ── Main page ──────────────────────────────────────────────────────────────

export default function TrainingPage() {
  const qc = useQueryClient();
  const { setActiveRun, clearActiveRun } = useTrainingStore();

  const [form, setForm] = useState<TrainingStartRequest>(DEFAULT_FORM);
  const set = <K extends keyof TrainingStartRequest>(key: K, val: TrainingStartRequest[K]) =>
    setForm((f) => ({ ...f, [key]: val }));

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
    <div className="space-y-6 max-w-[1400px] mx-auto">
      {/* Header */}
      <div>
        <h1 className="text-xl font-semibold text-text-primary">Training Control</h1>
        <p className="text-sm text-text-secondary mt-0.5">
          YOLO model training with live metrics, GPU monitoring, and full hyperparameter control
        </p>
      </div>

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
        {/* ── Config panel ─────────────────────────────────────────────── */}
        <div className="lg:col-span-1 space-y-3">

          {/* Core settings (always visible) */}
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
              {/* Experiment name */}
              <div>
                <label className="block text-xs text-text-secondary mb-1.5">Experiment Name</label>
                <input
                  className="input"
                  value={form.experiment_name}
                  onChange={(e) => set('experiment_name', e.target.value)}
                  placeholder="trichome-detection"
                />
              </div>

              {/* Model variant */}
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

              {/* Dataset YAML */}
              <div>
                <label className="block text-xs text-text-secondary mb-1.5">Dataset YAML Path</label>
                <input
                  className="input font-mono text-xs"
                  value={form.data_yaml}
                  onChange={(e) => set('data_yaml', e.target.value)}
                  placeholder="/data/datasets/trichome/data.yaml"
                />
              </div>

              {/* Epochs + Batch */}
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

              {/* Image size */}
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

              {/* AMP toggle */}
              <Toggle
                value={form.amp}
                onChange={(v) => set('amp', v)}
                label="Mixed Precision (FP16)"
                hint="Recommended for RTX 4060"
              />

              {/* Effective batch info */}
              <div className="code-block">
                Effective batch: {form.batch_size} × 4 accum ={' '}
                <span className="text-accent font-bold">{form.batch_size * 4}</span>
              </div>
            </div>
          </div>

          {/* ── LR Schedule ───────────────────────────────────────────── */}
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

          {/* ── Regularisation ────────────────────────────────────────── */}
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

          {/* ── Augmentation ──────────────────────────────────────────── */}
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

          {/* Start button */}
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

        {/* ── Charts + GPU ─────────────────────────────────────────────── */}
        <div className="lg:col-span-2 space-y-4">
          <MetricsChart runs={runs ?? []} />
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
                {runs.map((run) => (
                  <tr key={run.run_uuid} className="table-row">
                    <td className="py-2 pr-4">
                      <span className="font-mono text-xs text-text-muted">
                        {run.run_uuid.slice(0, 8)}…
                      </span>
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
                      {run.best_epoch}/{run.total_epochs}
                    </td>
                    <td className="py-2 text-right text-xs text-text-muted">
                      {formatDuration(run.duration_s)}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </div>
    </div>
  );
}
