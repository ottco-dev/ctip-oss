'use client';

import React, { useCallback, useEffect, useRef, useState } from 'react';
import { useMutation, useQuery } from '@tanstack/react-query';
import { useDropzone } from 'react-dropzone';
import {
  Upload,
  Loader2,
  FlaskConical,
  Microscope,
  BarChart3,
  Info,
  BrainCircuit,
  Play,
  CheckCircle2,
  XCircle,
  Download,
  ClipboardCheck,
} from 'lucide-react';
import { api, uploadFile } from '@/lib/api';
import { cn, formatConfidence, getConfidenceColor } from '@/lib/utils';

// ─────────────────────────────────────────────────────────────────────────────
// Types — Classify tab
// ─────────────────────────────────────────────────────────────────────────────

interface GeometricDescriptors {
  area_px: number;
  perimeter_px: number;
  circularity: number;
  eccentricity: number;
  solidity: number;
  major_axis_px: number;
  minor_axis_px: number;
  aspect_ratio: number;
  extent: number;
  convex_area_px: number;
}

interface StalkResult {
  has_visible_stalk: boolean;
  stalk_length_px: number | null;
  stalk_width_px: number | null;
  head_diameter_px: number | null;
  head_area_px: number | null;
  head_circularity: number | null;
}

interface MorphologyResponse {
  morphology_type: string;
  confidence: number;
  classification_method: string;
  geometric_features?: GeometricDescriptors;
  stalk?: StalkResult;
  processing_time_ms?: number;
}

interface MaturityResponse {
  stage: string;
  confidence: number;
  color_features?: {
    hue_mean?: number;
    saturation_mean?: number;
    value_mean?: number;
    amber_ratio?: number;
    translucency_score?: number;
  };
  processing_time_ms?: number;
  scientific_note?: string;
}

// ─────────────────────────────────────────────────────────────────────────────
// Types — Training tab
// ─────────────────────────────────────────────────────────────────────────────

interface TrainingConfig {
  dataset_path: string;
  model_arch: string;
  epochs: number;
  batch_size: number;
  learning_rate: number;
  dropout: number;
  val_split: number;
  early_stopping_patience: number;
  use_fp16: boolean;
  augment: boolean;
}

interface TrainingStartResponse {
  task_id: string;
  status: 'started';
}

type TrainingStatus = 'idle' | 'running' | 'complete' | 'error';

interface TrainingStatusResponse {
  task_id: string | null;
  status: TrainingStatus;
  epoch: number;
  total_epochs: number;
  train_loss: number;
  val_loss: number;
  val_accuracy: number;
  best_val_accuracy: number;
  elapsed_s: number;
}

interface EvaluateResponse {
  accuracy: number;
  per_class: Record<string, number>;
  confusion_matrix: number[][];
}

interface ExportResponse {
  onnx_path: string;
  export_time_s: number;
}

interface EpochPoint {
  epoch: number;
  val_accuracy: number;
}

// ─────────────────────────────────────────────────────────────────────────────
// Stage / type colour maps
// ─────────────────────────────────────────────────────────────────────────────

const STAGE_COLORS: Record<string, string> = {
  clear: '#60a5fa',
  cloudy: '#f9fafb',
  amber: '#f59e0b',
  degraded: '#a16207',
  mixed: '#8b5cf6',
  unknown: '#6b7280',
};

const TYPE_COLORS: Record<string, string> = {
  capitate_stalked: '#22d3ee',
  capitate_sessile: '#34d399',
  bulbous: '#a78bfa',
  non_glandular: '#fb923c',
  unknown: '#6b7280',
};

const TYPE_LABELS: Record<string, string> = {
  capitate_stalked: 'Capitate Stalked',
  capitate_sessile: 'Capitate Sessile',
  bulbous: 'Bulbous',
  non_glandular: 'Non-Glandular',
  unknown: 'Unknown',
};

const CLASS_DISPLAY: Record<string, string> = {
  CAPITATE_STALKED: 'Capitate Stalked',
  CAPITATE_SESSILE: 'Capitate Sessile',
  BULBOUS: 'Bulbous',
  NON_GLANDULAR: 'Non-Glandular',
};

const CLASS_COLORS: Record<string, string> = {
  CAPITATE_STALKED: '#22d3ee',
  CAPITATE_SESSILE: '#34d399',
  BULBOUS: '#a78bfa',
  NON_GLANDULAR: '#fb923c',
};

// ─────────────────────────────────────────────────────────────────────────────
// Metric row helper
// ─────────────────────────────────────────────────────────────────────────────

function MetricRow({
  label,
  value,
  unit = '',
}: {
  label: string;
  value: number | null | undefined;
  unit?: string;
}) {
  if (value == null) return null;
  return (
    <div className="flex items-center justify-between py-1.5 border-b border-border last:border-0">
      <span className="text-xs text-text-secondary">{label}</span>
      <span className="text-xs font-mono text-text-primary">
        {typeof value === 'number' ? value.toFixed(3) : value}
        {unit}
      </span>
    </div>
  );
}

// ─────────────────────────────────────────────────────────────────────────────
// formatElapsed helper
// ─────────────────────────────────────────────────────────────────────────────

function formatElapsed(seconds: number): string {
  const m = Math.floor(seconds / 60);
  const s = Math.floor(seconds % 60);
  return `${String(m).padStart(2, '0')}:${String(s).padStart(2, '0')}`;
}

// ─────────────────────────────────────────────────────────────────────────────
// Mini SVG accuracy chart (no recharts)
// ─────────────────────────────────────────────────────────────────────────────

function AccuracyChart({ points }: { points: EpochPoint[] }) {
  const W = 200;
  const H = 80;
  const PAD = { top: 8, right: 6, bottom: 20, left: 30 };
  const innerW = W - PAD.left - PAD.right;
  const innerH = H - PAD.top - PAD.bottom;

  if (points.length < 2) {
    return (
      <svg width={W} height={H} className="block">
        <rect x={PAD.left} y={PAD.top} width={innerW} height={innerH} fill="#161b22" rx={2} />
        <text x={W / 2} y={H / 2 + 4} textAnchor="middle" fill="#484f58" fontSize={10}>
          Waiting for data…
        </text>
      </svg>
    );
  }

  const maxEpoch = Math.max(...points.map((p) => p.epoch));
  const minEpoch = Math.min(...points.map((p) => p.epoch));
  const epochRange = Math.max(maxEpoch - minEpoch, 1);

  const xOf = (epoch: number) =>
    PAD.left + ((epoch - minEpoch) / epochRange) * innerW;
  const yOf = (acc: number) =>
    PAD.top + innerH - acc * innerH;

  const pathD = points
    .map((p, i) => `${i === 0 ? 'M' : 'L'} ${xOf(p.epoch).toFixed(1)} ${yOf(p.val_accuracy).toFixed(1)}`)
    .join(' ');

  // Y-axis labels
  const yLabels = [0, 0.5, 1.0];
  // X-axis labels
  const xLabels =
    maxEpoch <= 10
      ? points.map((p) => p.epoch)
      : [minEpoch, Math.round((minEpoch + maxEpoch) / 2), maxEpoch];

  return (
    <svg width={W} height={H} className="block overflow-visible">
      {/* background */}
      <rect x={PAD.left} y={PAD.top} width={innerW} height={innerH} fill="#0d1117" rx={2} />

      {/* gridlines + y-labels */}
      {yLabels.map((v) => {
        const y = yOf(v);
        return (
          <g key={v}>
            <line
              x1={PAD.left}
              y1={y}
              x2={PAD.left + innerW}
              y2={y}
              stroke="#21262d"
              strokeWidth={1}
            />
            <text x={PAD.left - 4} y={y + 3} textAnchor="end" fill="#484f58" fontSize={8}>
              {Math.round(v * 100)}
            </text>
          </g>
        );
      })}

      {/* x-axis labels */}
      {xLabels.map((ep) => (
        <text
          key={ep}
          x={xOf(ep)}
          y={PAD.top + innerH + 12}
          textAnchor="middle"
          fill="#484f58"
          fontSize={8}
        >
          {ep}
        </text>
      ))}

      {/* axes */}
      <line
        x1={PAD.left}
        y1={PAD.top}
        x2={PAD.left}
        y2={PAD.top + innerH}
        stroke="#484f58"
        strokeWidth={1}
      />
      <line
        x1={PAD.left}
        y1={PAD.top + innerH}
        x2={PAD.left + innerW}
        y2={PAD.top + innerH}
        stroke="#484f58"
        strokeWidth={1}
      />

      {/* line */}
      <path d={pathD} fill="none" stroke="#58a6ff" strokeWidth={1.5} strokeLinejoin="round" />

      {/* last dot */}
      {points.length > 0 && (
        <circle
          cx={xOf(points[points.length - 1].epoch)}
          cy={yOf(points[points.length - 1].val_accuracy)}
          r={2.5}
          fill="#58a6ff"
        />
      )}
    </svg>
  );
}

// ─────────────────────────────────────────────────────────────────────────────
// MorphologyCNNTraining component
// ─────────────────────────────────────────────────────────────────────────────

const DEFAULT_CONFIG: TrainingConfig = {
  dataset_path: '',
  model_arch: 'efficientnet_b0',
  epochs: 50,
  batch_size: 32,
  learning_rate: 0.0001,
  dropout: 0.3,
  val_split: 0.2,
  early_stopping_patience: 10,
  use_fp16: true,
  augment: true,
};

function MorphologyCNNTraining() {
  const [config, setConfig] = useState<TrainingConfig>(DEFAULT_CONFIG);
  const [taskId, setTaskId] = useState<string | null>(null);
  const [pollingEnabled, setPollingEnabled] = useState(false);
  const [epochHistory, setEpochHistory] = useState<EpochPoint[]>([]);
  const lastEpochRef = useRef<number>(-1);

  // Evaluate state
  const [evalModelPath, setEvalModelPath] = useState('');
  const [evalResult, setEvalResult] = useState<EvaluateResponse | null>(null);

  // Export state
  const [exportModelPath, setExportModelPath] = useState('');
  const [exportOutputPath, setExportOutputPath] = useState('');
  const [exportResult, setExportResult] = useState<ExportResponse | null>(null);

  // ── Training status poll ─────────────────────────────────────────────────

  const { data: statusData } = useQuery<TrainingStatusResponse>({
    queryKey: ['morphology-training-status'],
    queryFn: () =>
      api.get<TrainingStatusResponse>('/morphology/training/status').then((r) => r.data),
    refetchInterval: pollingEnabled ? 2000 : false,
    enabled: pollingEnabled,
  });

  // Append epoch history points
  useEffect(() => {
    if (!statusData) return;
    if (
      statusData.status === 'running' &&
      statusData.epoch > lastEpochRef.current
    ) {
      lastEpochRef.current = statusData.epoch;
      setEpochHistory((prev) => [
        ...prev,
        { epoch: statusData.epoch, val_accuracy: statusData.val_accuracy },
      ]);
    }
    if (statusData.status === 'complete' || statusData.status === 'error') {
      setPollingEnabled(false);
    }
  }, [statusData]);

  // ── Start mutation ────────────────────────────────────────────────────────

  const startMutation = useMutation<TrainingStartResponse, Error, TrainingConfig>({
    mutationFn: (cfg) =>
      api.post<TrainingStartResponse>('/morphology/training/start', cfg).then((r) => r.data),
    onSuccess: (data) => {
      setTaskId(data.task_id);
      setEpochHistory([]);
      lastEpochRef.current = -1;
      setPollingEnabled(true);
    },
  });

  // ── Evaluate mutation ─────────────────────────────────────────────────────

  const evalMutation = useMutation<EvaluateResponse, Error, string>({
    mutationFn: (modelPath) =>
      api
        .post<EvaluateResponse>('/morphology/training/evaluate', { model_path: modelPath })
        .then((r) => r.data),
    onSuccess: (data) => setEvalResult(data),
  });

  // ── Export mutation ───────────────────────────────────────────────────────

  const exportMutation = useMutation<ExportResponse, Error, { model_path: string; output_path: string }>({
    mutationFn: (body) =>
      api
        .post<ExportResponse>('/morphology/training/export', body)
        .then((r) => r.data),
    onSuccess: (data) => setExportResult(data),
  });

  const trainingStatus: TrainingStatus = statusData?.status ?? 'idle';
  const isRunning = trainingStatus === 'running';
  const isComplete = trainingStatus === 'complete';
  const isError = trainingStatus === 'error';
  const hasStarted = trainingStatus !== 'idle' && statusData != null;

  const handleStart = () => {
    if (!config.dataset_path.trim()) return;
    startMutation.mutate(config);
  };

  // ── Render ────────────────────────────────────────────────────────────────

  return (
    <div className="space-y-6">
      {/* ── Config card ─────────────────────────────────────────────────── */}
      <div className="card">
        <div className="card-header flex items-center gap-2">
          <BrainCircuit className="w-4 h-4" />
          Training Configuration
        </div>

        <div className="grid grid-cols-1 md:grid-cols-2 gap-x-8 gap-y-4 mt-1">
          {/* dataset_path — full width */}
          <div className="md:col-span-2 space-y-1">
            <label className="text-xs text-text-secondary">Dataset path</label>
            <input
              type="text"
              value={config.dataset_path}
              onChange={(e) => setConfig((c) => ({ ...c, dataset_path: e.target.value }))}
              placeholder="/data/morphology/labeled  (subdirs = class names)"
              className={cn(
                'w-full bg-[#0d1117] border border-[#21262d] rounded-md px-3 py-2',
                'text-sm text-[#e6edf3] placeholder-[#484f58] outline-none',
                'focus:border-[#58a6ff] transition-colors',
              )}
            />
          </div>

          {/* model_arch */}
          <div className="space-y-1">
            <label className="text-xs text-text-secondary">Architecture</label>
            <select
              value={config.model_arch}
              onChange={(e) => setConfig((c) => ({ ...c, model_arch: e.target.value }))}
              className={cn(
                'w-full bg-[#0d1117] border border-[#21262d] rounded-md px-3 py-2',
                'text-sm text-[#e6edf3] outline-none focus:border-[#58a6ff] transition-colors',
              )}
            >
              <option value="efficientnet_b0">EfficientNet-B0</option>
            </select>
          </div>

          {/* epochs */}
          <div className="space-y-1">
            <label className="text-xs text-text-secondary">Epochs</label>
            <input
              type="number"
              min={10}
              max={200}
              value={config.epochs}
              onChange={(e) =>
                setConfig((c) => ({ ...c, epochs: Math.max(10, Math.min(200, Number(e.target.value))) }))
              }
              className={cn(
                'w-full bg-[#0d1117] border border-[#21262d] rounded-md px-3 py-2',
                'text-sm text-[#e6edf3] outline-none focus:border-[#58a6ff] transition-colors',
              )}
            />
          </div>

          {/* batch_size */}
          <div className="space-y-1">
            <label className="text-xs text-text-secondary">
              Batch size{' '}
              <span className="text-[#484f58]">(reduce if OOM)</span>
            </label>
            <select
              value={config.batch_size}
              onChange={(e) => setConfig((c) => ({ ...c, batch_size: Number(e.target.value) }))}
              className={cn(
                'w-full bg-[#0d1117] border border-[#21262d] rounded-md px-3 py-2',
                'text-sm text-[#e6edf3] outline-none focus:border-[#58a6ff] transition-colors',
              )}
            >
              {[8, 16, 32, 64].map((v) => (
                <option key={v} value={v}>
                  {v}
                </option>
              ))}
            </select>
          </div>

          {/* learning_rate */}
          <div className="space-y-1">
            <label className="text-xs text-text-secondary">Learning rate</label>
            <select
              value={config.learning_rate}
              onChange={(e) => setConfig((c) => ({ ...c, learning_rate: Number(e.target.value) }))}
              className={cn(
                'w-full bg-[#0d1117] border border-[#21262d] rounded-md px-3 py-2',
                'text-sm text-[#e6edf3] outline-none focus:border-[#58a6ff] transition-colors',
              )}
            >
              <option value={1e-5}>0.00001</option>
              <option value={5e-5}>0.00005</option>
              <option value={1e-4}>0.0001</option>
              <option value={5e-4}>0.0005</option>
            </select>
          </div>

          {/* early_stopping_patience */}
          <div className="space-y-1">
            <label className="text-xs text-text-secondary">Early stopping patience</label>
            <input
              type="number"
              min={3}
              max={20}
              value={config.early_stopping_patience}
              onChange={(e) =>
                setConfig((c) => ({
                  ...c,
                  early_stopping_patience: Math.max(3, Math.min(20, Number(e.target.value))),
                }))
              }
              className={cn(
                'w-full bg-[#0d1117] border border-[#21262d] rounded-md px-3 py-2',
                'text-sm text-[#e6edf3] outline-none focus:border-[#58a6ff] transition-colors',
              )}
            />
          </div>

          {/* dropout slider */}
          <div className="space-y-1">
            <label className="text-xs text-text-secondary">
              Dropout{' '}
              <span className="font-mono text-[#58a6ff]">{config.dropout.toFixed(1)}</span>
            </label>
            <input
              type="range"
              min={0}
              max={0.5}
              step={0.1}
              value={config.dropout}
              onChange={(e) => setConfig((c) => ({ ...c, dropout: Number(e.target.value) }))}
              className="w-full accent-[#58a6ff]"
            />
            <div className="flex justify-between text-[10px] text-[#484f58]">
              <span>0.0</span>
              <span>0.5</span>
            </div>
          </div>

          {/* val_split slider */}
          <div className="space-y-1">
            <label className="text-xs text-text-secondary">
              Validation split{' '}
              <span className="font-mono text-[#58a6ff]">
                {Math.round(config.val_split * 100)}%
              </span>
            </label>
            <input
              type="range"
              min={0.1}
              max={0.3}
              step={0.05}
              value={config.val_split}
              onChange={(e) => setConfig((c) => ({ ...c, val_split: Number(e.target.value) }))}
              className="w-full accent-[#58a6ff]"
            />
            <div className="flex justify-between text-[10px] text-[#484f58]">
              <span>10%</span>
              <span>30%</span>
            </div>
          </div>

          {/* toggles row */}
          <div className="md:col-span-2 flex flex-wrap gap-6 pt-1">
            {/* use_fp16 */}
            <label className="flex items-center gap-2 cursor-pointer select-none">
              <button
                type="button"
                role="switch"
                aria-checked={config.use_fp16}
                onClick={() => setConfig((c) => ({ ...c, use_fp16: !c.use_fp16 }))}
                className={cn(
                  'relative inline-flex h-5 w-9 shrink-0 rounded-full border-2 border-transparent',
                  'transition-colors focus-visible:outline-none',
                  config.use_fp16 ? 'bg-[#58a6ff]' : 'bg-[#21262d]',
                )}
              >
                <span
                  className={cn(
                    'pointer-events-none inline-block h-4 w-4 rounded-full bg-white shadow',
                    'transform transition-transform',
                    config.use_fp16 ? 'translate-x-4' : 'translate-x-0',
                  )}
                />
              </button>
              <span className="text-xs text-text-secondary">
                FP16 mixed precision{' '}
                <span className="text-[#484f58]">(RTX 4060)</span>
              </span>
            </label>

            {/* augment */}
            <label className="flex items-center gap-2 cursor-pointer select-none">
              <button
                type="button"
                role="switch"
                aria-checked={config.augment}
                onClick={() => setConfig((c) => ({ ...c, augment: !c.augment }))}
                className={cn(
                  'relative inline-flex h-5 w-9 shrink-0 rounded-full border-2 border-transparent',
                  'transition-colors focus-visible:outline-none',
                  config.augment ? 'bg-[#58a6ff]' : 'bg-[#21262d]',
                )}
              >
                <span
                  className={cn(
                    'pointer-events-none inline-block h-4 w-4 rounded-full bg-white shadow',
                    'transform transition-transform',
                    config.augment ? 'translate-x-4' : 'translate-x-0',
                  )}
                />
              </button>
              <span className="text-xs text-text-secondary">
                Microscopy augmentations{' '}
                <span className="text-[#484f58]">(rotation, color jitter)</span>
              </span>
            </label>
          </div>
        </div>

        {/* Start button */}
        <div className="mt-5 flex items-center gap-3">
          <button
            onClick={handleStart}
            disabled={!config.dataset_path.trim() || startMutation.isPending || isRunning}
            className={cn(
              'flex items-center gap-2 px-4 py-2 rounded-md text-sm font-medium transition-colors',
              'bg-[#58a6ff] text-[#0d1117] hover:bg-[#79b8ff]',
              'disabled:opacity-40 disabled:cursor-not-allowed',
            )}
          >
            {startMutation.isPending ? (
              <Loader2 className="w-4 h-4 animate-spin" />
            ) : (
              <Play className="w-4 h-4" />
            )}
            Start Training
          </button>
          {taskId && (
            <span className="text-xs text-[#484f58] font-mono">task: {taskId}</span>
          )}
        </div>

        {startMutation.isError && (
          <div className="mt-3 text-sm text-red-400">{String(startMutation.error)}</div>
        )}
      </div>

      {/* ── Live training panel ──────────────────────────────────────────── */}
      {hasStarted && statusData && (
        <div className="card space-y-4">
          <div className="card-header flex items-center justify-between">
            <span className="flex items-center gap-2">
              <BarChart3 className="w-4 h-4" />
              Live Training
            </span>
            {/* Status badge */}
            {isRunning && (
              <span className="flex items-center gap-1.5 text-xs font-medium text-[#58a6ff]">
                <span className="w-2 h-2 rounded-full bg-[#58a6ff] animate-pulse" />
                Running
              </span>
            )}
            {isComplete && (
              <span className="flex items-center gap-1.5 text-xs font-medium text-green-400">
                <CheckCircle2 className="w-3.5 h-3.5" />
                Complete
              </span>
            )}
            {isError && (
              <span className="flex items-center gap-1.5 text-xs font-medium text-red-400">
                <XCircle className="w-3.5 h-3.5" />
                Error
              </span>
            )}
          </div>

          {/* Epoch progress */}
          <div className="space-y-1">
            <div className="flex justify-between text-xs text-text-secondary">
              <span>
                Epoch{' '}
                <span className="font-mono text-[#e6edf3]">{statusData.epoch}</span>
                {' / '}
                <span className="font-mono text-[#e6edf3]">{statusData.total_epochs}</span>
              </span>
              <span className="text-[#484f58]">{formatElapsed(statusData.elapsed_s)}</span>
            </div>
            <div className="h-2 bg-[#21262d] rounded-full overflow-hidden">
              <div
                className="h-full bg-[#58a6ff] rounded-full transition-all duration-500"
                style={{
                  width: `${statusData.total_epochs > 0 ? (statusData.epoch / statusData.total_epochs) * 100 : 0}%`,
                }}
              />
            </div>
          </div>

          {/* Metric rows */}
          <div className="grid grid-cols-2 gap-3 text-xs">
            <div className="bg-[#0d1117] rounded-md p-3 space-y-1">
              <div className="text-[#484f58]">Train loss</div>
              <div className="font-mono text-[#e6edf3] text-sm">{statusData.train_loss.toFixed(4)}</div>
            </div>
            <div className="bg-[#0d1117] rounded-md p-3 space-y-1">
              <div className="text-[#484f58]">Val loss</div>
              <div className="font-mono text-[#e6edf3] text-sm">{statusData.val_loss.toFixed(4)}</div>
            </div>
            <div className="bg-[#0d1117] rounded-md p-3 space-y-1">
              <div className="text-[#484f58]">Val accuracy</div>
              <div className="font-mono text-[#e6edf3] text-sm">
                {(statusData.val_accuracy * 100).toFixed(2)}%
              </div>
            </div>
            <div className="bg-[#0d1117] rounded-md p-3 space-y-1 ring-1 ring-green-500/30">
              <div className="text-[#484f58]">Best val accuracy</div>
              <div className="font-mono text-green-400 text-sm">
                {(statusData.best_val_accuracy * 100).toFixed(2)}%
              </div>
            </div>
          </div>

          {/* Accuracy chart */}
          <div>
            <div className="text-xs text-[#484f58] mb-1">Val accuracy over epochs</div>
            <AccuracyChart points={epochHistory} />
          </div>
        </div>
      )}

      {/* ── Evaluate section ─────────────────────────────────────────────── */}
      {isComplete && (
        <div className="card space-y-4">
          <div className="card-header flex items-center gap-2">
            <ClipboardCheck className="w-4 h-4" />
            Evaluate Model
          </div>

          <div className="flex gap-2">
            <input
              type="text"
              value={evalModelPath}
              onChange={(e) => setEvalModelPath(e.target.value)}
              placeholder="/models/morphology/best_model.pt"
              className={cn(
                'flex-1 bg-[#0d1117] border border-[#21262d] rounded-md px-3 py-2',
                'text-sm text-[#e6edf3] placeholder-[#484f58] outline-none',
                'focus:border-[#58a6ff] transition-colors',
              )}
            />
            <button
              onClick={() => evalMutation.mutate(evalModelPath)}
              disabled={!evalModelPath.trim() || evalMutation.isPending}
              className={cn(
                'px-4 py-2 rounded-md text-sm font-medium transition-colors',
                'bg-[#21262d] text-[#e6edf3] hover:bg-[#30363d]',
                'disabled:opacity-40 disabled:cursor-not-allowed',
                'flex items-center gap-2',
              )}
            >
              {evalMutation.isPending && <Loader2 className="w-3.5 h-3.5 animate-spin" />}
              Evaluate
            </button>
          </div>

          {evalMutation.isError && (
            <div className="text-sm text-red-400">{String(evalMutation.error)}</div>
          )}

          {evalResult && (
            <div className="space-y-4">
              {/* Overall accuracy */}
              <div className="flex items-center gap-3">
                <span className="text-3xl font-mono font-bold text-green-400">
                  {(evalResult.accuracy * 100).toFixed(1)}%
                </span>
                <span className="text-sm text-text-secondary">overall accuracy</span>
              </div>

              {/* Per-class bars */}
              <div className="space-y-2">
                <div className="text-xs text-[#484f58]">Per-class accuracy</div>
                {(
                  ['CAPITATE_STALKED', 'CAPITATE_SESSILE', 'BULBOUS', 'NON_GLANDULAR'] as const
                ).map((cls) => {
                  const val = evalResult.per_class[cls] ?? 0;
                  const color = CLASS_COLORS[cls] ?? '#484f58';
                  const label = CLASS_DISPLAY[cls] ?? cls;
                  return (
                    <div key={cls} className="space-y-0.5">
                      <div className="flex justify-between text-xs">
                        <span className="text-text-secondary">{label}</span>
                        <span className="font-mono" style={{ color }}>
                          {(val * 100).toFixed(1)}%
                        </span>
                      </div>
                      <svg width="100%" height="8">
                        <rect width="100%" height="8" rx="4" fill="#21262d" />
                        <rect
                          width={`${val * 100}%`}
                          height="8"
                          rx="4"
                          fill={color}
                          style={{ transition: 'width 0.4s ease' }}
                        />
                      </svg>
                    </div>
                  );
                })}
              </div>
            </div>
          )}
        </div>
      )}

      {/* ── Export to ONNX ───────────────────────────────────────────────── */}
      {isComplete && (
        <div className="card space-y-4">
          <div className="card-header flex items-center gap-2">
            <Download className="w-4 h-4" />
            Export to ONNX
          </div>

          <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
            <div className="space-y-1">
              <label className="text-xs text-text-secondary">Model path</label>
              <input
                type="text"
                value={exportModelPath}
                onChange={(e) => setExportModelPath(e.target.value)}
                placeholder="/models/morphology/best_model.pt"
                className={cn(
                  'w-full bg-[#0d1117] border border-[#21262d] rounded-md px-3 py-2',
                  'text-sm text-[#e6edf3] placeholder-[#484f58] outline-none',
                  'focus:border-[#58a6ff] transition-colors',
                )}
              />
            </div>
            <div className="space-y-1">
              <label className="text-xs text-text-secondary">Output path</label>
              <input
                type="text"
                value={exportOutputPath}
                onChange={(e) => setExportOutputPath(e.target.value)}
                placeholder="/models/morphology/model.onnx"
                className={cn(
                  'w-full bg-[#0d1117] border border-[#21262d] rounded-md px-3 py-2',
                  'text-sm text-[#e6edf3] placeholder-[#484f58] outline-none',
                  'focus:border-[#58a6ff] transition-colors',
                )}
              />
            </div>
          </div>

          <button
            onClick={() =>
              exportMutation.mutate({
                model_path: exportModelPath,
                output_path: exportOutputPath,
              })
            }
            disabled={
              !exportModelPath.trim() || !exportOutputPath.trim() || exportMutation.isPending
            }
            className={cn(
              'flex items-center gap-2 px-4 py-2 rounded-md text-sm font-medium transition-colors',
              'bg-[#21262d] text-[#e6edf3] hover:bg-[#30363d]',
              'disabled:opacity-40 disabled:cursor-not-allowed',
            )}
          >
            {exportMutation.isPending ? (
              <Loader2 className="w-4 h-4 animate-spin" />
            ) : (
              <Download className="w-4 h-4" />
            )}
            Export ONNX
          </button>

          {exportMutation.isError && (
            <div className="text-sm text-red-400">{String(exportMutation.error)}</div>
          )}

          {exportResult && (
            <div className="bg-green-500/10 border border-green-500/30 rounded-md p-4 space-y-1 text-sm">
              <div className="flex items-center gap-2 text-green-400 font-medium mb-2">
                <CheckCircle2 className="w-4 h-4" />
                Export successful
              </div>
              <div className="flex justify-between text-xs">
                <span className="text-[#484f58]">ONNX path</span>
                <span className="font-mono text-[#e6edf3] truncate max-w-[60%]">
                  {exportResult.onnx_path}
                </span>
              </div>
              <div className="flex justify-between text-xs">
                <span className="text-[#484f58]">Export time</span>
                <span className="font-mono text-[#e6edf3]">
                  {exportResult.export_time_s.toFixed(2)} s
                </span>
              </div>
            </div>
          )}
        </div>
      )}
    </div>
  );
}

// ─────────────────────────────────────────────────────────────────────────────
// Main page
// ─────────────────────────────────────────────────────────────────────────────

export default function MorphologyPage() {
  const [tab, setTab] = useState<'classify' | 'training'>('classify');

  // ── Classify tab state ────────────────────────────────────────────────────

  const imageRef = useRef<HTMLImageElement>(null);
  const [imageUrl, setImageUrl] = useState<string | null>(null);
  const [imageFile, setImageFile] = useState<File | null>(null);
  const [morphResult, setMorphResult] = useState<MorphologyResponse | null>(null);
  const [maturityResult, setMaturityResult] = useState<MaturityResponse | null>(null);

  // ── Mutations ──────────────────────────────────────────────────────────────

  const morphMutation = useMutation({
    mutationFn: (file: File) =>
      uploadFile('/morphology/instance', file, {
        include_geometric: 'true',
        include_stalk: 'true',
      }).then((r) => r.data as MorphologyResponse),
    onSuccess: (data) => setMorphResult(data),
  });

  const maturityMutation = useMutation({
    mutationFn: (file: File) =>
      uploadFile('/maturity/analyze/crop', file, { include_features: 'true' }).then(
        (r) => r.data as MaturityResponse,
      ),
    onSuccess: (data) => setMaturityResult(data),
  });

  // ── Dropzone ──────────────────────────────────────────────────────────────

  const onDrop = useCallback(
    (accepted: File[]) => {
      const file = accepted[0];
      if (!file) return;
      setImageFile(file);
      setMorphResult(null);
      setMaturityResult(null);

      const reader = new FileReader();
      reader.onload = (e) => setImageUrl(e.target?.result as string);
      reader.readAsDataURL(file);

      morphMutation.mutate(file);
      maturityMutation.mutate(file);
    },
    [morphMutation, maturityMutation],
  );

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept: { 'image/*': ['.jpg', '.jpeg', '.png', '.tif', '.tiff', '.bmp'] },
    maxFiles: 1,
    maxSize: 100 * 1024 * 1024,
  });

  const isLoading = morphMutation.isPending || maturityMutation.isPending;

  // ── Render ─────────────────────────────────────────────────────────────────

  return (
    <div className="space-y-6 max-w-[1400px] mx-auto">
      {/* Header */}
      <div>
        <h1 className="text-xl font-semibold text-text-primary flex items-center gap-2">
          <Microscope className="w-5 h-5 text-accent" />
          Trichome Analysis
        </h1>
        <p className="text-sm text-text-secondary mt-0.5">
          Morphology classification + maturity stage estimation from a single image
        </p>
      </div>

      {/* Tab bar */}
      <div className="flex gap-1 p-1 rounded-lg bg-[#161b22] border border-[#21262d] w-fit">
        {(
          [
            { key: 'classify', label: 'Classify', icon: FlaskConical },
            { key: 'training', label: 'Training', icon: BrainCircuit },
          ] as const
        ).map(({ key, label, icon: Icon }) => (
          <button
            key={key}
            onClick={() => setTab(key)}
            className={cn(
              'flex items-center gap-1.5 px-4 py-1.5 rounded-md text-sm font-medium transition-colors',
              tab === key
                ? 'bg-[#21262d] text-[#e6edf3]'
                : 'text-[#484f58] hover:text-[#e6edf3]',
            )}
          >
            <Icon className="w-3.5 h-3.5" />
            {label}
          </button>
        ))}
      </div>

      {/* ── Classify tab ──────────────────────────────────────────────────── */}
      {tab === 'classify' && (
        <>
          {/* Scientific caveat */}
          <div className="scientific-caveat">
            <Info className="w-4 h-4 shrink-0 mt-0.5" />
            <div>
              <strong className="block mb-0.5">Scientific Note</strong>
              Maturity stage describes <em>optical color state</em> only (clear → cloudy → amber →
              degraded). This does not quantify cannabinoid content. Chromatography (GC-MS, HPLC) is
              required for precise biochemical measurement.
            </div>
          </div>

          <div className="grid grid-cols-1 lg:grid-cols-5 gap-6">
            {/* Upload + preview — 3/5 width */}
            <div className="lg:col-span-3 space-y-4">
              {/* Dropzone */}
              <div
                {...getRootProps()}
                className={cn(
                  'border-2 border-dashed rounded-lg p-6 text-center cursor-pointer transition-colors',
                  isDragActive
                    ? 'border-accent bg-accent/5'
                    : 'border-border hover:border-accent/50',
                )}
              >
                <input {...getInputProps()} />
                <Upload className="w-8 h-8 mx-auto text-text-muted mb-2" />
                <p className="text-sm text-text-secondary">
                  {isDragActive ? 'Drop here' : 'Drop a trichome image or click to browse'}
                </p>
                <p className="text-xs text-text-muted mt-1">
                  JPEG · PNG · TIFF · BMP · up to 100 MB
                </p>
              </div>

              {/* Image preview */}
              {imageUrl && (
                <div className="card">
                  <div className="card-header">Preview</div>
                  <div className="relative">
                    <img
                      ref={imageRef}
                      src={imageUrl}
                      alt="Trichome"
                      className="w-full rounded-md object-contain max-h-96"
                    />
                    {isLoading && (
                      <div className="absolute inset-0 flex items-center justify-center bg-background/70 rounded-md">
                        <div className="flex flex-col items-center gap-2">
                          <Loader2 className="w-8 h-8 text-accent animate-spin" />
                          <span className="text-sm text-text-secondary">Analyzing…</span>
                        </div>
                      </div>
                    )}
                  </div>
                  <div className="mt-2 text-xs text-text-muted font-mono">{imageFile?.name}</div>
                </div>
              )}

              {/* Error display */}
              {(morphMutation.isError || maturityMutation.isError) && (
                <div className="card border-status-error/30 bg-status-error/5">
                  <div className="text-sm text-status-error">
                    {morphMutation.isError && (
                      <div>Morphology: {String(morphMutation.error)}</div>
                    )}
                    {maturityMutation.isError && (
                      <div>Maturity: {String(maturityMutation.error)}</div>
                    )}
                  </div>
                </div>
              )}
            </div>

            {/* Results panel — 2/5 width */}
            <div className="lg:col-span-2 space-y-4">
              {/* Morphology result */}
              <div className="card">
                <div className="card-header flex items-center gap-2">
                  <FlaskConical className="w-4 h-4" />
                  Morphology
                </div>

                {!morphResult && !morphMutation.isPending && (
                  <div className="text-sm text-text-muted py-4 text-center">
                    Upload an image to classify
                  </div>
                )}

                {morphMutation.isPending && (
                  <div className="flex items-center gap-2 text-sm text-text-secondary py-4">
                    <Loader2 className="w-4 h-4 animate-spin" />
                    Classifying…
                  </div>
                )}

                {morphResult && (
                  <div className="space-y-3">
                    {/* Type badge */}
                    <div className="flex items-center gap-3">
                      <div
                        className="w-3 h-3 rounded-full shrink-0"
                        style={{
                          backgroundColor: TYPE_COLORS[morphResult.morphology_type] ?? '#6b7280',
                        }}
                      />
                      <div>
                        <div className="text-base font-semibold text-text-primary">
                          {TYPE_LABELS[morphResult.morphology_type] ?? morphResult.morphology_type}
                        </div>
                        <div className="text-xs text-text-muted">
                          {morphResult.classification_method} classifier
                        </div>
                      </div>
                      <div
                        className="ml-auto text-sm font-mono font-bold"
                        style={{ color: getConfidenceColor(morphResult.confidence) }}
                      >
                        {formatConfidence(morphResult.confidence)}
                      </div>
                    </div>

                    {/* Confidence bar */}
                    <div className="h-1.5 bg-panel rounded-full overflow-hidden">
                      <div
                        className="h-full rounded-full transition-all"
                        style={{
                          width: `${morphResult.confidence * 100}%`,
                          backgroundColor: getConfidenceColor(morphResult.confidence),
                        }}
                      />
                    </div>

                    {/* Stalk info */}
                    {morphResult.stalk && (
                      <div className="code-block text-xs space-y-1">
                        <div className="flex justify-between">
                          <span>Stalk visible</span>
                          <span
                            className={
                              morphResult.stalk.has_visible_stalk
                                ? 'text-status-success'
                                : 'text-text-muted'
                            }
                          >
                            {morphResult.stalk.has_visible_stalk ? 'Yes' : 'No'}
                          </span>
                        </div>
                        {morphResult.stalk.stalk_length_px != null && (
                          <div className="flex justify-between">
                            <span>Stalk length</span>
                            <span>{morphResult.stalk.stalk_length_px.toFixed(1)} px</span>
                          </div>
                        )}
                        {morphResult.stalk.head_diameter_px != null && (
                          <div className="flex justify-between">
                            <span>Head diameter</span>
                            <span>{morphResult.stalk.head_diameter_px.toFixed(1)} px</span>
                          </div>
                        )}
                        {morphResult.stalk.head_circularity != null && (
                          <div className="flex justify-between">
                            <span>Head circularity</span>
                            <span>{morphResult.stalk.head_circularity.toFixed(3)}</span>
                          </div>
                        )}
                      </div>
                    )}

                    {/* Geometric descriptors */}
                    {morphResult.geometric_features && (
                      <details className="group">
                        <summary className="text-xs text-text-secondary cursor-pointer flex items-center gap-1 select-none">
                          <BarChart3 className="w-3 h-3" />
                          Geometric descriptors
                        </summary>
                        <div className="mt-2 space-y-0">
                          <MetricRow
                            label="Area"
                            value={morphResult.geometric_features.area_px}
                            unit=" px²"
                          />
                          <MetricRow
                            label="Perimeter"
                            value={morphResult.geometric_features.perimeter_px}
                            unit=" px"
                          />
                          <MetricRow
                            label="Circularity"
                            value={morphResult.geometric_features.circularity}
                          />
                          <MetricRow
                            label="Eccentricity"
                            value={morphResult.geometric_features.eccentricity}
                          />
                          <MetricRow
                            label="Solidity"
                            value={morphResult.geometric_features.solidity}
                          />
                          <MetricRow
                            label="Aspect ratio"
                            value={morphResult.geometric_features.aspect_ratio}
                          />
                          <MetricRow
                            label="Extent"
                            value={morphResult.geometric_features.extent}
                          />
                          <MetricRow
                            label="Major axis"
                            value={morphResult.geometric_features.major_axis_px}
                            unit=" px"
                          />
                          <MetricRow
                            label="Minor axis"
                            value={morphResult.geometric_features.minor_axis_px}
                            unit=" px"
                          />
                        </div>
                      </details>
                    )}

                    {morphResult.processing_time_ms != null && (
                      <div className="text-xs text-text-muted text-right">
                        {morphResult.processing_time_ms.toFixed(1)} ms
                      </div>
                    )}
                  </div>
                )}
              </div>

              {/* Maturity result */}
              <div className="card">
                <div className="card-header flex items-center gap-2">
                  <BarChart3 className="w-4 h-4" />
                  Maturity Stage
                </div>

                {!maturityResult && !maturityMutation.isPending && (
                  <div className="text-sm text-text-muted py-4 text-center">
                    Upload an image to estimate maturity
                  </div>
                )}

                {maturityMutation.isPending && (
                  <div className="flex items-center gap-2 text-sm text-text-secondary py-4">
                    <Loader2 className="w-4 h-4 animate-spin" />
                    Estimating…
                  </div>
                )}

                {maturityResult && (
                  <div className="space-y-3">
                    {/* Stage badge */}
                    <div className="flex items-center gap-3">
                      <div
                        className="w-3 h-3 rounded-full border border-border shrink-0"
                        style={{
                          backgroundColor: STAGE_COLORS[maturityResult.stage] ?? '#6b7280',
                        }}
                      />
                      <div>
                        <div className="text-base font-semibold text-text-primary capitalize">
                          {maturityResult.stage}
                        </div>
                        <div className="text-xs text-text-muted">Optical color state</div>
                      </div>
                      <div
                        className="ml-auto text-sm font-mono font-bold"
                        style={{ color: getConfidenceColor(maturityResult.confidence) }}
                      >
                        {formatConfidence(maturityResult.confidence)}
                      </div>
                    </div>

                    {/* Confidence bar */}
                    <div className="h-1.5 bg-panel rounded-full overflow-hidden">
                      <div
                        className="h-full rounded-full transition-all"
                        style={{
                          width: `${maturityResult.confidence * 100}%`,
                          backgroundColor: STAGE_COLORS[maturityResult.stage] ?? '#6b7280',
                        }}
                      />
                    </div>

                    {/* Stage scale */}
                    <div className="flex items-center gap-1 text-[10px] text-text-muted">
                      {['clear', 'cloudy', 'amber', 'degraded'].map((s) => (
                        <div
                          key={s}
                          className={cn(
                            'flex-1 text-center py-1 rounded-sm transition-all',
                            maturityResult.stage === s
                              ? 'ring-1 ring-offset-1 ring-offset-surface text-text-primary font-semibold'
                              : 'opacity-40',
                          )}
                          style={{
                            backgroundColor: STAGE_COLORS[s],
                            color: s === 'cloudy' ? '#111' : undefined,
                          }}
                        >
                          {s}
                        </div>
                      ))}
                    </div>

                    {/* Color features */}
                    {maturityResult.color_features && (
                      <details>
                        <summary className="text-xs text-text-secondary cursor-pointer select-none">
                          Color features
                        </summary>
                        <div className="mt-2">
                          <MetricRow
                            label="Hue mean"
                            value={maturityResult.color_features.hue_mean}
                            unit="°"
                          />
                          <MetricRow
                            label="Saturation"
                            value={maturityResult.color_features.saturation_mean}
                          />
                          <MetricRow
                            label="Brightness"
                            value={maturityResult.color_features.value_mean}
                          />
                          <MetricRow
                            label="Amber ratio"
                            value={maturityResult.color_features.amber_ratio}
                          />
                          <MetricRow
                            label="Translucency"
                            value={maturityResult.color_features.translucency_score}
                          />
                        </div>
                      </details>
                    )}

                    {maturityResult.scientific_note && (
                      <p className="text-[11px] text-text-muted italic">
                        {maturityResult.scientific_note}
                      </p>
                    )}

                    {maturityResult.processing_time_ms != null && (
                      <div className="text-xs text-text-muted text-right">
                        {maturityResult.processing_time_ms.toFixed(1)} ms
                      </div>
                    )}
                  </div>
                )}
              </div>

              {/* Type reference */}
              <div className="card">
                <div className="card-header">Trichome Type Reference</div>
                <div className="space-y-2 text-xs text-text-secondary">
                  {[
                    {
                      key: 'capitate_stalked',
                      label: 'Capitate Stalked',
                      desc: 'Elongated stalk + spherical head. 100–500 µm total height.',
                    },
                    {
                      key: 'capitate_sessile',
                      label: 'Capitate Sessile',
                      desc: 'Flat/absent stalk, head on surface. 25–100 µm head diameter.',
                    },
                    {
                      key: 'bulbous',
                      label: 'Bulbous',
                      desc: 'Very small, round, non-stalked. 10–30 µm.',
                    },
                    {
                      key: 'non_glandular',
                      label: 'Non-Glandular',
                      desc: 'Hair-like, no secretory head. Excluded from maturity analysis.',
                    },
                  ].map((t) => (
                    <div key={t.key} className="flex items-start gap-2">
                      <div
                        className="w-2 h-2 rounded-full mt-1 shrink-0"
                        style={{ backgroundColor: TYPE_COLORS[t.key] }}
                      />
                      <div>
                        <span className="font-medium text-text-primary">{t.label}</span>
                        {' — '}
                        {t.desc}
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            </div>
          </div>
        </>
      )}

      {/* ── Training tab ──────────────────────────────────────────────────── */}
      {tab === 'training' && <MorphologyCNNTraining />}
    </div>
  );
}
