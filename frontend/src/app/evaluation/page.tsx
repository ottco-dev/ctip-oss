'use client';

/**
 * Evaluation page — merged Calibration + Benchmarks.
 *
 * Tab 1 – Calibration: ECE computation, reliability diagram, per-bin analysis.
 *   Scientific basis: Guo et al. (2017). On Calibration of Modern Neural Networks. ICML 2017.
 *
 * Tab 2 – Benchmarks: inference latency benchmarking with FPS charts.
 *
 * Tab is controlled via ?tab=calibration|benchmarks query param.
 */

import React, { useState, Suspense } from 'react';
import { useRouter, useSearchParams } from 'next/navigation';
import { useMutation, useQuery } from '@tanstack/react-query';
import {
  FlaskConical,
  BarChart2,
  BarChart3,
  AlertTriangle,
  Info,
  Upload,
  Loader2,
  Play,
} from 'lucide-react';
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  Cell,
} from 'recharts';
import { useDropzone } from 'react-dropzone';
import { api } from '@/lib/api';
import { ReliabilityDiagram, type BinStats } from '@/components/charts/ReliabilityDiagram';
import { cn } from '@/lib/utils';

// ── Shared types ───────────────────────────────────────────────────────────────

type TabId = 'calibration' | 'benchmarks';

// ── Calibration types ──────────────────────────────────────────────────────────

interface CalibrationResponse {
  ece: number;
  mce: number;
  num_bins: number;
  total_samples: number;
  is_overconfident: boolean;
  overconfident_bin_fraction: number;
  bins: BinStats[];
  confidence_histogram: number[];
  interpretation: string;
  mlflow_run_id: string | null;
  source: string;
}

// ── Benchmarks types ───────────────────────────────────────────────────────────

interface ModelInfo {
  model_id: string;
  name: string;
  type: string;
  path?: string;
  format?: string;
  size_mb?: number;
  is_active?: boolean;
}

interface BenchmarkRun {
  model: string;
  n_images: number;
  mean_ms: number;
  min_ms: number;
  max_ms: number;
  fps: number;
  conf_threshold: number;
  timestamp: string;
}

// ── Reference benchmark data ───────────────────────────────────────────────────

const REFERENCE_BENCHMARKS = [
  { model: 'YOLO11n', imgsz: 640,  tiled: false, ms: 6,   fps: 150, vram: '0.6 GB' },
  { model: 'YOLO11n', imgsz: 1280, tiled: false, ms: 11,  fps: 90,  vram: '0.8 GB' },
  { model: 'YOLO11s', imgsz: 1280, tiled: false, ms: 13,  fps: 75,  vram: '1.2 GB' },
  { model: 'YOLO11s', imgsz: 4096, tiled: true,  ms: 150, fps: 7,   vram: '1.2 GB' },
  { model: 'YOLO11s+SAM2', imgsz: 1280, tiled: false, ms: 80, fps: 12, vram: '5.0 GB' },
];

// ── Calibration helpers ────────────────────────────────────────────────────────

/**
 * Parse a whitespace/comma-separated string of floats.
 * Returns null if any token is not a valid number.
 */
function parseFloatList(raw: string): number[] | null {
  const tokens = raw.trim().split(/[\s,]+/).filter(Boolean);
  const nums = tokens.map(Number);
  if (nums.some(isNaN)) return null;
  return nums;
}

/**
 * Parse a whitespace/comma-separated string of booleans.
 * Accepts: 1/0, true/false, yes/no (case-insensitive).
 */
function parseBoolList(raw: string): boolean[] | null {
  const tokens = raw.trim().split(/[\s,]+/).filter(Boolean);
  return tokens.map((t) => {
    const lc = t.toLowerCase();
    if (lc === '1' || lc === 'true' || lc === 'yes') return true;
    if (lc === '0' || lc === 'false' || lc === 'no') return false;
    return null;
  }) as boolean[] | null;
}

// ── Benchmarks sub-components ──────────────────────────────────────────────────

function CustomTooltip({ active, payload, label }: { active?: boolean; payload?: Array<{ name: string; value: number; color: string }>; label?: string }) {
  if (!active || !payload?.length) return null;
  return (
    <div
      className="rounded-lg px-3 py-2 text-xs"
      style={{ background: '#161b22', border: '1px solid #21262d', color: '#e6edf3' }}
    >
      <p className="font-medium mb-1">{label}</p>
      {payload.map((p) => (
        <p key={p.name} style={{ color: p.color }}>
          {p.name}: {typeof p.value === 'number' ? p.value.toFixed(1) : p.value}
          {p.name === 'fps' ? ' fps' : p.name.includes('ms') ? ' ms' : ''}
        </p>
      ))}
    </div>
  );
}

function BenchmarkCard({ run, rank }: { run: BenchmarkRun; rank: number }) {
  const fpsColor =
    run.fps >= 30 ? '#22c55e' : run.fps >= 10 ? '#eab308' : '#ef4444';

  const chartData = [
    { name: 'Min',  ms: run.min_ms },
    { name: 'Mean', ms: run.mean_ms },
    { name: 'Max',  ms: run.max_ms },
  ];

  const getBarColor = (name: string) => {
    if (name === 'Min')  return '#3b82f6';
    if (name === 'Mean') return '#22c55e';
    return '#ef4444';
  };

  return (
    <div
      className="rounded-xl p-5 space-y-4"
      style={{ background: '#0d1117', border: '1px solid #21262d' }}
    >
      <div className="flex items-center justify-between">
        <div>
          <div className="flex items-center gap-2">
            <span
              className="text-[10px] px-1.5 py-0.5 rounded font-mono font-bold"
              style={{ background: '#161b22', color: '#484f58' }}
            >
              #{rank}
            </span>
            <h3 className="text-sm font-semibold text-white">{run.model}</h3>
          </div>
          <p className="text-xs mt-0.5" style={{ color: '#484f58' }}>
            {run.n_images} images · conf={run.conf_threshold.toFixed(2)} · {run.timestamp}
          </p>
        </div>
        <div className="text-right">
          <p className="text-2xl font-bold font-mono" style={{ color: fpsColor }}>
            {run.fps.toFixed(1)}
          </p>
          <p className="text-xs" style={{ color: '#484f58' }}>FPS</p>
        </div>
      </div>

      <ResponsiveContainer width="100%" height={80}>
        <BarChart data={chartData} margin={{ top: 4, right: 4, bottom: 0, left: 0 }}>
          <CartesianGrid strokeDasharray="3 3" stroke="#21262d" vertical={false} />
          <XAxis dataKey="name" tick={{ fill: '#484f58', fontSize: 10 }} axisLine={false} tickLine={false} />
          <YAxis tick={{ fill: '#484f58', fontSize: 10 }} axisLine={false} tickLine={false} width={30} />
          <Tooltip content={<CustomTooltip />} cursor={{ fill: 'rgba(255,255,255,0.04)' }} />
          <Bar dataKey="ms" radius={[3, 3, 0, 0]} maxBarSize={40}>
            {chartData.map((entry) => (
              <Cell key={entry.name} fill={getBarColor(entry.name)} />
            ))}
          </Bar>
        </BarChart>
      </ResponsiveContainer>

      <div className="grid grid-cols-3 gap-2">
        {[
          { label: 'Min',  value: `${run.min_ms.toFixed(1)}ms`,  color: '#3b82f6' },
          { label: 'Mean', value: `${run.mean_ms.toFixed(1)}ms`, color: '#22c55e' },
          { label: 'Max',  value: `${run.max_ms.toFixed(1)}ms`,  color: '#ef4444' },
        ].map(({ label, value, color }) => (
          <div
            key={label}
            className="px-3 py-2 rounded-lg text-center"
            style={{ background: '#161b22' }}
          >
            <p className="text-sm font-bold font-mono" style={{ color }}>{value}</p>
            <p className="text-[10px]" style={{ color: '#484f58' }}>{label}</p>
          </div>
        ))}
      </div>
    </div>
  );
}

function ReferenceTable() {
  const refChartData = REFERENCE_BENCHMARKS.map((r) => ({
    name: `${r.model}\n${r.imgsz}px`,
    fps: r.fps,
    ms: r.ms,
  }));

  return (
    <div
      className="rounded-xl p-5 space-y-4"
      style={{ background: '#0d1117', border: '1px solid #21262d' }}
    >
      <h3 className="text-sm font-semibold text-white">
        Reference Benchmarks (RTX 4060, FP16, batch=1)
      </h3>

      <ResponsiveContainer width="100%" height={120}>
        <BarChart data={refChartData} margin={{ top: 4, right: 4, bottom: 0, left: 0 }}>
          <CartesianGrid strokeDasharray="3 3" stroke="#21262d" vertical={false} />
          <XAxis dataKey="name" tick={{ fill: '#484f58', fontSize: 9 }} axisLine={false} tickLine={false} />
          <YAxis tick={{ fill: '#484f58', fontSize: 10 }} axisLine={false} tickLine={false} width={28} />
          <Tooltip content={<CustomTooltip />} cursor={{ fill: 'rgba(255,255,255,0.04)' }} />
          <Bar dataKey="fps" radius={[3, 3, 0, 0]} maxBarSize={30}>
            {refChartData.map((entry, i) => (
              <Cell
                key={i}
                fill={entry.fps >= 30 ? '#22c55e' : entry.fps >= 10 ? '#eab308' : '#ef4444'}
              />
            ))}
          </Bar>
        </BarChart>
      </ResponsiveContainer>

      <table className="w-full text-xs">
        <thead>
          <tr style={{ borderBottom: '1px solid #21262d' }}>
            {['Model', 'Size', 'Tiled', 'ms/img', 'FPS', 'VRAM'].map((h) => (
              <th
                key={h}
                className={cn('pb-2 font-medium', h !== 'Model' ? 'text-right' : 'text-left')}
                style={{ color: '#484f58' }}
              >
                {h}
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {REFERENCE_BENCHMARKS.map((row, i) => (
            <tr
              key={i}
              className="transition-colors"
              style={{ borderBottom: '1px solid rgba(33,38,45,0.5)', color: '#8b949e' }}
            >
              <td className="py-2">{row.model}</td>
              <td className="py-2 text-right">{row.imgsz}px</td>
              <td className="py-2 text-right">{row.tiled ? '✓' : '—'}</td>
              <td className="py-2 text-right">~{row.ms}ms</td>
              <td
                className="py-2 text-right font-mono font-bold"
                style={{
                  color: row.fps >= 30 ? '#22c55e' : row.fps >= 10 ? '#eab308' : '#ef4444',
                }}
              >
                {row.fps}
              </td>
              <td className="py-2 text-right">{row.vram}</td>
            </tr>
          ))}
        </tbody>
      </table>
      <p className="text-[10px]" style={{ color: '#30363d' }}>
        Source: research/evaluation_methodology/benchmark_design.md
      </p>
    </div>
  );
}

// ── Calibration tab content ────────────────────────────────────────────────────

function CalibrationTab() {
  const [mode, setMode] = useState<'direct' | 'mlflow'>('direct');
  const [numBins, setNumBins] = useState(10);
  const [confText, setConfText] = useState('');
  const [correctText, setCorrectText] = useState('');
  const [runId, setRunId] = useState('');
  const [parseError, setParseError] = useState<string | null>(null);

  const mutation = useMutation<CalibrationResponse, Error, object>({
    mutationFn: (payload) =>
      api.post('/analytics/calibration', payload).then((r) => r.data),
  });

  const handleCompute = () => {
    setParseError(null);

    if (mode === 'direct') {
      const confs = parseFloatList(confText);
      if (!confs) {
        setParseError('Invalid confidence values. Use space or comma-separated floats, e.g. "0.9 0.7 0.5".');
        return;
      }
      const correct = parseBoolList(correctText);
      if (!correct) {
        setParseError('Invalid correctness flags. Use 1/0, true/false, or yes/no.');
        return;
      }
      mutation.mutate({ confidences: confs, is_correct: correct, num_bins: numBins });
    } else {
      if (!runId.trim()) {
        setParseError('Enter a MLflow run ID (32 hex characters).');
        return;
      }
      mutation.mutate({ mlflow_run_id: runId.trim(), num_bins: numBins });
    }
  };

  const result = mutation.data;

  return (
    <div className="space-y-6 max-w-[1200px]">
      <p className="text-sm text-text-secondary max-w-2xl">
        Compute Expected Calibration Error (ECE) and reliability diagrams to assess
        whether model confidence scores match observed accuracy.{' '}
        <span className="text-text-muted">
          Reference: Guo et al. (2017). On Calibration of Modern Neural Networks. ICML 2017.
        </span>
      </p>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Input panel */}
        <div className="space-y-4">
          <div className="card">
            <div className="card-header">Input</div>

            {/* Mode selector */}
            <div className="flex rounded-lg border border-border overflow-hidden mb-4">
              {(['direct', 'mlflow'] as const).map((m) => (
                <button
                  key={m}
                  className={cn(
                    'flex-1 py-2 text-sm font-medium transition-colors',
                    mode === m
                      ? 'bg-accent text-white'
                      : 'bg-surface-secondary text-text-secondary hover:bg-surface-tertiary',
                  )}
                  onClick={() => { setMode(m); setParseError(null); }}
                >
                  {m === 'direct' ? 'Raw Predictions' : 'MLflow Run'}
                </button>
              ))}
            </div>

            {mode === 'direct' ? (
              <div className="space-y-4">
                <div>
                  <label className="block text-xs text-text-secondary mb-1.5">
                    Confidence Scores
                    <span className="text-text-muted ml-1">(space or comma-separated floats in [0, 1])</span>
                  </label>
                  <textarea
                    className="input font-mono text-xs h-24 resize-y"
                    placeholder="0.95 0.82 0.73 0.65 0.51 0.48 0.38 0.20 0.11 …"
                    value={confText}
                    onChange={(e) => setConfText(e.target.value)}
                  />
                </div>
                <div>
                  <label className="block text-xs text-text-secondary mb-1.5">
                    Correctness Flags
                    <span className="text-text-muted ml-1">(1/0 or true/false, same length)</span>
                  </label>
                  <textarea
                    className="input font-mono text-xs h-24 resize-y"
                    placeholder="1 1 0 1 0 0 1 1 0 …"
                    value={correctText}
                    onChange={(e) => setCorrectText(e.target.value)}
                  />
                </div>
                <button
                  className="text-xs text-accent hover:text-accent/80 flex items-center gap-1"
                  onClick={() => {
                    setConfText('0.95 0.90 0.80 0.70 0.60 0.50 0.40 0.30 0.20 0.10');
                    setCorrectText('0 0 0 0 0 1 1 1 1 1');
                  }}
                >
                  <Info className="w-3 h-3" />
                  Load example (overconfident model)
                </button>
              </div>
            ) : (
              <div>
                <label className="block text-xs text-text-secondary mb-1.5">
                  MLflow Run ID
                  <span className="text-text-muted ml-1">(32-character hex)</span>
                </label>
                <input
                  className="input font-mono text-xs"
                  placeholder="a1b2c3d4e5f6a1b2c3d4e5f6a1b2c3d4"
                  value={runId}
                  onChange={(e) => setRunId(e.target.value)}
                />
                <p className="text-xs text-text-muted mt-2">
                  The run must have logged{' '}
                  <code className="font-mono text-text-secondary">predictions/confidence_scores.npy</code>{' '}
                  and{' '}
                  <code className="font-mono text-text-secondary">predictions/is_correct.npy</code>{' '}
                  as MLflow artifacts.
                </p>
              </div>
            )}

            {/* Bins selector */}
            <div className="mt-4">
              <div className="flex items-center justify-between mb-1.5">
                <label className="text-xs text-text-secondary">Bins</label>
                <span className="font-mono text-xs text-accent">{numBins}</span>
              </div>
              <input
                type="range"
                className="w-full accent-accent h-1.5"
                min={5}
                max={50}
                step={5}
                value={numBins}
                onChange={(e) => setNumBins(parseInt(e.target.value))}
              />
              <div className="flex justify-between text-[10px] text-text-muted mt-0.5">
                <span>5</span>
                <span>50</span>
              </div>
            </div>

            {parseError && (
              <div className="flex items-start gap-2 text-xs text-status-error bg-status-error/10 rounded p-2 mt-3">
                <AlertTriangle className="w-3.5 h-3.5 flex-shrink-0 mt-0.5" />
                {parseError}
              </div>
            )}

            {mutation.isError && (
              <div className="flex items-start gap-2 text-xs text-status-error bg-status-error/10 rounded p-2 mt-3">
                <AlertTriangle className="w-3.5 h-3.5 flex-shrink-0 mt-0.5" />
                {mutation.error.message}
              </div>
            )}

            <button
              className="btn-primary w-full mt-4 flex items-center justify-center gap-2"
              onClick={handleCompute}
              disabled={mutation.isPending}
            >
              <BarChart2 className="w-4 h-4" />
              {mutation.isPending ? 'Computing…' : 'Compute Calibration'}
            </button>
          </div>

          {/* Per-bin table */}
          {result && (
            <div className="card">
              <div className="card-header">Per-bin Detail</div>
              <div className="overflow-x-auto">
                <table className="w-full text-xs font-mono">
                  <thead>
                    <tr className="border-b border-border">
                      <th className="table-header text-left py-1.5 pr-3">Bin</th>
                      <th className="table-header text-right py-1.5 pr-3">Conf</th>
                      <th className="table-header text-right py-1.5 pr-3">Acc</th>
                      <th className="table-header text-right py-1.5 pr-3">Gap</th>
                      <th className="table-header text-right py-1.5 pr-3">n</th>
                      <th className="table-header text-right py-1.5">%</th>
                    </tr>
                  </thead>
                  <tbody>
                    {result.bins.filter((b) => !b.is_empty).map((b) => (
                      <tr key={b.bin_index} className="table-row text-xs">
                        <td className="py-1 pr-3 text-text-muted">
                          {b.confidence_lower.toFixed(2)}–{b.confidence_upper.toFixed(2)}
                        </td>
                        <td className="py-1 pr-3 text-right">{b.mean_confidence.toFixed(3)}</td>
                        <td className="py-1 pr-3 text-right">{b.accuracy.toFixed(3)}</td>
                        <td
                          className={cn(
                            'py-1 pr-3 text-right font-semibold',
                            b.is_overconfident ? 'text-orange-400' : 'text-blue-400',
                          )}
                        >
                          {b.gap > 0 ? '+' : ''}{b.gap.toFixed(3)}
                        </td>
                        <td className="py-1 pr-3 text-right text-text-muted">{b.count}</td>
                        <td className="py-1 text-right text-text-muted">
                          {(b.weight * 100).toFixed(1)}%
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          )}
        </div>

        {/* Reliability diagram panel */}
        <div className="card">
          <div className="card-header">Reliability Diagram</div>
          {!result ? (
            <div className="flex flex-col items-center justify-center py-16 text-text-muted">
              <BarChart2 className="w-12 h-12 mb-3 opacity-30" />
              <p className="text-sm">Enter predictions and click Compute</p>
            </div>
          ) : (
            <ReliabilityDiagram
              ece={result.ece}
              mce={result.mce}
              bins={result.bins}
              totalSamples={result.total_samples}
              isOverconfident={result.is_overconfident}
              interpretation={result.interpretation}
              width={440}
              height={340}
            />
          )}
        </div>
      </div>
    </div>
  );
}

// ── Benchmarks tab content ─────────────────────────────────────────────────────

function BenchmarksTab() {
  const [testImages, setTestImages] = useState<File[]>([]);
  const [confThreshold, setConfThreshold] = useState(0.35);
  const [results, setResults] = useState<BenchmarkRun[]>([]);

  const { data: modelsData } = useQuery({
    queryKey: ['inference-models'],
    queryFn: () => api.get('/inference/models').then((r) => r.data),
    staleTime: 60_000,
  });

  const models: ModelInfo[] = modelsData?.models ?? modelsData ?? [];
  const activeModel = models.find((m) => m.is_active);

  const benchmarkMutation = useMutation({
    mutationFn: async () => {
      if (testImages.length === 0) {
        throw new Error('Upload at least one test image');
      }

      const times: number[] = [];

      for (const file of testImages) {
        const formData = new FormData();
        formData.append('file', file);
        formData.append('conf_threshold', String(confThreshold));

        const start = performance.now();
        await api.post('/inference/detect', formData, {
          headers: { 'Content-Type': 'multipart/form-data' },
        });
        times.push(performance.now() - start);
      }

      const mean = times.reduce((a, b) => a + b, 0) / times.length;
      const sorted = [...times].sort((a, b) => a - b);

      return {
        model: activeModel?.name ?? activeModel?.model_id ?? 'active model',
        n_images: times.length,
        mean_ms: mean,
        min_ms: sorted[0],
        max_ms: sorted[sorted.length - 1],
        fps: 1000 / mean,
        conf_threshold: confThreshold,
        timestamp: new Date().toLocaleTimeString(),
      } as BenchmarkRun;
    },
    onSuccess: (data) => {
      setResults((prev) => [data, ...prev].slice(0, 10));
    },
  });

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop: (accepted: File[]) => setTestImages(accepted),
    accept: {
      'image/jpeg': ['.jpg', '.jpeg'],
      'image/png': ['.png'],
      'image/tiff': ['.tif', '.tiff'],
    },
    maxFiles: 20,
  });

  return (
    <div className="space-y-5">
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-5">
        {/* Config panel */}
        <div className="lg:col-span-1 space-y-4">
          <div
            className="rounded-xl p-5 space-y-4"
            style={{ background: '#0d1117', border: '1px solid #21262d' }}
          >
            <h2 className="text-sm font-semibold text-white">Benchmark Config</h2>

            {/* Active model display */}
            <div>
              <label className="text-xs mb-1.5 block" style={{ color: '#8b949e' }}>
                Active detection model
              </label>
              <div
                className="px-3 py-2 rounded-lg text-sm"
                style={{
                  background: '#161b22',
                  border: '1px solid #21262d',
                  color: activeModel ? '#e6edf3' : '#484f58',
                }}
              >
                {activeModel ? activeModel.name ?? activeModel.model_id : 'No model active'}
              </div>
              {models.length > 0 && (
                <p className="text-[10px] mt-1" style={{ color: '#484f58' }}>
                  {models.length} model{models.length !== 1 ? 's' : ''} available
                </p>
              )}
            </div>

            {/* Confidence threshold */}
            <div>
              <div className="flex items-center justify-between mb-1.5">
                <label className="text-xs" style={{ color: '#8b949e' }}>
                  Confidence threshold
                </label>
                <span className="text-xs font-mono" style={{ color: '#8b949e' }}>
                  {confThreshold.toFixed(2)}
                </span>
              </div>
              <input
                type="range"
                min={0.1}
                max={0.9}
                step={0.05}
                value={confThreshold}
                onChange={(e) => setConfThreshold(Number(e.target.value))}
                className="w-full h-1.5 appearance-none rounded cursor-pointer"
                style={{ background: '#21262d' }}
              />
            </div>

            {/* Test images upload */}
            <div>
              <label className="text-xs mb-1.5 block" style={{ color: '#8b949e' }}>
                Test images (max 20)
              </label>
              <div
                {...getRootProps()}
                className="flex flex-col items-center justify-center gap-2 h-24 rounded-lg border-2 border-dashed cursor-pointer transition-all"
                style={{
                  borderColor: isDragActive ? '#3b82f6' : '#21262d',
                  background: isDragActive ? 'rgba(59,130,246,0.1)' : '#161b22',
                }}
              >
                <input {...getInputProps()} />
                <Upload className="w-5 h-5" style={{ color: '#484f58' }} />
                <p className="text-xs text-center" style={{ color: '#484f58' }}>
                  {testImages.length > 0
                    ? `${testImages.length} image${testImages.length !== 1 ? 's' : ''} selected`
                    : 'Drop test images here'}
                </p>
              </div>
            </div>

            {/* Run button */}
            <button
              onClick={() => benchmarkMutation.mutate()}
              disabled={benchmarkMutation.isPending || testImages.length === 0}
              className="w-full flex items-center justify-center gap-2 py-2.5 rounded-lg text-sm font-medium transition-colors bg-blue-600 hover:bg-blue-500 disabled:opacity-50 disabled:cursor-not-allowed text-white"
            >
              {benchmarkMutation.isPending ? (
                <>
                  <Loader2 className="w-4 h-4 animate-spin" />
                  Running…
                </>
              ) : (
                <>
                  <Play className="w-4 h-4" />
                  Run Benchmark
                </>
              )}
            </button>

            {benchmarkMutation.isError && (
              <div
                className="flex items-start gap-2 px-3 py-2.5 rounded-lg"
                style={{ background: 'rgba(239,68,68,0.1)', border: '1px solid rgba(239,68,68,0.2)' }}
              >
                <AlertTriangle className="w-4 h-4 text-red-400 mt-0.5 flex-shrink-0" />
                <p className="text-xs text-red-400">
                  {(benchmarkMutation.error as Error)?.message ?? 'Benchmark failed'}
                </p>
              </div>
            )}
          </div>

          <ReferenceTable />
        </div>

        {/* Results */}
        <div className="lg:col-span-2 space-y-4">
          {benchmarkMutation.isPending && (
            <div
              className="rounded-xl p-8 text-center"
              style={{ background: '#0d1117', border: '1px solid #21262d' }}
            >
              <div
                className="w-8 h-8 border-2 border-blue-500 border-t-transparent rounded-full mx-auto mb-3"
                style={{ animation: 'spin 1s linear infinite' }}
              />
              <p className="text-sm" style={{ color: '#484f58' }}>
                Running inference on {testImages.length} image{testImages.length !== 1 ? 's' : ''}…
              </p>
            </div>
          )}

          {results.length === 0 && !benchmarkMutation.isPending && (
            <div
              className="rounded-xl p-8 text-center"
              style={{ background: '#0d1117', border: '1px solid #21262d' }}
            >
              <BarChart3 className="w-10 h-10 mx-auto mb-3 opacity-30" style={{ color: '#484f58' }} />
              <p className="text-sm" style={{ color: '#484f58' }}>
                Upload test images and run a benchmark to see latency results
              </p>
            </div>
          )}

          {results.map((run, i) => (
            <BenchmarkCard key={`${run.timestamp}-${i}`} run={run} rank={i + 1} />
          ))}
        </div>
      </div>
    </div>
  );
}

// ── Tab bar & routing ──────────────────────────────────────────────────────────

const TABS: { id: TabId; label: string; icon: React.ElementType }[] = [
  { id: 'calibration', label: 'Calibration', icon: BarChart2 },
  { id: 'benchmarks',  label: 'Benchmarks',  icon: BarChart3 },
];

function EvaluationInner() {
  const router = useRouter();
  const searchParams = useSearchParams();
  const rawTab = searchParams.get('tab');
  const activeTab: TabId =
    rawTab === 'calibration' || rawTab === 'benchmarks' ? rawTab : 'calibration';

  const switchTab = (tab: TabId) => {
    router.replace(`/evaluation?tab=${tab}`);
  };

  return (
    <div className="flex flex-col h-full overflow-y-auto">
      {/* Page header */}
      <div
        className="flex items-center gap-2 px-5 py-3"
        style={{ borderBottom: '1px solid #21262d' }}
      >
        <FlaskConical className="w-4 h-4 text-blue-400" />
        <h1 className="text-base font-semibold text-white">Evaluation</h1>
      </div>

      {/* Tab bar */}
      <div
        className="flex gap-0 px-5"
        style={{ borderBottom: '1px solid #21262d' }}
      >
        {TABS.map(({ id, label, icon: Icon }) => (
          <button
            key={id}
            onClick={() => switchTab(id)}
            className={cn(
              'flex items-center gap-2 px-4 py-3 text-sm font-medium transition-colors',
              activeTab === id
                ? 'text-blue-400 border-b-2 border-blue-400'
                : 'text-[#8b949e] hover:text-white border-b-2 border-transparent',
            )}
          >
            <Icon className="w-4 h-4" />
            {label}
          </button>
        ))}
      </div>

      {/* Tab content */}
      <div className="flex-1 p-5">
        {activeTab === 'calibration' && <CalibrationTab />}
        {activeTab === 'benchmarks'  && <BenchmarksTab />}
      </div>
    </div>
  );
}

// ── Default export wrapped in Suspense for useSearchParams ─────────────────────

export default function EvaluationPage() {
  return (
    <Suspense fallback={
      <div className="flex items-center justify-center h-full">
        <Loader2 className="w-6 h-6 animate-spin text-blue-400" />
      </div>
    }>
      <EvaluationInner />
    </Suspense>
  );
}
