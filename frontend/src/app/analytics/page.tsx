'use client';

/**
 * Analytics page — Model calibration analysis.
 *
 * Allows the user to:
 *   1. Paste raw confidence scores + correctness labels to compute ECE live.
 *   2. Enter a MLflow run ID to load prediction artifacts and compute calibration.
 *
 * Renders the ReliabilityDiagram component with full per-bin data.
 *
 * Scientific basis: Guo et al. (2017). On Calibration of Modern Neural Networks. ICML 2017.
 */

import { useState } from 'react';
import { useMutation } from '@tanstack/react-query';
import { BarChart2, AlertTriangle, Info, Upload } from 'lucide-react';
import { api } from '@/lib/api';
import { ReliabilityDiagram, type BinStats } from '@/components/charts/ReliabilityDiagram';
import { cn } from '@/lib/utils';

// ── Types ──────────────────────────────────────────────────────────────────

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

// ── Helpers ────────────────────────────────────────────────────────────────

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

// ── Main component ─────────────────────────────────────────────────────────

export default function AnalyticsPage() {
  const [mode, setMode] = useState<'direct' | 'mlflow'>('direct');
  const [numBins, setNumBins] = useState(10);

  // Direct mode
  const [confText, setConfText] = useState('');
  const [correctText, setCorrectText] = useState('');

  // MLflow mode
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
    <div className="space-y-6 max-w-[1200px] mx-auto">
      {/* Header */}
      <div>
        <h1 className="text-xl font-semibold text-text-primary flex items-center gap-2">
          <BarChart2 className="w-5 h-5 text-accent" />
          Model Calibration Analytics
        </h1>
        <p className="text-sm text-text-secondary mt-1 max-w-2xl">
          Compute Expected Calibration Error (ECE) and reliability diagrams to assess
          whether model confidence scores match observed accuracy.{' '}
          <span className="text-text-muted">
            Reference: Guo et al. (2017). On Calibration of Modern Neural Networks. ICML 2017.
          </span>
        </p>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* ── Input panel ───────────────────────────────────────────────── */}
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

                {/* Example data button */}
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

            {/* Parse error */}
            {parseError && (
              <div className="flex items-start gap-2 text-xs text-status-error bg-status-error/10 rounded p-2 mt-3">
                <AlertTriangle className="w-3.5 h-3.5 flex-shrink-0 mt-0.5" />
                {parseError}
              </div>
            )}

            {/* API error */}
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

          {/* ── Per-bin table ──────────────────────────────────────────── */}
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

        {/* ── Diagram panel ─────────────────────────────────────────────── */}
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
