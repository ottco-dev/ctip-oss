/**
 * ReliabilityDiagram — Calibration reliability diagram (ECE plot).
 *
 * Renders:
 *   1. Bar chart: observed accuracy per bin (blue = underconfident, orange = overconfident)
 *   2. Diagonal reference line: perfect calibration (confidence = accuracy)
 *   3. Confidence histogram overlay (semi-transparent grey bars)
 *   4. ECE / MCE scalar badge
 *   5. Interpretation text badge
 *
 * Data comes from POST /analytics/calibration response.
 *
 * Scientific basis:
 *   Guo et al. (2017). On Calibration of Modern Neural Networks. ICML 2017.
 */

'use client';

import { useMemo } from 'react';
import { cn } from '@/lib/utils';

// ── Types ──────────────────────────────────────────────────────────────────

export interface BinStats {
  bin_index: number;
  confidence_lower: number;
  confidence_upper: number;
  mean_confidence: number;
  accuracy: number;
  count: number;
  gap: number;
  abs_gap: number;
  weight: number;
  is_overconfident: boolean;
  is_empty: boolean;
}

export interface ReliabilityDiagramProps {
  /** ECE value [0, 1]. */
  ece: number;
  /** MCE value [0, 1]. */
  mce: number;
  /** Per-bin statistics (from CalibrationResponse.bins). */
  bins: BinStats[];
  /** Total sample count. */
  totalSamples: number;
  /** True = model overall overconfident. */
  isOverconfident: boolean;
  /** Human-readable quality assessment. */
  interpretation: string;
  /** Canvas width in px (default 480). */
  width?: number;
  /** Canvas height in px (default 320). */
  height?: number;
  className?: string;
}

// ── Constants ──────────────────────────────────────────────────────────────

const MARGIN = { top: 20, right: 24, bottom: 48, left: 48 };

const COLOR_OVER   = '#f97316';  // orange-500 — overconfident bin
const COLOR_UNDER  = '#3b82f6';  // blue-500   — underconfident bin
const COLOR_HIST   = 'rgba(156,163,175,0.25)';  // grey-400 @ 25% — histogram
const COLOR_DIAG   = '#4ade80';  // green-400  — perfect calibration line
const COLOR_GAP    = 'rgba(239,68,68,0.15)';    // red fill between bar and diagonal

// ── Component ──────────────────────────────────────────────────────────────

export function ReliabilityDiagram({
  ece,
  mce,
  bins,
  totalSamples,
  isOverconfident,
  interpretation,
  width = 480,
  height = 320,
  className,
}: ReliabilityDiagramProps) {
  const plotW = width - MARGIN.left - MARGIN.right;
  const plotH = height - MARGIN.top - MARGIN.bottom;
  const numBins = bins.length;
  const binWidth = plotW / numBins;
  const maxCount = useMemo(() => Math.max(...bins.map((b) => b.count), 1), [bins]);

  // Scale helpers
  const xScale = (v: number) => v * plotW;                  // [0,1] → px
  const yScale = (v: number) => plotH - v * plotH;           // [0,1] → px (inverted)

  // Diagonal ticks
  const ticks = [0, 0.2, 0.4, 0.6, 0.8, 1.0];

  // ECE quality colour
  const eceColor =
    ece < 0.02 ? 'text-status-success' :
    ece < 0.05 ? 'text-accent' :
    ece < 0.10 ? 'text-status-warning' :
    'text-status-error';

  return (
    <div className={cn('space-y-3', className)}>
      {/* Metric badges */}
      <div className="flex flex-wrap items-center gap-3">
        <div className="flex items-center gap-1.5">
          <span className="text-xs text-text-muted">ECE</span>
          <span className={cn('font-mono text-sm font-semibold', eceColor)}>
            {(ece * 100).toFixed(2)}%
          </span>
        </div>
        <div className="flex items-center gap-1.5">
          <span className="text-xs text-text-muted">MCE</span>
          <span className="font-mono text-sm font-semibold text-text-primary">
            {(mce * 100).toFixed(2)}%
          </span>
        </div>
        <div className="flex items-center gap-1.5">
          <span className="text-xs text-text-muted">n</span>
          <span className="font-mono text-sm text-text-secondary">
            {totalSamples.toLocaleString()}
          </span>
        </div>
        <span
          className={cn(
            'text-xs px-2 py-0.5 rounded-full font-medium',
            isOverconfident
              ? 'bg-orange-500/15 text-orange-400'
              : 'bg-blue-500/15 text-blue-400',
          )}
        >
          {isOverconfident ? '↑ Overconfident' : '↓ Underconfident'}
        </span>
      </div>

      {/* SVG diagram */}
      <svg
        width={width}
        height={height}
        viewBox={`0 0 ${width} ${height}`}
        className="overflow-visible"
        aria-label="Reliability diagram"
      >
        <g transform={`translate(${MARGIN.left},${MARGIN.top})`}>
          {/* ── Background grid ──────────────────────────────────────────── */}
          {ticks.map((t) => (
            <line
              key={t}
              x1={0}
              x2={plotW}
              y1={yScale(t)}
              y2={yScale(t)}
              stroke="#374151"
              strokeWidth={0.5}
              strokeDasharray="3,3"
            />
          ))}
          {ticks.map((t) => (
            <line
              key={t}
              x1={xScale(t)}
              x2={xScale(t)}
              y1={0}
              y2={plotH}
              stroke="#374151"
              strokeWidth={0.5}
              strokeDasharray="3,3"
            />
          ))}

          {/* ── Confidence histogram (background) ───────────────────────── */}
          {bins.map((b, i) => {
            const hBarH = (b.count / maxCount) * plotH;
            return (
              <rect
                key={`hist-${i}`}
                x={i * binWidth + 1}
                y={plotH - hBarH}
                width={binWidth - 2}
                height={hBarH}
                fill={COLOR_HIST}
              />
            );
          })}

          {/* ── Gap fill (between bar and diagonal) ─────────────────────── */}
          {bins.map((b, i) => {
            if (b.is_empty) return null;
            const x0 = i * binWidth + binWidth * 0.05;
            const x1 = x0 + binWidth * 0.9;
            const diagY = yScale(b.mean_confidence);
            const accY  = yScale(b.accuracy);
            return (
              <rect
                key={`gap-${i}`}
                x={x0}
                y={Math.min(diagY, accY)}
                width={x1 - x0}
                height={Math.abs(diagY - accY)}
                fill={COLOR_GAP}
              />
            );
          })}

          {/* ── Accuracy bars ────────────────────────────────────────────── */}
          {bins.map((b, i) => {
            if (b.is_empty) return null;
            const barH = b.accuracy * plotH;
            const fill = b.is_overconfident ? COLOR_OVER : COLOR_UNDER;
            return (
              <g key={`bar-${i}`}>
                <rect
                  x={i * binWidth + binWidth * 0.1}
                  y={yScale(b.accuracy)}
                  width={binWidth * 0.8}
                  height={barH}
                  fill={fill}
                  opacity={0.85}
                  rx={1}
                />
                <title>
                  Bin {i + 1}: conf={b.mean_confidence.toFixed(3)} acc={b.accuracy.toFixed(3)} n={b.count}
                </title>
              </g>
            );
          })}

          {/* ── Perfect calibration diagonal ────────────────────────────── */}
          <line
            x1={xScale(0)}
            y1={yScale(0)}
            x2={xScale(1)}
            y2={yScale(1)}
            stroke={COLOR_DIAG}
            strokeWidth={1.5}
            strokeDasharray="6,3"
          />

          {/* ── Axes ─────────────────────────────────────────────────────── */}
          {/* X axis */}
          <line x1={0} x2={plotW} y1={plotH} y2={plotH} stroke="#6b7280" strokeWidth={1} />
          {/* Y axis */}
          <line x1={0} x2={0} y1={0} y2={plotH} stroke="#6b7280" strokeWidth={1} />

          {/* X ticks + labels */}
          {ticks.map((t) => (
            <g key={`xt-${t}`}>
              <line x1={xScale(t)} x2={xScale(t)} y1={plotH} y2={plotH + 4} stroke="#6b7280" strokeWidth={1} />
              <text
                x={xScale(t)}
                y={plotH + 16}
                textAnchor="middle"
                fill="#9ca3af"
                fontSize={10}
              >
                {t.toFixed(1)}
              </text>
            </g>
          ))}

          {/* Y ticks + labels */}
          {ticks.map((t) => (
            <g key={`yt-${t}`}>
              <line x1={-4} x2={0} y1={yScale(t)} y2={yScale(t)} stroke="#6b7280" strokeWidth={1} />
              <text
                x={-8}
                y={yScale(t) + 4}
                textAnchor="end"
                fill="#9ca3af"
                fontSize={10}
              >
                {t.toFixed(1)}
              </text>
            </g>
          ))}

          {/* Axis labels */}
          <text
            x={plotW / 2}
            y={plotH + 40}
            textAnchor="middle"
            fill="#6b7280"
            fontSize={11}
          >
            Confidence
          </text>
          <text
            x={-plotH / 2}
            y={-36}
            textAnchor="middle"
            fill="#6b7280"
            fontSize={11}
            transform="rotate(-90)"
          >
            Accuracy
          </text>
        </g>
      </svg>

      {/* Colour legend */}
      <div className="flex flex-wrap items-center gap-4 text-xs text-text-muted">
        <div className="flex items-center gap-1.5">
          <span className="w-3 h-3 rounded-sm" style={{ background: COLOR_OVER }} />
          Overconfident bin
        </div>
        <div className="flex items-center gap-1.5">
          <span className="w-3 h-3 rounded-sm" style={{ background: COLOR_UNDER }} />
          Underconfident bin
        </div>
        <div className="flex items-center gap-1.5">
          <span
            className="w-6 border-t-2 border-dashed"
            style={{ borderColor: COLOR_DIAG }}
          />
          Perfect calibration
        </div>
        <div className="flex items-center gap-1.5">
          <span className="w-3 h-3 rounded-sm" style={{ background: 'rgba(156,163,175,0.6)' }} />
          Sample count
        </div>
      </div>

      {/* Interpretation */}
      <p className="text-xs text-text-muted leading-relaxed border-l-2 border-border pl-3">
        {interpretation}
      </p>
    </div>
  );
}
