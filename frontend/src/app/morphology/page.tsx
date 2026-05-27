'use client';

import React, { useCallback, useRef, useState } from 'react';
import { useMutation } from '@tanstack/react-query';
import { useDropzone } from 'react-dropzone';
import { Upload, Loader2, FlaskConical, Microscope, BarChart3, Info } from 'lucide-react';
import { uploadFile } from '@/lib/api';
import { cn, formatConfidence, getConfidenceColor } from '@/lib/utils';

// ─────────────────────────────────────────────────────────────────────────────
// Types
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
// Stage colour map
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

// ─────────────────────────────────────────────────────────────────────────────
// Metric row helper
// ─────────────────────────────────────────────────────────────────────────────

function MetricRow({ label, value, unit = '' }: { label: string; value: number | null | undefined; unit?: string }) {
  if (value == null) return null;
  return (
    <div className="flex items-center justify-between py-1.5 border-b border-border last:border-0">
      <span className="text-xs text-text-secondary">{label}</span>
      <span className="text-xs font-mono text-text-primary">
        {typeof value === 'number' ? value.toFixed(3) : value}{unit}
      </span>
    </div>
  );
}

// ─────────────────────────────────────────────────────────────────────────────
// Main page
// ─────────────────────────────────────────────────────────────────────────────

export default function MorphologyPage() {
  const imageRef = useRef<HTMLImageElement>(null);
  const [imageUrl, setImageUrl] = useState<string | null>(null);
  const [imageFile, setImageFile] = useState<File | null>(null);
  const [morphResult, setMorphResult] = useState<MorphologyResponse | null>(null);
  const [maturityResult, setMaturityResult] = useState<MaturityResponse | null>(null);

  // ── Mutations ──────────────────────────────────────────────────────────────

  const morphMutation = useMutation({
    mutationFn: (file: File) =>
      uploadFile('/morphology/instance', file, { include_geometric: 'true', include_stalk: 'true' }).then(
        (r) => r.data as MorphologyResponse,
      ),
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

      // Preview
      const reader = new FileReader();
      reader.onload = (e) => setImageUrl(e.target?.result as string);
      reader.readAsDataURL(file);

      // Fire both analyses in parallel
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

      {/* Scientific caveat */}
      <div className="scientific-caveat">
        <Info className="w-4 h-4 shrink-0 mt-0.5" />
        <div>
          <strong className="block mb-0.5">Scientific Note</strong>
          Maturity stage describes <em>optical color state</em> only (clear → cloudy → amber → degraded).
          This does not quantify cannabinoid content. Chromatography (GC-MS, HPLC) is required
          for precise biochemical measurement.
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
              isDragActive ? 'border-accent bg-accent/5' : 'border-border hover:border-accent/50',
            )}
          >
            <input {...getInputProps()} />
            <Upload className="w-8 h-8 mx-auto text-text-muted mb-2" />
            <p className="text-sm text-text-secondary">
              {isDragActive ? 'Drop here' : 'Drop a trichome image or click to browse'}
            </p>
            <p className="text-xs text-text-muted mt-1">JPEG · PNG · TIFF · BMP · up to 100 MB</p>
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
                {morphMutation.isError && <div>Morphology: {String(morphMutation.error)}</div>}
                {maturityMutation.isError && <div>Maturity: {String(maturityMutation.error)}</div>}
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
                    style={{ backgroundColor: TYPE_COLORS[morphResult.morphology_type] ?? '#6b7280' }}
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
                      <span className={morphResult.stalk.has_visible_stalk ? 'text-status-success' : 'text-text-muted'}>
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
                      <MetricRow label="Area" value={morphResult.geometric_features.area_px} unit=" px²" />
                      <MetricRow label="Perimeter" value={morphResult.geometric_features.perimeter_px} unit=" px" />
                      <MetricRow label="Circularity" value={morphResult.geometric_features.circularity} />
                      <MetricRow label="Eccentricity" value={morphResult.geometric_features.eccentricity} />
                      <MetricRow label="Solidity" value={morphResult.geometric_features.solidity} />
                      <MetricRow label="Aspect ratio" value={morphResult.geometric_features.aspect_ratio} />
                      <MetricRow label="Extent" value={morphResult.geometric_features.extent} />
                      <MetricRow label="Major axis" value={morphResult.geometric_features.major_axis_px} unit=" px" />
                      <MetricRow label="Minor axis" value={morphResult.geometric_features.minor_axis_px} unit=" px" />
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
                    style={{ backgroundColor: STAGE_COLORS[maturityResult.stage] ?? '#6b7280' }}
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
                      style={{ backgroundColor: STAGE_COLORS[s], color: s === 'cloudy' ? '#111' : undefined }}
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
                      <MetricRow label="Hue mean" value={maturityResult.color_features.hue_mean} unit="°" />
                      <MetricRow label="Saturation" value={maturityResult.color_features.saturation_mean} />
                      <MetricRow label="Brightness" value={maturityResult.color_features.value_mean} />
                      <MetricRow label="Amber ratio" value={maturityResult.color_features.amber_ratio} />
                      <MetricRow label="Translucency" value={maturityResult.color_features.translucency_score} />
                    </div>
                  </details>
                )}

                {maturityResult.scientific_note && (
                  <p className="text-[11px] text-text-muted italic">{maturityResult.scientific_note}</p>
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
    </div>
  );
}
