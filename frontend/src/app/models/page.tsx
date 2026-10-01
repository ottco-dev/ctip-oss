"use client";

import React, { useState } from "react";
import { useQuery, useMutation, useQueryClient } from "@tanstack/react-query";
import {
  BoxSelect,
  Download,
  CheckCircle2,
  Cpu,
  ChevronDown,
  ChevronRight,
  ExternalLink,
  RefreshCw,
  Loader2,
  Package,
} from "lucide-react";
import { api } from "@/lib/api";
import { cn, formatBytes } from "@/lib/utils";

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

interface ModelVersion {
  id: number;
  name: string;
  model_type: string;
  framework: string;
  variant: string;
  file_path: string | null;
  file_size_bytes: number | null;
  vram_required_gb: number | null;
  metrics: Record<string, number>;
  created_at: string;
  is_downloaded: boolean;
  is_active: boolean;
  description?: string;
  source_url?: string;
}

// ---------------------------------------------------------------------------
// VRAM indicator bar
// ---------------------------------------------------------------------------

function VramBar({ requiredGb, totalGb = 8 }: { requiredGb: number; totalGb?: number }) {
  const pct = Math.min(100, (requiredGb / totalGb) * 100);
  const color =
    pct > 85 ? "bg-red-500" : pct > 65 ? "bg-yellow-500" : "bg-accent";

  return (
    <div className="flex items-center gap-2">
      <div className="flex-1 h-1.5 rounded-full overflow-hidden" style={{ background: 'var(--border)' }}>
        <div className={cn("h-full transition-all", color)} style={{ width: `${pct}%` }} />
      </div>
      <span className="text-[10px] font-mono w-12 text-right" style={{ color: 'var(--text-muted)' }}>
        {requiredGb.toFixed(1)} GB
      </span>
    </div>
  );
}

// ---------------------------------------------------------------------------
// Metric badge
// ---------------------------------------------------------------------------

function MetricBadge({ label, value }: { label: string; value: number }) {
  return (
    <div
      className="flex flex-col items-center px-2.5 py-1.5 rounded"
      style={{ background: 'var(--panel)', border: '1px solid var(--border)' }}
    >
      <span className="text-[9px] uppercase tracking-wide" style={{ color: 'var(--text-muted)' }}>
        {label}
      </span>
      <span className="text-sm font-bold font-mono text-text-primary mt-0.5">
        {(value * 100).toFixed(1)}
      </span>
    </div>
  );
}

// ---------------------------------------------------------------------------
// Model card
// ---------------------------------------------------------------------------

function ModelCard({
  model,
  onDownload,
  onActivate,
  isDownloading,
  isActivating,
}: {
  model: ModelVersion;
  onDownload: (id: number) => void;
  onActivate: (id: number) => void;
  isDownloading: boolean;
  isActivating: boolean;
}) {
  const [expanded, setExpanded] = useState(false);

  const typeColor: Record<string, string> = {
    detection: "text-accent-text bg-accent/10 border-accent/20",
    segmentation: "text-purple-400 bg-purple-500/10 border-purple-500/20",
    maturity: "text-amber-400 bg-amber-500/10 border-amber-500/20",
    morphology: "text-green-400 bg-green-500/10 border-green-500/20",
    vlm: "text-pink-400 bg-pink-500/10 border-pink-500/20",
  };

  const tc = typeColor[model.model_type] ?? "text-gray-400 bg-gray-500/10 border-gray-500/20";

  return (
    <div
      className={cn("rounded-xl border transition-all")}
      style={{
        border: model.is_active ? '1px solid rgba(47,158,58,0.4)' : '1px solid var(--border)',
        background: model.is_active ? 'rgba(47,158,58,0.05)' : 'var(--surface)',
      }}
    >
      {/* Header */}
      <div className="flex items-center gap-3 px-4 py-3">
        {model.is_active && (
          <div className="w-1.5 h-1.5 rounded-full bg-accent-hover flex-shrink-0" />
        )}

        <div className="flex-1 min-w-0">
          <div className="flex items-center gap-2 flex-wrap">
            <h3 className="text-sm font-semibold text-text-primary">{model.name}</h3>
            <span className={cn("text-[10px] px-1.5 py-0.5 rounded border font-medium capitalize", tc)}>
              {model.model_type}
            </span>
            <span className="text-[10px]" style={{ color: 'var(--text-muted)' }}>
              {model.variant} · {model.framework}
            </span>
          </div>
          {model.description && (
            <p className="text-xs mt-0.5 truncate" style={{ color: 'var(--text-muted)' }}>
              {model.description}
            </p>
          )}
        </div>

        {model.vram_required_gb !== null && (
          <div className="hidden sm:block w-36">
            <VramBar requiredGb={model.vram_required_gb} />
          </div>
        )}

        {Object.keys(model.metrics).length > 0 && (
          <div className="hidden lg:flex items-center gap-1.5">
            {model.metrics.map50 !== undefined && (
              <MetricBadge label="mAP50" value={model.metrics.map50} />
            )}
            {model.metrics.precision !== undefined && (
              <MetricBadge label="P" value={model.metrics.precision} />
            )}
          </div>
        )}

        <div className="flex items-center gap-1">
          {!model.is_downloaded ? (
            <button
              onClick={() => onDownload(model.id)}
              disabled={isDownloading}
              className="flex items-center gap-1 px-2.5 py-1.5 rounded-lg text-xs font-medium
                bg-accent hover:bg-accent-hover text-white transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
            >
              {isDownloading ? (
                <Loader2 className="w-3.5 h-3.5 animate-spin" />
              ) : (
                <Download className="w-3.5 h-3.5" />
              )}
              {isDownloading ? "Downloading…" : "Download"}
            </button>
          ) : !model.is_active ? (
            <button
              onClick={() => onActivate(model.id)}
              disabled={isActivating}
              className="flex items-center gap-1 px-2.5 py-1.5 rounded-lg text-xs font-medium
                border transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
              style={{ borderColor: 'var(--border)', color: 'var(--text-secondary)' }}
            >
              {isActivating ? (
                <Loader2 className="w-3.5 h-3.5 animate-spin" />
              ) : (
                <CheckCircle2 className="w-3.5 h-3.5" />
              )}
              {isActivating ? "Activating…" : "Activate"}
            </button>
          ) : (
            <span className="flex items-center gap-1 px-2.5 py-1.5 text-xs text-accent-text font-medium">
              <CheckCircle2 className="w-3.5 h-3.5" />
              Active
            </span>
          )}

          <button
            onClick={() => setExpanded((v) => !v)}
            className="p-1.5 rounded transition-colors"
            style={{ color: 'var(--text-muted)' }}
          >
            {expanded ? (
              <ChevronDown className="w-4 h-4" />
            ) : (
              <ChevronRight className="w-4 h-4" />
            )}
          </button>
        </div>
      </div>

      {/* Expanded details */}
      {expanded && (
        <div
          className="px-4 pb-4 pt-3 space-y-3"
          style={{ borderTop: '1px solid var(--border)' }}
        >
          <div className="grid grid-cols-2 sm:grid-cols-4 gap-3 text-xs">
            {[
              { label: "Framework", value: model.framework },
              {
                label: "VRAM Required",
                value: model.vram_required_gb ? `${model.vram_required_gb} GB` : "—",
              },
              {
                label: "File Size",
                value: model.file_size_bytes ? formatBytes(model.file_size_bytes) : "—",
              },
              { label: "Status", value: model.is_downloaded ? "Downloaded" : "Not downloaded" },
            ].map(({ label, value }) => (
              <div key={label} className="space-y-0.5">
                <p className="text-[10px] uppercase tracking-wide" style={{ color: 'var(--text-muted)' }}>{label}</p>
                <p style={{ color: 'var(--text-secondary)' }}>{value}</p>
              </div>
            ))}
          </div>

          {Object.keys(model.metrics).length > 0 && (
            <div>
              <p className="text-[10px] uppercase tracking-wide mb-2" style={{ color: 'var(--text-muted)' }}>
                Metrics
              </p>
              <div className="flex flex-wrap gap-2">
                {Object.entries(model.metrics).map(([key, val]) => (
                  <div
                    key={key}
                    className="px-2 py-1 rounded text-xs"
                    style={{ background: 'var(--panel)', border: '1px solid var(--border)' }}
                  >
                    <span style={{ color: 'var(--text-muted)' }}>{key}: </span>
                    <span className="font-mono text-text-primary">
                      {typeof val === "number" && val <= 1
                        ? (val * 100).toFixed(1) + "%"
                        : val}
                    </span>
                  </div>
                ))}
              </div>
            </div>
          )}

          {model.file_path && (
            <div>
              <p className="text-[10px] uppercase tracking-wide mb-1" style={{ color: 'var(--text-muted)' }}>Path</p>
              <code
                className="text-[10px] text-green-300/80 px-2 py-1 rounded block truncate"
                style={{ background: 'var(--panel)' }}
              >
                {model.file_path}
              </code>
            </div>
          )}

          {model.source_url && (
            <a
              href={model.source_url}
              target="_blank"
              rel="noopener noreferrer"
              className="inline-flex items-center gap-1 text-xs text-accent-text hover:text-accent-text"
            >
              <ExternalLink className="w-3 h-3" />
              Source / Weights
            </a>
          )}
        </div>
      )}
    </div>
  );
}

// ---------------------------------------------------------------------------
// Main models page
// ---------------------------------------------------------------------------

export default function ModelsPage() {
  const [filterType, setFilterType] = useState<string>("all");
  const [downloadingId, setDownloadingId] = useState<number | null>(null);
  const [activatingId, setActivatingId] = useState<number | null>(null);
  const queryClient = useQueryClient();

  const {
    data: models = [],
    isLoading,
    isError,
    error,
    refetch,
  } = useQuery<ModelVersion[]>({
    queryKey: ["models"],
    queryFn: () => api.get("/models").then((r) => r.data),
    staleTime: 60_000,
  });

  const downloadMutation = useMutation({
    mutationFn: (id: number) => api.post(`/models/${id}/download`).then((r) => r.data),
    onMutate: (id) => setDownloadingId(id),
    onSettled: () => setDownloadingId(null),
    onSuccess: () => queryClient.invalidateQueries({ queryKey: ["models"] }),
  });

  const activateMutation = useMutation({
    mutationFn: (id: number) => api.put(`/models/${id}/activate`).then((r) => r.data),
    onMutate: (id) => setActivatingId(id),
    onSettled: () => setActivatingId(null),
    onSuccess: () => queryClient.invalidateQueries({ queryKey: ["models"] }),
  });

  const filteredModels =
    filterType === "all" ? models : models.filter((m) => m.model_type === filterType);

  const types = ["all", ...Array.from(new Set(models.map((m) => m.model_type)))];

  return (
    <div className="flex flex-col h-full overflow-y-auto">
      {/* Header */}
      <div
        className="flex items-center justify-between px-5 py-3"
        style={{ borderBottom: '1px solid var(--border)' }}
      >
        <div className="flex items-center gap-2">
          <BoxSelect className="w-4 h-4 text-accent-text" />
          <h1 className="text-base font-semibold text-text-primary">Model Registry</h1>
        </div>
        <div className="flex items-center gap-3">
          <div
            className="flex gap-1 p-0.5 rounded-lg"
            style={{ background: 'var(--panel)', border: '1px solid var(--border)' }}
          >
            {types.map((t) => (
              <button
                key={t}
                onClick={() => setFilterType(t)}
                className={cn(
                  "px-2.5 py-1 rounded text-xs font-medium capitalize transition-all",
                  filterType === t
                    ? "text-text-primary"
                    : "hover:text-text-primary"
                )}
                style={{
                  background: filterType === t ? 'var(--surface)' : 'transparent',
                  color: filterType === t ? 'var(--text-primary)' : 'var(--text-muted)',
                }}
              >
                {t}
              </button>
            ))}
          </div>
          <button
            onClick={() => refetch()}
            className="p-1.5 rounded transition-colors"
            style={{ color: 'var(--text-muted)' }}
          >
            <RefreshCw className="w-4 h-4" />
          </button>
        </div>
      </div>

      {/* VRAM summary */}
      <div
        className="px-5 py-3"
        style={{ borderBottom: '1px solid var(--border)', background: 'var(--panel)' }}
      >
        <div className="flex items-center gap-6 text-xs">
          <div className="flex items-center gap-1.5">
            <Cpu className="w-3.5 h-3.5 text-accent-text" />
            <span style={{ color: 'var(--text-muted)' }}>RTX 4060 — 8 GB VRAM</span>
          </div>
          <div className="flex items-center gap-1.5">
            <span className="w-2 h-2 rounded-full bg-accent-hover" />
            <span className="text-accent-text">Safe (≤ 5.2 GB)</span>
          </div>
          <div className="flex items-center gap-1.5">
            <span className="w-2 h-2 rounded-full bg-yellow-400" />
            <span className="text-yellow-400">Caution (5.2–6.8 GB)</span>
          </div>
          <div className="flex items-center gap-1.5">
            <span className="w-2 h-2 rounded-full bg-red-400" />
            <span className="text-red-400">Too large (&gt; 6.8 GB)</span>
          </div>
          <span className="ml-auto" style={{ color: 'var(--text-muted)' }}>
            Only 1 GPU task runs at a time (asyncio.Semaphore)
          </span>
        </div>
      </div>

      {/* Content */}
      <div className="flex-1 px-5 py-4 space-y-3">
        {/* Loading */}
        {isLoading && (
          <div className="flex items-center justify-center py-16">
            <Loader2 className="w-6 h-6 text-accent-text animate-spin" />
          </div>
        )}

        {/* Error */}
        {isError && (
          <div
            className="flex items-start gap-3 px-4 py-3 rounded-lg"
            style={{ background: 'rgba(239,68,68,0.1)', border: '1px solid rgba(239,68,68,0.2)' }}
          >
            <div>
              <p className="text-sm font-medium text-red-400">Failed to load models</p>
              <p className="text-xs mt-0.5" style={{ color: 'rgba(252,165,165,0.7)' }}>
                {(error as Error)?.message ?? "Unknown error"}
              </p>
            </div>
          </div>
        )}

        {/* Empty state */}
        {!isLoading && !isError && models.length === 0 && (
          <div className="text-center py-16" style={{ color: 'var(--text-muted)' }}>
            <Package className="w-10 h-10 mx-auto mb-3 opacity-30" />
            <p className="text-sm font-medium">No models registered yet</p>
            <p className="text-xs mt-1">Models registered via the API will appear here</p>
          </div>
        )}

        {/* Filtered empty */}
        {!isLoading && !isError && models.length > 0 && filteredModels.length === 0 && (
          <div className="text-center py-16" style={{ color: 'var(--text-muted)' }}>
            <BoxSelect className="w-8 h-8 mx-auto mb-3 opacity-30" />
            <p className="text-sm">No {filterType} models in registry</p>
          </div>
        )}

        {/* Model list */}
        {filteredModels.map((model) => (
          <ModelCard
            key={model.id}
            model={model}
            onDownload={(id) => downloadMutation.mutate(id)}
            onActivate={(id) => activateMutation.mutate(id)}
            isDownloading={downloadingId === model.id}
            isActivating={activatingId === model.id}
          />
        ))}
      </div>
    </div>
  );
}
