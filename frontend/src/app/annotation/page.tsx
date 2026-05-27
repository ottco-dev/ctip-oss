"use client";

import React, { useState } from "react";
import { useQuery, useMutation, useQueryClient } from "@tanstack/react-query";
import {
  Tag,
  Clock,
  CheckCircle2,
  XCircle,
  AlertTriangle,
  Loader2,
  BarChart3,
  RefreshCw,
  ChevronRight,
  Brain,
  FlaskConical,
} from "lucide-react";
import { api } from "@/lib/api";
import { cn, timeAgo, formatConfidence } from "@/lib/utils";

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

interface ReviewItem {
  id: number;
  image_path?: string;
  filename?: string;
  maturity_stage?: string;
  clear_fraction?: number;
  cloudy_fraction?: number;
  amber_fraction?: number;
  vlm_confidence?: number;
  confidence?: number;
  hallucination_flags?: string[];
  review_priority?: number;
  priority?: number;
  queued_at?: string;
  created_at?: string;
  vlm_backend?: string;
  status?: string;
}

interface AnnotationStats {
  total_pending?: number;
  pending_count?: number;
  total_reviewed?: number;
  reviewed_count?: number;
  throughput_per_hour?: number;
  avg_priority?: number;
  high_priority_count?: number;
}

interface AnnotationJob {
  id?: string;
  job_uuid?: string;
  job_type?: string;
  type?: string;
  status: string;
  progress?: number;
  processed_items?: number;
  total_items?: number;
}

// Normalize the item fields from various response shapes
function normalizeItem(item: ReviewItem): ReviewItem {
  return {
    ...item,
    filename: item.filename ?? item.image_path?.split("/").pop() ?? `Item ${item.id}`,
    vlm_confidence: item.vlm_confidence ?? item.confidence ?? 0,
    review_priority: item.review_priority ?? item.priority ?? 0,
    queued_at: item.queued_at ?? item.created_at ?? new Date().toISOString(),
    hallucination_flags: item.hallucination_flags ?? [],
    clear_fraction: item.clear_fraction ?? 0,
    cloudy_fraction: item.cloudy_fraction ?? 0,
    amber_fraction: item.amber_fraction ?? 0,
  };
}

// ---------------------------------------------------------------------------
// Priority badge
// ---------------------------------------------------------------------------

function PriorityBadge({ priority }: { priority: number }) {
  const configs = [
    { label: "Low", bg: "rgba(107,114,128,0.2)", color: "#9ca3af" },
    { label: "Med", bg: "rgba(59,130,246,0.2)", color: "#60a5fa" },
    { label: "High", bg: "rgba(234,179,8,0.2)", color: "#eab308" },
    { label: "Crit", bg: "rgba(239,68,68,0.2)", color: "#ef4444" },
  ];
  const config = configs[priority] ?? configs[0];

  return (
    <span
      className="text-[10px] px-1.5 py-0.5 rounded font-medium uppercase"
      style={{ background: config.bg, color: config.color }}
    >
      {config.label}
    </span>
  );
}

// ---------------------------------------------------------------------------
// Fraction bar
// ---------------------------------------------------------------------------

function FractionBar({ clear, cloudy, amber }: { clear: number; cloudy: number; amber: number }) {
  return (
    <div className="flex h-1.5 rounded-full overflow-hidden gap-[1px] w-full">
      <div className="bg-blue-400" style={{ width: `${clear * 100}%` }} />
      <div className="bg-gray-300" style={{ width: `${cloudy * 100}%` }} />
      <div className="bg-amber-400" style={{ width: `${amber * 100}%` }} />
    </div>
  );
}

// ---------------------------------------------------------------------------
// Review row
// ---------------------------------------------------------------------------

function ReviewRow({
  item,
  onApprove,
  onReject,
}: {
  item: ReviewItem;
  onApprove: (id: number) => void;
  onReject: (id: number) => void;
}) {
  const norm = normalizeItem(item);
  const conf = norm.vlm_confidence ?? 0;
  const confColor = conf >= 0.7 ? "#22c55e" : conf >= 0.5 ? "#eab308" : "#ef4444";

  return (
    <div
      className="group flex items-center gap-3 px-4 py-3 rounded-xl transition-all"
      style={{ background: '#0d1117', border: '1px solid #21262d' }}
    >
      <PriorityBadge priority={norm.review_priority ?? 0} />

      <div className="flex-1 min-w-0">
        <p className="text-sm truncate" style={{ color: '#8b949e' }}>{norm.filename}</p>
        <div className="flex items-center gap-3 mt-1">
          {norm.maturity_stage && (
            <span
              className="text-[10px] px-1.5 py-0.5 rounded font-medium capitalize"
              style={{
                background:
                  norm.maturity_stage === "amber"
                    ? "rgba(245,158,11,0.2)"
                    : norm.maturity_stage === "cloudy"
                    ? "rgba(107,114,128,0.2)"
                    : norm.maturity_stage === "clear"
                    ? "rgba(59,130,246,0.2)"
                    : "rgba(168,85,247,0.2)",
                color:
                  norm.maturity_stage === "amber"
                    ? "#f59e0b"
                    : norm.maturity_stage === "cloudy"
                    ? "#9ca3af"
                    : norm.maturity_stage === "clear"
                    ? "#60a5fa"
                    : "#a855f7",
              }}
            >
              {norm.maturity_stage}
            </span>
          )}
          {norm.vlm_backend && (
            <span className="text-[10px]" style={{ color: '#484f58' }}>{norm.vlm_backend}</span>
          )}
          {(norm.hallucination_flags?.length ?? 0) > 0 && (
            <div className="flex items-center gap-0.5 text-yellow-400">
              <AlertTriangle className="w-3 h-3" />
              <span className="text-[10px]">{norm.hallucination_flags!.length} flags</span>
            </div>
          )}
        </div>
      </div>

      {/* Fraction bar */}
      {(norm.clear_fraction! + norm.cloudy_fraction! + norm.amber_fraction!) > 0 && (
        <div className="w-24 space-y-0.5">
          <FractionBar
            clear={norm.clear_fraction!}
            cloudy={norm.cloudy_fraction!}
            amber={norm.amber_fraction!}
          />
          <div className="flex justify-between text-[9px] font-mono" style={{ color: '#484f58' }}>
            <span>{Math.round((norm.clear_fraction ?? 0) * 100)}</span>
            <span>{Math.round((norm.cloudy_fraction ?? 0) * 100)}</span>
            <span>{Math.round((norm.amber_fraction ?? 0) * 100)}</span>
          </div>
        </div>
      )}

      {/* Confidence */}
      <span className="text-xs font-mono w-10 text-right" style={{ color: confColor }}>
        {formatConfidence(conf)}
      </span>

      {/* Time */}
      <span className="text-[10px] w-14 text-right" style={{ color: '#484f58' }}>
        {norm.queued_at ? timeAgo(new Date(norm.queued_at!).getTime() / 1000) : "—"}
      </span>

      {/* Actions */}
      <div className="flex items-center gap-1 opacity-0 group-hover:opacity-100 transition-opacity">
        <button
          onClick={() => onApprove(item.id)}
          className="p-1.5 rounded transition-colors"
          style={{ color: '#484f58' }}
          title="Approve"
        >
          <CheckCircle2 className="w-3.5 h-3.5 hover:text-green-400" />
        </button>
        <button
          onClick={() => onReject(item.id)}
          className="p-1.5 rounded transition-colors"
          style={{ color: '#484f58' }}
          title="Reject"
        >
          <XCircle className="w-3.5 h-3.5 hover:text-red-400" />
        </button>
      </div>
    </div>
  );
}

// ---------------------------------------------------------------------------
// Auto-label panel
// ---------------------------------------------------------------------------

function AutoLabelPanel() {
  const queryClient = useQueryClient();
  const [backend, setBackend] = useState("moondream");
  const [batchSize, setBatchSize] = useState(50);
  const [datasetId, setDatasetId] = useState<number | null>(null);

  const startMutation = useMutation({
    mutationFn: () =>
      api
        .post("/annotation/auto-label", { dataset_id: datasetId, backend, batch_size: batchSize })
        .then((r) => r.data),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["annotation-queue"] });
      queryClient.invalidateQueries({ queryKey: ["annotation-jobs"] });
    },
  });

  return (
    <div className="p-4 space-y-4">
      {/* Dataset ID */}
      <div>
        <label className="text-xs mb-1.5 block" style={{ color: '#484f58' }}>Target Dataset ID</label>
        <input
          type="number"
          placeholder="Dataset ID"
          value={datasetId ?? ""}
          onChange={(e) => setDatasetId(e.target.value ? Number(e.target.value) : null)}
          className="w-full px-3 py-1.5 text-sm rounded-lg focus:outline-none"
          style={{ background: '#0d1117', border: '1px solid #21262d', color: '#8b949e' }}
        />
      </div>

      {/* VLM Backend */}
      <div>
        <label className="text-xs mb-1.5 block" style={{ color: '#484f58' }}>VLM Backend</label>
        <div className="flex gap-1.5">
          {[
            { id: "moondream", label: "Moondream", vram: "2.1 GB" },
            { id: "florence2", label: "Florence-2", vram: "3.5 GB" },
            { id: "qwen2vl", label: "Qwen2-VL", vram: "5.5 GB" },
          ].map((m) => (
            <button
              key={m.id}
              onClick={() => setBackend(m.id)}
              className="flex-1 px-2 py-1.5 rounded-lg text-xs font-medium transition-all border text-center"
              style={{
                background: backend === m.id ? 'rgba(168,85,247,0.2)' : 'transparent',
                border: backend === m.id ? '1px solid rgba(168,85,247,0.4)' : '1px solid #21262d',
                color: backend === m.id ? '#c084fc' : '#484f58',
              }}
            >
              <div>{m.label}</div>
              <div className="text-[9px] opacity-70">{m.vram}</div>
            </button>
          ))}
        </div>
      </div>

      {/* Batch size */}
      <div>
        <div className="flex items-center justify-between mb-1.5">
          <label className="text-xs" style={{ color: '#484f58' }}>Batch Size</label>
          <span className="text-xs font-mono" style={{ color: '#8b949e' }}>{batchSize}</span>
        </div>
        <input
          type="range"
          min={10}
          max={500}
          step={10}
          value={batchSize}
          onChange={(e) => setBatchSize(Number(e.target.value))}
          className="w-full h-1.5 appearance-none rounded cursor-pointer"
          style={{ background: '#21262d' }}
        />
      </div>

      {/* Start button */}
      <button
        onClick={() => startMutation.mutate()}
        disabled={!datasetId || startMutation.isPending}
        className="w-full flex items-center justify-center gap-2 py-2 rounded-lg text-sm font-medium transition-all"
        style={{
          background: !datasetId || startMutation.isPending
            ? 'rgba(168,85,247,0.3)'
            : 'rgb(147,51,234)',
          color: !datasetId || startMutation.isPending ? 'rgba(192,132,252,0.5)' : 'white',
          cursor: !datasetId || startMutation.isPending ? 'not-allowed' : 'pointer',
        }}
      >
        {startMutation.isPending ? (
          <>
            <Loader2 className="w-4 h-4 animate-spin" />
            Labeling…
          </>
        ) : (
          <>
            <Brain className="w-4 h-4" />
            Start Auto-Label
          </>
        )}
      </button>

      {!datasetId && (
        <p className="text-[10px] text-center" style={{ color: '#484f58' }}>
          Enter a dataset ID first
        </p>
      )}

      {startMutation.isError && (
        <p className="text-xs text-red-400">
          {(startMutation.error as Error)?.message ?? "Failed to start auto-labeling"}
        </p>
      )}

      {startMutation.isSuccess && (
        <p className="text-xs text-green-400">Auto-labeling job started</p>
      )}

      {/* Kappa info */}
      <div
        className="px-3 py-2.5 rounded-lg space-y-1.5 mt-2"
        style={{ background: '#161b22', border: '1px solid #21262d' }}
      >
        <h4 className="text-xs font-medium" style={{ color: '#8b949e' }}>Kappa Agreement</h4>
        <p className="text-[10px] leading-relaxed" style={{ color: '#484f58' }}>
          Cohen&apos;s κ measures inter-annotator consistency. κ ≥ 0.80 is target for training data quality.
        </p>
        <div className="flex items-center gap-2 mt-1">
          <div
            className="flex-1 h-1.5 rounded-full overflow-hidden"
            style={{ background: '#21262d' }}
          >
            <div className="h-full bg-green-500 rounded-full" style={{ width: '76%' }} />
          </div>
          <span className="text-xs font-mono text-green-400">0.76</span>
        </div>
        <p className="text-[9px]" style={{ color: '#484f58' }}>VLM vs. human (estimated)</p>
      </div>
    </div>
  );
}

// ---------------------------------------------------------------------------
// Stats tab
// ---------------------------------------------------------------------------

function StatsTab() {
  const { data, isLoading } = useQuery({
    queryKey: ["annotation-stats"],
    queryFn: () => api.get("/annotation/stats").then((r) => r.data),
    refetchInterval: 10_000,
  });

  if (isLoading) {
    return (
      <div className="flex items-center justify-center py-16">
        <Loader2 className="w-6 h-6 text-blue-400 animate-spin" />
      </div>
    );
  }

  if (!data) {
    return (
      <div className="text-center py-16" style={{ color: '#484f58' }}>
        <BarChart3 className="w-8 h-8 mx-auto mb-3 opacity-30" />
        <p className="text-sm">No annotation statistics available</p>
      </div>
    );
  }

  const stats: AnnotationStats = data;

  return (
    <div className="space-y-4">
      <div className="grid grid-cols-2 gap-3">
        {[
          {
            label: "Pending",
            value: stats.total_pending ?? stats.pending_count ?? 0,
            color: "#eab308",
          },
          {
            label: "Reviewed",
            value: stats.total_reviewed ?? stats.reviewed_count ?? 0,
            color: "#22c55e",
          },
          {
            label: "Throughput",
            value: `${(stats.throughput_per_hour ?? 0).toFixed(1)}/hr`,
            color: "#3b82f6",
          },
          {
            label: "High Priority",
            value: stats.high_priority_count ?? 0,
            color: "#ef4444",
          },
        ].map(({ label, value, color }) => (
          <div
            key={label}
            className="px-4 py-3 rounded-xl"
            style={{ background: '#0d1117', border: '1px solid #21262d' }}
          >
            <p className="text-xs mb-1" style={{ color: '#484f58' }}>{label}</p>
            <p className="text-2xl font-bold font-mono" style={{ color }}>{value}</p>
          </div>
        ))}
      </div>

      {/* Agreement */}
      {data.agreement !== undefined && (
        <div
          className="px-4 py-3 rounded-xl"
          style={{ background: '#0d1117', border: '1px solid #21262d' }}
        >
          <p className="text-xs mb-2" style={{ color: '#484f58' }}>Inter-Annotator Agreement (Cohen&apos;s κ)</p>
          <div className="flex items-center gap-3">
            <div className="flex-1 h-2 rounded-full overflow-hidden" style={{ background: '#21262d' }}>
              <div
                className="h-full rounded-full"
                style={{
                  width: `${Math.min(100, data.agreement * 100)}%`,
                  background: data.agreement >= 0.8 ? '#22c55e' : data.agreement >= 0.6 ? '#eab308' : '#ef4444',
                }}
              />
            </div>
            <span className="text-sm font-mono font-bold text-white">
              {data.agreement.toFixed(2)}
            </span>
          </div>
        </div>
      )}
    </div>
  );
}

// ---------------------------------------------------------------------------
// Main annotation page
// ---------------------------------------------------------------------------

export default function AnnotationPage() {
  const queryClient = useQueryClient();
  const [tab, setTab] = useState<"queue" | "jobs" | "stats">("queue");

  const {
    data: queueData,
    isLoading: queueLoading,
    refetch: refetchQueue,
  } = useQuery({
    queryKey: ["annotation-queue"],
    queryFn: () => api.get("/annotation/queue").then((r) => r.data),
    refetchInterval: 10_000,
    staleTime: 5_000,
  });

  const { data: jobsData = [] } = useQuery({
    queryKey: ["annotation-jobs"],
    queryFn: () => api.get("/annotation/jobs").then((r) => r.data),
    refetchInterval: 5_000,
    enabled: tab === "jobs",
  });

  const approveMutation = useMutation({
    mutationFn: (itemId: number) =>
      api.put(`/annotation/queue/${itemId}`, { status: "approved" }).then((r) => r.data),
    onSuccess: () => queryClient.invalidateQueries({ queryKey: ["annotation-queue"] }),
  });

  const rejectMutation = useMutation({
    mutationFn: (itemId: number) =>
      api.put(`/annotation/queue/${itemId}`, { status: "rejected" }).then((r) => r.data),
    onSuccess: () => queryClient.invalidateQueries({ queryKey: ["annotation-queue"] }),
  });

  // Normalize queue response — handle {items: [...]} or [...] or {queue: [...]}
  const rawItems: ReviewItem[] = Array.isArray(queueData)
    ? queueData
    : queueData?.items ?? queueData?.queue ?? [];

  const stats: AnnotationStats = queueData?.stats ?? {};
  const pendingCount = stats.total_pending ?? stats.pending_count ?? rawItems.length;

  const sortedItems = [...rawItems].sort(
    (a, b) => ((b.review_priority ?? b.priority ?? 0) - (a.review_priority ?? a.priority ?? 0))
  );

  const jobs: AnnotationJob[] = Array.isArray(jobsData)
    ? jobsData
    : jobsData?.jobs ?? [];

  return (
    <div className="flex h-full gap-0">
      {/* Main content */}
      <div className="flex-1 flex flex-col min-w-0">
        {/* Header */}
        <div
          className="flex items-center justify-between px-5 py-3"
          style={{ borderBottom: '1px solid #21262d' }}
        >
          <div className="flex items-center gap-2">
            <Tag className="w-4 h-4 text-purple-400" />
            <h1 className="text-base font-semibold text-white">Annotation</h1>
          </div>

          <div className="flex items-center gap-3">
            <div
              className="flex gap-1 p-0.5 rounded-lg"
              style={{ background: '#161b22', border: '1px solid #21262d' }}
            >
              {(["queue", "jobs", "stats"] as const).map((t) => (
                <button
                  key={t}
                  onClick={() => setTab(t)}
                  className="px-3 py-1 rounded text-xs font-medium transition-all capitalize"
                  style={{
                    background: tab === t ? '#0d1117' : 'transparent',
                    color: tab === t ? '#e6edf3' : '#484f58',
                  }}
                >
                  {t}{t === "queue" && pendingCount > 0 ? ` (${pendingCount})` : ""}
                </button>
              ))}
            </div>

            <button
              onClick={() => refetchQueue()}
              className="p-1.5 rounded transition-colors"
              style={{ color: '#484f58' }}
            >
              <RefreshCw className="w-4 h-4" />
            </button>
          </div>
        </div>

        {/* Stats bar */}
        {Object.keys(stats).length > 0 && (
          <div
            className="flex items-center gap-6 px-5 py-2.5"
            style={{ borderBottom: '1px solid #21262d', background: '#161b22' }}
          >
            {[
              { label: "Pending", value: stats.total_pending ?? stats.pending_count ?? 0, color: "#eab308" },
              { label: "Reviewed", value: stats.total_reviewed ?? stats.reviewed_count ?? 0, color: "#22c55e" },
              {
                label: "Throughput",
                value: `${(stats.throughput_per_hour ?? 0).toFixed(1)}/hr`,
                color: "#3b82f6",
              },
              { label: "High Priority", value: stats.high_priority_count ?? 0, color: "#ef4444" },
            ].map(({ label, value, color }) => (
              <div key={label} className="flex items-center gap-1.5">
                <span className="text-xs" style={{ color: '#484f58' }}>{label}:</span>
                <span className="text-xs font-bold font-mono" style={{ color }}>{value}</span>
              </div>
            ))}
          </div>
        )}

        {/* Tab content */}
        <div className="flex-1 overflow-y-auto px-5 py-4">
          {tab === "queue" && (
            <div className="space-y-2">
              {/* Human-in-loop notice */}
              <div
                className="flex items-start gap-2 px-3 py-2.5 rounded-lg mb-4"
                style={{ background: 'rgba(59,130,246,0.1)', border: '1px solid rgba(59,130,246,0.2)' }}
              >
                <FlaskConical className="w-4 h-4 text-blue-400 flex-shrink-0 mt-0.5" />
                <p className="text-xs" style={{ color: 'rgba(191,219,254,0.8)' }}>
                  <strong>Human-in-loop enforced:</strong> VLM auto-labels require human approval
                  before entering the training dataset. Review, approve or reject each item below.
                </p>
              </div>

              {queueLoading ? (
                <div className="flex items-center justify-center py-16">
                  <Loader2 className="w-6 h-6 animate-spin text-blue-400" />
                </div>
              ) : sortedItems.length === 0 ? (
                <div className="text-center py-16" style={{ color: '#484f58' }}>
                  <Tag className="w-10 h-10 mx-auto mb-3 opacity-30" />
                  <p className="text-sm font-medium">Queue is empty</p>
                  <p className="text-xs mt-1">Run VLM auto-labeling to populate the queue</p>
                </div>
              ) : (
                sortedItems.map((item) => (
                  <ReviewRow
                    key={item.id}
                    item={item}
                    onApprove={(id) => approveMutation.mutate(id)}
                    onReject={(id) => rejectMutation.mutate(id)}
                  />
                ))
              )}
            </div>
          )}

          {tab === "jobs" && (
            <div className="space-y-2">
              {jobs.length === 0 ? (
                <div className="text-center py-16" style={{ color: '#484f58' }}>
                  <Clock className="w-8 h-8 mx-auto mb-3 opacity-30" />
                  <p className="text-sm">No annotation jobs yet</p>
                </div>
              ) : (
                jobs.map((job, i) => (
                  <div
                    key={job.id ?? job.job_uuid ?? i}
                    className="flex items-center gap-3 px-4 py-3 rounded-xl"
                    style={{ background: '#0d1117', border: '1px solid #21262d' }}
                  >
                    <div
                      className={cn("w-2 h-2 rounded-full flex-shrink-0")}
                      style={{
                        background:
                          job.status === "running"
                            ? "#3b82f6"
                            : job.status === "completed"
                            ? "#22c55e"
                            : job.status === "failed"
                            ? "#ef4444"
                            : "#6b7280",
                        animation: job.status === "running" ? "pulse 2s infinite" : undefined,
                      }}
                    />
                    <div className="flex-1 min-w-0">
                      <p className="text-sm" style={{ color: '#8b949e' }}>
                        {job.job_type ?? job.type ?? "Annotation Job"}
                      </p>
                      <p className="text-xs" style={{ color: '#484f58' }}>
                        {job.processed_items ?? 0} / {job.total_items ?? "?"} processed
                      </p>
                    </div>
                    {job.progress !== undefined && (
                      <div
                        className="w-20 h-1.5 rounded-full overflow-hidden"
                        style={{ background: '#21262d' }}
                      >
                        <div
                          className="h-full bg-blue-500 transition-all"
                          style={{ width: `${job.progress}%` }}
                        />
                      </div>
                    )}
                    <span className="text-xs font-mono" style={{ color: '#484f58' }}>
                      {job.status}
                    </span>
                  </div>
                ))
              )}
            </div>
          )}

          {tab === "stats" && <StatsTab />}
        </div>
      </div>

      {/* Right sidebar: VLM controls */}
      <div
        className="w-72 flex-shrink-0 overflow-y-auto"
        style={{ borderLeft: '1px solid #21262d' }}
      >
        <div className="px-4 py-3 flex items-center gap-2" style={{ borderBottom: '1px solid #21262d' }}>
          <Brain className="w-4 h-4 text-purple-400" />
          <h2 className="text-sm font-semibold text-white">Auto-Label</h2>
        </div>
        <AutoLabelPanel />
      </div>
    </div>
  );
}
