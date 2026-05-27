"use client";

import React, { useState } from "react";
import { useQuery, useMutation, useQueryClient } from "@tanstack/react-query";
import {
  FlaskConical,
  Plus,
  RefreshCw,
  Loader2,
  Trash2,
  ArchiveRestore,
  Archive,
  AlertTriangle,
  TrendingUp,
  X,
} from "lucide-react";
import { api } from "@/lib/api";
import { cn, timeAgo } from "@/lib/utils";

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

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

// ---------------------------------------------------------------------------
// Create experiment modal
// ---------------------------------------------------------------------------

function CreateModal({ onClose }: { onClose: () => void }) {
  const queryClient = useQueryClient();
  const [form, setForm] = useState<CreateExperimentRequest>({
    name: "",
    description: "",
    tags: [],
  });
  const [tagInput, setTagInput] = useState("");

  const createMutation = useMutation({
    mutationFn: () => api.post("/experiments", form).then((r) => r.data),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["experiments"] });
      onClose();
    },
  });

  const addTag = () => {
    const t = tagInput.trim();
    if (t && !form.tags?.includes(t)) {
      setForm({ ...form, tags: [...(form.tags ?? []), t] });
      setTagInput("");
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
                onKeyDown={(e) => e.key === "Enter" && (e.preventDefault(), addTag())}
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
            ) : "Create Experiment"}
          </button>
        </div>

        {createMutation.isError && (
          <p className="text-xs text-red-400">
            {(createMutation.error as Error)?.message ?? "Failed to create experiment"}
          </p>
        )}
      </div>
    </div>
  );
}

// ---------------------------------------------------------------------------
// Experiment card
// ---------------------------------------------------------------------------

function ExperimentCard({ experiment }: { experiment: Experiment }) {
  const queryClient = useQueryClient();
  const isArchived = experiment.is_archived ?? experiment.archived ?? false;

  const archiveMutation = useMutation({
    mutationFn: () =>
      api.put(`/experiments/${experiment.id}`, { is_archived: !isArchived }).then((r) => r.data),
    onSuccess: () => queryClient.invalidateQueries({ queryKey: ["experiments"] }),
  });

  const deleteMutation = useMutation({
    mutationFn: () => api.delete(`/experiments/${experiment.id}`).then((r) => r.data),
    onSuccess: () => queryClient.invalidateQueries({ queryKey: ["experiments"] }),
  });

  const createdAt =
    experiment.created_at !== undefined
      ? typeof experiment.created_at === "number"
        ? timeAgo(experiment.created_at)
        : timeAgo(new Date(experiment.created_at as string).getTime() / 1000)
      : null;

  const runCount = experiment.run_count ?? 0;
  const bestMap50 = experiment.best_map50;

  return (
    <div
      className={cn("rounded-xl p-4 space-y-3 transition-all")}
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
            {experiment.status && experiment.status !== "active" && !isArchived && (
              <span
                className="text-[10px] px-1.5 py-0.5 rounded font-medium capitalize"
                style={{
                  background:
                    experiment.status === "running"
                      ? "rgba(59,130,246,0.2)"
                      : "rgba(34,197,94,0.2)",
                  color:
                    experiment.status === "running" ? "#60a5fa" : "#22c55e",
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
            title={isArchived ? "Restore" : "Archive"}
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

      {/* Run bar indicator */}
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

// ---------------------------------------------------------------------------
// Main experiments page
// ---------------------------------------------------------------------------

export default function ExperimentsPage() {
  const [showCreate, setShowCreate] = useState(false);
  const [showArchived, setShowArchived] = useState(false);
  const queryClient = useQueryClient();

  const {
    data,
    isLoading,
    isError,
    error,
    refetch,
  } = useQuery({
    queryKey: ["experiments"],
    queryFn: () => api.get("/experiments").then((r) => r.data),
    staleTime: 30_000,
    refetchInterval: 15_000,
  });

  const experiments: Experiment[] = Array.isArray(data)
    ? data
    : data?.experiments ?? data?.items ?? [];

  const active = experiments.filter((e) => !(e.is_archived ?? e.archived));
  const archived = experiments.filter((e) => e.is_archived ?? e.archived);

  const displayed = showArchived ? experiments : active;

  // Summary stats
  const totalRuns = experiments.reduce((sum, e) => sum + (e.run_count ?? 0), 0);
  const bestMap = experiments.reduce((best, e) => {
    const m = e.best_map50 ?? 0;
    return m > best ? m : best;
  }, 0);

  return (
    <div className="flex flex-col h-full overflow-y-auto">
      {/* Header */}
      <div
        className="flex items-center justify-between px-5 py-3"
        style={{ borderBottom: '1px solid #21262d' }}
      >
        <div className="flex items-center gap-2">
          <FlaskConical className="w-4 h-4 text-purple-400" />
          <h1 className="text-base font-semibold text-white">Experiments</h1>
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

      {/* Summary stats */}
      {experiments.length > 0 && (
        <div
          className="flex items-center gap-6 px-5 py-2.5"
          style={{ borderBottom: '1px solid #21262d', background: '#161b22' }}
        >
          {[
            { label: "Experiments", value: active.length, color: "#e6edf3" },
            { label: "Total Runs", value: totalRuns, color: "#8b949e" },
            {
              label: "Best mAP50",
              value: bestMap > 0 ? `${(bestMap * 100).toFixed(1)}%` : "—",
              color: "#22c55e",
            },
            { label: "Archived", value: archived.length, color: "#484f58" },
          ].map(({ label, value, color }) => (
            <div key={label} className="flex items-center gap-1.5">
              <span className="text-xs" style={{ color: '#484f58' }}>{label}:</span>
              <span className="text-xs font-bold font-mono" style={{ color }}>{value}</span>
            </div>
          ))}
        </div>
      )}

      <div className="flex-1 px-5 py-4">
        {/* Loading */}
        {isLoading && (
          <div className="flex items-center justify-center py-16">
            <Loader2 className="w-6 h-6 text-blue-400 animate-spin" />
          </div>
        )}

        {/* Error */}
        {isError && (
          <div
            className="flex items-start gap-3 px-4 py-3 rounded-lg mb-4"
            style={{ background: 'rgba(239,68,68,0.1)', border: '1px solid rgba(239,68,68,0.2)' }}
          >
            <AlertTriangle className="w-4 h-4 text-red-400 mt-0.5" />
            <div>
              <p className="text-sm font-medium text-red-400">Failed to load experiments</p>
              <p className="text-xs mt-0.5" style={{ color: 'rgba(252,165,165,0.7)' }}>
                {(error as Error)?.message ?? "Unknown error"}
              </p>
            </div>
          </div>
        )}

        {/* Empty state */}
        {!isLoading && !isError && displayed.length === 0 && (
          <div className="text-center py-20" style={{ color: '#484f58' }}>
            <FlaskConical className="w-12 h-12 mx-auto mb-4 opacity-30" />
            <p className="text-base font-medium">
              {showArchived ? "No archived experiments" : "No experiments yet"}
            </p>
            <p className="text-sm mt-1">
              {!showArchived && 'Click "New Experiment" to create your first experiment'}
            </p>
          </div>
        )}

        {/* Experiment grid */}
        {!isLoading && !isError && displayed.length > 0 && (
          <div className="grid grid-cols-1 md:grid-cols-2 xl:grid-cols-3 gap-4">
            {displayed.map((exp) => (
              <ExperimentCard key={exp.id} experiment={exp} />
            ))}
          </div>
        )}
      </div>

      {showCreate && <CreateModal onClose={() => setShowCreate(false)} />}
    </div>
  );
}
