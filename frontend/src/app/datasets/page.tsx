"use client";

import React, { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { useQuery, useMutation, useQueryClient } from "@tanstack/react-query";
import {
  Database,
  Plus,
  RefreshCw,
  LayoutGrid,
  List,
  ChevronRight,
  X,
  Loader2,
  Layers,
  FolderSearch,
  CheckCircle2,
  AlertCircle,
  Trash2,
  ExternalLink,
  HardDrive,
  FileStack,
  ArrowRight,
} from "lucide-react";
import Link from "next/link";
import { api } from "@/lib/api";
import { cn, timeAgo } from "@/lib/utils";
import type { DatasetSummary, SampleSummary } from "@/lib/types";
import { ImageGrid, SelectionToolbar } from "@/components/datasets/ImageGrid";
import { FilterPanel, DEFAULT_FILTERS } from "@/components/datasets/FilterPanel";
import type { DatasetFilters } from "@/components/datasets/FilterPanel";
import { UploadZone } from "@/components/datasets/UploadZone";

// ---------------------------------------------------------------------------
// Streaming-format types
// ---------------------------------------------------------------------------

type StreamingFormat = "zarr" | "hdf5";

interface ConvertRequest {
  source_path: string;
  output_path: string;
  format: StreamingFormat;
  image_size: number;
  val_split: number;
  test_split: number;
}

interface ConvertStartedResponse {
  task_id: string;
}

type ConversionStatus = "running" | "complete" | "error";

interface ConversionStatusResponse {
  task_id: string;
  status: ConversionStatus;
  progress_pct?: number;
  error?: string;
}

interface StreamingStoreStats {
  total_images: number;
  chunk_count?: number;
  store_size_mb: number;
  format: StreamingFormat;
}

interface RecentConversion {
  source_path: string;
  output_path: string;
  format: StreamingFormat;
  stats: StreamingStoreStats;
  converted_at: number;
}

// ---------------------------------------------------------------------------
// Create dataset modal
// ---------------------------------------------------------------------------

function CreateDatasetModal({
  onClose,
  onCreate,
}: {
  onClose: () => void;
  onCreate: (name: string, description: string) => void;
}) {
  const [name, setName] = useState("");
  const [description, setDescription] = useState("");

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/60 backdrop-blur-sm">
      <div className="w-full max-w-md bg-[var(--color-panel)] border border-[var(--color-border)] rounded-xl shadow-xl p-6">
        <div className="flex items-center justify-between mb-5">
          <h2 className="text-lg font-semibold text-white">New Dataset</h2>
          <button onClick={onClose} className="text-[var(--color-text-muted)] hover:text-white">
            <X className="w-5 h-5" />
          </button>
        </div>

        <div className="space-y-4">
          <div>
            <label className="block text-sm font-medium text-[var(--color-text-secondary)] mb-1.5">
              Name <span className="text-red-400">*</span>
            </label>
            <input
              type="text"
              value={name}
              onChange={(e) => setName(e.target.value)}
              placeholder="e.g. strain_og_kush_batch1"
              className="w-full px-3 py-2 bg-[var(--color-surface)] border border-[var(--color-border)]
                rounded-lg text-sm text-[var(--color-text-secondary)] placeholder:text-[var(--color-text-muted)]
                focus:outline-none focus:border-blue-500/60 focus:ring-1 focus:ring-blue-500/20"
            />
          </div>

          <div>
            <label className="block text-sm font-medium text-[var(--color-text-secondary)] mb-1.5">
              Description
            </label>
            <textarea
              value={description}
              onChange={(e) => setDescription(e.target.value)}
              rows={3}
              placeholder="Strain, microscope, magnification, conditions…"
              className="w-full px-3 py-2 bg-[var(--color-surface)] border border-[var(--color-border)]
                rounded-lg text-sm text-[var(--color-text-secondary)] placeholder:text-[var(--color-text-muted)]
                focus:outline-none focus:border-blue-500/60 focus:ring-1 focus:ring-blue-500/20 resize-none"
            />
          </div>

          <div className="flex gap-3 pt-2">
            <button
              onClick={onClose}
              className="flex-1 py-2 rounded-lg text-sm border border-[var(--color-border)] text-[var(--color-text-muted)]
                hover:text-[var(--color-text-secondary)] hover:border-[var(--color-border-hover)] transition-colors"
            >
              Cancel
            </button>
            <button
              onClick={() => {
                if (name.trim()) {
                  onCreate(name.trim(), description.trim());
                  onClose();
                }
              }}
              disabled={!name.trim()}
              className="flex-1 py-2 rounded-lg text-sm font-medium bg-blue-600 hover:bg-blue-500 text-white
                disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
            >
              Create Dataset
            </button>
          </div>
        </div>
      </div>
    </div>
  );
}

// ---------------------------------------------------------------------------
// Dataset card (list view)
// ---------------------------------------------------------------------------

function DatasetCard({ dataset }: { dataset: DatasetSummary }) {
  return (
    <Link
      href={`/datasets/${dataset.id}`}
      className="group flex items-center gap-4 px-4 py-3 rounded-xl bg-[var(--color-surface)]
        border border-[var(--color-border)] hover:border-[var(--color-border-hover)] transition-all"
    >
      <div className="w-10 h-10 rounded-lg bg-blue-500/10 flex items-center justify-center flex-shrink-0">
        <Database className="w-5 h-5 text-blue-400" />
      </div>

      <div className="flex-1 min-w-0">
        <div className="flex items-center gap-2">
          <h3 className="text-sm font-medium text-white truncate">{dataset.name}</h3>
          {dataset.status && (
            <span
              className={cn(
                "text-[10px] px-1.5 py-0.5 rounded font-medium uppercase tracking-wide",
                dataset.status === "ready"
                  ? "bg-green-500/20 text-green-400"
                  : "bg-yellow-500/20 text-yellow-400"
              )}
            >
              {dataset.status}
            </span>
          )}
        </div>
        <div className="flex items-center gap-3 mt-0.5 text-xs text-[var(--color-text-muted)]">
          <span>{dataset.num_samples.toLocaleString()} images</span>
          {dataset.created_at && <span>Created {timeAgo(dataset.created_at)}</span>}
        </div>
      </div>

      {/* Class distribution mini-bar */}
      {dataset.class_names && dataset.class_names.length > 0 && (
        <div className="hidden sm:flex items-center gap-1 text-xs text-[var(--color-text-muted)]">
          <span>{dataset.class_names.length} classes</span>
        </div>
      )}

      <ChevronRight className="w-4 h-4 text-[var(--color-text-muted)] group-hover:text-[var(--color-text-secondary)] transition-colors" />
    </Link>
  );
}

// ---------------------------------------------------------------------------
// Streaming tab — converter + inspector
// ---------------------------------------------------------------------------

function FormatBadge({ format }: { format: StreamingFormat }) {
  return (
    <span
      className={cn(
        "text-[10px] px-1.5 py-0.5 rounded font-medium uppercase tracking-wide font-mono",
        format === "zarr"
          ? "bg-purple-500/20 text-purple-300"
          : "bg-cyan-500/20 text-cyan-300"
      )}
    >
      {format}
    </span>
  );
}

function SplitPreview({
  valSplit,
  testSplit,
}: {
  valSplit: number;
  testSplit: number;
}) {
  const trainPct = Math.round((1 - valSplit - testSplit) * 100);
  const valPct = Math.round(valSplit * 100);
  const testPct = Math.round(testSplit * 100);
  return (
    <p className="text-xs text-[#484f58] mt-1">
      Train:{" "}
      <span className="text-green-400 font-medium">~{trainPct}%</span>
      {"  ·  "}Val:{" "}
      <span className="text-blue-400 font-medium">{valPct}%</span>
      {"  ·  "}Test:{" "}
      <span className="text-yellow-400 font-medium">{testPct}%</span>
    </p>
  );
}

function ConversionJobCard({
  taskId,
  onComplete,
}: {
  taskId: string;
  onComplete: (stats: StreamingStoreStats) => void;
}) {
  const [status, setStatus] = useState<ConversionStatus>("running");
  const [progressPct, setProgressPct] = useState<number | undefined>(undefined);
  const [errorMsg, setErrorMsg] = useState<string | undefined>(undefined);
  const [completedStats, setCompletedStats] = useState<StreamingStoreStats | null>(null);
  const intervalRef = useRef<ReturnType<typeof setInterval> | null>(null);

  useEffect(() => {
    if (status !== "running") return;

    const poll = async () => {
      try {
        const res = await api.get<ConversionStatusResponse>(
          `/datasets/convert/${taskId}`
        );
        const data = res.data;
        setStatus(data.status);
        if (data.progress_pct !== undefined) setProgressPct(data.progress_pct);
        if (data.error) setErrorMsg(data.error);
      } catch {
        // network hiccup — keep polling
      }
    };

    poll();
    intervalRef.current = setInterval(poll, 2000);
    return () => {
      if (intervalRef.current) clearInterval(intervalRef.current);
    };
  }, [taskId, status]);

  // Stop polling once done
  useEffect(() => {
    if (status !== "running" && intervalRef.current) {
      clearInterval(intervalRef.current);
      intervalRef.current = null;
    }
  }, [status]);

  const statusColor =
    status === "complete"
      ? "text-green-400"
      : status === "error"
      ? "text-red-400"
      : "text-blue-400";

  const statusLabel =
    status === "complete" ? "Complete" : status === "error" ? "Error" : "Running";

  return (
    <div className="rounded-lg border border-[#21262d] bg-[#0d1117] px-4 py-3 space-y-2">
      <div className="flex items-center justify-between">
        <span className="text-xs font-mono text-[#484f58]">
          task:{" "}
          <span className="text-[#e6edf3]">{taskId.slice(0, 12)}…</span>
        </span>
        <span className={cn("flex items-center gap-1.5 text-xs font-medium", statusColor)}>
          {status === "running" && (
            <Loader2 className="w-3.5 h-3.5 animate-spin" />
          )}
          {status === "complete" && <CheckCircle2 className="w-3.5 h-3.5" />}
          {status === "error" && <AlertCircle className="w-3.5 h-3.5" />}
          {statusLabel}
        </span>
      </div>

      {/* Progress bar */}
      {status === "running" && (
        <div className="w-full h-1.5 rounded-full bg-[#161b22] overflow-hidden">
          {progressPct !== undefined ? (
            <div
              className="h-full rounded-full bg-blue-500 transition-all duration-500"
              style={{ width: `${progressPct}%` }}
            />
          ) : (
            /* indeterminate */
            <div className="h-full rounded-full bg-blue-500 animate-[indeterminate_1.5s_ease-in-out_infinite] w-1/3" />
          )}
        </div>
      )}

      {progressPct !== undefined && status === "running" && (
        <p className="text-[11px] text-[#484f58]">{progressPct.toFixed(0)}% processed</p>
      )}

      {status === "error" && errorMsg && (
        <p className="text-xs text-red-400 bg-red-500/10 rounded px-2 py-1.5">{errorMsg}</p>
      )}
    </div>
  );
}

function StoreInspector({
  initialPath,
  onInspected,
}: {
  initialPath?: string;
  onInspected?: (stats: StreamingStoreStats, path: string) => void;
}) {
  const [path, setPath] = useState(initialPath ?? "");

  const {
    data: stats,
    isFetching,
    isError,
    error,
    refetch,
  } = useQuery<StreamingStoreStats>({
    queryKey: ["streaming-stats", path],
    queryFn: () =>
      api
        .get<StreamingStoreStats>("/datasets/streaming/stats", {
          params: { path },
        })
        .then((r) => r.data),
    enabled: false,
    retry: false,
  });

  useEffect(() => {
    if (initialPath && initialPath !== path) {
      setPath(initialPath);
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [initialPath]);

  const handleInspect = () => {
    if (!path.trim()) return;
    refetch().then((result) => {
      if (result.data && onInspected) {
        onInspected(result.data, path);
      }
    });
  };

  return (
    <div>
      <div className="flex gap-2">
        <input
          type="text"
          value={path}
          onChange={(e) => setPath(e.target.value)}
          onKeyDown={(e) => {
            if (e.key === "Enter") handleInspect();
          }}
          placeholder="/data/datasets/trichome_zarr"
          className="flex-1 px-3 py-2 bg-[#0d1117] border border-[#21262d]
            rounded-lg text-sm text-[#e6edf3] placeholder:text-[#484f58]
            focus:outline-none focus:border-blue-500/60 focus:ring-1 focus:ring-blue-500/20"
        />
        <button
          onClick={handleInspect}
          disabled={!path.trim() || isFetching}
          className="flex items-center gap-1.5 px-4 py-2 rounded-lg text-sm font-medium
            bg-blue-600 hover:bg-blue-500 text-white disabled:opacity-50
            disabled:cursor-not-allowed transition-colors"
        >
          {isFetching ? (
            <Loader2 className="w-3.5 h-3.5 animate-spin" />
          ) : (
            <FolderSearch className="w-3.5 h-3.5" />
          )}
          Inspect
        </button>
      </div>

      {isError && (
        <p className="mt-2 text-xs text-red-400 bg-red-500/10 rounded px-2 py-1.5">
          {(error as Error).message}
        </p>
      )}

      {stats && (
        <div className="mt-3 grid grid-cols-2 sm:grid-cols-4 gap-3">
          <div className="rounded-lg bg-[#0d1117] border border-[#21262d] px-3 py-2.5">
            <p className="text-[10px] uppercase tracking-wider text-[#484f58] mb-1">
              Total images
            </p>
            <p className="text-sm font-semibold text-[#e6edf3]">
              {stats.total_images.toLocaleString()}
            </p>
          </div>
          <div className="rounded-lg bg-[#0d1117] border border-[#21262d] px-3 py-2.5">
            <p className="text-[10px] uppercase tracking-wider text-[#484f58] mb-1">
              Store size
            </p>
            <p className="text-sm font-semibold text-[#e6edf3]">
              {stats.store_size_mb.toFixed(1)} MB
            </p>
          </div>
          <div className="rounded-lg bg-[#0d1117] border border-[#21262d] px-3 py-2.5">
            <p className="text-[10px] uppercase tracking-wider text-[#484f58] mb-1">
              Format
            </p>
            <FormatBadge format={stats.format} />
          </div>
          {stats.chunk_count !== undefined && (
            <div className="rounded-lg bg-[#0d1117] border border-[#21262d] px-3 py-2.5">
              <p className="text-[10px] uppercase tracking-wider text-[#484f58] mb-1">
                Chunks
              </p>
              <p className="text-sm font-semibold text-[#e6edf3]">
                {stats.chunk_count.toLocaleString()}
              </p>
            </div>
          )}
        </div>
      )}
    </div>
  );
}

function StreamingTab() {
  // ── Convert form state ──────────────────────────────────────────
  const [sourcePath, setSourcePath] = useState("");
  const [outputPath, setOutputPath] = useState("");
  const [format, setFormat] = useState<StreamingFormat>("zarr");
  const [imageSize, setImageSize] = useState<number>(640);
  const [valSplit, setValSplit] = useState<number>(0.15);
  const [testSplit, setTestSplit] = useState<number>(0.10);

  // ── Active job ──────────────────────────────────────────────────
  const [activeTaskId, setActiveTaskId] = useState<string | null>(null);
  const [jobStatus, setJobStatus] = useState<ConversionStatus | null>(null);

  // ── Auto-inspect output path after completion ───────────────────
  const [inspectTriggerPath, setInspectTriggerPath] = useState<string | undefined>(
    undefined
  );

  // ── Recent conversions (session-scoped) ─────────────────────────
  const [recentConversions, setRecentConversions] = useState<RecentConversion[]>([]);

  // Derive output path suggestion when source changes
  useEffect(() => {
    if (sourcePath && !outputPath) {
      const base = sourcePath.replace(/\/+$/, "");
      setOutputPath(`${base}_${format}`);
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [sourcePath]);

  // Keep output path suffix in sync when format changes
  useEffect(() => {
    const other: StreamingFormat = format === "zarr" ? "hdf5" : "zarr";
    if (outputPath.endsWith(`_${other}`)) {
      setOutputPath(outputPath.slice(0, -other.length) + format);
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [format]);

  // ── Convert mutation ────────────────────────────────────────────
  const convertMutation = useMutation<
    ConvertStartedResponse,
    Error,
    ConvertRequest
  >({
    mutationFn: (body) =>
      api.post<ConvertStartedResponse>("/datasets/convert", body).then((r) => r.data),
    onSuccess: (data) => {
      setActiveTaskId(data.task_id);
      setJobStatus("running");
    },
  });

  const handleConvert = () => {
    if (!sourcePath.trim() || !outputPath.trim()) return;
    convertMutation.mutate({
      source_path: sourcePath.trim(),
      output_path: outputPath.trim(),
      format,
      image_size: imageSize,
      val_split: valSplit,
      test_split: testSplit,
    });
  };

  const handleJobComplete = useCallback(
    (stats: StreamingStoreStats) => {
      setJobStatus("complete");
      setInspectTriggerPath(outputPath.trim());
      setRecentConversions((prev) => [
        {
          source_path: sourcePath.trim(),
          output_path: outputPath.trim(),
          format,
          stats,
          converted_at: Math.floor(Date.now() / 1000),
        },
        ...prev,
      ]);
    },
    [sourcePath, outputPath, format]
  );

  const handleInspected = useCallback(
    (stats: StreamingStoreStats, path: string) => {
      // If we just completed a conversion update recent list with real stats
      setRecentConversions((prev) =>
        prev.map((r) =>
          r.output_path === path ? { ...r, stats } : r
        )
      );
    },
    []
  );

  const trainPct = Math.round((1 - valSplit - testSplit) * 100);

  return (
    <div className="flex-1 overflow-y-auto px-6 py-6 space-y-6 max-w-4xl mx-auto w-full">
      {/* ── Converter card ─────────────────────────────────────── */}
      <div className="rounded-xl border border-[#21262d] bg-[#161b22] p-5 space-y-5">
        <div className="flex items-center gap-2">
          <Layers className="w-4 h-4 text-purple-400" />
          <h2 className="text-sm font-semibold text-[#e6edf3]">
            Convert YOLO Dataset to Streaming Format
          </h2>
        </div>

        {/* Source / output */}
        <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
          <div>
            <label className="block text-xs font-medium text-[#e6edf3] mb-1.5">
              Source path
            </label>
            <input
              type="text"
              value={sourcePath}
              onChange={(e) => setSourcePath(e.target.value)}
              placeholder="/data/datasets/trichome"
              className="w-full px-3 py-2 bg-[#0d1117] border border-[#21262d]
                rounded-lg text-sm text-[#e6edf3] placeholder:text-[#484f58]
                focus:outline-none focus:border-blue-500/60 focus:ring-1 focus:ring-blue-500/20"
            />
            <p className="mt-1 text-[11px] text-[#484f58]">
              Path to YOLO dataset root (contains images/ and labels/)
            </p>
          </div>
          <div>
            <label className="block text-xs font-medium text-[#e6edf3] mb-1.5">
              Output path
            </label>
            <input
              type="text"
              value={outputPath}
              onChange={(e) => setOutputPath(e.target.value)}
              placeholder="/data/datasets/trichome_zarr"
              className="w-full px-3 py-2 bg-[#0d1117] border border-[#21262d]
                rounded-lg text-sm text-[#e6edf3] placeholder:text-[#484f58]
                focus:outline-none focus:border-blue-500/60 focus:ring-1 focus:ring-blue-500/20"
            />
            <p className="mt-1 text-[11px] text-[#484f58]">
              Destination store (auto-filled from source)
            </p>
          </div>
        </div>

        {/* Format selector */}
        <div>
          <label className="block text-xs font-medium text-[#e6edf3] mb-2">
            Format
          </label>
          <div className="grid grid-cols-1 sm:grid-cols-2 gap-3">
            {(["zarr", "hdf5"] as StreamingFormat[]).map((f) => (
              <button
                key={f}
                onClick={() => setFormat(f)}
                className={cn(
                  "flex items-start gap-3 rounded-lg border px-4 py-3 text-left transition-all",
                  format === f
                    ? f === "zarr"
                      ? "border-purple-500/60 bg-purple-500/10"
                      : "border-cyan-500/60 bg-cyan-500/10"
                    : "border-[#21262d] hover:border-[#30363d]"
                )}
              >
                <div
                  className={cn(
                    "mt-0.5 w-3 h-3 rounded-full border-2 flex-shrink-0",
                    format === f
                      ? f === "zarr"
                        ? "border-purple-400 bg-purple-400"
                        : "border-cyan-400 bg-cyan-400"
                      : "border-[#484f58]"
                  )}
                />
                <div>
                  <p
                    className={cn(
                      "text-sm font-medium uppercase tracking-wide font-mono",
                      format === f ? "text-[#e6edf3]" : "text-[#484f58]"
                    )}
                  >
                    {f}
                  </p>
                  <p className="text-[11px] text-[#484f58] mt-0.5">
                    {f === "zarr"
                      ? "Streaming chunks — best for sequential training passes"
                      : "Random access — best for mixed batching"}
                  </p>
                </div>
              </button>
            ))}
          </div>
        </div>

        {/* Image size + splits row */}
        <div className="grid grid-cols-1 sm:grid-cols-3 gap-4">
          {/* Image size */}
          <div>
            <label className="block text-xs font-medium text-[#e6edf3] mb-1.5">
              Image size (px)
            </label>
            <select
              value={imageSize}
              onChange={(e) => setImageSize(Number(e.target.value))}
              className="w-full px-3 py-2 bg-[#0d1117] border border-[#21262d]
                rounded-lg text-sm text-[#e6edf3]
                focus:outline-none focus:border-blue-500/60 focus:ring-1 focus:ring-blue-500/20"
            >
              <option value={320}>320</option>
              <option value={640}>640</option>
              <option value={1280}>1280</option>
            </select>
          </div>

          {/* Val split */}
          <div>
            <label className="block text-xs font-medium text-[#e6edf3] mb-1.5">
              Val split
            </label>
            <input
              type="range"
              min={0.05}
              max={0.30}
              step={0.05}
              value={valSplit}
              onChange={(e) => setValSplit(Number(e.target.value))}
              className="w-full accent-blue-500"
            />
            <p className="text-[11px] text-[#484f58] mt-1">
              {Math.round(valSplit * 100)}% validation
            </p>
          </div>

          {/* Test split */}
          <div>
            <label className="block text-xs font-medium text-[#e6edf3] mb-1.5">
              Test split
            </label>
            <input
              type="range"
              min={0.05}
              max={0.20}
              step={0.05}
              value={testSplit}
              onChange={(e) => setTestSplit(Number(e.target.value))}
              className="w-full accent-yellow-500"
            />
            <p className="text-[11px] text-[#484f58] mt-1">
              {Math.round(testSplit * 100)}% test
            </p>
          </div>
        </div>

        {/* Derived split summary */}
        <div className="flex items-center justify-between">
          <SplitPreview valSplit={valSplit} testSplit={testSplit} />

          <button
            onClick={handleConvert}
            disabled={
              !sourcePath.trim() ||
              !outputPath.trim() ||
              convertMutation.isPending ||
              (activeTaskId !== null && jobStatus === "running")
            }
            className="flex items-center gap-2 px-5 py-2 rounded-lg text-sm font-medium
              bg-blue-600 hover:bg-blue-500 text-white disabled:opacity-50
              disabled:cursor-not-allowed transition-colors"
          >
            {convertMutation.isPending ? (
              <Loader2 className="w-3.5 h-3.5 animate-spin" />
            ) : (
              <ArrowRight className="w-3.5 h-3.5" />
            )}
            Convert
          </button>
        </div>

        {convertMutation.isError && (
          <p className="text-xs text-red-400 bg-red-500/10 rounded px-3 py-2">
            {convertMutation.error.message}
          </p>
        )}
      </div>

      {/* ── Active job card ────────────────────────────────────── */}
      {activeTaskId && (
        <div className="rounded-xl border border-[#21262d] bg-[#161b22] p-5 space-y-3">
          <div className="flex items-center gap-2">
            <FileStack className="w-4 h-4 text-blue-400" />
            <h2 className="text-sm font-semibold text-[#e6edf3]">
              Active Conversion Job
            </h2>
          </div>
          <ConversionJobCard
            key={activeTaskId}
            taskId={activeTaskId}
            onComplete={handleJobComplete}
          />
        </div>
      )}

      {/* ── Store inspector ────────────────────────────────────── */}
      <div className="rounded-xl border border-[#21262d] bg-[#161b22] p-5 space-y-3">
        <div className="flex items-center gap-2">
          <HardDrive className="w-4 h-4 text-cyan-400" />
          <h2 className="text-sm font-semibold text-[#e6edf3]">
            Inspect Streaming Store
          </h2>
        </div>
        <StoreInspector
          initialPath={inspectTriggerPath}
          onInspected={handleInspected}
        />
      </div>

      {/* ── Recent conversions ─────────────────────────────────── */}
      {recentConversions.length > 0 && (
        <div className="rounded-xl border border-[#21262d] bg-[#161b22] p-5 space-y-3">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-2">
              <Database className="w-4 h-4 text-[#484f58]" />
              <h2 className="text-sm font-semibold text-[#e6edf3]">
                Recent Conversions
              </h2>
              <span className="text-[10px] px-1.5 py-0.5 rounded bg-[#21262d] text-[#484f58]">
                session
              </span>
            </div>
            <button
              onClick={() => setRecentConversions([])}
              className="flex items-center gap-1 text-xs text-[#484f58] hover:text-red-400 transition-colors"
            >
              <Trash2 className="w-3 h-3" />
              Clear
            </button>
          </div>

          <div className="space-y-1.5">
            {recentConversions.map((rc, idx) => {
              const basename = rc.source_path.split("/").pop() ?? rc.source_path;
              return (
                <div
                  key={idx}
                  className="flex items-center gap-3 px-3 py-2.5 rounded-lg
                    bg-[#0d1117] border border-[#21262d] text-sm"
                >
                  <div className="flex-1 min-w-0">
                    <p className="text-[#e6edf3] truncate text-xs font-medium">
                      {basename}
                    </p>
                    <p className="text-[11px] text-[#484f58] truncate mt-0.5">
                      {rc.output_path}
                    </p>
                  </div>
                  <FormatBadge format={rc.format} />
                  <div className="text-right flex-shrink-0">
                    <p className="text-xs text-[#e6edf3]">
                      {rc.stats.total_images.toLocaleString()} imgs
                    </p>
                    <p className="text-[11px] text-[#484f58]">
                      {rc.stats.store_size_mb.toFixed(1)} MB
                    </p>
                  </div>
                  <button
                    onClick={() => setInspectTriggerPath(rc.output_path)}
                    className="flex items-center gap-1 text-[11px] text-blue-400
                      hover:text-blue-300 transition-colors flex-shrink-0"
                    title="Inspect this store"
                  >
                    <ExternalLink className="w-3 h-3" />
                    Inspect
                  </button>
                </div>
              );
            })}
          </div>
        </div>
      )}
    </div>
  );
}

// ---------------------------------------------------------------------------
// Main Datasets page
// ---------------------------------------------------------------------------

type PageTab = "datasets" | "streaming";

export default function DatasetsPage() {
  const queryClient = useQueryClient();

  const [pageTab, setPageTab] = useState<PageTab>("datasets");
  const [selectedDatasetId, setSelectedDatasetId] = useState<number | null>(null);
  const [viewMode, setViewMode] = useState<"list" | "grid">("list");
  const [showCreateModal, setShowCreateModal] = useState(false);
  const [showUpload, setShowUpload] = useState(false);
  const [filters, setFilters] = useState<DatasetFilters>(DEFAULT_FILTERS);
  const [selectedSampleIds, setSelectedSampleIds] = useState<Set<number>>(new Set());
  const [columnCount, setColumnCount] = useState(5);

  // ---------------------------------------------------------------------------
  // Queries
  // ---------------------------------------------------------------------------

  const {
    data: datasets = [],
    isLoading: datasetsLoading,
    refetch: refetchDatasets,
  } = useQuery<DatasetSummary[]>({
    queryKey: ["datasets"],
    queryFn: () => api.get("/datasets").then((r) => r.data),
    staleTime: 30_000,
  });

  const {
    data: samples = [],
    isLoading: samplesLoading,
    refetch: refetchSamples,
  } = useQuery<SampleSummary[]>({
    queryKey: ["dataset-samples", selectedDatasetId, filters.split],
    queryFn: () =>
      selectedDatasetId
        ? api
            .get(`/datasets/${selectedDatasetId}/samples`, {
              params: {
                split: filters.split !== "all" ? filters.split : undefined,
                limit: 2000,
              },
            })
            .then((r) => r.data)
        : Promise.resolve([]),
    enabled: selectedDatasetId !== null,
    staleTime: 60_000,
  });

  // ---------------------------------------------------------------------------
  // Mutations
  // ---------------------------------------------------------------------------

  const createDatasetMutation = useMutation({
    mutationFn: ({ name, description }: { name: string; description: string }) =>
      api.post("/datasets", { name, description }).then((r) => r.data),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["datasets"] });
    },
  });

  // ---------------------------------------------------------------------------
  // Filtered samples (client-side)
  // ---------------------------------------------------------------------------

  const filteredSamples = useMemo(() => {
    return samples.filter((s) => {
      if (filters.search && !s.filename.toLowerCase().includes(filters.search.toLowerCase()))
        return false;
      if (filters.reviewed === "reviewed" && !s.reviewed) return false;
      if (filters.reviewed === "unreviewed" && s.reviewed) return false;
      const q = s.quality_score ?? 1;
      if (q < filters.minQuality || q > filters.maxQuality) return false;
      if (s.num_annotations < filters.minAnnotations) return false;
      return true;
    });
  }, [samples, filters]);

  // ---------------------------------------------------------------------------
  // Selection handlers
  // ---------------------------------------------------------------------------

  const handleSelect = useCallback(
    (id: number, multiSelect: boolean) => {
      setSelectedSampleIds((prev) => {
        const next = new Set(prev);
        if (multiSelect) {
          if (next.has(id)) next.delete(id);
          else next.add(id);
        } else {
          if (next.has(id) && next.size === 1) next.clear();
          else {
            next.clear();
            next.add(id);
          }
        }
        return next;
      });
    },
    []
  );

  const handleSelectAll = useCallback(() => {
    setSelectedSampleIds(new Set(filteredSamples.map((s) => s.id)));
  }, [filteredSamples]);

  const handleDeselectAll = useCallback(() => {
    setSelectedSampleIds(new Set());
  }, []);

  const selectedDataset = datasets.find((d) => d.id === selectedDatasetId);

  // ---------------------------------------------------------------------------
  // Render
  // ---------------------------------------------------------------------------

  return (
    <div className="flex flex-col h-full">
      {/* ── Top-level tab bar ──────────────────────────────────── */}
      <div className="flex items-center gap-1 px-4 pt-3 pb-0 border-b border-[var(--color-border)] flex-shrink-0">
        <button
          onClick={() => setPageTab("datasets")}
          className={cn(
            "flex items-center gap-1.5 px-3 py-2 text-sm font-medium rounded-t-lg transition-all border-b-2",
            pageTab === "datasets"
              ? "text-white border-blue-500"
              : "text-[#484f58] border-transparent hover:text-[#e6edf3]"
          )}
        >
          <Database className="w-3.5 h-3.5" />
          Datasets
        </button>
        <button
          onClick={() => setPageTab("streaming")}
          className={cn(
            "flex items-center gap-1.5 px-3 py-2 text-sm font-medium rounded-t-lg transition-all border-b-2",
            pageTab === "streaming"
              ? "text-white border-purple-500"
              : "text-[#484f58] border-transparent hover:text-[#e6edf3]"
          )}
        >
          <Layers className="w-3.5 h-3.5" />
          Streaming Formats
        </button>
      </div>

      {/* ── Tab content ────────────────────────────────────────── */}
      {pageTab === "streaming" ? (
        <StreamingTab />
      ) : (
        <div className="flex flex-1 min-h-0">
          {/* Left panel: dataset list */}
          <div className="w-64 flex-shrink-0 border-r border-[var(--color-border)] flex flex-col">
            {/* Header */}
            <div className="flex items-center justify-between px-4 py-3 border-b border-[var(--color-border)]">
              <div className="flex items-center gap-2">
                <Database className="w-4 h-4 text-blue-400" />
                <h2 className="text-sm font-semibold text-white">Datasets</h2>
              </div>
              <button
                onClick={() => setShowCreateModal(true)}
                className="p-1 rounded hover:bg-[var(--color-surface)] text-[var(--color-text-muted)] hover:text-white transition-colors"
                title="Create dataset"
              >
                <Plus className="w-4 h-4" />
              </button>
            </div>

            {/* List */}
            <div className="flex-1 overflow-y-auto py-2 px-2 space-y-0.5">
              {datasetsLoading ? (
                <div className="flex items-center justify-center py-8">
                  <Loader2 className="w-5 h-5 text-[var(--color-text-muted)] animate-spin" />
                </div>
              ) : datasets.length === 0 ? (
                <div className="text-center py-8 px-3">
                  <p className="text-sm text-[var(--color-text-muted)]">No datasets yet</p>
                  <button
                    onClick={() => setShowCreateModal(true)}
                    className="mt-2 text-xs text-blue-400 hover:text-blue-300"
                  >
                    Create your first dataset →
                  </button>
                </div>
              ) : (
                datasets.map((dataset) => (
                  <button
                    key={dataset.id}
                    onClick={() => {
                      setSelectedDatasetId(dataset.id);
                      setSelectedSampleIds(new Set());
                    }}
                    className={cn(
                      "w-full text-left px-3 py-2 rounded-lg transition-all",
                      "text-sm flex items-center gap-2.5 group",
                      selectedDatasetId === dataset.id
                        ? "bg-blue-600/20 text-white border border-blue-500/30"
                        : "text-[var(--color-text-muted)] hover:text-[var(--color-text-secondary)] hover:bg-[var(--color-surface)]"
                    )}
                  >
                    <Database className="w-3.5 h-3.5 flex-shrink-0" />
                    <div className="min-w-0 flex-1">
                      <p className="truncate font-medium">{dataset.name}</p>
                      <p className="text-[10px] text-[var(--color-text-muted)]">
                        {dataset.num_samples.toLocaleString()} images
                      </p>
                    </div>
                  </button>
                ))
              )}
            </div>

            {/* Footer */}
            <div className="border-t border-[var(--color-border)] px-4 py-3">
              <button
                onClick={() => refetchDatasets()}
                className="flex items-center gap-1.5 text-xs text-[var(--color-text-muted)] hover:text-[var(--color-text-secondary)] transition-colors"
              >
                <RefreshCw className="w-3 h-3" />
                Refresh
              </button>
            </div>
          </div>

          {/* Right panel: sample browser */}
          <div className="flex-1 flex flex-col min-w-0">
            {selectedDatasetId ? (
              <>
                {/* Dataset toolbar */}
                <div className="flex items-center justify-between px-5 py-3 border-b border-[var(--color-border)]">
                  <div>
                    <h1 className="text-base font-semibold text-white">
                      {selectedDataset?.name ?? "Dataset"}
                    </h1>
                    <p className="text-xs text-[var(--color-text-muted)]">
                      {samples.length.toLocaleString()} images
                      {filteredSamples.length !== samples.length &&
                        ` · ${filteredSamples.length} matching filters`}
                    </p>
                  </div>

                  <div className="flex items-center gap-2">
                    {/* Column count */}
                    <div className="flex items-center gap-1 text-xs text-[var(--color-text-muted)]">
                      <span>Cols:</span>
                      {[3, 4, 5, 6, 8].map((n) => (
                        <button
                          key={n}
                          onClick={() => setColumnCount(n)}
                          className={cn(
                            "px-1.5 py-0.5 rounded transition-colors",
                            columnCount === n
                              ? "text-white bg-blue-600"
                              : "hover:text-[var(--color-text-secondary)]"
                          )}
                        >
                          {n}
                        </button>
                      ))}
                    </div>

                    {/* Upload toggle */}
                    <button
                      onClick={() => setShowUpload((v) => !v)}
                      className={cn(
                        "flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-xs font-medium transition-all border",
                        showUpload
                          ? "bg-blue-600/20 border-blue-500/40 text-blue-300"
                          : "border-[var(--color-border)] text-[var(--color-text-muted)] hover:text-white hover:border-[var(--color-border-hover)]"
                      )}
                    >
                      <Plus className="w-3.5 h-3.5" />
                      Upload
                    </button>

                    {/* View toggle */}
                    <div className="flex border border-[var(--color-border)] rounded-lg overflow-hidden">
                      <button
                        onClick={() => setViewMode("grid")}
                        className={cn(
                          "p-1.5 transition-colors",
                          viewMode === "grid"
                            ? "bg-blue-600 text-white"
                            : "text-[var(--color-text-muted)] hover:text-white"
                        )}
                      >
                        <LayoutGrid className="w-4 h-4" />
                      </button>
                      <button
                        onClick={() => setViewMode("list")}
                        className={cn(
                          "p-1.5 transition-colors",
                          viewMode === "list"
                            ? "bg-blue-600 text-white"
                            : "text-[var(--color-text-muted)] hover:text-white"
                        )}
                      >
                        <List className="w-4 h-4" />
                      </button>
                    </div>

                    <button
                      onClick={() => refetchSamples()}
                      className="p-1.5 rounded text-[var(--color-text-muted)] hover:text-white hover:bg-[var(--color-surface)] transition-colors"
                    >
                      <RefreshCw className="w-4 h-4" />
                    </button>
                  </div>
                </div>

                {/* Upload zone (collapsible) */}
                {showUpload && (
                  <div className="px-5 py-4 border-b border-[var(--color-border)] bg-[var(--color-surface)]">
                    <UploadZone
                      datasetId={selectedDatasetId}
                      onUploadComplete={() => {
                        queryClient.invalidateQueries({
                          queryKey: ["dataset-samples", selectedDatasetId],
                        });
                        queryClient.invalidateQueries({ queryKey: ["datasets"] });
                      }}
                    />
                  </div>
                )}

                {/* Filters */}
                <div className="px-5 py-3 border-b border-[var(--color-border)]">
                  <FilterPanel
                    filters={filters}
                    onChange={setFilters}
                    totalCount={samples.length}
                    filteredCount={filteredSamples.length}
                  />
                </div>

                {/* Selection toolbar */}
                {selectedSampleIds.size > 0 && (
                  <div className="px-5 py-2 border-b border-[var(--color-border)]">
                    <SelectionToolbar
                      count={selectedSampleIds.size}
                      total={filteredSamples.length}
                      onSelectAll={handleSelectAll}
                      onDeselectAll={handleDeselectAll}
                    />
                  </div>
                )}

                {/* Image grid / list */}
                <div className="flex-1 overflow-hidden px-5 py-4">
                  {samplesLoading ? (
                    <div className="flex items-center justify-center py-20">
                      <Loader2 className="w-6 h-6 text-[var(--color-text-muted)] animate-spin" />
                    </div>
                  ) : viewMode === "grid" ? (
                    <ImageGrid
                      samples={filteredSamples}
                      selectedIds={selectedSampleIds}
                      onSelect={handleSelect}
                      onOpen={(s) => {
                        // TODO: open ImageViewer
                        console.log("Open:", s.filename);
                      }}
                      columnCount={columnCount}
                    />
                  ) : (
                    /* List view */
                    <div className="space-y-1 overflow-y-auto max-h-[calc(100vh-280px)] pr-1">
                      {filteredSamples.length === 0 ? (
                        <div className="text-center py-12 text-[var(--color-text-muted)]">
                          No samples match the current filters
                        </div>
                      ) : (
                        filteredSamples.map((sample) => (
                          <div
                            key={sample.id}
                            onClick={() => handleSelect(sample.id, false)}
                            className={cn(
                              "flex items-center gap-3 px-3 py-2 rounded-lg cursor-pointer transition-all border",
                              selectedSampleIds.has(sample.id)
                                ? "border-blue-500/40 bg-blue-500/5"
                                : "border-transparent hover:border-[var(--color-border)] hover:bg-[var(--color-surface)]"
                            )}
                          >
                            <span className="text-xs font-mono text-[var(--color-text-muted)] w-6">
                              {sample.split.charAt(0).toUpperCase()}
                            </span>
                            <span className="flex-1 text-sm text-[var(--color-text-secondary)] truncate">
                              {sample.filename}
                            </span>
                            <span className="text-xs text-[var(--color-text-muted)]">
                              {sample.num_annotations} ann.
                            </span>
                            {sample.quality_score !== undefined && (
                              <span className="text-xs font-mono text-green-400">
                                {Math.round((sample.quality_score ?? 0) * 100)}%
                              </span>
                            )}
                          </div>
                        ))
                      )}
                    </div>
                  )}
                </div>
              </>
            ) : (
              /* No dataset selected */
              <div className="flex-1 flex flex-col items-center justify-center gap-4 text-[var(--color-text-muted)]">
                <Database className="w-12 h-12 opacity-30" />
                <div className="text-center">
                  <p className="text-base font-medium">Select a dataset</p>
                  <p className="text-sm mt-1">or create a new one to get started</p>
                </div>
                <button
                  onClick={() => setShowCreateModal(true)}
                  className="flex items-center gap-2 px-4 py-2 rounded-lg bg-blue-600 hover:bg-blue-500 text-white text-sm font-medium transition-colors"
                >
                  <Plus className="w-4 h-4" />
                  Create Dataset
                </button>
              </div>
            )}
          </div>
        </div>
      )}

      {/* Create modal */}
      {showCreateModal && (
        <CreateDatasetModal
          onClose={() => setShowCreateModal(false)}
          onCreate={(name, description) =>
            createDatasetMutation.mutate({ name, description })
          }
        />
      )}
    </div>
  );
}
