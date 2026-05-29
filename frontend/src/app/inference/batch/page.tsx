"use client";

/**
 * Batch Inference — dataset-scale trichome detection.
 *
 * Submits multiple images to POST /inference/detect/batch (multipart, files[])
 * and presents a results table with per-image stats plus aggregate KPIs.
 */

import React, { useCallback, useState } from "react";
import {
  Upload,
  Loader2,
  AlertTriangle,
  Cpu,
  Download,
  Trash2,
  BarChart3,
  ArrowLeft,
  CheckCircle2,
  XCircle,
  Layers,
} from "lucide-react";
import { useDropzone } from "react-dropzone";
import Link from "next/link";
import { api } from "@/lib/api";
import { cn, formatConfidence } from "@/lib/utils";

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

interface BatchDetectionResultItem {
  filename: string;
  width?: number;
  height?: number;
  num_detections: number;
  inference_time_ms?: number;
  processing_time_ms?: number;
  mean_confidence?: number;
  status: "ok" | "error";
  error?: string;
  detections?: unknown[];
}

interface BatchDetectionResponse {
  results: BatchDetectionResultItem[];
  total_images: number;
  total_detections: number;
  avg_inference_time_ms?: number;
  failed_count?: number;
}

type ProcessingStatus = "idle" | "running" | "done" | "error";

interface FileResult extends BatchDetectionResultItem {
  /** local index for table key */
  _idx: number;
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

function inferenceMs(item: BatchDetectionResultItem): number | null {
  return item.inference_time_ms ?? item.processing_time_ms ?? null;
}

// ---------------------------------------------------------------------------
// Stats card
// ---------------------------------------------------------------------------

function StatCard({
  label,
  value,
  color,
}: {
  label: string;
  value: string | number;
  color?: string;
}) {
  return (
    <div
      className="px-4 py-3 rounded-xl flex-1 min-w-0"
      style={{ background: "#0d1117", border: "1px solid #21262d" }}
    >
      <p className="text-[10px] uppercase tracking-wide mb-1" style={{ color: "#484f58" }}>
        {label}
      </p>
      <p
        className="text-2xl font-bold font-mono truncate"
        style={{ color: color ?? "#e6edf3" }}
      >
        {value}
      </p>
    </div>
  );
}

// ---------------------------------------------------------------------------
// Results table row
// ---------------------------------------------------------------------------

function ResultRow({ item, idx }: { item: FileResult; idx: number }) {
  const ms = inferenceMs(item);
  const isOk = item.status === "ok";

  return (
    <tr
      className={cn("border-b text-sm transition-colors", idx % 2 === 0 ? "" : "")}
      style={{ borderColor: "#21262d" }}
    >
      {/* Status */}
      <td className="px-3 py-2.5 w-8">
        {isOk ? (
          <CheckCircle2 className="w-3.5 h-3.5 text-green-400" />
        ) : (
          <XCircle className="w-3.5 h-3.5 text-red-400" />
        )}
      </td>

      {/* Filename */}
      <td className="px-3 py-2.5 max-w-xs">
        <span
          className="block truncate text-xs font-mono"
          title={item.filename}
          style={{ color: "#8b949e" }}
        >
          {item.filename}
        </span>
        {!isOk && (
          <span className="block text-[10px] text-red-400 truncate" title={item.error}>
            {item.error}
          </span>
        )}
      </td>

      {/* Dimensions */}
      <td className="px-3 py-2.5 text-xs font-mono text-right" style={{ color: "#484f58" }}>
        {item.width && item.height ? `${item.width}×${item.height}` : "—"}
      </td>

      {/* Detections */}
      <td className="px-3 py-2.5 text-right">
        <span
          className="text-sm font-bold font-mono"
          style={{ color: isOk ? "#60a5fa" : "#484f58" }}
        >
          {isOk ? item.num_detections : "—"}
        </span>
      </td>

      {/* Mean confidence */}
      <td className="px-3 py-2.5 text-right">
        <span
          className="text-xs font-mono"
          style={{
            color:
              item.mean_confidence == null
                ? "#484f58"
                : item.mean_confidence >= 0.7
                ? "#22c55e"
                : item.mean_confidence >= 0.5
                ? "#eab308"
                : "#ef4444",
          }}
        >
          {item.mean_confidence != null ? formatConfidence(item.mean_confidence) : "—"}
        </span>
      </td>

      {/* Inference time */}
      <td className="px-3 py-2.5 text-right text-xs font-mono" style={{ color: "#484f58" }}>
        {ms != null ? `${ms.toFixed(0)} ms` : "—"}
      </td>
    </tr>
  );
}

// ---------------------------------------------------------------------------
// Page component
// ---------------------------------------------------------------------------

export default function BatchInferencePage() {
  const [files, setFiles] = useState<File[]>([]);
  const [confThreshold, setConfThreshold] = useState(0.35);
  const [modelVariant, setModelVariant] = useState<"yolo11n" | "yolo11s" | "yolo11m">("yolo11s");
  const [useTiled, setUseTiled] = useState(false);
  const [status, setStatus] = useState<ProcessingStatus>("idle");
  const [progress, setProgress] = useState({ done: 0, total: 0 });
  const [results, setResults] = useState<FileResult[]>([]);
  const [aggregated, setAggregated] = useState<BatchDetectionResponse | null>(null);
  const [errorMsg, setErrorMsg] = useState<string | null>(null);

  // -- Dropzone -----------------------------------------------------------------
  const onDrop = useCallback((accepted: File[]) => {
    setFiles((prev) => {
      const existingNames = new Set(prev.map((f) => f.name));
      const fresh = accepted.filter((f) => !existingNames.has(f.name));
      return [...prev, ...fresh];
    });
    setStatus("idle");
    setResults([]);
    setAggregated(null);
    setErrorMsg(null);
  }, []);

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept: {
      "image/jpeg": [".jpg", ".jpeg"],
      "image/png": [".png"],
      "image/tiff": [".tif", ".tiff"],
    },
    multiple: true,
  });

  // -- Submit -------------------------------------------------------------------
  const runBatch = async () => {
    if (files.length === 0) return;

    setStatus("running");
    setProgress({ done: 0, total: files.length });
    setResults([]);
    setAggregated(null);
    setErrorMsg(null);

    const formData = new FormData();
    files.forEach((f) => formData.append("files", f));
    formData.append("conf_threshold", String(confThreshold));
    formData.append("model_variant", modelVariant);
    formData.append("use_tiled", String(useTiled));

    try {
      // Stream-like progress simulation while waiting for response
      const interval = setInterval(() => {
        setProgress((p) => ({
          ...p,
          done: Math.min(p.done + 1, Math.max(1, Math.floor(p.total * 0.8))),
        }));
      }, 600);

      const response = await api.post<BatchDetectionResponse>(
        "/inference/detect/batch",
        formData,
        {
          headers: { "Content-Type": "multipart/form-data" },
          timeout: 600_000, // 10 min for large batches
        },
      );

      clearInterval(interval);

      const data = response.data;
      setAggregated(data);

      const mapped: FileResult[] = (data.results ?? []).map((r, i) => ({ ...r, _idx: i }));
      setResults(mapped);
      setProgress({ done: files.length, total: files.length });
      setStatus("done");
    } catch (err: unknown) {
      const msg = err instanceof Error ? err.message : "Batch detection failed";
      setErrorMsg(msg);
      setStatus("error");
    }
  };

  // -- Export -------------------------------------------------------------------
  const exportJson = () => {
    if (!aggregated) return;
    const blob = new Blob([JSON.stringify(aggregated, null, 2)], {
      type: "application/json",
    });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = `batch_results_${Date.now()}.json`;
    a.click();
    URL.revokeObjectURL(url);
  };

  // -- Derived stats ------------------------------------------------------------
  const totalDets = aggregated?.total_detections ?? 0;
  const avgInfMs = aggregated?.avg_inference_time_ms;
  const avgDetsPerImage =
    aggregated && aggregated.total_images > 0
      ? (totalDets / aggregated.total_images).toFixed(1)
      : "—";

  // ---------------------------------------------------------------------------
  return (
    <div className="flex flex-col h-full" style={{ background: "#010409" }}>
      {/* ── Header ── */}
      <div
        className="flex items-center gap-4 px-5 py-3 flex-shrink-0"
        style={{ borderBottom: "1px solid #21262d", background: "#0d1117" }}
      >
        <Link
          href="/inference"
          className="flex items-center gap-1.5 text-xs transition-colors"
          style={{ color: "#484f58" }}
        >
          <ArrowLeft className="w-3.5 h-3.5" />
          Inference
        </Link>
        <span style={{ color: "#21262d" }}>/</span>
        <div className="flex items-center gap-2">
          <Layers className="w-4 h-4 text-blue-400" />
          <h1 className="text-base font-semibold text-white">Batch Detection</h1>
        </div>
      </div>

      <div className="flex flex-1 min-h-0 overflow-hidden">
        {/* ── Left: config + dropzone ── */}
        <div
          className="w-72 flex-shrink-0 flex flex-col overflow-y-auto"
          style={{ borderRight: "1px solid #21262d" }}
        >
          <div className="p-4 space-y-5">
            {/* Dropzone */}
            <div>
              <p className="text-xs font-medium mb-2" style={{ color: "#8b949e" }}>
                Images
              </p>
              <div
                {...getRootProps()}
                className={cn(
                  "flex flex-col items-center justify-center gap-3 p-5 rounded-xl",
                  "border-2 border-dashed cursor-pointer transition-all",
                )}
                style={{
                  borderColor: isDragActive ? "#3b82f6" : "#21262d",
                  background: isDragActive ? "rgba(59,130,246,0.08)" : "transparent",
                }}
              >
                <input {...getInputProps()} />
                <Upload className="w-7 h-7" style={{ color: isDragActive ? "#3b82f6" : "#484f58" }} />
                <div className="text-center">
                  <p className="text-xs font-medium" style={{ color: "#8b949e" }}>
                    {isDragActive ? "Drop images here" : "Drop or click to add images"}
                  </p>
                  <p className="text-[10px] mt-0.5" style={{ color: "#484f58" }}>
                    PNG · JPG · TIFF
                  </p>
                </div>
              </div>

              {files.length > 0 && (
                <div className="mt-2 flex items-center justify-between">
                  <span className="text-xs" style={{ color: "#484f58" }}>
                    {files.length} file{files.length !== 1 ? "s" : ""} queued
                  </span>
                  <button
                    onClick={() => {
                      setFiles([]);
                      setResults([]);
                      setAggregated(null);
                      setStatus("idle");
                    }}
                    className="text-[10px] flex items-center gap-1 transition-colors"
                    style={{ color: "#484f58" }}
                  >
                    <Trash2 className="w-3 h-3" />
                    Clear all
                  </button>
                </div>
              )}
            </div>

            {/* Configuration */}
            <div className="space-y-4">
              <p className="text-xs font-medium" style={{ color: "#8b949e" }}>
                Configuration
              </p>

              {/* Conf threshold */}
              <div>
                <div className="flex items-center justify-between mb-1.5">
                  <label className="text-xs" style={{ color: "#484f58" }}>
                    Confidence Threshold
                  </label>
                  <span className="text-xs font-mono" style={{ color: "#8b949e" }}>
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
                  style={{ background: "#21262d" }}
                />
                <div className="flex justify-between text-[10px] mt-0.5" style={{ color: "#484f58" }}>
                  <span>0.1</span>
                  <span>0.9</span>
                </div>
              </div>

              {/* Model variant */}
              <div>
                <label className="text-xs mb-1.5 block" style={{ color: "#484f58" }}>
                  Model Variant
                </label>
                <select
                  value={modelVariant}
                  onChange={(e) =>
                    setModelVariant(e.target.value as "yolo11n" | "yolo11s" | "yolo11m")
                  }
                  className="w-full px-3 py-1.5 text-xs rounded-lg focus:outline-none"
                  style={{
                    background: "#0d1117",
                    border: "1px solid #21262d",
                    color: "#8b949e",
                  }}
                >
                  <option value="yolo11n">yolo11n — Nano (fastest)</option>
                  <option value="yolo11s">yolo11s — Small (default)</option>
                  <option value="yolo11m">yolo11m — Medium (best accuracy)</option>
                </select>
              </div>

              {/* Tiled toggle */}
              <div className="flex items-center justify-between">
                <div>
                  <p className="text-xs" style={{ color: "#8b949e" }}>
                    Tiled Inference
                  </p>
                  <p className="text-[10px]" style={{ color: "#484f58" }}>
                    Recommended for images &gt; 1280px
                  </p>
                </div>
                <button
                  onClick={() => setUseTiled((v) => !v)}
                  className={cn(
                    "relative inline-flex h-5 w-9 items-center rounded-full transition-colors",
                  )}
                  style={{ background: useTiled ? "#2563eb" : "#21262d" }}
                  role="switch"
                  aria-checked={useTiled}
                >
                  <span
                    className={cn(
                      "inline-block h-3.5 w-3.5 transform rounded-full bg-white transition-transform",
                      useTiled ? "translate-x-4" : "translate-x-0.5",
                    )}
                  />
                </button>
              </div>
            </div>

            {/* Submit */}
            <button
              onClick={runBatch}
              disabled={files.length === 0 || status === "running"}
              className={cn(
                "w-full flex items-center justify-center gap-2 py-2.5 rounded-lg",
                "text-sm font-medium transition-all",
              )}
              style={{
                background:
                  files.length === 0 || status === "running"
                    ? "rgba(37,99,235,0.3)"
                    : "#1d4ed8",
                color:
                  files.length === 0 || status === "running"
                    ? "rgba(147,197,253,0.5)"
                    : "white",
                cursor:
                  files.length === 0 || status === "running" ? "not-allowed" : "pointer",
              }}
            >
              {status === "running" ? (
                <>
                  <Loader2 className="w-4 h-4 animate-spin" />
                  Running…
                </>
              ) : (
                <>
                  <Cpu className="w-4 h-4" />
                  Run Batch Detection
                </>
              )}
            </button>

            {files.length === 0 && (
              <p className="text-[10px] text-center" style={{ color: "#484f58" }}>
                Add at least one image
              </p>
            )}
          </div>
        </div>

        {/* ── Right: progress + results ── */}
        <div className="flex-1 flex flex-col min-w-0 overflow-hidden">
          {/* Progress bar (while running) */}
          {status === "running" && (
            <div
              className="px-5 py-3 flex-shrink-0"
              style={{ borderBottom: "1px solid #21262d" }}
            >
              <div className="flex items-center justify-between mb-1.5 text-xs">
                <span style={{ color: "#8b949e" }}>Processing images…</span>
                <span className="font-mono" style={{ color: "#484f58" }}>
                  {progress.done} / {progress.total}
                </span>
              </div>
              <div className="h-1.5 rounded-full overflow-hidden" style={{ background: "#21262d" }}>
                <div
                  className="h-full bg-blue-500 transition-all duration-500"
                  style={{
                    width: `${progress.total > 0 ? (progress.done / progress.total) * 100 : 0}%`,
                  }}
                />
              </div>
            </div>
          )}

          {/* Error banner */}
          {status === "error" && errorMsg && (
            <div
              className="mx-5 mt-4 flex items-start gap-2 px-4 py-3 rounded-xl flex-shrink-0"
              style={{
                background: "rgba(239,68,68,0.1)",
                border: "1px solid rgba(239,68,68,0.2)",
              }}
            >
              <AlertTriangle className="w-4 h-4 text-red-400 flex-shrink-0 mt-0.5" />
              <div>
                <p className="text-sm font-medium text-red-400">Batch detection failed</p>
                <p className="text-xs mt-0.5" style={{ color: "rgba(252,165,165,0.7)" }}>
                  {errorMsg}
                </p>
              </div>
            </div>
          )}

          {/* Stats + export (when done) */}
          {status === "done" && aggregated && (
            <div
              className="px-5 pt-4 pb-3 flex-shrink-0"
              style={{ borderBottom: "1px solid #21262d" }}
            >
              <div className="flex items-center justify-between mb-3">
                <div className="flex items-center gap-2">
                  <BarChart3 className="w-4 h-4 text-blue-400" />
                  <h2 className="text-sm font-semibold text-white">Batch Summary</h2>
                </div>
                <button
                  onClick={exportJson}
                  className="flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-xs font-medium transition-all"
                  style={{ background: "rgba(34,197,94,0.15)", color: "#4ade80", border: "1px solid rgba(34,197,94,0.25)" }}
                >
                  <Download className="w-3.5 h-3.5" />
                  Export JSON
                </button>
              </div>

              <div className="flex gap-3">
                <StatCard
                  label="Total Images"
                  value={aggregated.total_images}
                  color="#60a5fa"
                />
                <StatCard
                  label="Total Detections"
                  value={totalDets}
                  color="#a78bfa"
                />
                <StatCard
                  label="Avg Inference"
                  value={avgInfMs != null ? `${avgInfMs.toFixed(0)} ms` : "—"}
                  color="#8b949e"
                />
                <StatCard
                  label="Avg Dets / Image"
                  value={avgDetsPerImage}
                  color="#34d399"
                />
              </div>

              {/* Scientific caveat */}
              <div
                className="flex items-start gap-2 px-3 py-2 rounded-lg mt-3"
                style={{
                  background: "rgba(234,179,8,0.08)",
                  border: "1px solid rgba(234,179,8,0.15)",
                }}
              >
                <AlertTriangle className="w-3.5 h-3.5 text-yellow-400 flex-shrink-0 mt-0.5" />
                <p className="text-[11px]" style={{ color: "rgba(254,243,199,0.75)" }}>
                  Maturity stage reflects optical properties only. No inference about THC, CBD or
                  other cannabinoid concentrations can be made from visual appearance.
                </p>
              </div>
            </div>
          )}

          {/* Idle placeholder */}
          {status === "idle" && files.length === 0 && (
            <div className="flex-1 flex items-center justify-center">
              <div className="text-center" style={{ color: "#484f58" }}>
                <Layers className="w-12 h-12 mx-auto mb-4 opacity-20" />
                <p className="text-sm font-medium">No images loaded</p>
                <p className="text-xs mt-1 opacity-60">
                  Add images via drag & drop or the file picker on the left
                </p>
              </div>
            </div>
          )}

          {status === "idle" && files.length > 0 && (
            <div className="flex-1 flex items-center justify-center">
              <div className="text-center" style={{ color: "#484f58" }}>
                <Cpu className="w-10 h-10 mx-auto mb-3 opacity-30" />
                <p className="text-sm">
                  {files.length} image{files.length !== 1 ? "s" : ""} ready
                </p>
                <p className="text-xs mt-1 opacity-60">Press &quot;Run Batch Detection&quot; to start</p>
              </div>
            </div>
          )}

          {/* Results table */}
          {results.length > 0 && (
            <div className="flex-1 overflow-y-auto">
              <table className="w-full text-left border-collapse">
                <thead className="sticky top-0" style={{ background: "#0d1117" }}>
                  <tr style={{ borderBottom: "1px solid #21262d" }}>
                    <th className="px-3 py-2.5 w-8" />
                    <th
                      className="px-3 py-2.5 text-[10px] font-medium uppercase tracking-wide"
                      style={{ color: "#484f58" }}
                    >
                      Filename
                    </th>
                    <th
                      className="px-3 py-2.5 text-[10px] font-medium uppercase tracking-wide text-right"
                      style={{ color: "#484f58" }}
                    >
                      Dimensions
                    </th>
                    <th
                      className="px-3 py-2.5 text-[10px] font-medium uppercase tracking-wide text-right"
                      style={{ color: "#484f58" }}
                    >
                      Detections
                    </th>
                    <th
                      className="px-3 py-2.5 text-[10px] font-medium uppercase tracking-wide text-right"
                      style={{ color: "#484f58" }}
                    >
                      Avg Conf
                    </th>
                    <th
                      className="px-3 py-2.5 text-[10px] font-medium uppercase tracking-wide text-right"
                      style={{ color: "#484f58" }}
                    >
                      Inference Time
                    </th>
                  </tr>
                </thead>
                <tbody>
                  {results.map((item) => (
                    <ResultRow key={item._idx} item={item} idx={item._idx} />
                  ))}
                </tbody>
              </table>
            </div>
          )}

          {/* Running skeleton */}
          {status === "running" && results.length === 0 && (
            <div className="flex-1 flex items-center justify-center">
              <div className="text-center" style={{ color: "#484f58" }}>
                <Loader2 className="w-8 h-8 animate-spin mx-auto mb-3 text-blue-400" />
                <p className="text-sm">Running batch inference…</p>
                <p className="text-xs mt-1 opacity-60">
                  {progress.done} of {progress.total} images processed
                </p>
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
