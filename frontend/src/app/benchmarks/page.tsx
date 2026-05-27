"use client";

import React, { useState } from "react";
import { useMutation, useQuery } from "@tanstack/react-query";
import {
  BarChart3,
  Loader2,
  Play,
  AlertTriangle,
  Upload,
} from "lucide-react";
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  Cell,
} from "recharts";
import { useDropzone } from "react-dropzone";
import { api } from "@/lib/api";
import { cn } from "@/lib/utils";

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

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

// ---------------------------------------------------------------------------
// Reference benchmarks (static data from benchmark_design.md)
// ---------------------------------------------------------------------------

const REFERENCE_BENCHMARKS = [
  { model: "YOLO11n", imgsz: 640, tiled: false, ms: 6, fps: 150, vram: "0.6 GB" },
  { model: "YOLO11n", imgsz: 1280, tiled: false, ms: 11, fps: 90, vram: "0.8 GB" },
  { model: "YOLO11s", imgsz: 1280, tiled: false, ms: 13, fps: 75, vram: "1.2 GB" },
  { model: "YOLO11s", imgsz: 4096, tiled: true, ms: 150, fps: 7, vram: "1.2 GB" },
  { model: "YOLO11s+SAM2", imgsz: 1280, tiled: false, ms: 80, fps: 12, vram: "5.0 GB" },
];

// ---------------------------------------------------------------------------
// Custom bar chart tooltip
// ---------------------------------------------------------------------------

function CustomTooltip({ active, payload, label }: any) {
  if (!active || !payload?.length) return null;
  return (
    <div
      className="rounded-lg px-3 py-2 text-xs"
      style={{ background: '#161b22', border: '1px solid #21262d', color: '#e6edf3' }}
    >
      <p className="font-medium mb-1">{label}</p>
      {payload.map((p: any) => (
        <p key={p.name} style={{ color: p.color }}>
          {p.name}: {typeof p.value === 'number' ? p.value.toFixed(1) : p.value}
          {p.name === 'fps' ? ' fps' : p.name.includes('ms') ? ' ms' : ''}
        </p>
      ))}
    </div>
  );
}

// ---------------------------------------------------------------------------
// Benchmark result card
// ---------------------------------------------------------------------------

function BenchmarkCard({ run, rank }: { run: BenchmarkRun; rank: number }) {
  const fpsColor =
    run.fps >= 30 ? "#22c55e" : run.fps >= 10 ? "#eab308" : "#ef4444";

  const chartData = [
    { name: "Min", ms: run.min_ms },
    { name: "Mean", ms: run.mean_ms },
    { name: "Max", ms: run.max_ms },
  ];

  const getBarColor = (name: string) => {
    if (name === "Min") return "#3b82f6";
    if (name === "Mean") return "#22c55e";
    return "#ef4444";
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

      {/* Bar chart */}
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

      {/* Stats row */}
      <div className="grid grid-cols-3 gap-2">
        {[
          { label: "Min", value: `${run.min_ms.toFixed(1)}ms`, color: "#3b82f6" },
          { label: "Mean", value: `${run.mean_ms.toFixed(1)}ms`, color: "#22c55e" },
          { label: "Max", value: `${run.max_ms.toFixed(1)}ms`, color: "#ef4444" },
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

// ---------------------------------------------------------------------------
// Reference table
// ---------------------------------------------------------------------------

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

      {/* FPS bar chart */}
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
                fill={entry.fps >= 30 ? "#22c55e" : entry.fps >= 10 ? "#eab308" : "#ef4444"}
              />
            ))}
          </Bar>
        </BarChart>
      </ResponsiveContainer>

      <table className="w-full text-xs">
        <thead>
          <tr style={{ borderBottom: '1px solid #21262d' }}>
            {["Model", "Size", "Tiled", "ms/img", "FPS", "VRAM"].map((h) => (
              <th
                key={h}
                className={cn("pb-2 font-medium", h !== "Model" ? "text-right" : "text-left")}
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
              <td className="py-2 text-right">{row.tiled ? "✓" : "—"}</td>
              <td className="py-2 text-right">~{row.ms}ms</td>
              <td
                className="py-2 text-right font-mono font-bold"
                style={{
                  color: row.fps >= 30 ? "#22c55e" : row.fps >= 10 ? "#eab308" : "#ef4444",
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

// ---------------------------------------------------------------------------
// Main benchmarks page
// ---------------------------------------------------------------------------

export default function BenchmarksPage() {
  const [testImages, setTestImages] = useState<File[]>([]);
  const [confThreshold, setConfThreshold] = useState(0.35);
  const [results, setResults] = useState<BenchmarkRun[]>([]);

  // Fetch available inference models
  const { data: modelsData } = useQuery({
    queryKey: ["inference-models"],
    queryFn: () => api.get("/inference/models").then((r) => r.data),
    staleTime: 60_000,
  });

  const models: ModelInfo[] = modelsData?.models ?? modelsData ?? [];
  const activeModel = models.find((m) => m.is_active);

  // Run benchmark by sending test images to detect endpoint
  const benchmarkMutation = useMutation({
    mutationFn: async () => {
      if (testImages.length === 0) {
        throw new Error("Upload at least one test image");
      }

      const times: number[] = [];

      for (const file of testImages) {
        const formData = new FormData();
        formData.append("file", file);
        formData.append("conf_threshold", String(confThreshold));

        const start = performance.now();
        await api.post("/inference/detect", formData, {
          headers: { "Content-Type": "multipart/form-data" },
        });
        times.push(performance.now() - start);
      }

      const mean = times.reduce((a, b) => a + b, 0) / times.length;
      const sorted = [...times].sort((a, b) => a - b);

      return {
        model: activeModel?.name ?? activeModel?.model_id ?? "active model",
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

  const onDrop = (accepted: File[]) => {
    setTestImages(accepted);
  };

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept: {
      "image/jpeg": [".jpg", ".jpeg"],
      "image/png": [".png"],
      "image/tiff": [".tif", ".tiff"],
    },
    maxFiles: 20,
  });

  return (
    <div className="flex flex-col h-full overflow-y-auto">
      {/* Header */}
      <div
        className="flex items-center gap-2 px-5 py-3"
        style={{ borderBottom: '1px solid #21262d' }}
      >
        <BarChart3 className="w-4 h-4 text-blue-400" />
        <h1 className="text-base font-semibold text-white">Benchmarks</h1>
      </div>

      <div className="p-5 space-y-5">
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
                  style={{ background: '#161b22', border: '1px solid #21262d', color: activeModel ? '#e6edf3' : '#484f58' }}
                >
                  {activeModel ? activeModel.name ?? activeModel.model_id : "No model active"}
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
                      : "Drop test images here"}
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
                    {(benchmarkMutation.error as Error)?.message ?? "Benchmark failed"}
                  </p>
                </div>
              )}
            </div>

            {/* Reference table */}
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
    </div>
  );
}
