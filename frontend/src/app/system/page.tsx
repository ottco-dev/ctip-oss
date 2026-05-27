"use client";

import React, { useEffect, useRef, useState } from "react";
import { useQuery } from "@tanstack/react-query";
import {
  Monitor,
  Cpu,
  HardDrive,
  Activity,
  Wifi,
  WifiOff,
  RefreshCw,
  Terminal,
  MemoryStick,
} from "lucide-react";
import { api } from "@/lib/api";
import { cn } from "@/lib/utils";
import { useSystemStore } from "@/store/systemStore";

// ---------------------------------------------------------------------------
// Types from the actual API responses
// ---------------------------------------------------------------------------

interface GpuData {
  available?: boolean;
  device_name?: string;
  gpu_name?: string;
  vram_total_gb?: number;
  vram_used_gb?: number;
  vram_free_gb?: number;
  vram_used_pct?: number;
  gpu_utilization_pct?: number | null;
  temperature_c?: number;
  power_draw_w?: number;
  // Legacy fields from ws messages
  used_mb?: number;
  total_mb?: number;
  free_mb?: number;
  utilization_pct?: number;
}

interface CpuRamData {
  cpu_count?: number;
  cpu_utilization_pct?: number;
  cpu_pct?: number;
  ram_total_gb?: number;
  ram_used_gb?: number;
  ram_used_pct?: number;
  disk_total_gb?: number;
  disk_free_gb?: number;
  // Legacy mb-based fields
  total_mb?: number;
  used_mb?: number;
}

interface SystemInfoFull {
  gpu?: GpuData;
  cpu_ram?: CpuRamData;
  gpu_name?: string;
  cpu_name?: string;
  disk_free_gb?: number;
  timestamp?: number;
  config?: Record<string, unknown>;
}

interface SystemQueueData {
  depth?: number;
  queue_depth?: number;
  gpu_task_running?: boolean | string;
  active_ws_connections?: number;
  ws_count?: number;
}

// ---------------------------------------------------------------------------
// Ring gauge
// ---------------------------------------------------------------------------

function RingGauge({
  pct,
  size = 80,
  strokeWidth = 8,
  color = "#60a5fa",
  label,
  value,
}: {
  pct: number;
  size?: number;
  strokeWidth?: number;
  color?: string;
  label: string;
  value: string;
}) {
  const r = (size - strokeWidth * 2) / 2;
  const cx = size / 2;
  const cy = size / 2;
  const circ = 2 * Math.PI * r;
  const dash = (Math.min(100, Math.max(0, pct)) / 100) * circ;

  return (
    <div className="flex flex-col items-center gap-2">
      <div className="relative" style={{ width: size, height: size }}>
        <svg width={size} height={size}>
          <circle
            cx={cx}
            cy={cy}
            r={r}
            fill="none"
            stroke="rgba(255,255,255,0.08)"
            strokeWidth={strokeWidth}
          />
          <circle
            cx={cx}
            cy={cy}
            r={r}
            fill="none"
            stroke={color}
            strokeWidth={strokeWidth}
            strokeDasharray={`${dash} ${circ - dash}`}
            strokeLinecap="round"
            strokeDashoffset={circ * 0.25}
            style={{ transition: "stroke-dasharray 0.4s ease" }}
          />
        </svg>
        <div className="absolute inset-0 flex items-center justify-center">
          <span className="text-sm font-bold font-mono text-white">{Math.round(pct)}%</span>
        </div>
      </div>
      <div className="text-center">
        <p className="text-xs font-medium" style={{ color: '#8b949e' }}>{label}</p>
        <p className="text-[10px] font-mono" style={{ color: '#484f58' }}>{value}</p>
      </div>
    </div>
  );
}

// ---------------------------------------------------------------------------
// Metric card
// ---------------------------------------------------------------------------

function MetricCard({
  icon: Icon,
  title,
  value,
  subtitle,
  color = "#60a5fa",
  barPct,
}: {
  icon: React.ElementType;
  title: string;
  value: string;
  subtitle?: string;
  color?: string;
  barPct?: number;
}) {
  return (
    <div
      className="rounded-xl p-4 space-y-3"
      style={{ background: '#0d1117', border: '1px solid #21262d' }}
    >
      <div className="flex items-center gap-2">
        <Icon className="w-4 h-4" style={{ color }} />
        <span className="text-xs font-medium uppercase tracking-wide" style={{ color: '#484f58' }}>
          {title}
        </span>
      </div>
      <p className="text-2xl font-bold font-mono" style={{ color }}>{value}</p>
      {barPct !== undefined && (
        <div className="h-1 rounded-full overflow-hidden" style={{ background: '#21262d' }}>
          <div
            className="h-full rounded-full transition-all"
            style={{
              width: `${Math.min(100, barPct)}%`,
              background: barPct > 85 ? '#ef4444' : barPct > 65 ? '#eab308' : color,
            }}
          />
        </div>
      )}
      {subtitle && (
        <p className="text-xs" style={{ color: '#484f58' }}>{subtitle}</p>
      )}
    </div>
  );
}

// ---------------------------------------------------------------------------
// Sparkline
// ---------------------------------------------------------------------------

function Sparkline({ values, color = "#60a5fa" }: { values: number[]; color?: string }) {
  if (values.length < 2) return null;
  const h = 40;
  const w = 200;
  const max = Math.max(...values, 1);
  const pts = values.map((v, i) => {
    const x = (i / (values.length - 1)) * w;
    const y = h - (v / max) * h;
    return `${x},${y}`;
  });

  return (
    <svg width={w} height={h} className="overflow-visible">
      <polyline
        points={pts.join(" ")}
        fill="none"
        stroke={color}
        strokeWidth={1.5}
        strokeLinejoin="round"
        strokeLinecap="round"
        opacity={0.9}
      />
    </svg>
  );
}

// ---------------------------------------------------------------------------
// Log terminal
// ---------------------------------------------------------------------------

function LogTerminal() {
  const [logs, setLogs] = useState<string[]>([
    "[system] Trichome Analysis started",
    "[backend] FastAPI app running on :8000",
    "[gpu] RTX 4060 detected, 8192 MB VRAM",
  ]);
  const endRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    endRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [logs]);

  useQuery({
    queryKey: ["system-logs"],
    queryFn: async () => {
      try {
        const r = await api.get("/system/logs?limit=5");
        const newEntries: string[] = r.data.entries ?? r.data.logs ?? [];
        if (newEntries.length > 0) {
          setLogs((prev) => [...prev, ...newEntries].slice(-200));
        }
        return r.data;
      } catch {
        return null;
      }
    },
    refetchInterval: 5_000,
  });

  return (
    <div
      className="rounded-xl flex flex-col overflow-hidden"
      style={{ background: '#080b10', border: '1px solid #21262d', height: '14rem' }}
    >
      <div
        className="flex items-center gap-2 px-3 py-2 flex-shrink-0"
        style={{ borderBottom: '1px solid #21262d', background: '#0d1117' }}
      >
        <Terminal className="w-3.5 h-3.5 text-green-400" />
        <span className="text-xs font-medium uppercase tracking-wide" style={{ color: '#484f58' }}>
          System Log
        </span>
        <div className="flex gap-1 ml-auto">
          <div className="w-2 h-2 rounded-full bg-red-400" />
          <div className="w-2 h-2 rounded-full bg-yellow-400" />
          <div className="w-2 h-2 rounded-full bg-green-400" />
        </div>
      </div>
      <div className="flex-1 overflow-y-auto p-3 font-mono text-[10px] text-green-300/80 space-y-0.5">
        {logs.map((line, i) => (
          <div key={i} className="leading-relaxed">
            <span className="text-green-500/50 mr-2 select-none">{">"}</span>
            {line}
          </div>
        ))}
        <div ref={endRef} />
      </div>
    </div>
  );
}

// ---------------------------------------------------------------------------
// Queue panel
// ---------------------------------------------------------------------------

function QueuePanel() {
  const { gpuTaskRunning, queueDepth } = useSystemStore();

  const { data: queueData } = useQuery<SystemQueueData>({
    queryKey: ["system-queue"],
    queryFn: () => api.get("/system/queue").then((r) => r.data),
    refetchInterval: 5_000,
  });

  const depth = queueData?.depth ?? queueData?.queue_depth ?? queueDepth ?? 0;
  const gpuRunning = queueData?.gpu_task_running ?? gpuTaskRunning;
  const wsCount = queueData?.active_ws_connections ?? queueData?.ws_count ?? 0;

  return (
    <div
      className="rounded-xl p-4 space-y-3"
      style={{ background: '#0d1117', border: '1px solid #21262d' }}
    >
      <div className="flex items-center gap-2">
        <Activity className="w-4 h-4 text-purple-400" />
        <span className="text-xs font-medium uppercase tracking-wide" style={{ color: '#484f58' }}>
          Task Queue
        </span>
      </div>
      <div className="space-y-3">
        {[
          {
            label: "GPU Task",
            value: gpuRunning ? "Running" : "Idle",
            color: gpuRunning ? "#3b82f6" : "#484f58",
            dot: gpuRunning ? "bg-blue-400 animate-pulse" : "bg-gray-600",
          },
          {
            label: "Queue Depth",
            value: String(depth),
            color: depth > 3 ? "#ef4444" : depth > 0 ? "#eab308" : "#22c55e",
            dot: null,
          },
          {
            label: "WS Connections",
            value: String(wsCount),
            color: "#a78bfa",
            dot: null,
          },
        ].map(({ label, value, color, dot }) => (
          <div key={label} className="flex items-center justify-between">
            <span className="text-sm" style={{ color: '#484f58' }}>{label}</span>
            <div className="flex items-center gap-1.5">
              {dot && <div className={cn("w-2 h-2 rounded-full", dot)} />}
              <span className="text-sm font-mono font-bold" style={{ color }}>{value}</span>
            </div>
          </div>
        ))}
      </div>
      <div
        className="pt-2 text-[10px]"
        style={{ borderTop: '1px solid #21262d', color: '#484f58' }}
      >
        Semaphore: max 1 concurrent GPU task (RTX 4060 VRAM guard)
      </div>
    </div>
  );
}

// ---------------------------------------------------------------------------
// Main system page
// ---------------------------------------------------------------------------

export default function SystemPage() {
  const { wsConnected } = useSystemStore();
  const [gpuHistory, setGpuHistory] = useState<number[]>([]);
  const [vramHistory, setVramHistory] = useState<number[]>([]);

  // Poll full system info every 3 seconds
  const { data: sysInfo, refetch, isFetching } = useQuery<SystemInfoFull>({
    queryKey: ["system-info"],
    queryFn: () => api.get("/system/info").then((r) => r.data),
    refetchInterval: 3_000,
    staleTime: 2_500,
  });

  const gpu = sysInfo?.gpu;
  const cpuRam = sysInfo?.cpu_ram;

  // GPU percentages — normalize from both gb and mb fields
  const vramTotalGb = gpu?.vram_total_gb ?? (gpu?.total_mb ? gpu.total_mb / 1024 : 8);
  const vramUsedGb = gpu?.vram_used_gb ?? (gpu?.used_mb ? gpu.used_mb / 1024 : 0);
  const vramPct = gpu?.vram_used_pct ?? (vramTotalGb > 0 ? (vramUsedGb / vramTotalGb) * 100 : 0);
  const gpuUtil = gpu?.gpu_utilization_pct ?? gpu?.utilization_pct ?? 0;
  const tempC = gpu?.temperature_c;

  // CPU / RAM
  const cpuPct = cpuRam?.cpu_utilization_pct ?? cpuRam?.cpu_pct ?? 0;
  const ramTotalGb = cpuRam?.ram_total_gb ?? (cpuRam?.total_mb ? cpuRam.total_mb / 1024 : 0);
  const ramUsedGb = cpuRam?.ram_used_gb ?? (cpuRam?.used_mb ? cpuRam.used_mb / 1024 : 0);
  const ramPct = cpuRam?.ram_used_pct ?? (ramTotalGb > 0 ? (ramUsedGb / ramTotalGb) * 100 : 0);
  const diskFreeGb = cpuRam?.disk_free_gb ?? sysInfo?.disk_free_gb ?? 0;
  const diskTotalGb = cpuRam?.disk_total_gb ?? 0;
  const diskUsedPct = diskTotalGb > 0 ? ((diskTotalGb - diskFreeGb) / diskTotalGb) * 100 : 0;

  // Accumulate sparkline history
  useEffect(() => {
    if (!gpu) return;
    setGpuHistory((prev) => [...prev.slice(-59), gpuUtil ?? 0]);
    setVramHistory((prev) => [...prev.slice(-59), vramPct]);
  }, [gpu, gpuUtil, vramPct]);

  const gpuName = gpu?.device_name ?? gpu?.gpu_name ?? sysInfo?.gpu_name ?? "RTX 4060";
  const cpuName = sysInfo?.cpu_name ?? "i5-13400F";

  return (
    <div className="flex flex-col h-full overflow-y-auto">
      {/* Header */}
      <div
        className="flex items-center justify-between px-5 py-3"
        style={{ borderBottom: '1px solid #21262d' }}
      >
        <div className="flex items-center gap-2">
          <Monitor className="w-4 h-4 text-blue-400" />
          <h1 className="text-base font-semibold text-white">System Monitor</h1>
        </div>
        <div className="flex items-center gap-3">
          <div className="flex items-center gap-1.5 text-xs">
            {wsConnected ? (
              <>
                <Wifi className="w-3.5 h-3.5 text-green-400" />
                <span className="text-green-400">Live</span>
              </>
            ) : (
              <>
                <WifiOff className="w-3.5 h-3.5 text-red-400" />
                <span className="text-red-400">Disconnected</span>
              </>
            )}
          </div>
          <button
            onClick={() => refetch()}
            className={cn("p-1.5 rounded transition-colors", isFetching && "animate-spin")}
            style={{ color: '#484f58' }}
          >
            <RefreshCw className="w-4 h-4" />
          </button>
        </div>
      </div>

      <div className="p-5 space-y-5">
        {/* GPU section */}
        <div
          className="rounded-xl p-5"
          style={{ background: '#0d1117', border: '1px solid #21262d' }}
        >
          <h2 className="text-sm font-semibold text-white mb-5">GPU — {gpuName}</h2>
          <div className="flex items-start gap-8">
            {/* Rings */}
            <div className="flex gap-6">
              <RingGauge
                pct={vramPct}
                color={vramPct > 85 ? "#ef4444" : vramPct > 65 ? "#eab308" : "#60a5fa"}
                label="VRAM"
                value={`${vramUsedGb.toFixed(1)} / ${vramTotalGb.toFixed(1)} GB`}
              />
              <RingGauge
                pct={gpuUtil ?? 0}
                color="#a78bfa"
                label="GPU Util"
                value={`${(gpuUtil ?? 0).toFixed(0)}%`}
              />
              {tempC && (
                <RingGauge
                  pct={(tempC / 95) * 100}
                  color={tempC > 80 ? "#ef4444" : "#34d399"}
                  label="Temp"
                  value={`${tempC}°C`}
                />
              )}
            </div>

            {/* Sparklines */}
            <div className="flex-1 space-y-4">
              <div>
                <div className="flex items-center justify-between mb-1">
                  <span className="text-[10px]" style={{ color: '#484f58' }}>GPU Utilization</span>
                  <span className="text-[10px] font-mono text-purple-400">60s window</span>
                </div>
                <Sparkline values={gpuHistory} color="#a78bfa" />
              </div>
              <div>
                <div className="flex items-center justify-between mb-1">
                  <span className="text-[10px]" style={{ color: '#484f58' }}>VRAM Usage</span>
                  <span className="text-[10px] font-mono text-blue-400">60s window</span>
                </div>
                <Sparkline values={vramHistory} color="#60a5fa" />
              </div>
            </div>

            {/* GPU detail stats */}
            {gpu && (
              <div className="space-y-2 min-w-[140px]">
                {[
                  {
                    label: "Power Draw",
                    value: gpu.power_draw_w ? `${gpu.power_draw_w}W` : "—",
                  },
                  {
                    label: "Free VRAM",
                    value: gpu.vram_free_gb
                      ? `${gpu.vram_free_gb.toFixed(1)} GB`
                      : gpu.free_mb
                      ? `${(gpu.free_mb / 1024).toFixed(1)} GB`
                      : "—",
                  },
                  { label: "Device", value: gpuName },
                ].map(({ label, value }) => (
                  <div key={label} className="flex items-center justify-between gap-4">
                    <span className="text-[10px]" style={{ color: '#484f58' }}>{label}</span>
                    <span className="text-[10px] font-mono" style={{ color: '#8b949e' }}>{value}</span>
                  </div>
                ))}
              </div>
            )}
          </div>
        </div>

        {/* CPU + RAM + Disk */}
        <div className="grid grid-cols-3 gap-4">
          <MetricCard
            icon={Cpu}
            title="CPU"
            value={`${cpuPct.toFixed(0)}%`}
            subtitle={cpuName}
            color="#22c55e"
            barPct={cpuPct}
          />
          <MetricCard
            icon={MemoryStick}
            title="RAM"
            value={ramUsedGb > 0 ? `${ramUsedGb.toFixed(1)} GB` : "—"}
            subtitle={ramTotalGb > 0 ? `${ramTotalGb.toFixed(0)} GB total · ${Math.round(ramPct)}% used` : "16 GB"}
            color="#eab308"
            barPct={ramPct}
          />
          <MetricCard
            icon={HardDrive}
            title="Disk Free"
            value={diskFreeGb > 0 ? `${diskFreeGb.toFixed(0)} GB` : "—"}
            subtitle="Free space on model volume"
            color="#a78bfa"
            barPct={diskTotalGb > 0 ? 100 - diskUsedPct : undefined}
          />
        </div>

        {/* Queue + Log */}
        <div className="grid grid-cols-3 gap-4">
          <QueuePanel />
          <div className="col-span-2">
            <LogTerminal />
          </div>
        </div>

        {/* Services */}
        <div
          className="rounded-xl p-5"
          style={{ background: '#0d1117', border: '1px solid #21262d' }}
        >
          <h2 className="text-sm font-semibold text-white mb-4">Services</h2>
          <div className="grid grid-cols-2 sm:grid-cols-4 gap-3">
            {[
              { name: "FastAPI Backend", port: 8000, status: "running" },
              { name: "Next.js Frontend", port: 3000, status: "running" },
              { name: "MLflow Tracking", port: 5000, status: "running" },
              { name: "CVAT", port: 8080, status: "stopped", profile: "annotation" },
            ].map((svc) => (
              <div
                key={svc.name}
                className="px-3 py-2.5 rounded-lg"
                style={{ background: '#161b22', border: '1px solid #21262d' }}
              >
                <div className="flex items-center gap-1.5 mb-1">
                  <div
                    className="w-1.5 h-1.5 rounded-full"
                    style={{ background: svc.status === "running" ? '#22c55e' : '#374151' }}
                  />
                  <span
                    className="text-[10px] font-medium"
                    style={{ color: svc.status === "running" ? '#22c55e' : '#484f58' }}
                  >
                    {svc.status}
                  </span>
                </div>
                <p className="text-xs" style={{ color: '#8b949e' }}>{svc.name}</p>
                <p className="text-[10px] font-mono" style={{ color: '#484f58' }}>:{svc.port}</p>
                {svc.profile && (
                  <p className="text-[9px] mt-0.5" style={{ color: '#484f58' }}>
                    profile: {svc.profile}
                  </p>
                )}
              </div>
            ))}
          </div>
        </div>
      </div>
    </div>
  );
}
