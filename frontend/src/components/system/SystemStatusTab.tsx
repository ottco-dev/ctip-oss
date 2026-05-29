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
  Zap,
  Thermometer,
  Server,
  Database,
  Settings2,
  CheckCircle2,
  XCircle,
  Clock,
} from "lucide-react";
import { api } from "@/lib/api";
import { cn } from "@/lib/utils";
import { useSystemStore } from "@/store/systemStore";

// ---------------------------------------------------------------------------
// API types
// ---------------------------------------------------------------------------

interface GpuData {
  available?: boolean;
  device_name?: string;
  vram_total_gb?: number;
  vram_used_gb?: number;
  vram_reserved_gb?: number;
  vram_free_gb?: number;
  vram_used_pct?: number;
  gpu_utilization_pct?: number | null;
  memory_utilization_pct?: number | null;
  temperature_c?: number;
  power_draw_w?: number;
  power_limit_w?: number;
  compute_capability?: string;
  multi_processor_count?: number;
}

interface CpuRamData {
  cpu_count?: number;
  cpu_utilization_pct?: number;
  ram_total_gb?: number;
  ram_used_gb?: number;
  ram_free_gb?: number;
  ram_used_pct?: number;
  disk_total_gb?: number;
  disk_free_gb?: number;
  disk_used_pct?: number;
}

interface PlatformData {
  os?: string;
  os_version?: string;
  python_version?: string;
  hostname?: string;
}

interface ConfigData {
  api_host?: string;
  api_port?: number;
  database_url?: string;
  mlflow_uri?: string;
  default_vlm?: string;
  cuda_device?: string;
  vram_limit_gb?: number;
}

interface SystemInfoFull {
  timestamp?: number;
  platform?: PlatformData;
  gpu?: GpuData;
  cpu_ram?: CpuRamData;
  config?: ConfigData;
}

interface GpuSemaphore {
  max_concurrent?: number;
  available_slots?: number;
  busy?: boolean;
  waiting_requests?: number;
}

interface QueueData {
  gpu_task_running?: boolean | null;
  gpu_queue_depth?: number;
  total_active_jobs?: number;
  gpu_semaphore?: GpuSemaphore;
  jobs?: {
    pending: number;
    running: number;
    completed: number;
    failed: number;
  };
}

interface ServiceStatus {
  name: string;
  port: number;
  status: "running" | "stopped";
  profile: string | null;
  url: string;
}

interface ServicesData {
  services: ServiceStatus[];
}

interface LogEntry {
  level?: string;
  msg?: string;
  ts?: string | null;
  // string fallback
  [key: string]: unknown;
}

// ---------------------------------------------------------------------------
// Ring-gauge SVG component
// ---------------------------------------------------------------------------

function RingGauge({
  pct,
  size = 88,
  strokeWidth = 9,
  color,
  label,
  value,
  sublabel,
}: {
  pct: number;
  size?: number;
  strokeWidth?: number;
  color: string;
  label: string;
  value: string;
  sublabel?: string;
}) {
  const r = (size - strokeWidth * 2) / 2;
  const circ = 2 * Math.PI * r;
  const fill = (Math.min(100, Math.max(0, pct)) / 100) * circ;

  return (
    <div className="flex flex-col items-center gap-1.5">
      <div className="relative" style={{ width: size, height: size }}>
        <svg width={size} height={size} style={{ transform: "rotate(-90deg)" }}>
          <circle
            cx={size / 2} cy={size / 2} r={r}
            fill="none" stroke="rgba(255,255,255,0.06)" strokeWidth={strokeWidth}
          />
          <circle
            cx={size / 2} cy={size / 2} r={r}
            fill="none" stroke={color} strokeWidth={strokeWidth}
            strokeLinecap="round"
            strokeDasharray={`${fill} ${circ - fill}`}
            style={{ transition: "stroke-dasharray 0.5s ease" }}
          />
        </svg>
        <div className="absolute inset-0 flex items-center justify-center">
          <span className="text-sm font-bold font-mono text-white">{Math.round(pct)}%</span>
        </div>
      </div>
      <div className="text-center">
        <p className="text-xs font-semibold" style={{ color: "#8b949e" }}>{label}</p>
        <p className="text-[10px] font-mono leading-tight" style={{ color: "#484f58" }}>{value}</p>
        {sublabel && (
          <p className="text-[9px] leading-tight mt-0.5" style={{ color: "#30363d" }}>{sublabel}</p>
        )}
      </div>
    </div>
  );
}

// ---------------------------------------------------------------------------
// Sparkline
// ---------------------------------------------------------------------------

function Sparkline({
  values,
  color,
  height = 36,
  width = 180,
}: {
  values: number[];
  color: string;
  height?: number;
  width?: number;
}) {
  if (values.length < 2) {
    return <div style={{ width, height, opacity: 0.2 }} className="rounded" />;
  }
  const max = Math.max(...values, 1);
  const pts = values.map((v, i) => {
    const x = (i / (values.length - 1)) * width;
    const y = height - (v / max) * height;
    return `${x.toFixed(1)},${y.toFixed(1)}`;
  });

  return (
    <svg width={width} height={height} className="overflow-visible">
      <defs>
        <linearGradient id={`sg-${color.replace("#", "")}`} x1="0" y1="0" x2="0" y2="1">
          <stop offset="0%" stopColor={color} stopOpacity={0.3} />
          <stop offset="100%" stopColor={color} stopOpacity={0} />
        </linearGradient>
      </defs>
      <polygon
        points={`0,${height} ${pts.join(" ")} ${width},${height}`}
        fill={`url(#sg-${color.replace("#", "")})`}
      />
      <polyline
        points={pts.join(" ")}
        fill="none" stroke={color} strokeWidth={1.5}
        strokeLinejoin="round" strokeLinecap="round"
      />
    </svg>
  );
}

// ---------------------------------------------------------------------------
// Stat row in a panel
// ---------------------------------------------------------------------------

function StatRow({ label, value, color = "#8b949e" }: { label: string; value: string; color?: string }) {
  return (
    <div className="flex items-center justify-between py-1">
      <span className="text-xs" style={{ color: "#484f58" }}>{label}</span>
      <span className="text-xs font-mono font-medium" style={{ color }}>{value}</span>
    </div>
  );
}

// ---------------------------------------------------------------------------
// Panel wrapper
// ---------------------------------------------------------------------------

function Panel({
  title,
  icon: Icon,
  iconColor = "#60a5fa",
  children,
  className,
}: {
  title: string;
  icon: React.ElementType;
  iconColor?: string;
  children: React.ReactNode;
  className?: string;
}) {
  return (
    <div
      className={cn("rounded-xl overflow-hidden", className)}
      style={{ background: "#0d1117", border: "1px solid #21262d" }}
    >
      <div
        className="flex items-center gap-2 px-4 py-2.5"
        style={{ borderBottom: "1px solid #21262d", background: "#161b22" }}
      >
        <Icon className="w-3.5 h-3.5" style={{ color: iconColor }} />
        <span className="text-xs font-semibold uppercase tracking-widest" style={{ color: "#8b949e" }}>
          {title}
        </span>
      </div>
      <div className="p-4">{children}</div>
    </div>
  );
}

// ---------------------------------------------------------------------------
// GPU Panel
// ---------------------------------------------------------------------------

function GpuPanel({
  gpu,
  gpuUtilHistory,
  vramHistory,
}: {
  gpu: GpuData | undefined;
  gpuUtilHistory: number[];
  vramHistory: number[];
}) {
  if (!gpu?.available) {
    return (
      <Panel title="GPU" icon={Zap} iconColor="#a78bfa">
        <p className="text-sm text-center py-4" style={{ color: "#484f58" }}>
          {gpu?.available === false ? "No CUDA GPU detected" : "Loading…"}
        </p>
      </Panel>
    );
  }

  const vramTotalGb = gpu.vram_total_gb ?? 8;
  const vramUsedGb = gpu.vram_used_gb ?? 0;
  const vramReservedGb = gpu.vram_reserved_gb ?? 0;
  const vramFreeGb = gpu.vram_free_gb ?? vramTotalGb;
  const vramPct = gpu.vram_used_pct ?? (vramTotalGb > 0 ? (vramUsedGb / vramTotalGb) * 100 : 0);
  const gpuUtil = gpu.gpu_utilization_pct ?? 0;
  const tempC = gpu.temperature_c;
  const powerW = gpu.power_draw_w;
  const powerLimitW = gpu.power_limit_w;
  const powerPct = powerW && powerLimitW ? (powerW / powerLimitW) * 100 : 0;

  const vramColor = vramPct > 85 ? "#ef4444" : vramPct > 65 ? "#eab308" : "#60a5fa";
  const tempColor = tempC ? (tempC > 80 ? "#ef4444" : tempC > 65 ? "#eab308" : "#34d399") : "#34d399";

  return (
    <Panel title={`GPU — ${gpu.device_name ?? "RTX 4060"}`} icon={Zap} iconColor="#a78bfa">
      <div className="flex items-start gap-6">
        {/* Rings */}
        <div className="flex gap-5 flex-shrink-0">
          <RingGauge
            pct={vramPct}
            color={vramColor}
            label="VRAM"
            value={`${vramUsedGb.toFixed(1)} / ${vramTotalGb.toFixed(1)} GB`}
            sublabel={`${vramFreeGb.toFixed(1)} GB free`}
          />
          <RingGauge
            pct={gpuUtil}
            color="#a78bfa"
            label="Compute"
            value={gpu.gpu_utilization_pct != null ? `${gpuUtil}%` : "n/a"}
            sublabel={gpu.gpu_utilization_pct == null ? "pynvml off" : undefined}
          />
          {tempC != null && (
            <RingGauge
              pct={(tempC / 100) * 100}
              color={tempColor}
              label="Temp"
              value={`${tempC}°C`}
            />
          )}
          {powerW != null && powerLimitW != null && (
            <RingGauge
              pct={powerPct}
              color="#f97316"
              label="Power"
              value={`${powerW}W`}
              sublabel={`/ ${powerLimitW}W`}
            />
          )}
        </div>

        {/* Sparklines */}
        <div className="flex-1 space-y-3 min-w-0">
          <div>
            <div className="flex items-center justify-between mb-1">
              <span className="text-[10px]" style={{ color: "#484f58" }}>VRAM Usage (60s)</span>
              <span className="text-[10px] font-mono" style={{ color: vramColor }}>
                {vramPct.toFixed(1)}%
              </span>
            </div>
            <Sparkline values={vramHistory} color={vramColor} />
          </div>
          <div>
            <div className="flex items-center justify-between mb-1">
              <span className="text-[10px]" style={{ color: "#484f58" }}>GPU Compute (60s)</span>
              <span className="text-[10px] font-mono" style={{ color: "#a78bfa" }}>
                {gpu.gpu_utilization_pct != null ? `${gpuUtil}%` : "—"}
              </span>
            </div>
            <Sparkline values={gpuUtilHistory} color="#a78bfa" />
          </div>
        </div>

        {/* Detail stats */}
        <div className="space-y-0 min-w-[160px] flex-shrink-0 divide-y" style={{ borderColor: "#21262d" }}>
          <StatRow label="Reserved VRAM" value={`${vramReservedGb.toFixed(1)} GB`} color="#60a5fa" />
          <StatRow label="Free VRAM" value={`${vramFreeGb.toFixed(1)} GB`} />
          <StatRow label="Compute Cap." value={gpu.compute_capability ?? "—"} />
          <StatRow label="SMs" value={gpu.multi_processor_count != null ? String(gpu.multi_processor_count) : "—"} />
          {powerW != null && <StatRow label="Power Draw" value={`${powerW}W`} color="#f97316" />}
        </div>
      </div>
    </Panel>
  );
}

// ---------------------------------------------------------------------------
// CPU / RAM / Disk row
// ---------------------------------------------------------------------------

function ResourceBar({
  label,
  icon: Icon,
  pct,
  value,
  sub,
  color,
}: {
  label: string;
  icon: React.ElementType;
  pct: number;
  value: string;
  sub: string;
  color: string;
}) {
  const barColor = pct > 85 ? "#ef4444" : pct > 65 ? "#eab308" : color;
  return (
    <div
      className="rounded-xl p-4 space-y-3"
      style={{ background: "#0d1117", border: "1px solid #21262d" }}
    >
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-1.5">
          <Icon className="w-3.5 h-3.5" style={{ color }} />
          <span className="text-xs font-semibold uppercase tracking-wide" style={{ color: "#484f58" }}>
            {label}
          </span>
        </div>
        <span className="text-lg font-bold font-mono" style={{ color }}>{value}</span>
      </div>
      <div className="h-1.5 rounded-full overflow-hidden" style={{ background: "#21262d" }}>
        <div
          className="h-full rounded-full transition-all duration-500"
          style={{ width: `${Math.min(100, pct)}%`, background: barColor }}
        />
      </div>
      <p className="text-[10px]" style={{ color: "#484f58" }}>{sub}</p>
    </div>
  );
}

// ---------------------------------------------------------------------------
// Queue panel
// ---------------------------------------------------------------------------

function QueuePanel({ data }: { data: QueueData | undefined }) {
  const sem = data?.gpu_semaphore;
  const jobs = data?.jobs;
  const gpuBusy = sem?.busy ?? (data?.gpu_task_running ?? false);
  const pendingDepth = data?.gpu_queue_depth ?? 0;
  const totalActive = data?.total_active_jobs ?? 0;

  return (
    <Panel title="Task Queue" icon={Activity} iconColor="#a78bfa">
      <div className="space-y-0 divide-y" style={{ borderColor: "#21262d" }}>
        <div className="flex items-center justify-between py-2">
          <span className="text-xs" style={{ color: "#484f58" }}>GPU Semaphore</span>
          <div className="flex items-center gap-1.5">
            <div className={cn("w-2 h-2 rounded-full", gpuBusy ? "bg-blue-400 animate-pulse" : "bg-gray-600")} />
            <span className="text-xs font-mono font-bold" style={{ color: gpuBusy ? "#60a5fa" : "#484f58" }}>
              {gpuBusy ? "BUSY" : "IDLE"}
            </span>
          </div>
        </div>
        <div className="flex items-center justify-between py-2">
          <span className="text-xs" style={{ color: "#484f58" }}>Slots Available</span>
          <span className="text-xs font-mono font-bold" style={{ color: "#22c55e" }}>
            {sem?.available_slots ?? 1} / {sem?.max_concurrent ?? 1}
          </span>
        </div>
        <div className="flex items-center justify-between py-2">
          <span className="text-xs" style={{ color: "#484f58" }}>GPU Queue Depth</span>
          <span className="text-xs font-mono font-bold" style={{ color: pendingDepth > 0 ? "#eab308" : "#484f58" }}>
            {pendingDepth}
          </span>
        </div>
        <div className="flex items-center justify-between py-2">
          <span className="text-xs" style={{ color: "#484f58" }}>Active Jobs</span>
          <span className="text-xs font-mono font-bold" style={{ color: totalActive > 0 ? "#a78bfa" : "#484f58" }}>
            {totalActive}
          </span>
        </div>
        {jobs && (
          <>
            <div className="flex items-center justify-between py-2">
              <span className="text-xs" style={{ color: "#484f58" }}>Completed</span>
              <span className="text-xs font-mono" style={{ color: "#22c55e" }}>{jobs.completed}</span>
            </div>
            <div className="flex items-center justify-between py-2">
              <span className="text-xs" style={{ color: "#484f58" }}>Failed</span>
              <span className="text-xs font-mono" style={{ color: jobs.failed > 0 ? "#ef4444" : "#484f58" }}>
                {jobs.failed}
              </span>
            </div>
          </>
        )}
      </div>
      <p className="text-[9px] mt-3 pt-2" style={{ borderTop: "1px solid #21262d", color: "#30363d" }}>
        RTX 4060 semaphore: max 1 concurrent GPU task (VRAM guard)
      </p>
    </Panel>
  );
}

// ---------------------------------------------------------------------------
// Services panel — live health from /system/services
// ---------------------------------------------------------------------------

function ServicesPanel() {
  const { data, isLoading } = useQuery<ServicesData>({
    queryKey: ["system-services"],
    queryFn: () => api.get("/system/services").then((r) => r.data),
    refetchInterval: 10_000,
    staleTime: 8_000,
  });

  const services = data?.services ?? [];

  const SERVICE_ICONS: Record<string, React.ElementType> = {
    "FastAPI Backend": Server,
    "Next.js Frontend": Monitor,
    "nginx Proxy": Activity,
    MLflow: Database,
    "Label Studio": Settings2,
    CVAT: Settings2,
  };

  return (
    <Panel title="Services" icon={Server} iconColor="#22c55e">
      {isLoading ? (
        <div className="text-xs py-2" style={{ color: "#484f58" }}>Checking services…</div>
      ) : (
        <div className="grid grid-cols-2 sm:grid-cols-3 gap-2">
          {services.map((svc) => {
            const Icon = SERVICE_ICONS[svc.name] ?? Server;
            const up = svc.status === "running";
            return (
              <div
                key={svc.name}
                className="rounded-lg p-3 flex items-start gap-2"
                style={{
                  background: up ? "rgba(34,197,94,0.04)" : "#161b22",
                  border: `1px solid ${up ? "rgba(34,197,94,0.2)" : "#21262d"}`,
                }}
              >
                <div className="mt-0.5 flex-shrink-0">
                  {up ? (
                    <CheckCircle2 className="w-3.5 h-3.5" style={{ color: "#22c55e" }} />
                  ) : (
                    <XCircle className="w-3.5 h-3.5" style={{ color: "#484f58" }} />
                  )}
                </div>
                <div className="min-w-0">
                  <p className="text-xs font-medium truncate" style={{ color: up ? "#e6edf3" : "#484f58" }}>
                    {svc.name}
                  </p>
                  <p className="text-[10px] font-mono" style={{ color: up ? "#22c55e" : "#374151" }}>
                    :{svc.port}
                  </p>
                  {svc.profile && (
                    <p className="text-[9px]" style={{ color: "#30363d" }}>
                      docker: {svc.profile}
                    </p>
                  )}
                </div>
              </div>
            );
          })}
        </div>
      )}
    </Panel>
  );
}

// ---------------------------------------------------------------------------
// Runtime config panel
// ---------------------------------------------------------------------------

function ConfigPanel({ config, platform: plat }: { config: ConfigData | undefined; platform: PlatformData | undefined }) {
  if (!config && !plat) return null;

  const rows = [
    { label: "Hostname", value: plat?.hostname ?? "—" },
    { label: "OS", value: plat?.os ?? "—" },
    { label: "Python", value: plat?.python_version ?? "—" },
    { label: "Default VLM", value: config?.default_vlm ?? "—", color: "#a78bfa" },
    { label: "CUDA Device", value: config?.cuda_device != null ? `cuda:${config.cuda_device}` : "—", color: "#60a5fa" },
    { label: "VRAM Limit", value: config?.vram_limit_gb != null ? `${config.vram_limit_gb} GB` : "—" },
    { label: "MLflow URI", value: config?.mlflow_uri ?? "—" },
    { label: "Database", value: config?.database_url ?? "—" },
  ];

  return (
    <Panel title="Runtime Config" icon={Settings2} iconColor="#f97316">
      <div className="divide-y" style={{ borderColor: "#21262d" }}>
        {rows.map(({ label, value, color }) => (
          <StatRow key={label} label={label} value={value} color={color} />
        ))}
      </div>
    </Panel>
  );
}

// ---------------------------------------------------------------------------
// Log terminal
// ---------------------------------------------------------------------------

function LogTerminal() {
  const [lines, setLines] = useState<string[]>([
    "[system] TrichomeLab backend started",
    "[gpu] RTX 4060 detected — 8.16 GB VRAM",
    "[api] FastAPI listening on :8000",
  ]);
  const endRef = useRef<HTMLDivElement>(null);

  useQuery({
    queryKey: ["system-logs"],
    queryFn: async () => {
      try {
        const r = await api.get("/system/logs?limit=20");
        const entries: LogEntry[] = r.data.entries ?? [];
        if (entries.length > 0) {
          const newLines = entries.map((e) => {
            if (typeof e === "string") return e as string;
            const lvl = e.level ?? "INFO";
            const msg = e.msg ?? JSON.stringify(e);
            return `[${lvl.toLowerCase()}] ${msg}`;
          });
          setLines((prev) => [...prev, ...newLines].slice(-300));
        }
        return r.data;
      } catch {
        return null;
      }
    },
    refetchInterval: 5_000,
  });

  useEffect(() => {
    endRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [lines]);

  return (
    <div
      className="rounded-xl flex flex-col overflow-hidden"
      style={{ background: "#080b10", border: "1px solid #21262d", height: "13rem" }}
    >
      <div
        className="flex items-center gap-2 px-3 py-2 flex-shrink-0"
        style={{ borderBottom: "1px solid #21262d", background: "#0d1117" }}
      >
        <Terminal className="w-3.5 h-3.5 text-green-400" />
        <span className="text-[10px] font-semibold uppercase tracking-widest" style={{ color: "#484f58" }}>
          Backend Log
        </span>
        <div className="flex gap-1 ml-auto">
          <div className="w-1.5 h-1.5 rounded-full bg-red-400/60" />
          <div className="w-1.5 h-1.5 rounded-full bg-yellow-400/60" />
          <div className="w-1.5 h-1.5 rounded-full bg-green-400/60" />
        </div>
      </div>
      <div className="flex-1 overflow-y-auto px-3 py-2 space-y-0.5" style={{ fontFamily: "'JetBrains Mono', monospace" }}>
        {lines.map((line, i) => {
          const isError = line.includes("[error]") || line.includes("ERROR");
          const isWarn = line.includes("[warn]") || line.includes("WARNING");
          return (
            <div
              key={i}
              className="text-[10px] leading-relaxed"
              style={{ color: isError ? "#f87171" : isWarn ? "#fbbf24" : "#4ade80" }}
            >
              <span style={{ color: "#1f6b36", marginRight: "0.5rem", userSelect: "none" }}>›</span>
              {line}
            </div>
          );
        })}
        <div ref={endRef} />
      </div>
    </div>
  );
}

// ---------------------------------------------------------------------------
// Main export
// ---------------------------------------------------------------------------

export function SystemStatusTab() {
  const { wsConnected } = useSystemStore();
  const [gpuUtilHistory, setGpuUtilHistory] = useState<number[]>([]);
  const [vramHistory, setVramHistory] = useState<number[]>([]);

  const { data: sysInfo, refetch, isFetching } = useQuery<SystemInfoFull>({
    queryKey: ["system-info"],
    queryFn: () => api.get("/system/info").then((r) => r.data),
    refetchInterval: 3_000,
    staleTime: 2_500,
  });

  const { data: queueData } = useQuery<QueueData>({
    queryKey: ["system-queue"],
    queryFn: () => api.get("/system/queue").then((r) => r.data),
    refetchInterval: 4_000,
  });

  const gpu = sysInfo?.gpu;
  const cpuRam = sysInfo?.cpu_ram;

  useEffect(() => {
    if (!gpu?.available) return;
    const util = gpu.gpu_utilization_pct ?? 0;
    const vramPct = gpu.vram_used_pct ?? 0;
    setGpuUtilHistory((p) => [...p.slice(-59), util]);
    setVramHistory((p) => [...p.slice(-59), vramPct]);
  }, [gpu]);

  const cpuPct = cpuRam?.cpu_utilization_pct ?? 0;
  const ramUsedGb = cpuRam?.ram_used_gb ?? 0;
  const ramTotalGb = cpuRam?.ram_total_gb ?? 16;
  const ramPct = cpuRam?.ram_used_pct ?? 0;
  const diskFreeGb = cpuRam?.disk_free_gb ?? 0;
  const diskTotalGb = cpuRam?.disk_total_gb ?? 0;
  const diskUsedPct = cpuRam?.disk_used_pct ?? 0;

  return (
    <div className="flex flex-col h-full overflow-y-auto">
      {/* Header */}
      <div
        className="flex items-center justify-between px-5 py-3 flex-shrink-0"
        style={{ borderBottom: "1px solid #21262d", background: "#161b22" }}
      >
        <div className="flex items-center gap-3">
          <Monitor className="w-4 h-4 text-blue-400" />
          <h1 className="text-sm font-semibold text-white">System Monitor</h1>
          {sysInfo?.platform?.hostname && (
            <span
              className="text-[10px] px-2 py-0.5 rounded font-mono"
              style={{ background: "#0d1117", color: "#484f58", border: "1px solid #21262d" }}
            >
              {sysInfo.platform.hostname}
            </span>
          )}
        </div>
        <div className="flex items-center gap-3">
          <div className="flex items-center gap-1.5 text-xs">
            {wsConnected ? (
              <>
                <Wifi className="w-3.5 h-3.5 text-green-400" />
                <span style={{ color: "#22c55e" }}>Live</span>
              </>
            ) : (
              <>
                <WifiOff className="w-3.5 h-3.5 text-red-400" />
                <span style={{ color: "#ef4444" }}>Offline</span>
              </>
            )}
          </div>
          {sysInfo?.timestamp && (
            <div className="flex items-center gap-1 text-[10px]" style={{ color: "#484f58" }}>
              <Clock className="w-3 h-3" />
              <span>{new Date(sysInfo.timestamp * 1000).toLocaleTimeString()}</span>
            </div>
          )}
          <button
            onClick={() => refetch()}
            className={cn("p-1.5 rounded transition-colors hover:bg-panel", isFetching && "animate-spin")}
            style={{ color: "#484f58" }}
          >
            <RefreshCw className="w-4 h-4" />
          </button>
        </div>
      </div>

      <div className="p-5 space-y-4">
        {/* GPU */}
        <GpuPanel gpu={gpu} gpuUtilHistory={gpuUtilHistory} vramHistory={vramHistory} />

        {/* CPU / RAM / Disk */}
        <div className="grid grid-cols-3 gap-4">
          <ResourceBar
            label="CPU"
            icon={Cpu}
            pct={cpuPct}
            value={`${cpuPct.toFixed(0)}%`}
            sub={`${cpuRam?.cpu_count ?? 16} logical cores · ${cpuPct.toFixed(1)}% utilization`}
            color="#22c55e"
          />
          <ResourceBar
            label="RAM"
            icon={MemoryStick}
            pct={ramPct}
            value={`${ramUsedGb.toFixed(1)} GB`}
            sub={`${ramUsedGb.toFixed(1)} / ${ramTotalGb.toFixed(1)} GB · ${Math.round(ramPct)}% used`}
            color="#eab308"
          />
          <ResourceBar
            label="Disk"
            icon={HardDrive}
            pct={diskUsedPct}
            value={`${diskFreeGb.toFixed(0)} GB`}
            sub={`${diskFreeGb.toFixed(0)} free / ${diskTotalGb.toFixed(0)} GB total · ${diskUsedPct.toFixed(0)}% used`}
            color="#a78bfa"
          />
        </div>

        {/* Queue + Log side by side */}
        <div className="grid grid-cols-3 gap-4">
          <QueuePanel data={queueData} />
          <div className="col-span-2">
            <LogTerminal />
          </div>
        </div>

        {/* Services + Config side by side */}
        <div className="grid grid-cols-5 gap-4">
          <div className="col-span-3">
            <ServicesPanel />
          </div>
          <div className="col-span-2">
            <ConfigPanel config={sysInfo?.config} platform={sysInfo?.platform} />
          </div>
        </div>
      </div>
    </div>
  );
}
