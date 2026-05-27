"use client";

import React, { useEffect, useRef, useState, useCallback } from "react";
import { useQuery } from "@tanstack/react-query";
import {
  Activity,
  Terminal,
  Cpu,
  Server,
  Database,
  Zap,
  Tag,
  Video,
  BarChart3,
  RefreshCw,
  Loader2,
  Circle,
  ChevronDown,
  ChevronRight,
  Filter,
  Trash2,
} from "lucide-react";
import { api, wsUrl } from "@/lib/api";
import { useWebSocket } from "@/hooks/useWebSocket";

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

interface LogEntry {
  ts: number;
  level: "DEBUG" | "INFO" | "WARNING" | "ERROR" | "CRITICAL";
  logger: string;
  msg: string;
}

interface SubsystemStatus {
  name: string;
  label: string;
  icon: React.ElementType;
  endpoint: string;
  color: string;
}

// ---------------------------------------------------------------------------
// Subsystem definitions
// ---------------------------------------------------------------------------

const SUBSYSTEMS: SubsystemStatus[] = [
  {
    name: "backend",
    label: "Backend API",
    icon: Server,
    endpoint: "/system/health",
    color: "#3b82f6",
  },
  {
    name: "database",
    label: "Database",
    icon: Database,
    endpoint: "/system/health",
    color: "#22c55e",
  },
  {
    name: "gpu",
    label: "GPU / CUDA",
    icon: Cpu,
    endpoint: "/system/gpu",
    color: "#a855f7",
  },
  {
    name: "vlm",
    label: "VLM Labeling",
    icon: Zap,
    endpoint: "/vlm/status",
    color: "#f59e0b",
  },
  {
    name: "annotation",
    label: "Annotation Queue",
    icon: Tag,
    endpoint: "/annotation/stats",
    color: "#ec4899",
  },
  {
    name: "training",
    label: "Training Engine",
    icon: BarChart3,
    endpoint: "/training/status",
    color: "#06b6d4",
  },
  {
    name: "video",
    label: "Video Pipeline",
    icon: Video,
    endpoint: "/video/status",
    color: "#84cc16",
  },
  {
    name: "detection",
    label: "Detection",
    icon: Activity,
    endpoint: "/detect/status",
    color: "#f97316",
  },
];

// ---------------------------------------------------------------------------
// Log level config
// ---------------------------------------------------------------------------

const LEVEL_COLORS: Record<string, { bg: string; text: string }> = {
  DEBUG: { bg: "rgba(107,114,128,0.15)", text: "#6b7280" },
  INFO: { bg: "rgba(59,130,246,0.15)", text: "#60a5fa" },
  WARNING: { bg: "rgba(234,179,8,0.15)", text: "#eab308" },
  ERROR: { bg: "rgba(239,68,68,0.15)", text: "#ef4444" },
  CRITICAL: { bg: "rgba(239,68,68,0.25)", text: "#f87171" },
};

function levelNum(level: string): number {
  return { DEBUG: 0, INFO: 1, WARNING: 2, ERROR: 3, CRITICAL: 4 }[level] ?? 0;
}

// ---------------------------------------------------------------------------
// Subsystem status badge
// ---------------------------------------------------------------------------

function SubsystemCard({ sub }: { sub: SubsystemStatus }) {
  const { data, isLoading, isError } = useQuery({
    queryKey: ["subsystem", sub.name],
    queryFn: () => api.get(sub.endpoint).then((r) => r.data).catch(() => null),
    retry: false,
    refetchInterval: 15_000,
    staleTime: 10_000,
  });

  const Icon = sub.icon;
  const status = isLoading ? "checking" : isError || data === null ? "down" : "up";

  return (
    <div
      className="flex items-center gap-3 px-3 py-2.5 rounded-xl transition-all"
      style={{
        background: "#0d1117",
        border: `1px solid ${status === "up" ? "rgba(34,197,94,0.2)" : status === "down" ? "rgba(239,68,68,0.2)" : "#21262d"}`,
      }}
    >
      <div
        className="w-8 h-8 rounded-lg flex items-center justify-center flex-shrink-0"
        style={{ background: `${sub.color}18` }}
      >
        <Icon className="w-4 h-4" style={{ color: sub.color }} />
      </div>

      <div className="flex-1 min-w-0">
        <p className="text-xs font-medium text-white truncate">{sub.label}</p>
        <p
          className="text-[10px]"
          style={{
            color:
              status === "up" ? "#22c55e" : status === "down" ? "#ef4444" : "#6b7280",
          }}
        >
          {status === "checking" ? "Checking…" : status === "up" ? "Online" : "Offline / Not loaded"}
        </p>
      </div>

      <div
        className="w-2 h-2 rounded-full flex-shrink-0"
        style={{
          background:
            status === "up" ? "#22c55e" : status === "down" ? "#ef4444" : "#6b7280",
          boxShadow: status === "up" ? "0 0 6px #22c55e" : undefined,
        }}
      />
    </div>
  );
}

// ---------------------------------------------------------------------------
// Log row
// ---------------------------------------------------------------------------

function LogRow({ entry }: { entry: LogEntry }) {
  const cfg = LEVEL_COLORS[entry.level] ?? LEVEL_COLORS.INFO;
  const time = new Date(entry.ts * 1000);
  const timeStr = time.toLocaleTimeString("de-DE", {
    hour: "2-digit",
    minute: "2-digit",
    second: "2-digit",
  });

  return (
    <div className="flex items-start gap-2 py-0.5 px-2 text-[11px] font-mono hover:bg-white/[0.02] rounded">
      <span className="flex-shrink-0 text-[10px] pt-0.5 w-16" style={{ color: "#484f58" }}>
        {timeStr}
      </span>
      <span
        className="flex-shrink-0 px-1 py-0.5 rounded text-[9px] uppercase font-bold w-14 text-center"
        style={{ background: cfg.bg, color: cfg.text }}
      >
        {entry.level}
      </span>
      <span className="flex-shrink-0 text-[10px] pt-0.5 max-w-[100px] truncate"
        style={{ color: "#6b7280" }} title={entry.logger}>
        {entry.logger.split(".").pop()}
      </span>
      <span className="flex-1 pt-0.5 break-words" style={{ color: "#8b949e" }}>
        {entry.msg}
      </span>
    </div>
  );
}

// ---------------------------------------------------------------------------
// Main processes page
// ---------------------------------------------------------------------------

type FilterLevel = "ALL" | "DEBUG" | "INFO" | "WARNING" | "ERROR";

export default function ProcessesPage() {
  const [logs, setLogs] = useState<LogEntry[]>([]);
  const [filterLevel, setFilterLevel] = useState<FilterLevel>("INFO");
  const [autoScroll, setAutoScroll] = useState(true);
  const [wsConnected, setWsConnected] = useState(false);
  const logsEndRef = useRef<HTMLDivElement>(null);

  const handleMessage = useCallback((data: unknown) => {
    const msg = data as { type?: string; entries?: LogEntry[] };
    if (msg.type === "log_history" && Array.isArray(msg.entries)) {
      setLogs(msg.entries);
    } else if (msg.type === "log_batch" && Array.isArray(msg.entries)) {
      setLogs((prev) => [...prev.slice(-1000), ...msg.entries!]);
    }
  }, []);

  useWebSocket("/ws/logs", "processes-tray", {
    onMessage: handleMessage,
    onConnect: () => setWsConnected(true),
    onDisconnect: () => setWsConnected(false),
  });

  // Auto-scroll
  useEffect(() => {
    if (autoScroll) {
      logsEndRef.current?.scrollIntoView({ behavior: "smooth" });
    }
  }, [logs, autoScroll]);

  const filteredLogs = logs.filter((e) =>
    filterLevel === "ALL" ? true : levelNum(e.level) >= levelNum(filterLevel)
  );

  const errorCount = logs.filter((e) => e.level === "ERROR" || e.level === "CRITICAL").length;
  const warnCount = logs.filter((e) => e.level === "WARNING").length;

  return (
    <div className="flex h-full">
      {/* Left: subsystem status */}
      <div
        className="w-56 flex-shrink-0 flex flex-col"
        style={{ borderRight: "1px solid #21262d" }}
      >
        <div
          className="flex items-center gap-2 px-4 py-3"
          style={{ borderBottom: "1px solid #21262d" }}
        >
          <Activity className="w-4 h-4 text-green-400" />
          <h2 className="text-sm font-semibold text-white">Subsystems</h2>
        </div>

        <div className="flex-1 overflow-y-auto p-3 space-y-2">
          {SUBSYSTEMS.map((sub) => (
            <SubsystemCard key={sub.name} sub={sub} />
          ))}
        </div>
      </div>

      {/* Right: log stream */}
      <div className="flex-1 flex flex-col min-w-0">
        {/* Log header */}
        <div
          className="flex items-center justify-between px-4 py-3 gap-3"
          style={{ borderBottom: "1px solid #21262d" }}
        >
          <div className="flex items-center gap-2">
            <Terminal className="w-4 h-4 text-green-400" />
            <h1 className="text-base font-semibold text-white">Live Logs</h1>
            {wsConnected ? (
              <span
                className="flex items-center gap-1 text-[10px] px-2 py-0.5 rounded-full"
                style={{ background: "rgba(34,197,94,0.15)", color: "#4ade80" }}
              >
                <span className="w-1.5 h-1.5 rounded-full bg-green-400 animate-pulse" />
                Live
              </span>
            ) : (
              <span
                className="flex items-center gap-1 text-[10px] px-2 py-0.5 rounded-full"
                style={{ background: "rgba(107,114,128,0.15)", color: "#6b7280" }}
              >
                <Loader2 className="w-2.5 h-2.5 animate-spin" />
                Connecting
              </span>
            )}
          </div>

          <div className="flex items-center gap-2">
            {/* Error/warn counters */}
            {errorCount > 0 && (
              <span
                className="text-[10px] px-2 py-0.5 rounded-full font-mono"
                style={{ background: "rgba(239,68,68,0.15)", color: "#ef4444" }}
              >
                {errorCount} error{errorCount !== 1 ? "s" : ""}
              </span>
            )}
            {warnCount > 0 && (
              <span
                className="text-[10px] px-2 py-0.5 rounded-full font-mono"
                style={{ background: "rgba(234,179,8,0.15)", color: "#eab308" }}
              >
                {warnCount} warn
              </span>
            )}

            {/* Level filter */}
            <div className="flex items-center gap-1 p-0.5 rounded-lg"
              style={{ background: "#161b22", border: "1px solid #21262d" }}>
              {(["ALL", "INFO", "WARNING", "ERROR"] as FilterLevel[]).map((lvl) => (
                <button
                  key={lvl}
                  onClick={() => setFilterLevel(lvl)}
                  className="px-2 py-0.5 rounded text-[10px] font-medium transition-all"
                  style={{
                    background: filterLevel === lvl ? "#21262d" : "transparent",
                    color: filterLevel === lvl
                      ? lvl === "ERROR" ? "#ef4444" : lvl === "WARNING" ? "#eab308" : "#e6edf3"
                      : "#484f58",
                  }}
                >
                  {lvl}
                </button>
              ))}
            </div>

            {/* Auto-scroll toggle */}
            <button
              onClick={() => setAutoScroll((v) => !v)}
              className="flex items-center gap-1 text-[10px] px-2 py-1 rounded-lg transition-colors"
              style={{
                background: autoScroll ? "rgba(59,130,246,0.15)" : "transparent",
                color: autoScroll ? "#60a5fa" : "#484f58",
                border: "1px solid #21262d",
              }}
              title={autoScroll ? "Auto-scroll on" : "Auto-scroll off"}
            >
              <ChevronDown className="w-3 h-3" />
              Scroll
            </button>

            {/* Clear */}
            <button
              onClick={() => setLogs([])}
              className="p-1.5 rounded-lg transition-colors"
              style={{ color: "#484f58" }}
              title="Clear logs"
            >
              <Trash2 className="w-3.5 h-3.5" />
            </button>
          </div>
        </div>

        {/* Stats bar */}
        <div
          className="flex items-center gap-5 px-4 py-2 text-[11px]"
          style={{ borderBottom: "1px solid #21262d", background: "#161b22" }}
        >
          <span style={{ color: "#484f58" }}>
            Total: <span className="text-white font-mono">{logs.length}</span>
          </span>
          <span style={{ color: "#484f58" }}>
            Showing: <span className="text-white font-mono">{filteredLogs.length}</span>
          </span>
          <span style={{ color: "#484f58" }}>
            Filter: <span className="font-mono" style={{ color: filterLevel === "ERROR" ? "#ef4444" : filterLevel === "WARNING" ? "#eab308" : "#60a5fa" }}>{filterLevel}</span>
          </span>
        </div>

        {/* Log stream */}
        <div
          className="flex-1 overflow-y-auto py-2 font-mono text-[11px]"
          style={{ background: "#0d1117" }}
          onScroll={(e) => {
            const el = e.currentTarget;
            const atBottom = el.scrollTop + el.clientHeight >= el.scrollHeight - 40;
            if (!atBottom && autoScroll) setAutoScroll(false);
          }}
        >
          {filteredLogs.length === 0 ? (
            <div className="flex flex-col items-center justify-center h-full gap-3"
              style={{ color: "#484f58" }}>
              <Terminal className="w-10 h-10 opacity-20" />
              <p className="text-sm">No logs yet</p>
              {!wsConnected && (
                <p className="text-xs">Waiting for WebSocket connection…</p>
              )}
            </div>
          ) : (
            <>
              {filteredLogs.map((entry, i) => (
                <LogRow key={`${entry.ts}-${i}`} entry={entry} />
              ))}
              <div ref={logsEndRef} />
            </>
          )}
        </div>
      </div>
    </div>
  );
}
