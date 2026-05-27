"use client";

import React, { useEffect, useRef, useState, useCallback } from "react";
import { useQuery, useMutation, useQueryClient } from "@tanstack/react-query";
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
  ChevronDown,
  Filter,
  Trash2,
  Container,
  Play,
  Square,
  RotateCcw,
  ChevronRight,
  FileText,
  Settings,
  ExternalLink,
} from "lucide-react";
import { api } from "@/lib/api";
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

type FilterLevel = "ALL" | "INFO" | "WARNING" | "ERROR";

interface ContainerSummary {
  id: string;
  name: string;
  image: string;
  status: string;
  state: string;
  ports: string;
  running: boolean;
  compose_project: string | null;
  compose_service: string | null;
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
  { name: "backend", label: "Backend API", icon: Server, endpoint: "/system/health", color: "#3b82f6" },
  { name: "database", label: "Database", icon: Database, endpoint: "/system/health", color: "#22c55e" },
  { name: "gpu", label: "GPU / CUDA", icon: Cpu, endpoint: "/system/gpu", color: "#a855f7" },
  { name: "vlm", label: "VLM Labeling", icon: Zap, endpoint: "/vlm/status", color: "#f59e0b" },
  { name: "annotation", label: "Annotation Queue", icon: Tag, endpoint: "/annotation/status", color: "#ec4899" },
  { name: "video", label: "Video Pipeline", icon: Video, endpoint: "/video/status", color: "#06b6d4" },
  { name: "analytics", label: "Analytics", icon: BarChart3, endpoint: "/analytics/status", color: "#84cc16" },
];

function levelNum(level: string): number {
  return { DEBUG: 0, INFO: 1, WARNING: 2, ERROR: 3, CRITICAL: 4 }[level] ?? 0;
}

// ---------------------------------------------------------------------------
// SubsystemCard
// ---------------------------------------------------------------------------

function SubsystemCard({ sub }: { sub: SubsystemStatus }) {
  const { data, isLoading } = useQuery({
    queryKey: ["subsystem", sub.name],
    queryFn: () => api.get(sub.endpoint).then((r) => r.data),
    retry: false,
    refetchInterval: 10_000,
  });

  const ok = !isLoading && !!data;

  return (
    <div className="flex items-center gap-2.5 p-2 rounded-lg" style={{ background: "#0d1117", border: "1px solid #21262d" }}>
      <div className="p-1.5 rounded-md" style={{ background: ok ? `${sub.color}18` : "#21262d" }}>
        <sub.icon className="w-3.5 h-3.5" style={{ color: ok ? sub.color : "#484f58" }} />
      </div>
      <div className="flex-1 min-w-0">
        <div className="text-xs font-medium truncate" style={{ color: ok ? "#e6edf3" : "#8b949e" }}>{sub.label}</div>
        <div className="text-[10px]" style={{ color: ok ? "#4ade80" : isLoading ? "#484f58" : "#ef4444" }}>
          {isLoading ? "checking…" : ok ? "online" : "offline"}
        </div>
      </div>
      <div className="w-1.5 h-1.5 rounded-full shrink-0" style={{ background: ok ? "#4ade80" : isLoading ? "#484f58" : "#ef4444" }} />
    </div>
  );
}

// ---------------------------------------------------------------------------
// LogRow
// ---------------------------------------------------------------------------

function LogRow({ entry }: { entry: LogEntry }) {
  const color = { DEBUG: "#484f58", INFO: "#8b949e", WARNING: "#eab308", ERROR: "#ef4444", CRITICAL: "#f87171" }[entry.level] ?? "#8b949e";
  const ts = new Date(entry.ts * 1000).toLocaleTimeString("en-GB", { hour12: false, hour: "2-digit", minute: "2-digit", second: "2-digit" });

  return (
    <div className="flex items-start gap-2 py-0.5 px-2 text-[11px] font-mono hover:bg-white/[0.02] rounded">
      <span className="shrink-0 text-[#484f58]">{ts}</span>
      <span className="shrink-0 w-14" style={{ color }}>{entry.level}</span>
      <span className="shrink-0 text-[#3b82f6] truncate max-w-[120px]">{entry.logger}</span>
      <span className="text-[#8b949e] break-words min-w-0">{entry.msg}</span>
    </div>
  );
}

// ---------------------------------------------------------------------------
// ContainersPanel
// ---------------------------------------------------------------------------

function ContainersPanel() {
  const qc = useQueryClient();
  const [expanded, setExpanded] = useState<string | null>(null);
  const [containerLogs, setContainerLogs] = useState<Record<string, string[]>>({});
  const [logStreaming, setLogStreaming] = useState<string | null>(null);
  const [composeLog, setComposeLog] = useState<string[]>([]);
  const [composeRunning, setComposeRunning] = useState(false);
  const [composeOk, setComposeOk] = useState<boolean | null>(null);
  const [showEnv, setShowEnv] = useState(false);
  const composeLogRef = useRef<HTMLPreElement>(null);
  const esRef = useRef<EventSource | null>(null);

  const { data: containers = [], isLoading, refetch } = useQuery<ContainerSummary[]>({
    queryKey: ["containers"],
    queryFn: () => api.get("/containers?all=true").then((r) => r.data),
    refetchInterval: 8_000,
  });

  const { data: composeConfig } = useQuery({
    queryKey: ["compose-config"],
    queryFn: () => api.get("/containers/compose/config").then((r) => r.data),
    refetchInterval: 30_000,
  });

  const actionMutation = useMutation({
    mutationFn: ({ name, action }: { name: string; action: "start" | "stop" | "restart" }) =>
      api.post(`/containers/${name}/${action}`),
    onSettled: () => { setTimeout(() => refetch(), 1500); },
  });

  useEffect(() => { if (composeLogRef.current) composeLogRef.current.scrollTop = composeLogRef.current.scrollHeight; }, [composeLog]);
  useEffect(() => () => { esRef.current?.close(); }, []);

  const fetchLogs = async (name: string) => {
    const res = await api.get(`/containers/${name}/logs?tail=100`);
    setContainerLogs(prev => ({ ...prev, [name]: (res.data.logs as string).split('\n') }));
  };

  const streamLogs = (name: string) => {
    if (logStreaming === name) {
      // already streaming this one — stop
      setLogStreaming(null);
      return;
    }
    setLogStreaming(name);
    setContainerLogs(prev => ({ ...prev, [name]: [] }));
    const es = new EventSource(`/api/v1/containers/${name}/logs/stream?tail=50`);
    es.onmessage = (e) => {
      if (e.data === '[STREAM_END]') { setLogStreaming(null); es.close(); return; }
      setContainerLogs(prev => ({ ...prev, [name]: [...(prev[name] ?? []), e.data] }));
    };
    es.onerror = () => { setLogStreaming(null); es.close(); };
  };

  const runCompose = (action: "up" | "down") => {
    esRef.current?.close();
    setComposeLog([]);
    setComposeRunning(true);
    setComposeOk(null);

    const endpoint = action === "up"
      ? "/api/v1/containers/compose/up/stream?profile=annotation"
      : null;

    if (!endpoint) {
      // down has no stream — just call REST
      api.post("/containers/compose/down?profile=annotation").then(r => {
        setComposeOk(r.data.ok);
        setComposeLog([r.data.detail]);
        setComposeRunning(false);
        setTimeout(() => refetch(), 2000);
      });
      return;
    }

    const es = new EventSource(endpoint);
    esRef.current = es;
    es.onmessage = (e) => {
      if ((e.data as string).startsWith('[DONE:')) {
        setComposeOk((e.data as string).includes('[DONE:OK]'));
        setComposeRunning(false);
        es.close(); esRef.current = null;
        setTimeout(() => refetch(), 2000);
      } else {
        setComposeLog(prev => [...prev, e.data]);
      }
    };
    es.onerror = () => { setComposeOk(false); setComposeRunning(false); es.close(); esRef.current = null; };
  };

  const running = containers.filter(c => c.running);
  const stopped = containers.filter(c => !c.running);

  const rawEnv: Record<string, string> = composeConfig?.raw_env ?? {};
  const sensitiveKeys = new Set(['LABEL_STUDIO_API_KEY', 'CVAT_PASSWORD', 'POSTGRES_PASSWORD', 'SECRET_KEY']);

  return (
    <div className="flex flex-col h-full overflow-hidden">
      {/* Header */}
      <div className="flex items-center justify-between px-4 py-3 shrink-0" style={{ borderBottom: "1px solid #21262d" }}>
        <div className="flex items-center gap-2">
          <Container className="w-4 h-4 text-accent" />
          <h1 className="text-base font-semibold text-white">Containers</h1>
          <span className="text-[10px] px-2 py-0.5 rounded-full font-mono"
            style={{ background: "rgba(34,197,94,0.1)", color: "#4ade80" }}>
            {running.length} running
          </span>
          {stopped.length > 0 && (
            <span className="text-[10px] px-2 py-0.5 rounded-full font-mono"
              style={{ background: "#161b22", color: "#8b949e" }}>
              {stopped.length} stopped
            </span>
          )}
        </div>
        <button onClick={() => refetch()} disabled={isLoading}
          className="flex items-center gap-1.5 text-xs px-2 py-1.5 rounded-lg transition-colors"
          style={{ background: "#161b22", border: "1px solid #21262d", color: "#8b949e" }}>
          <RefreshCw className={`w-3.5 h-3.5 ${isLoading ? 'animate-spin' : ''}`} />
          Refresh
        </button>
      </div>

      <div className="flex-1 overflow-y-auto">
        {/* Compose actions */}
        <div className="p-4 space-y-3" style={{ borderBottom: "1px solid #21262d" }}>
          <p className="text-[10px] font-semibold uppercase tracking-widest" style={{ color: "#484f58" }}>Annotation Stack</p>
          <div className="flex gap-2">
            <button onClick={() => runCompose("up")} disabled={composeRunning}
              className="flex items-center gap-1.5 text-xs px-3 py-1.5 rounded-lg transition-colors flex-1"
              style={{ background: "rgba(35,134,54,0.15)", border: "1px solid rgba(35,134,54,0.3)", color: "#22c55e" }}>
              {composeRunning ? <Loader2 className="w-3.5 h-3.5 animate-spin" /> : <Play className="w-3.5 h-3.5" />}
              Start (--profile annotation)
            </button>
            <button onClick={() => runCompose("down")} disabled={composeRunning}
              className="flex items-center gap-1.5 text-xs px-3 py-1.5 rounded-lg transition-colors"
              style={{ background: "rgba(239,68,68,0.08)", border: "1px solid rgba(239,68,68,0.2)", color: "#ef4444" }}>
              <Square className="w-3.5 h-3.5" />
              Stop all
            </button>
          </div>

          {/* Compose live log */}
          {(composeLog.length > 0 || composeRunning) && (
            <div className="rounded-md overflow-hidden"
              style={{ border: `1px solid ${composeOk === false ? 'rgba(239,68,68,0.3)' : composeOk === true ? 'rgba(34,197,94,0.3)' : '#21262d'}` }}>
              <div className="flex items-center gap-2 px-3 py-1.5" style={{ background: "#161b22", borderBottom: "1px solid #21262d" }}>
                <div className="flex gap-1.5">
                  <div className="w-2 h-2 rounded-full bg-[#ff5f57]" />
                  <div className="w-2 h-2 rounded-full bg-[#febc2e]" />
                  <div className="w-2 h-2 rounded-full bg-[#28c840]" />
                </div>
                <span className="text-[10px] font-mono text-text-muted flex-1">docker compose up</span>
                {composeRunning && <Loader2 className="w-3 h-3 animate-spin text-accent" />}
                {composeOk === true && <span className="text-[10px] text-status-success">✓ done</span>}
                {composeOk === false && <span className="text-[10px] text-status-error">✗ failed</span>}
              </div>
              <pre ref={composeLogRef}
                className="p-2 font-mono text-[10px] leading-relaxed overflow-y-auto max-h-40 whitespace-pre-wrap break-all"
                style={{ background: "#0d1117", color: "#8b949e" }}>
                {composeLog.join('\n')}
                {composeRunning && <span className="animate-pulse text-accent">▌</span>}
              </pre>
            </div>
          )}
        </div>

        {/* Container list */}
        <div className="p-4 space-y-2">
          <p className="text-[10px] font-semibold uppercase tracking-widest mb-3" style={{ color: "#484f58" }}>All containers</p>

          {isLoading && containers.length === 0 && (
            <div className="flex items-center justify-center py-8 gap-2 text-text-muted">
              <Loader2 className="w-4 h-4 animate-spin" /> Loading…
            </div>
          )}

          {containers.length === 0 && !isLoading && (
            <div className="text-center py-8 text-text-muted text-sm">No containers found</div>
          )}

          {containers.map(c => {
            const isExpanded = expanded === c.name;
            const logs = containerLogs[c.name] ?? [];

            return (
              <div key={c.name} className="rounded-lg overflow-hidden"
                style={{ border: `1px solid ${c.running ? 'rgba(34,197,94,0.2)' : '#21262d'}` }}>
                {/* Container row */}
                <div className="flex items-center gap-2 px-3 py-2.5"
                  style={{ background: c.running ? 'rgba(34,197,94,0.04)' : '#0d1117' }}>
                  {/* Status dot */}
                  <div className={`w-2 h-2 rounded-full shrink-0 ${c.running ? 'bg-status-success' : 'bg-border'}`} />

                  {/* Name + service */}
                  <div className="flex-1 min-w-0">
                    <div className="flex items-center gap-2">
                      <span className="font-mono text-xs text-text-primary truncate">{c.name}</span>
                      {c.compose_service && (
                        <span className="text-[10px] px-1.5 py-0.5 rounded font-mono shrink-0"
                          style={{ background: "#21262d", color: "#8b949e" }}>
                          {c.compose_service}
                        </span>
                      )}
                    </div>
                    <div className="text-[10px] text-text-muted truncate">{c.image}</div>
                  </div>

                  {/* Ports */}
                  {c.ports && (
                    <span className="text-[10px] font-mono text-text-muted shrink-0 hidden md:block max-w-[140px] truncate">
                      {c.ports}
                    </span>
                  )}

                  {/* Status badge */}
                  <span className={`text-[10px] px-1.5 py-0.5 rounded font-medium shrink-0 ${
                    c.running
                      ? 'text-status-success bg-[rgba(34,197,94,0.1)] border border-[rgba(34,197,94,0.2)]'
                      : 'text-text-muted bg-panel border border-border'
                  }`}>{c.state}</span>

                  {/* Actions */}
                  <div className="flex items-center gap-1 shrink-0">
                    {!c.running && (
                      <button title="Start"
                        onClick={() => actionMutation.mutate({ name: c.name, action: "start" })}
                        className="p-1 rounded hover:bg-[rgba(34,197,94,0.15)] text-status-success transition-colors">
                        <Play className="w-3.5 h-3.5" />
                      </button>
                    )}
                    {c.running && (
                      <button title="Stop"
                        onClick={() => actionMutation.mutate({ name: c.name, action: "stop" })}
                        className="p-1 rounded hover:bg-[rgba(239,68,68,0.15)] text-status-error transition-colors">
                        <Square className="w-3.5 h-3.5" />
                      </button>
                    )}
                    <button title="Restart"
                      onClick={() => actionMutation.mutate({ name: c.name, action: "restart" })}
                      className="p-1 rounded hover:bg-panel text-text-muted transition-colors">
                      <RotateCcw className="w-3.5 h-3.5" />
                    </button>
                    {/* Toggle logs */}
                    <button title="Logs"
                      onClick={() => {
                        if (!isExpanded) { fetchLogs(c.name); }
                        setExpanded(isExpanded ? null : c.name);
                      }}
                      className={`p-1 rounded transition-colors ${isExpanded ? 'text-accent' : 'text-text-muted hover:bg-panel'}`}>
                      <FileText className="w-3.5 h-3.5" />
                    </button>
                    <button title={logStreaming === c.name ? "Stop streaming" : "Live logs"}
                      onClick={() => { setExpanded(c.name); streamLogs(c.name); }}
                      className={`p-1 rounded transition-colors ${logStreaming === c.name ? 'text-accent' : 'text-text-muted hover:bg-panel'}`}>
                      <Activity className={`w-3.5 h-3.5 ${logStreaming === c.name ? 'animate-pulse' : ''}`} />
                    </button>
                  </div>

                  <button onClick={() => setExpanded(isExpanded ? null : c.name)}
                    className="p-0.5 text-text-muted shrink-0">
                    <ChevronRight className={`w-3.5 h-3.5 transition-transform ${isExpanded ? 'rotate-90' : ''}`} />
                  </button>
                </div>

                {/* Expanded log panel */}
                {isExpanded && (
                  <div style={{ borderTop: "1px solid #21262d" }}>
                    <div className="flex items-center justify-between px-3 py-1.5"
                      style={{ background: "#161b22", borderBottom: "1px solid #21262d" }}>
                      <span className="text-[10px] font-mono text-text-muted">
                        {c.name} — logs {logStreaming === c.name && <span className="text-accent animate-pulse">● live</span>}
                      </span>
                      <div className="flex gap-1">
                        <button onClick={() => fetchLogs(c.name)}
                          className="text-[10px] px-2 py-0.5 rounded text-text-muted hover:text-text-primary border border-border">
                          refresh
                        </button>
                        <button onClick={() => streamLogs(c.name)}
                          className={`text-[10px] px-2 py-0.5 rounded border border-border ${logStreaming === c.name ? 'text-accent border-accent/30' : 'text-text-muted hover:text-text-primary'}`}>
                          {logStreaming === c.name ? 'stop' : 'live'}
                        </button>
                      </div>
                    </div>
                    <pre className="p-3 font-mono text-[10px] leading-relaxed overflow-y-auto max-h-48 whitespace-pre-wrap break-all"
                      style={{ background: "#0d1117", color: "#8b949e" }}>
                      {logs.length ? logs.join('\n') : 'No logs loaded yet — click refresh or live.'}
                    </pre>
                  </div>
                )}
              </div>
            );
          })}
        </div>

        {/* Env config section */}
        {composeConfig && (
          <div className="px-4 pb-4">
            <button onClick={() => setShowEnv(v => !v)}
              className="flex items-center gap-2 w-full text-left py-2 text-[10px] font-semibold uppercase tracking-widest"
              style={{ color: "#484f58" }}>
              <Settings className="w-3 h-3" />
              .env configuration
              <ChevronRight className={`w-3 h-3 ml-auto transition-transform ${showEnv ? 'rotate-90' : ''}`} />
            </button>
            {showEnv && (
              <div className="rounded-lg overflow-hidden mt-1"
                style={{ border: "1px solid #21262d", background: "#0d1117" }}>
                <div className="p-2 space-y-0.5 max-h-60 overflow-y-auto">
                  {Object.entries(rawEnv).map(([k, v]) => (
                    <div key={k} className="flex items-start gap-2 py-0.5 font-mono text-[10px]">
                      <span className="text-[#3b82f6] shrink-0">{k}</span>
                      <span className="text-text-muted">=</span>
                      <span className="text-text-secondary break-all">
                        {sensitiveKeys.has(k) ? '••••••••' : v || '(empty)'}
                      </span>
                    </div>
                  ))}
                </div>
              </div>
            )}
          </div>
        )}
      </div>
    </div>
  );
}

// ---------------------------------------------------------------------------
// Main page
// ---------------------------------------------------------------------------

type Tab = "logs" | "containers";

export default function ProcessesPage() {
  const [logs, setLogs] = useState<LogEntry[]>([]);
  const [filterLevel, setFilterLevel] = useState<FilterLevel>("INFO");
  const [autoScroll, setAutoScroll] = useState(true);
  const [wsConnected, setWsConnected] = useState(false);
  const [tab, setTab] = useState<Tab>("containers");
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

  useEffect(() => {
    if (autoScroll && tab === "logs") {
      logsEndRef.current?.scrollIntoView({ behavior: "smooth" });
    }
  }, [logs, autoScroll, tab]);

  const filteredLogs = logs.filter((e) =>
    filterLevel === "ALL" ? true : levelNum(e.level) >= levelNum(filterLevel)
  );
  const errorCount = logs.filter((e) => e.level === "ERROR" || e.level === "CRITICAL").length;
  const warnCount = logs.filter((e) => e.level === "WARNING").length;

  return (
    <div className="flex h-full">
      {/* Left: subsystem status */}
      <div className="w-52 shrink-0 flex flex-col" style={{ borderRight: "1px solid #21262d" }}>
        <div className="flex items-center gap-2 px-4 py-3" style={{ borderBottom: "1px solid #21262d" }}>
          <Activity className="w-4 h-4 text-green-400" />
          <h2 className="text-sm font-semibold text-white">Subsystems</h2>
        </div>
        <div className="flex-1 overflow-y-auto p-3 space-y-2">
          {SUBSYSTEMS.map((sub) => <SubsystemCard key={sub.name} sub={sub} />)}
        </div>
      </div>

      {/* Right: tabbed panel */}
      <div className="flex-1 flex flex-col min-w-0">
        {/* Tab bar */}
        <div className="flex items-center gap-1 px-3 py-1.5 shrink-0" style={{ borderBottom: "1px solid #21262d", background: "#161b22" }}>
          <button onClick={() => setTab("containers")}
            className={`flex items-center gap-1.5 px-3 py-1.5 rounded-md text-xs font-medium transition-colors ${
              tab === "containers" ? "bg-panel text-text-primary" : "text-text-muted hover:text-text-secondary"
            }`}>
            <Container className="w-3.5 h-3.5" />
            Containers
          </button>
          <button onClick={() => setTab("logs")}
            className={`flex items-center gap-1.5 px-3 py-1.5 rounded-md text-xs font-medium transition-colors ${
              tab === "logs" ? "bg-panel text-text-primary" : "text-text-muted hover:text-text-secondary"
            }`}>
            <Terminal className="w-3.5 h-3.5" />
            Live Logs
            {errorCount > 0 && (
              <span className="ml-1 text-[10px] px-1.5 py-0.5 rounded-full font-mono"
                style={{ background: "rgba(239,68,68,0.15)", color: "#ef4444" }}>
                {errorCount}
              </span>
            )}
          </button>

          {/* WS indicator */}
          {tab === "logs" && (
            <div className="ml-auto flex items-center gap-1">
              {wsConnected
                ? <span className="flex items-center gap-1 text-[10px] px-2 py-0.5 rounded-full" style={{ background: "rgba(34,197,94,0.15)", color: "#4ade80" }}>
                    <span className="w-1.5 h-1.5 rounded-full bg-green-400 animate-pulse" /> Live
                  </span>
                : <span className="flex items-center gap-1 text-[10px] px-2 py-0.5 rounded-full" style={{ background: "rgba(107,114,128,0.15)", color: "#6b7280" }}>
                    <Loader2 className="w-2.5 h-2.5 animate-spin" /> Connecting
                  </span>
              }
            </div>
          )}
        </div>

        {/* Containers tab */}
        {tab === "containers" && <ContainersPanel />}

        {/* Logs tab */}
        {tab === "logs" && (
          <div className="flex-1 flex flex-col min-w-0 overflow-hidden">
            {/* Log toolbar */}
            <div className="flex items-center justify-between px-4 py-2 gap-3 shrink-0" style={{ borderBottom: "1px solid #21262d" }}>
              <div className="flex items-center gap-2 text-[11px]" style={{ color: "#484f58" }}>
                <span>Total: <span className="text-white font-mono">{logs.length}</span></span>
                <span>Showing: <span className="text-white font-mono">{filteredLogs.length}</span></span>
                {errorCount > 0 && <span style={{ color: "#ef4444" }}>{errorCount} errors</span>}
                {warnCount > 0 && <span style={{ color: "#eab308" }}>{warnCount} warnings</span>}
              </div>
              <div className="flex items-center gap-2">
                <div className="flex items-center gap-1 p-0.5 rounded-lg" style={{ background: "#161b22", border: "1px solid #21262d" }}>
                  {(["ALL", "INFO", "WARNING", "ERROR"] as FilterLevel[]).map((lvl) => (
                    <button key={lvl} onClick={() => setFilterLevel(lvl)}
                      className="px-2 py-0.5 rounded text-[10px] font-medium transition-all"
                      style={{
                        background: filterLevel === lvl ? "#21262d" : "transparent",
                        color: filterLevel === lvl ? (lvl === "ERROR" ? "#ef4444" : lvl === "WARNING" ? "#eab308" : "#e6edf3") : "#484f58",
                      }}>
                      {lvl}
                    </button>
                  ))}
                </div>
                <button onClick={() => setAutoScroll(v => !v)}
                  className="flex items-center gap-1 text-[10px] px-2 py-1 rounded-lg transition-colors"
                  style={{ background: autoScroll ? "rgba(59,130,246,0.15)" : "transparent", color: autoScroll ? "#60a5fa" : "#484f58", border: "1px solid #21262d" }}>
                  <ChevronDown className="w-3 h-3" /> Scroll
                </button>
                <button onClick={() => setLogs([])} className="p-1.5 rounded-lg" style={{ color: "#484f58" }} title="Clear">
                  <Trash2 className="w-3.5 h-3.5" />
                </button>
              </div>
            </div>

            {/* Log stream */}
            <div className="flex-1 overflow-y-auto py-2 font-mono text-[11px]" style={{ background: "#0d1117" }}
              onScroll={(e) => {
                const el = e.currentTarget;
                const atBottom = el.scrollTop + el.clientHeight >= el.scrollHeight - 40;
                if (!atBottom && autoScroll) setAutoScroll(false);
              }}>
              {filteredLogs.length === 0 ? (
                <div className="flex flex-col items-center justify-center h-full gap-3" style={{ color: "#484f58" }}>
                  <Terminal className="w-10 h-10 opacity-20" />
                  <p className="text-sm">No logs yet</p>
                </div>
              ) : (
                <>
                  {filteredLogs.map((entry, i) => <LogRow key={`${entry.ts}-${i}`} entry={entry} />)}
                  <div ref={logsEndRef} />
                </>
              )}
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
