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
  Trash2,
  Container,
  Play,
  Square,
  RotateCcw,
  ChevronRight,
  FileText,
  Settings,
  Bell,
  BellRing,
  CheckCircle,
  XCircle,
  Clock,
  Download,
  PackageOpen,
  Bot,
  Key,
  ExternalLink,
  Globe,
  AlertCircle,
  Eye,
  EyeOff,
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

type TaskStatus = "queued" | "running" | "done" | "error" | "port_conflict";

interface PortConflictInfo {
  port: number;
  service: string;
  env_var: string;
}

interface BgTaskStatus {
  id: string;
  status: TaskStatus;
  started_at: number;
  finished_at: number | null;
  ok: boolean | null;
  log: string[];
  profile: string;
  elapsed_seconds: number | null;
  port_conflict: PortConflictInfo | null;
}

interface SubsystemStatus {
  name: string;
  label: string;
  icon: React.ElementType;
  endpoint: string;
  color: string;
}

// ---------------------------------------------------------------------------
// Notification helper
// ---------------------------------------------------------------------------

function requestNotifPermission() {
  if (typeof window === "undefined") return;
  if ("Notification" in window && Notification.permission === "default") {
    Notification.requestPermission();
  }
}

function sendBrowserNotif(title: string, body: string, ok: boolean) {
  if (typeof window === "undefined") return;
  if (!("Notification" in window) || Notification.permission !== "granted") return;
  try {
    new Notification(title, {
      body,
      icon: ok ? undefined : undefined,
      tag: "ctip-compose",
    });
  } catch {
    // ignore — some browsers block non-HTTPS notifications
  }
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
// Toast notification (in-app)
// ---------------------------------------------------------------------------

interface ToastProps {
  ok: boolean;
  msg: string;
  elapsed: number | null;
  onClose: () => void;
}

function ComposeToast({ ok, msg, elapsed, onClose }: ToastProps) {
  useEffect(() => {
    const t = setTimeout(onClose, 12_000);
    return () => clearTimeout(t);
  }, [onClose]);

  return (
    <div
      className="fixed bottom-6 right-6 z-50 flex items-start gap-3 px-4 py-3 rounded-xl shadow-2xl max-w-sm"
      style={{
        background: ok ? "rgba(22,27,34,0.97)" : "rgba(22,27,34,0.97)",
        border: `1px solid ${ok ? "rgba(34,197,94,0.4)" : "rgba(239,68,68,0.4)"}`,
        backdropFilter: "blur(8px)",
      }}
    >
      {ok
        ? <CheckCircle className="w-5 h-5 shrink-0 mt-0.5" style={{ color: "#4ade80" }} />
        : <XCircle className="w-5 h-5 shrink-0 mt-0.5" style={{ color: "#ef4444" }} />}
      <div className="flex-1 min-w-0">
        <div className="text-sm font-semibold" style={{ color: ok ? "#4ade80" : "#ef4444" }}>
          {ok ? "Stack ready" : "Stack failed"}
        </div>
        <div className="text-xs mt-0.5" style={{ color: "#8b949e" }}>{msg}</div>
        {elapsed !== null && (
          <div className="text-[10px] mt-1 flex items-center gap-1" style={{ color: "#484f58" }}>
            <Clock className="w-3 h-3" /> {elapsed}s
          </div>
        )}
      </div>
      <button onClick={onClose} className="text-[#484f58] hover:text-text-muted shrink-0 text-lg leading-none">×</button>
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

  // SSE compose (legacy — keeps connection open)
  const [composeLog, setComposeLog] = useState<string[]>([]);
  const [composeRunning, setComposeRunning] = useState(false);
  const [composeOk, setComposeOk] = useState<boolean | null>(null);

  // Background task
  const [bgTaskId, setBgTaskId] = useState<string | null>(null);
  const [bgTask, setBgTask] = useState<BgTaskStatus | null>(null);
  const [toast, setToast] = useState<{ ok: boolean; msg: string; elapsed: number | null } | null>(null);
  const [showBgLog, setShowBgLog] = useState(false);
  const bgPollRef = useRef<ReturnType<typeof setInterval> | null>(null);
  const bgLogRef = useRef<HTMLPreElement>(null);

  // Port conflict dialog state
  const [portConflict, setPortConflict] = useState<PortConflictInfo | null>(null);
  const [portInput, setPortInput] = useState<string>("");
  const [portUpdating, setPortUpdating] = useState(false);
  const [portError, setPortError] = useState<string>("");

  const [showEnv, setShowEnv] = useState(false);
  const composeLogRef = useRef<HTMLPreElement>(null);
  const esRef = useRef<EventSource | null>(null);

  // Container remove confirmation
  const [confirmRm, setConfirmRm] = useState<string | null>(null);
  const [rmPending, setRmPending] = useState(false);

  // Docker daemon modal
  const [daemonModalDismissed, setDaemonModalDismissed] = useState(false);
  const [startingDaemon, setStartingDaemon] = useState(false);
  const [daemonStartMsg, setDaemonStartMsg] = useState<string | null>(null);

  // Request notification permission on mount
  useEffect(() => { requestNotifPermission(); }, []);

  const { data: daemonStatus, refetch: refetchDaemon } = useQuery({
    queryKey: ["docker-daemon"],
    queryFn: () => api.get("/containers/daemon").then((r) => r.data),
    refetchInterval: 15_000,
    retry: false,
  });

  const daemonOffline = daemonStatus && !daemonStatus.available;

  const handleStartDaemon = async () => {
    setStartingDaemon(true);
    setDaemonStartMsg(null);
    try {
      const r = await api.post("/containers/daemon/start");
      setDaemonStartMsg(r.data.message);
      if (r.data.started) {
        setTimeout(() => { refetchDaemon(); setDaemonModalDismissed(true); }, 1500);
      }
    } catch {
      setDaemonStartMsg("Request failed — try running the command manually.");
    } finally {
      setStartingDaemon(false);
    }
  };

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

  const handleConfirmRm = async () => {
    if (!confirmRm) return;
    setRmPending(true);
    try {
      await api.delete(`/containers/${confirmRm}`);
      setTimeout(() => refetch(), 1500);
    } catch {
      // error is shown implicitly via refetch
    } finally {
      setRmPending(false);
      setConfirmRm(null);
    }
  };

  // Auto-scroll compose log
  useEffect(() => { if (composeLogRef.current) composeLogRef.current.scrollTop = composeLogRef.current.scrollHeight; }, [composeLog]);
  // Auto-scroll bg task log
  useEffect(() => { if (bgLogRef.current) bgLogRef.current.scrollTop = bgLogRef.current.scrollHeight; }, [bgTask?.log]);
  // Cleanup SSE on unmount
  useEffect(() => () => { esRef.current?.close(); }, []);

  // ── Per-container pull ───────────────────────────────────────────────────

  const [pullingContainer, setPullingContainer] = useState<string | null>(null);
  const [pullResult, setPullResult] = useState<Record<string, { ok: boolean; msg: string }>>({});

  const pullContainerImage = useCallback(async (name: string) => {
    setPullingContainer(name);
    try {
      const res = await api.post(`/containers/${name}/pull`);
      setPullResult(prev => ({ ...prev, [name]: { ok: res.data.ok, msg: res.data.detail } }));
      setTimeout(() => refetch(), 2000);
    } catch (err: unknown) {
      const msg = (err as { response?: { data?: { detail?: string } } })?.response?.data?.detail ?? "pull failed";
      setPullResult(prev => ({ ...prev, [name]: { ok: false, msg } }));
    } finally {
      setPullingContainer(null);
    }
  }, [refetch]);

  // ── Background task polling ──────────────────────────────────────────────

  const stopBgPoll = useCallback(() => {
    if (bgPollRef.current) { clearInterval(bgPollRef.current); bgPollRef.current = null; }
  }, []);

  const pollTask = useCallback(async (id: string) => {
    try {
      const res = await api.get(`/containers/compose/task/${id}`);
      const task: BgTaskStatus = res.data;
      setBgTask(task);

      if (task.status === "port_conflict" && task.port_conflict) {
        // Stop polling — waiting for user input
        stopBgPoll();
        setPortConflict(task.port_conflict);
        setPortInput(String(task.port_conflict.port + 1));
        setPortError("");
        return;
      }

      if (task.status === "done" || task.status === "error") {
        stopBgPoll();
        const ok = task.ok === true;
        const msg = ok
          ? "Annotation stack is up and running."
          : `docker compose exited with errors. Check the log for details.`;
        // Browser notification
        sendBrowserNotif(ok ? "✅ CTIP — Stack Ready" : "❌ CTIP — Stack Failed", msg, ok);
        // In-app toast
        setToast({ ok, msg, elapsed: task.elapsed_seconds });
        // Refresh container list
        setTimeout(() => refetch(), 2000);
      }
    } catch {
      // transient error — keep polling
    }
  }, [stopBgPoll, refetch]);

  const startBgTask = useCallback(async (profile = "annotation", reinstall = false) => {
    try {
      stopBgPoll();
      setBgTask(null);
      setBgTaskId(null);
      setShowBgLog(true);
      // Request notification permission before starting
      if (typeof window !== "undefined" && "Notification" in window && Notification.permission === "default") {
        await Notification.requestPermission();
      }
      const endpoint = reinstall
        ? `/containers/compose/reinstall/background?profile=${profile}`
        : `/containers/compose/up/background?profile=${profile}`;
      const res = await api.post(endpoint);
      const { task_id } = res.data;
      setBgTaskId(task_id);
      // Initial poll immediately
      await pollTask(task_id);
      // Then every 3 seconds
      bgPollRef.current = setInterval(() => pollTask(task_id), 3_000);
    } catch (err: unknown) {
      const msg = (err as { response?: { data?: { detail?: string } } })?.response?.data?.detail ?? String(err);
      setToast({ ok: false, msg, elapsed: null });
    }
  }, [stopBgPoll, pollTask]);

  useEffect(() => () => stopBgPoll(), [stopBgPoll]);

  // ── SSE compose (keeps connection open) ──────────────────────────────────

  const fetchLogs = async (name: string) => {
    const res = await api.get(`/containers/${name}/logs?tail=100`);
    setContainerLogs(prev => ({ ...prev, [name]: (res.data.logs as string).split('\n') }));
  };

  const streamLogs = (name: string) => {
    if (logStreaming === name) {
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

  const runComposeDown = () => {
    api.post("/containers/compose/down?profile=annotation").then(r => {
      setToast({ ok: r.data.ok, msg: r.data.detail?.slice(0, 200) ?? "done", elapsed: null });
      setTimeout(() => refetch(), 2000);
    });
  };

  const running = containers.filter(c => c.running);
  const stopped = containers.filter(c => !c.running);

  const rawEnv: Record<string, string> = composeConfig?.raw_env ?? {};
  const sensitiveKeys = new Set(['LABEL_STUDIO_API_KEY', 'CVAT_PASSWORD', 'POSTGRES_PASSWORD', 'SECRET_KEY']);

  const bgIsActive = bgTask?.status === "queued" || bgTask?.status === "running";

  // Notification permission display
  const notifGranted = typeof window !== "undefined" && "Notification" in window && Notification.permission === "granted";
  const notifBlocked = typeof window !== "undefined" && "Notification" in window && Notification.permission === "denied";

  // ── Port conflict dialog handler ──────────────────────────────────────────
  const applyPortChange = useCallback(async () => {
    if (!portConflict) return;
    const newPort = parseInt(portInput, 10);
    if (isNaN(newPort) || newPort < 1024 || newPort > 65535) {
      setPortError("Port must be between 1024 and 65535");
      return;
    }
    setPortUpdating(true);
    setPortError("");
    try {
      await api.patch("/containers/compose/ports", {
        env_var: portConflict.env_var,
        port: newPort,
      });
      setPortConflict(null);
      setPortInput("");
      // Retry reinstall automatically
      await startBgTask(bgTask?.profile ?? "annotation", true);
    } catch (err: unknown) {
      const msg = (err as { response?: { data?: { detail?: string } } })?.response?.data?.detail ?? String(err);
      setPortError(msg);
    } finally {
      setPortUpdating(false);
    }
  }, [portConflict, portInput, bgTask, startBgTask]);

  // Suppress unused variable warnings for SSE compose state
  void composeLog;
  void composeRunning;
  void composeOk;
  void setComposeLog;
  void setComposeRunning;
  void setComposeOk;
  void qc;
  void bgTaskId;
  void setBgTaskId;

  return (
    <div className="flex flex-col h-full overflow-hidden">
      {/* Docker daemon offline modal */}
      {daemonOffline && !daemonModalDismissed && (
        <div
          style={{
            position: "fixed", inset: 0, zIndex: 100,
            background: "rgba(0,0,0,0.72)", backdropFilter: "blur(4px)",
            display: "flex", alignItems: "center", justifyContent: "center",
          }}
        >
          <div style={{
            background: "#161b22", border: "1px solid #30363d", borderRadius: 12,
            padding: "28px 32px", maxWidth: 440, width: "90%",
          }}>
            <div style={{ display: "flex", alignItems: "center", gap: 10, marginBottom: 12 }}>
              <AlertCircle className="w-5 h-5" style={{ color: "#f59e0b", flexShrink: 0 }} />
              <span style={{ fontSize: 16, fontWeight: 700, color: "#e6edf3" }}>
                Docker daemon offline
              </span>
            </div>
            <p style={{ fontSize: 13, color: "#8b949e", marginBottom: 16, lineHeight: 1.6 }}>
              {daemonStatus?.error ?? "Could not connect to the Docker daemon."}
            </p>

            {daemonStatus?.fix && (
              <div style={{ marginBottom: 16 }}>
                <div style={{ fontSize: 11, color: "#484f58", marginBottom: 6, textTransform: "uppercase", letterSpacing: "0.05em" }}>
                  {daemonStatus.kind === "permission" ? "Fix — add user to docker group" : daemonStatus.kind === "stopped" ? "Start docker" : "Reference"}
                </div>
                <code style={{
                  display: "block", background: "#0d1117", border: "1px solid #21262d",
                  borderRadius: 6, padding: "8px 12px", fontSize: 12,
                  color: "#79c0ff", fontFamily: "monospace", wordBreak: "break-all",
                }}>
                  {daemonStatus.fix}
                </code>
              </div>
            )}

            {daemonStartMsg && (
              <p style={{ fontSize: 12, color: daemonStartMsg.includes("started") ? "#4ade80" : "#f87171", marginBottom: 12 }}>
                {daemonStartMsg}
              </p>
            )}

            <div style={{ display: "flex", gap: 8, justifyContent: "flex-end" }}>
              <button
                onClick={() => setDaemonModalDismissed(true)}
                style={{
                  padding: "6px 14px", fontSize: 13, borderRadius: 6, cursor: "pointer",
                  background: "transparent", border: "1px solid #30363d", color: "#8b949e",
                }}
              >
                Dismiss
              </button>
              {daemonStatus?.kind === "stopped" && (
                <button
                  onClick={handleStartDaemon}
                  disabled={startingDaemon}
                  style={{
                    padding: "6px 14px", fontSize: 13, borderRadius: 6, cursor: startingDaemon ? "not-allowed" : "pointer",
                    background: "var(--accent)", border: "none", color: "#fff",
                    opacity: startingDaemon ? 0.6 : 1,
                  }}
                >
                  {startingDaemon ? "Starting…" : "Start Docker"}
                </button>
              )}
            </div>
          </div>
        </div>
      )}

      {/* In-app toast */}
      {toast && (
        <ComposeToast
          ok={toast.ok}
          msg={toast.msg}
          elapsed={toast.elapsed}
          onClose={() => setToast(null)}
        />
      )}

      {/* Container remove confirmation modal */}
      {confirmRm && (
        <div className="fixed inset-0 z-50 flex items-center justify-center" style={{ background: "rgba(0,0,0,0.65)", backdropFilter: "blur(4px)" }}>
          <div className="w-full max-w-sm rounded-2xl p-6 shadow-2xl" style={{ background: "var(--panel)", border: "1px solid rgba(239,68,68,0.4)" }}>
            <div className="flex items-center gap-3 mb-4">
              <div className="p-2 rounded-lg" style={{ background: "rgba(239,68,68,0.12)" }}>
                <Trash2 className="w-5 h-5" style={{ color: "#ef4444" }} />
              </div>
              <div>
                <p className="text-sm font-semibold" style={{ color: "var(--text-primary)" }}>Remove container?</p>
                <p className="text-xs font-mono mt-0.5" style={{ color: "#ef4444" }}>{confirmRm}</p>
              </div>
            </div>
            <p className="text-xs mb-5" style={{ color: "var(--text-secondary)" }}>
              This will <strong>stop and remove</strong> the container. Images and volumes are preserved.
              The container can be recreated via{" "}
              <span className="font-mono" style={{ color: "var(--accent-text)" }}>docker compose up</span>.
            </p>
            <div className="flex gap-3">
              <button
                onClick={() => setConfirmRm(null)}
                disabled={rmPending}
                className="flex-1 py-2 rounded-lg text-xs font-medium transition-colors"
                style={{ background: "var(--surface)", color: "var(--text-secondary)", border: "1px solid var(--border)" }}
              >
                Cancel
              </button>
              <button
                onClick={handleConfirmRm}
                disabled={rmPending}
                className="flex-1 py-2 rounded-lg text-xs font-semibold flex items-center justify-center gap-1.5 transition-colors"
                style={{ background: "#ef4444", color: "#fff", opacity: rmPending ? 0.6 : 1 }}
              >
                {rmPending ? <><Loader2 className="w-3 h-3 animate-spin" />Removing…</> : <><Trash2 className="w-3 h-3" />Remove</>}
              </button>
            </div>
          </div>
        </div>
      )}

      {/* Port conflict modal */}
      {portConflict && (
        <div className="fixed inset-0 z-50 flex items-center justify-center" style={{ background: "rgba(0,0,0,0.6)", backdropFilter: "blur(4px)" }}>
          <div
            className="w-full max-w-md rounded-2xl p-6 shadow-2xl"
            style={{ background: "#161b22", border: "1px solid rgba(239,68,68,0.4)" }}
          >
            {/* Header */}
            <div className="flex items-center gap-3 mb-4">
              <div className="p-2 rounded-lg" style={{ background: "rgba(239,68,68,0.15)" }}>
                <XCircle className="w-5 h-5" style={{ color: "#ef4444" }} />
              </div>
              <div>
                <div className="text-sm font-semibold" style={{ color: "#e6edf3" }}>Port Already In Use</div>
                <div className="text-xs" style={{ color: "#8b949e" }}>
                  Host port <span className="font-mono font-bold" style={{ color: "#ef4444" }}>{portConflict.port}</span> is occupied by another process
                </div>
              </div>
            </div>

            {/* Service info */}
            <div className="rounded-lg p-3 mb-4" style={{ background: "#0d1117", border: "1px solid #21262d" }}>
              <div className="text-xs mb-1" style={{ color: "#8b949e" }}>Service</div>
              <div className="text-sm font-medium" style={{ color: "#e6edf3" }}>{portConflict.service}</div>
              {portConflict.env_var && (
                <div className="text-[11px] font-mono mt-1" style={{ color: "#3b82f6" }}>.env: {portConflict.env_var}=&quot;{portConflict.port}&quot;</div>
              )}
            </div>

            {/* Port input */}
            <div className="mb-4">
              <label className="block text-xs font-medium mb-1.5" style={{ color: "#8b949e" }}>
                Choose a different host port
              </label>
              <div className="flex gap-2">
                <input
                  type="number"
                  min={1024}
                  max={65535}
                  value={portInput}
                  onChange={(e) => { setPortInput(e.target.value); setPortError(""); }}
                  onKeyDown={(e) => e.key === "Enter" && applyPortChange()}
                  className="flex-1 rounded-lg px-3 py-2 text-sm font-mono outline-none"
                  style={{
                    background: "#0d1117",
                    border: `1px solid ${portError ? "#ef4444" : "#30363d"}`,
                    color: "#e6edf3",
                  }}
                  placeholder={`e.g. ${portConflict.port + 10}`}
                  autoFocus
                />
              </div>
              {portError && (
                <div className="text-xs mt-1.5" style={{ color: "#ef4444" }}>{portError}</div>
              )}
              <div className="text-xs mt-1.5" style={{ color: "#484f58" }}>
                The new port will be saved to <span className="font-mono">.env</span> and the stack will retry automatically.
              </div>
            </div>

            {/* Actions */}
            <div className="flex gap-2">
              <button
                onClick={() => { setPortConflict(null); setPortInput(""); setPortError(""); }}
                disabled={portUpdating}
                className="flex-1 py-2 rounded-lg text-xs font-medium transition-colors"
                style={{ background: "#21262d", color: "#8b949e", border: "1px solid #30363d" }}
              >
                Cancel
              </button>
              <button
                onClick={applyPortChange}
                disabled={portUpdating || !portInput}
                className="flex-1 py-2 rounded-lg text-xs font-semibold flex items-center justify-center gap-2 transition-colors"
                style={{
                  background: portUpdating ? "#1f2937" : "#3b82f6",
                  color: portUpdating ? "#6b7280" : "#fff",
                  border: "none",
                  cursor: portUpdating ? "not-allowed" : "pointer",
                }}
              >
                {portUpdating ? (
                  <><Loader2 className="w-3 h-3 animate-spin" />Applying…</>
                ) : (
                  <><RefreshCw className="w-3 h-3" />Apply &amp; Retry</>
                )}
              </button>
            </div>
          </div>
        </div>
      )}

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
        <div className="flex items-center gap-2">
          {/* Notification status indicator */}
          {notifBlocked && (
            <span className="text-[10px] flex items-center gap-1 px-2 py-0.5 rounded" style={{ color: "#ef4444", background: "rgba(239,68,68,0.08)" }}>
              <Bell className="w-3 h-3" /> Notifications blocked
            </span>
          )}
          {notifGranted && (
            <span className="text-[10px] flex items-center gap-1 px-2 py-0.5 rounded" style={{ color: "#4ade80", background: "rgba(34,197,94,0.08)" }}>
              <BellRing className="w-3 h-3" /> Notifications on
            </span>
          )}
          <button onClick={() => refetch()} disabled={isLoading}
            className="flex items-center gap-1.5 text-xs px-2 py-1.5 rounded-lg transition-colors"
            style={{ background: "#161b22", border: "1px solid #21262d", color: "#8b949e" }}>
            <RefreshCw className={`w-3.5 h-3.5 ${isLoading ? 'animate-spin' : ''}`} />
            Refresh
          </button>
        </div>
      </div>

      <div className="flex-1 overflow-y-auto">
        {/* Compose actions */}
        <div className="p-4 space-y-3" style={{ borderBottom: "1px solid #21262d" }}>
          <p className="text-[10px] font-semibold uppercase tracking-widest" style={{ color: "#484f58" }}>Annotation Stack</p>

          <div className="flex gap-2 flex-wrap">
            {/* Background start */}
            <button onClick={() => startBgTask("annotation")} disabled={bgIsActive}
              className="flex items-center gap-1.5 text-xs px-3 py-1.5 rounded-lg transition-colors flex-1 min-w-0"
              style={{
                background: bgIsActive ? "rgba(59,130,246,0.08)" : "rgba(35,134,54,0.15)",
                border: `1px solid ${bgIsActive ? "rgba(59,130,246,0.3)" : "rgba(35,134,54,0.3)"}`,
                color: bgIsActive ? "#60a5fa" : "#22c55e",
              }}>
              {bgIsActive
                ? <><Loader2 className="w-3.5 h-3.5 animate-spin" /> Running…</>
                : <><BellRing className="w-3.5 h-3.5" /> Start + Notify</>}
            </button>

            {/* Reinstall (pull + force-recreate) */}
            <button onClick={() => startBgTask("annotation", true)} disabled={bgIsActive}
              className="flex items-center gap-1.5 text-xs px-3 py-1.5 rounded-lg transition-colors"
              style={{ background: "rgba(168,85,247,0.08)", border: "1px solid rgba(168,85,247,0.25)", color: "#a855f7" }}>
              <PackageOpen className="w-3.5 h-3.5" />
              Reinstall all
            </button>

            {/* Stop all */}
            <button onClick={runComposeDown}
              className="flex items-center gap-1.5 text-xs px-3 py-1.5 rounded-lg transition-colors"
              style={{ background: "rgba(239,68,68,0.08)", border: "1px solid rgba(239,68,68,0.2)", color: "#ef4444" }}>
              <Square className="w-3.5 h-3.5" />
              Stop all
            </button>
          </div>

          {/* Background task status panel */}
          {bgTask && (
            <div className="rounded-md overflow-hidden"
              style={{ border: `1px solid ${
                bgTask.status === "error" ? 'rgba(239,68,68,0.3)'
                : bgTask.status === "port_conflict" ? 'rgba(251,146,60,0.5)'
                : bgTask.status === "done" ? 'rgba(34,197,94,0.3)'
                : 'rgba(59,130,246,0.3)'}` }}>
              {/* Title bar */}
              <div className="flex items-center gap-2 px-3 py-1.5 cursor-pointer select-none"
                style={{ background: "#161b22", borderBottom: showBgLog ? "1px solid #21262d" : "none" }}
                onClick={() => setShowBgLog(v => !v)}>
                <div className="flex gap-1.5">
                  <div className="w-2 h-2 rounded-full bg-[#ff5f57]" />
                  <div className="w-2 h-2 rounded-full bg-[#febc2e]" />
                  <div className="w-2 h-2 rounded-full bg-[#28c840]" />
                </div>
                <span className="text-[10px] font-mono text-text-muted flex-1">
                  docker compose up — background
                  {bgTask.elapsed_seconds !== null && (
                    <span className="ml-2 opacity-60">{bgTask.elapsed_seconds}s</span>
                  )}
                </span>
                <div className="flex items-center gap-2">
                  {bgIsActive && <Loader2 className="w-3 h-3 animate-spin text-[#60a5fa]" />}
                  {bgTask.status === "done" && bgTask.ok && <CheckCircle className="w-3 h-3 text-[#4ade80]" />}
                  {bgTask.status === "error" && <XCircle className="w-3 h-3 text-[#ef4444]" />}
                  {bgTask.status === "port_conflict" && (
                    <button
                      onClick={(e) => { e.stopPropagation(); setPortConflict(bgTask.port_conflict); setPortInput(String((bgTask.port_conflict?.port ?? 3004) + 1)); setPortError(""); }}
                      className="text-[10px] font-semibold px-2 py-0.5 rounded flex items-center gap-1"
                      style={{ background: "rgba(251,146,60,0.2)", color: "#fb923c", border: "1px solid rgba(251,146,60,0.4)" }}
                    >
                      ⚠ Port conflict — fix
                    </button>
                  )}
                  <span className="text-[10px] font-mono px-1.5 py-0.5 rounded"
                    style={{
                      background: bgIsActive ? "rgba(59,130,246,0.12)"
                        : bgTask.status === "port_conflict" ? "rgba(251,146,60,0.12)"
                        : bgTask.ok ? "rgba(34,197,94,0.12)"
                        : "rgba(239,68,68,0.12)",
                      color: bgIsActive ? "#60a5fa"
                        : bgTask.status === "port_conflict" ? "#fb923c"
                        : bgTask.ok ? "#4ade80"
                        : "#ef4444",
                    }}>
                    {bgTask.status}
                  </span>
                  <ChevronRight className={`w-3 h-3 text-text-muted transition-transform ${showBgLog ? 'rotate-90' : ''}`} />
                </div>
              </div>

              {/* Log output */}
              {showBgLog && (
                <pre ref={bgLogRef}
                  className="p-2 font-mono text-[10px] leading-relaxed overflow-y-auto max-h-48 whitespace-pre-wrap break-all"
                  style={{ background: "#0d1117", color: "#8b949e" }}>
                  {bgTask.log.length > 0
                    ? bgTask.log.join('\n')
                    : bgIsActive
                      ? "Waiting for output…"
                      : "No output captured."}
                  {bgIsActive && <span className="animate-pulse text-[#60a5fa]">▌</span>}
                </pre>
              )}
            </div>
          )}

          {/* Notification hint (if not yet granted) */}
          {!notifGranted && !notifBlocked && (
            <p className="text-[10px]" style={{ color: "#484f58" }}>
              💡 Click &ldquo;Start + Notify&rdquo; — you&apos;ll be asked for notification permission so CTIP can alert you when the stack is ready, even if you navigate away.
            </p>
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
                    {/* Pull latest image */}
                    <button
                      title="Pull latest image & restart"
                      disabled={pullingContainer === c.name}
                      onClick={() => pullContainerImage(c.name)}
                      className={`p-1 rounded transition-colors ${
                        pullResult[c.name]?.ok === true
                          ? 'text-status-success'
                          : pullResult[c.name]?.ok === false
                            ? 'text-status-error'
                            : 'text-text-muted hover:bg-panel'
                      }`}>
                      {pullingContainer === c.name
                        ? <Loader2 className="w-3.5 h-3.5 animate-spin" />
                        : <Download className="w-3.5 h-3.5" />}
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
                    {/* Remove container — requires 2-click confirm */}
                    <button
                      title="Remove container (stop + rm)"
                      onClick={() => setConfirmRm(c.name)}
                      className="p-1 rounded hover:bg-[rgba(239,68,68,0.12)] text-text-muted hover:text-status-error transition-colors"
                    >
                      <Trash2 className="w-3.5 h-3.5" />
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
                    {/* Pull result banner */}
                    {pullResult[c.name] && (
                      <div className="px-3 py-1.5 text-[10px] font-mono flex items-center gap-2"
                        style={{
                          background: pullResult[c.name].ok ? "rgba(34,197,94,0.06)" : "rgba(239,68,68,0.06)",
                          borderBottom: "1px solid #21262d",
                          color: pullResult[c.name].ok ? "#4ade80" : "#ef4444",
                        }}>
                        {pullResult[c.name].ok ? <CheckCircle className="w-3 h-3 shrink-0" /> : <XCircle className="w-3 h-3 shrink-0" />}
                        <span className="truncate">{pullResult[c.name].msg}</span>
                      </div>
                    )}
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
// VLM Providers panel
// ---------------------------------------------------------------------------

interface ProviderStatus {
  provider_id: string;
  name: string;
  kind: "local" | "remote";
  tier: "free" | "freemium" | "paid";
  available: boolean;
  has_api_key: boolean;
  env_var: string;
  models: string[];
  default_model: string;
  vram_gb: number | null;
  cost_per_1k_tokens: number | null;
  rate_limit_rpm: number | null;
  free_tier_note: string;
  signup_url: string;
  is_active: boolean;
}

interface ActiveProviderInfo {
  provider_id: string;
  model: string | null;
  kind: string;
  tier: string;
  name: string;
}

function TierBadge({ tier }: { tier: string }) {
  const colors: Record<string, { bg: string; text: string }> = {
    free:      { bg: "rgba(34,197,94,0.12)",  text: "#4ade80" },
    freemium:  { bg: "rgba(59,130,246,0.12)", text: "#60a5fa" },
    paid:      { bg: "rgba(251,146,60,0.12)", text: "#fb923c" },
    local:     { bg: "rgba(167,139,250,0.12)", text: "#a78bfa" },
  };
  const c = colors[tier] ?? { bg: "rgba(107,114,128,0.12)", text: "#6b7280" };
  return (
    <span className="text-[9px] font-semibold px-1.5 py-0.5 rounded-full uppercase tracking-wide"
      style={{ background: c.bg, color: c.text }}>
      {tier}
    </span>
  );
}

function ProviderCard({
  provider,
  onActivate,
  activating,
}: {
  provider: ProviderStatus;
  onActivate: (id: string, model?: string) => void;
  activating: boolean;
}) {
  const [showKey, setShowKey] = useState(false);
  const [keyInput, setKeyInput] = useState("");
  const [keyExpanded, setKeyExpanded] = useState(false);
  const [keyStatus, setKeyStatus] = useState<"idle" | "saving" | "saved" | "error">("idle");
  const [selectedModel, setSelectedModel] = useState(provider.default_model);
  const queryClient = useQueryClient();

  const saveKey = async () => {
    if (!keyInput.trim()) return;
    setKeyStatus("saving");
    try {
      await api.post(`/vlm/providers/${provider.provider_id}/configure`, {
        api_key: keyInput.trim(),
        model: selectedModel !== provider.default_model ? selectedModel : undefined,
      });
      setKeyStatus("saved");
      setKeyInput("");
      setKeyExpanded(false);
      await queryClient.invalidateQueries({ queryKey: ["vlm-providers"] });
      setTimeout(() => setKeyStatus("idle"), 2500);
    } catch {
      setKeyStatus("error");
      setTimeout(() => setKeyStatus("idle"), 3000);
    }
  };

  const isActive = provider.is_active;
  const borderColor = isActive ? "#22c55e" : provider.available ? "#21262d" : "#21262d";
  const statusDot = provider.available ? "#22c55e" : provider.has_api_key ? "#eab308" : "#374151";

  return (
    <div
      className="rounded-xl p-4 flex flex-col gap-3 transition-all"
      style={{
        background: isActive ? "rgba(34,197,94,0.05)" : "#0d1117",
        border: `1px solid ${borderColor}`,
        opacity: provider.available ? 1 : 0.72,
      }}
    >
      {/* Header row */}
      <div className="flex items-start justify-between gap-2">
        <div className="flex items-center gap-2 min-w-0">
          <div className="w-2 h-2 rounded-full shrink-0" style={{ background: statusDot }} />
          <span className="text-sm font-semibold text-white truncate">{provider.name}</span>
        </div>
        <div className="flex items-center gap-1.5 shrink-0">
          <TierBadge tier={provider.kind === "local" ? "local" : provider.tier} />
          {provider.kind === "local" && <TierBadge tier="local" />}
          {isActive && (
            <span className="text-[9px] font-bold px-1.5 py-0.5 rounded-full uppercase"
              style={{ background: "rgba(34,197,94,0.2)", color: "#22c55e" }}>
              active
            </span>
          )}
        </div>
      </div>

      {/* Info row */}
      <div className="flex flex-wrap gap-x-4 gap-y-1 text-[10px]" style={{ color: "#8b949e" }}>
        {provider.vram_gb !== null && (
          <span><span className="text-white font-mono">{provider.vram_gb} GB</span> VRAM</span>
        )}
        {provider.cost_per_1k_tokens !== null && (
          <span><span className="text-white font-mono">${provider.cost_per_1k_tokens}</span>/1k tok</span>
        )}
        {provider.rate_limit_rpm !== null && (
          <span><span className="text-white font-mono">{provider.rate_limit_rpm}</span> RPM</span>
        )}
        {provider.free_tier_note && (
          <span className="text-[9px]" style={{ color: "#4ade80" }}>{provider.free_tier_note}</span>
        )}
      </div>

      {/* Model selector (when multiple models) */}
      {provider.models.length > 1 && (
        <select
          className="w-full text-[11px] rounded-lg px-2 py-1.5"
          style={{ background: "#161b22", border: "1px solid #30363d", color: "#e6edf3" }}
          value={selectedModel}
          onChange={(e) => setSelectedModel(e.target.value)}
        >
          {provider.models.map((m) => (
            <option key={m} value={m}>{m}</option>
          ))}
        </select>
      )}

      {/* Actions */}
      <div className="flex items-center gap-2">
        {/* Activate button */}
        {!isActive && provider.available && (
          <button
            onClick={() => onActivate(provider.provider_id,
              selectedModel !== provider.default_model ? selectedModel : undefined)}
            disabled={activating}
            className="flex-1 flex items-center justify-center gap-1.5 py-1.5 rounded-lg text-[11px] font-medium transition-colors"
            style={{
              background: "rgba(34,197,94,0.12)",
              border: "1px solid rgba(34,197,94,0.25)",
              color: "#4ade80",
            }}
          >
            {activating ? <Loader2 className="w-3 h-3 animate-spin" /> : <CheckCircle className="w-3 h-3" />}
            Activate
          </button>
        )}
        {isActive && (
          <div className="flex-1 flex items-center justify-center gap-1.5 py-1.5 rounded-lg text-[11px] font-medium"
            style={{ background: "rgba(34,197,94,0.08)", border: "1px solid rgba(34,197,94,0.2)", color: "#22c55e" }}>
            <CheckCircle className="w-3 h-3" /> Active provider
          </div>
        )}
        {!provider.available && provider.kind === "remote" && (
          <div className="flex-1 flex items-center justify-center gap-1.5 py-1.5 rounded-lg text-[11px]"
            style={{ background: "rgba(107,114,128,0.08)", border: "1px solid #21262d", color: "#6b7280" }}>
            <Key className="w-3 h-3" /> API key required
          </div>
        )}

        {/* Configure key toggle */}
        {provider.kind === "remote" && provider.env_var && (
          <button
            onClick={() => setKeyExpanded(v => !v)}
            className="p-1.5 rounded-lg transition-colors"
            style={{
              background: keyExpanded ? "rgba(59,130,246,0.12)" : "transparent",
              border: "1px solid #21262d",
              color: keyExpanded ? "#60a5fa" : "#484f58",
            }}
            title="Configure API key"
          >
            <Key className="w-3.5 h-3.5" />
          </button>
        )}

        {/* Signup link */}
        {provider.signup_url && (
          <a
            href={provider.signup_url}
            target="_blank"
            rel="noopener noreferrer"
            className="p-1.5 rounded-lg transition-colors"
            style={{ border: "1px solid #21262d", color: "#484f58" }}
            title="Sign up / get API key"
          >
            <ExternalLink className="w-3.5 h-3.5" />
          </a>
        )}
      </div>

      {/* API key form (collapsible) */}
      {keyExpanded && provider.kind === "remote" && (
        <div className="flex flex-col gap-2 pt-2" style={{ borderTop: "1px solid #21262d" }}>
          <p className="text-[10px]" style={{ color: "#8b949e" }}>
            Set <span className="font-mono text-white">{provider.env_var}</span> — persisted to <span className="font-mono">.env</span>
          </p>
          <div className="flex gap-2">
            <div className="flex-1 relative">
              <input
                type={showKey ? "text" : "password"}
                className="w-full text-[11px] rounded-lg px-2.5 py-1.5 pr-8 font-mono"
                style={{ background: "#161b22", border: "1px solid #30363d", color: "#e6edf3" }}
                placeholder="Paste API key…"
                value={keyInput}
                onChange={(e) => setKeyInput(e.target.value)}
                onKeyDown={(e) => e.key === "Enter" && saveKey()}
              />
              <button
                className="absolute right-1.5 top-1/2 -translate-y-1/2"
                style={{ color: "#484f58" }}
                onClick={() => setShowKey(v => !v)}
                type="button"
              >
                {showKey ? <EyeOff className="w-3 h-3" /> : <Eye className="w-3 h-3" />}
              </button>
            </div>
            <button
              onClick={saveKey}
              disabled={keyStatus === "saving" || !keyInput.trim()}
              className="px-3 py-1.5 rounded-lg text-[11px] font-medium transition-colors"
              style={{
                background: keyStatus === "saved" ? "rgba(34,197,94,0.15)" : "rgba(59,130,246,0.15)",
                border: "1px solid rgba(59,130,246,0.3)",
                color: keyStatus === "saved" ? "#4ade80" : "#60a5fa",
              }}
            >
              {keyStatus === "saving" ? <Loader2 className="w-3 h-3 animate-spin" />
                : keyStatus === "saved" ? "Saved ✓"
                : keyStatus === "error" ? "Error"
                : "Save"}
            </button>
          </div>
          {provider.has_api_key && (
            <p className="text-[9px]" style={{ color: "#4ade80" }}>
              ✓ API key is configured. Enter a new value to replace it.
            </p>
          )}
        </div>
      )}
    </div>
  );
}

function VLMProvidersPanel() {
  const queryClient = useQueryClient();
  const [activatingId, setActivatingId] = useState<string | null>(null);
  const [switchError, setSwitchError] = useState<string | null>(null);

  const { data: providers = [], isLoading, error } = useQuery<ProviderStatus[]>({
    queryKey: ["vlm-providers"],
    queryFn: async () => {
      const r = await api.get<ProviderStatus[]>("/vlm/providers");
      return r.data;
    },
    refetchInterval: 30_000,
  });

  const { data: active } = useQuery<ActiveProviderInfo>({
    queryKey: ["vlm-active"],
    queryFn: async () => {
      const r = await api.get<ActiveProviderInfo>("/vlm/providers/active");
      return r.data;
    },
    refetchInterval: 30_000,
  });

  const handleActivate = async (providerId: string, model?: string) => {
    setActivatingId(providerId);
    setSwitchError(null);
    try {
      await api.post("/vlm/providers/active", { provider_id: providerId, model: model ?? null });
      await queryClient.invalidateQueries({ queryKey: ["vlm-providers"] });
      await queryClient.invalidateQueries({ queryKey: ["vlm-active"] });
    } catch (err: unknown) {
      const msg = (err as { response?: { data?: { detail?: string } } })?.response?.data?.detail ?? "Failed to switch provider";
      setSwitchError(msg);
      setTimeout(() => setSwitchError(null), 5000);
    } finally {
      setActivatingId(null);
    }
  };

  const local = providers.filter(p => p.kind === "local");
  const remote = providers.filter(p => p.kind === "remote");

  return (
    <div className="flex-1 overflow-y-auto p-5" style={{ background: "#0d1117" }}>
      {/* Header */}
      <div className="flex items-center justify-between mb-5">
        <div>
          <h2 className="text-sm font-semibold text-white flex items-center gap-2">
            <Bot className="w-4 h-4 text-blue-400" />
            VLM Providers
          </h2>
          {active && (
            <p className="text-[11px] mt-0.5" style={{ color: "#8b949e" }}>
              Active: <span className="text-white font-medium">{active.name}</span>
              {active.model && <span className="ml-1 font-mono text-[10px]" style={{ color: "#4ade80" }}> — {active.model}</span>}
            </p>
          )}
        </div>
        <button
          onClick={() => queryClient.invalidateQueries({ queryKey: ["vlm-providers"] })}
          className="p-1.5 rounded-lg transition-colors"
          style={{ border: "1px solid #21262d", color: "#484f58" }}
          title="Refresh"
        >
          <RefreshCw className="w-3.5 h-3.5" />
        </button>
      </div>

      {/* Error banner */}
      {switchError && (
        <div className="flex items-center gap-2 mb-4 px-3 py-2 rounded-lg text-xs"
          style={{ background: "rgba(239,68,68,0.1)", border: "1px solid rgba(239,68,68,0.3)", color: "#f87171" }}>
          <AlertCircle className="w-3.5 h-3.5 shrink-0" />
          {switchError}
        </div>
      )}

      {/* Loading */}
      {isLoading && (
        <div className="flex items-center justify-center h-40 gap-2" style={{ color: "#484f58" }}>
          <Loader2 className="w-5 h-5 animate-spin" />
          <span className="text-xs">Loading providers…</span>
        </div>
      )}

      {/* API error */}
      {error && !isLoading && (
        <div className="flex items-center gap-2 px-3 py-2 rounded-lg text-xs"
          style={{ background: "rgba(239,68,68,0.1)", border: "1px solid rgba(239,68,68,0.3)", color: "#f87171" }}>
          <AlertCircle className="w-3.5 h-3.5" />
          Could not load providers — backend may be unreachable
        </div>
      )}

      {/* Local providers */}
      {local.length > 0 && (
        <section className="mb-6">
          <h3 className="text-[10px] font-semibold uppercase tracking-widest mb-3"
            style={{ color: "#a78bfa" }}>
            Local (On-Device)
          </h3>
          <div className="grid grid-cols-1 gap-3 sm:grid-cols-2 xl:grid-cols-3">
            {local.map(p => (
              <ProviderCard
                key={p.provider_id}
                provider={p}
                onActivate={handleActivate}
                activating={activatingId === p.provider_id}
              />
            ))}
          </div>
        </section>
      )}

      {/* Remote providers */}
      {remote.length > 0 && (
        <section>
          <h3 className="text-[10px] font-semibold uppercase tracking-widest mb-3"
            style={{ color: "#60a5fa" }}>
            Remote API Providers
          </h3>
          {/* Free tier highlight */}
          {remote.some(p => p.tier === "free") && (
            <div className="flex items-center gap-2 mb-3 px-3 py-2 rounded-lg text-[11px]"
              style={{ background: "rgba(34,197,94,0.06)", border: "1px solid rgba(34,197,94,0.2)", color: "#4ade80" }}>
              <Globe className="w-3.5 h-3.5 shrink-0" />
              Free-tier providers (Groq, Google) require no credit card — recommended for testing.
            </div>
          )}
          <div className="grid grid-cols-1 gap-3 sm:grid-cols-2 xl:grid-cols-3">
            {remote.map(p => (
              <ProviderCard
                key={p.provider_id}
                provider={p}
                onActivate={handleActivate}
                activating={activatingId === p.provider_id}
              />
            ))}
          </div>
        </section>
      )}

      {/* Empty state */}
      {!isLoading && !error && providers.length === 0 && (
        <div className="flex flex-col items-center justify-center h-40 gap-3" style={{ color: "#484f58" }}>
          <Bot className="w-10 h-10 opacity-20" />
          <p className="text-sm">No providers found</p>
        </div>
      )}
    </div>
  );
}

// ---------------------------------------------------------------------------
// ProcessesTab (extracted from ProcessesPage)
// ---------------------------------------------------------------------------

type InnerTab = "logs" | "containers" | "vlm";

export function ProcessesTab() {
  const [logs, setLogs] = useState<LogEntry[]>([]);
  const [filterLevel, setFilterLevel] = useState<FilterLevel>("INFO");
  const [autoScroll, setAutoScroll] = useState(true);
  const [wsConnected, setWsConnected] = useState(false);
  const [tab, setTab] = useState<InnerTab>("containers");
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
          <button onClick={() => setTab("vlm")}
            className={`flex items-center gap-1.5 px-3 py-1.5 rounded-md text-xs font-medium transition-colors ${
              tab === "vlm" ? "bg-panel text-text-primary" : "text-text-muted hover:text-text-secondary"
            }`}>
            <Bot className="w-3.5 h-3.5" />
            VLM Providers
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

        {/* VLM Providers tab */}
        {tab === "vlm" && <VLMProvidersPanel />}

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
