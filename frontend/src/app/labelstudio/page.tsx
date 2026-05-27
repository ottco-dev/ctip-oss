"use client";

import React, { useState } from "react";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import {
  Tags,
  Link2,
  Link2Off,
  RefreshCw,
  Loader2,
  CheckCircle2,
  AlertCircle,
  ExternalLink,
  ArrowDownToLine,
  ArrowUpFromLine,
  FolderOpen,
  List,
  Settings,
  Upload,
} from "lucide-react";
import { api } from "@/lib/api";

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

interface LSStatus {
  host: string;
  api_key: string;
  connected: boolean;
  last_check: number;
  project_count: number;
}

interface LSProject {
  id: number;
  title: string;
  task_count?: number;
  task_number?: number;
  annotation_count?: number;
  num_tasks_with_annotations?: number;
}

interface LSTask {
  id: number;
  data?: { image?: string };
  is_labeled?: boolean;
  total_annotations?: number;
}

// ---------------------------------------------------------------------------
// Connection panel
// ---------------------------------------------------------------------------

function ConnectionPanel({
  status,
  onConnect,
}: {
  status: LSStatus | undefined;
  onConnect: (host: string, apiKey: string) => Promise<void>;
}) {
  const [host, setHost] = useState(status?.host ?? "http://localhost:8090");
  const [apiKey, setApiKey] = useState("");
  const [isPending, setIsPending] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [success, setSuccess] = useState(false);

  const handleConnect = async () => {
    setIsPending(true);
    setError(null);
    setSuccess(false);
    try {
      await onConnect(host, apiKey);
      setSuccess(true);
    } catch (e: unknown) {
      const err = e as { response?: { data?: { detail?: string } }; message?: string };
      setError(err.response?.data?.detail ?? err.message ?? "Connection failed");
    } finally {
      setIsPending(false);
    }
  };

  return (
    <div className="space-y-4">
      <div>
        <label className="text-xs mb-1.5 block" style={{ color: "#8b949e" }}>
          Label Studio URL
        </label>
        <input
          type="text"
          value={host}
          onChange={(e) => setHost(e.target.value)}
          placeholder="http://localhost:8090"
          className="w-full px-3 py-2 text-sm rounded-lg focus:outline-none"
          style={{ background: "#161b22", border: "1px solid #21262d", color: "#e6edf3" }}
        />
      </div>
      <div>
        <label className="text-xs mb-1.5 block" style={{ color: "#8b949e" }}>
          API Key
        </label>
        <input
          type="password"
          value={apiKey}
          onChange={(e) => setApiKey(e.target.value)}
          placeholder="Label Studio API token"
          className="w-full px-3 py-2 text-sm rounded-lg focus:outline-none"
          style={{ background: "#161b22", border: "1px solid #21262d", color: "#e6edf3" }}
        />
        <p className="text-[10px] mt-1" style={{ color: "#484f58" }}>
          Account → Access Token in Label Studio UI
        </p>
      </div>

      {error && (
        <div className="flex items-start gap-2 px-3 py-2.5 rounded-lg"
          style={{ background: "rgba(239,68,68,0.1)", border: "1px solid rgba(239,68,68,0.2)" }}>
          <AlertCircle className="w-4 h-4 text-red-400 flex-shrink-0 mt-0.5" />
          <p className="text-xs text-red-400">{error}</p>
        </div>
      )}
      {success && (
        <div className="flex items-start gap-2 px-3 py-2.5 rounded-lg"
          style={{ background: "rgba(34,197,94,0.1)", border: "1px solid rgba(34,197,94,0.2)" }}>
          <CheckCircle2 className="w-4 h-4 text-green-400 flex-shrink-0 mt-0.5" />
          <p className="text-xs text-green-400">Connected successfully!</p>
        </div>
      )}

      <button
        onClick={handleConnect}
        disabled={isPending || !host}
        className="w-full flex items-center justify-center gap-2 py-2 rounded-lg text-sm font-medium transition-all"
        style={{
          background: isPending || !host ? "rgba(37,99,235,0.3)" : "#1d4ed8",
          color: isPending || !host ? "rgba(147,197,253,0.5)" : "white",
          cursor: isPending || !host ? "not-allowed" : "pointer",
        }}
      >
        {isPending ? (
          <><Loader2 className="w-4 h-4 animate-spin" />Connecting…</>
        ) : (
          <><Link2 className="w-4 h-4" />Connect</>
        )}
      </button>

      <div className="px-3 py-2.5 rounded-lg text-xs space-y-1.5"
        style={{ background: "#0d1117", border: "1px solid #21262d" }}>
        <p className="font-medium" style={{ color: "#8b949e" }}>Quick Start</p>
        <p style={{ color: "#484f58" }}>
          1. Start Label Studio:{" "}
          <code className="text-green-400 text-[10px]">docker-compose up label-studio</code>
        </p>
        <p style={{ color: "#484f58" }}>
          2. Open{" "}
          <a href="http://localhost:8090" target="_blank" rel="noreferrer" className="text-blue-400 underline">
            localhost:8090
          </a>{" "}
          → create account
        </p>
        <p style={{ color: "#484f58" }}>3. Copy API key from Account → Access Token</p>
      </div>
    </div>
  );
}

// ---------------------------------------------------------------------------
// Tasks drawer
// ---------------------------------------------------------------------------

function TasksDrawer({ projectId, onClose }: { projectId: number; onClose: () => void }) {
  const [page, setPage] = useState(1);
  const PAGE_SIZE = 25;

  const { data, isLoading } = useQuery({
    queryKey: ["ls-tasks", projectId, page],
    queryFn: () =>
      api.get(`/labelstudio/tasks/${projectId}?page=${page}&page_size=${PAGE_SIZE}`).then((r) => r.data),
  });

  const tasks: LSTask[] = data?.tasks ?? [];
  const total: number = data?.total ?? 0;
  const totalPages = Math.ceil(total / PAGE_SIZE);

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center"
      style={{ background: "rgba(0,0,0,0.75)" }}
      onClick={onClose}>
      <div
        className="w-full max-w-xl max-h-[75vh] rounded-2xl flex flex-col overflow-hidden"
        style={{ background: "#161b22", border: "1px solid #30363d" }}
        onClick={(e) => e.stopPropagation()}
      >
        <div className="flex items-center justify-between px-5 py-3.5"
          style={{ borderBottom: "1px solid #21262d" }}>
          <div className="flex items-center gap-2">
            <List className="w-4 h-4 text-blue-400" />
            <h2 className="text-sm font-semibold text-white">Project #{projectId} — {total} tasks</h2>
          </div>
          <button onClick={onClose} className="text-sm px-3 py-1 rounded-lg"
            style={{ background: "#21262d", color: "#8b949e" }}>
            Close
          </button>
        </div>

        <div className="flex-1 overflow-y-auto px-4 py-3 space-y-1.5">
          {isLoading ? (
            <div className="flex items-center justify-center py-16">
              <Loader2 className="w-6 h-6 text-blue-400 animate-spin" />
            </div>
          ) : tasks.length === 0 ? (
            <p className="text-center text-sm py-12" style={{ color: "#484f58" }}>No tasks</p>
          ) : (
            tasks.map((task) => (
              <div key={task.id}
                className="flex items-center gap-3 px-3 py-2 rounded-lg"
                style={{ background: "#0d1117", border: "1px solid #21262d" }}>
                <span className="text-xs font-mono w-12 text-right" style={{ color: "#484f58" }}>#{task.id}</span>
                <span className="flex-1 text-xs truncate" style={{ color: "#8b949e" }}>
                  {task.data?.image?.split("/").pop() ?? "—"}
                </span>
                {task.is_labeled
                  ? <CheckCircle2 className="w-3.5 h-3.5 text-green-400 flex-shrink-0" />
                  : <div className="w-3.5 h-3.5 rounded-full border flex-shrink-0"
                      style={{ borderColor: "#21262d" }} />
                }
                <span className="text-[10px] w-14 text-right" style={{ color: "#484f58" }}>
                  {task.total_annotations ?? 0} ann.
                </span>
              </div>
            ))
          )}
        </div>

        {totalPages > 1 && (
          <div className="flex items-center justify-between px-5 py-3"
            style={{ borderTop: "1px solid #21262d" }}>
            <button disabled={page <= 1} onClick={() => setPage((p) => p - 1)}
              className="text-xs px-3 py-1 rounded disabled:opacity-40"
              style={{ background: "#21262d", color: "#8b949e" }}>Prev</button>
            <span className="text-xs" style={{ color: "#484f58" }}>
              {page} / {totalPages}
            </span>
            <button disabled={page >= totalPages} onClick={() => setPage((p) => p + 1)}
              className="text-xs px-3 py-1 rounded disabled:opacity-40"
              style={{ background: "#21262d", color: "#8b949e" }}>Next</button>
          </div>
        )}
      </div>
    </div>
  );
}

// ---------------------------------------------------------------------------
// Project card
// ---------------------------------------------------------------------------

function ProjectCard({
  project,
  onImport,
  onViewTasks,
  importing,
  importMsg,
}: {
  project: LSProject;
  onImport: (id: number) => void;
  onViewTasks: (id: number) => void;
  importing: boolean;
  importMsg?: string;
}) {
  const taskCount = project.task_count ?? project.task_number ?? 0;
  const annCount = project.annotation_count ?? project.num_tasks_with_annotations ?? 0;
  const pct = taskCount > 0 ? Math.round((annCount / taskCount) * 100) : 0;

  return (
    <div className="rounded-xl p-4 space-y-3"
      style={{ background: "#0d1117", border: "1px solid #21262d" }}>
      <div className="flex items-start justify-between gap-3">
        <div>
          <p className="text-sm font-medium text-white truncate">{project.title}</p>
          <p className="text-xs mt-0.5" style={{ color: "#484f58" }}>#{project.id}</p>
        </div>
        <span className="text-xs px-2 py-0.5 rounded-full font-medium flex-shrink-0"
          style={{
            background: pct === 100 ? "rgba(34,197,94,0.15)" : "rgba(59,130,246,0.15)",
            color: pct === 100 ? "#22c55e" : "#60a5fa",
          }}>
          {pct}%
        </span>
      </div>

      {/* Progress */}
      <div className="space-y-1">
        <div className="h-1.5 rounded-full overflow-hidden" style={{ background: "#21262d" }}>
          <div className="h-full rounded-full transition-all"
            style={{
              width: `${pct}%`,
              background: pct === 100 ? "#22c55e" : "linear-gradient(90deg,#3b82f6,#60a5fa)",
            }} />
        </div>
        <div className="flex justify-between text-[10px]" style={{ color: "#484f58" }}>
          <span>{annCount} annotated</span>
          <span>{taskCount} tasks</span>
        </div>
      </div>

      {importMsg && (
        <p className="text-xs" style={{ color: importMsg.startsWith("✓") ? "#4ade80" : "#f87171" }}>
          {importMsg}
        </p>
      )}

      <div className="flex gap-2">
        <button onClick={() => onViewTasks(project.id)}
          className="flex-1 flex items-center justify-center gap-1.5 py-1.5 rounded-lg text-xs font-medium transition-colors"
          style={{ background: "transparent", border: "1px solid #21262d", color: "#8b949e" }}>
          <List className="w-3.5 h-3.5" />Tasks
        </button>
        <button onClick={() => onImport(project.id)}
          disabled={importing || annCount === 0}
          className="flex-1 flex items-center justify-center gap-1.5 py-1.5 rounded-lg text-xs font-medium transition-all"
          style={{
            background: importing || annCount === 0 ? "rgba(34,197,94,0.08)" : "rgba(34,197,94,0.2)",
            color: importing || annCount === 0 ? "rgba(74,222,128,0.4)" : "#4ade80",
            cursor: importing || annCount === 0 ? "not-allowed" : "pointer",
          }}>
          {importing ? <Loader2 className="w-3.5 h-3.5 animate-spin" /> : <ArrowDownToLine className="w-3.5 h-3.5" />}
          Import
        </button>
      </div>
    </div>
  );
}

// ---------------------------------------------------------------------------
// Main page
// ---------------------------------------------------------------------------

export default function LabelStudioPage() {
  const queryClient = useQueryClient();
  const [panel, setPanel] = useState<"projects" | "settings">("projects");
  const [tasksProjectId, setTasksProjectId] = useState<number | null>(null);
  const [importingId, setImportingId] = useState<number | null>(null);
  const [importResults, setImportResults] = useState<Record<number, string>>({});

  const { data: status, refetch: refetchStatus } = useQuery<LSStatus>({
    queryKey: ["ls-status"],
    queryFn: () => api.get("/labelstudio/status").then((r) => r.data),
    refetchInterval: 30_000,
  });

  const { data: projectsData, isLoading: projectsLoading, refetch: refetchProjects } = useQuery({
    queryKey: ["ls-projects"],
    queryFn: () => api.get("/labelstudio/projects").then((r) => r.data),
    enabled: status?.connected === true,
    staleTime: 30_000,
  });

  const connectMutation = useMutation({
    mutationFn: ({ host, apiKey }: { host: string; apiKey: string }) =>
      api.post("/labelstudio/connect", { host, api_key: apiKey }).then((r) => r.data),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["ls-status"] });
      queryClient.invalidateQueries({ queryKey: ["ls-projects"] });
      setPanel("projects");
    },
  });

  const handleImport = async (projectId: number) => {
    setImportingId(projectId);
    setImportResults((p) => ({ ...p, [projectId]: "" }));
    try {
      const r = await api.post(`/labelstudio/import/${projectId}`);
      setImportResults((p) => ({
        ...p,
        [projectId]: `✓ ${r.data.imported_to_queue ?? 0} queued for review`,
      }));
      queryClient.invalidateQueries({ queryKey: ["annotation-queue"] });
    } catch (e: unknown) {
      const err = e as { response?: { data?: { detail?: string } }; message?: string };
      setImportResults((p) => ({
        ...p,
        [projectId]: `✗ ${err.response?.data?.detail ?? err.message ?? "Import failed"}`,
      }));
    } finally {
      setImportingId(null);
    }
  };

  const connected = status?.connected ?? false;
  const projects: LSProject[] = projectsData?.projects ?? [];

  return (
    <div className="flex h-full">
      {/* Main */}
      <div className="flex-1 flex flex-col min-w-0">
        {/* Header */}
        <div className="flex items-center justify-between px-5 py-3"
          style={{ borderBottom: "1px solid #21262d" }}>
          <div className="flex items-center gap-3">
            <Tags className="w-4 h-4 text-blue-400" />
            <h1 className="text-base font-semibold text-white">Label Studio</h1>
            {connected ? (
              <span className="flex items-center gap-1.5 text-xs px-2.5 py-1 rounded-full"
                style={{ background: "rgba(34,197,94,0.15)", color: "#4ade80" }}>
                <span className="w-1.5 h-1.5 rounded-full bg-green-400 animate-pulse" />
                Connected — {status?.host}
              </span>
            ) : (
              <span className="flex items-center gap-1.5 text-xs px-2.5 py-1 rounded-full"
                style={{ background: "rgba(239,68,68,0.1)", color: "#f87171" }}>
                <Link2Off className="w-3 h-3" />
                Not connected
              </span>
            )}
          </div>
          <div className="flex items-center gap-2">
            {connected && (
              <>
                <a href={status?.host} target="_blank" rel="noreferrer"
                  className="flex items-center gap-1.5 text-xs px-3 py-1.5 rounded-lg"
                  style={{ border: "1px solid #21262d", color: "#8b949e" }}>
                  <ExternalLink className="w-3.5 h-3.5" />
                  Open LS
                </a>
                <button onClick={() => refetchProjects()}
                  className="p-2 rounded-lg" style={{ color: "#484f58" }}>
                  <RefreshCw className="w-4 h-4" />
                </button>
              </>
            )}
            <button
              onClick={() => setPanel((p) => p === "settings" ? "projects" : "settings")}
              className="p-2 rounded-lg transition-colors"
              style={{ background: panel === "settings" ? "#21262d" : "transparent", color: panel === "settings" ? "#e6edf3" : "#484f58" }}>
              <Settings className="w-4 h-4" />
            </button>
          </div>
        </div>

        {/* Stats bar */}
        {connected && (
          <div className="flex items-center gap-6 px-5 py-2"
            style={{ borderBottom: "1px solid #21262d", background: "#161b22" }}>
            <div className="flex items-center gap-1.5">
              <span className="text-xs" style={{ color: "#484f58" }}>Projects:</span>
              <span className="text-xs font-bold text-white">{status?.project_count ?? projects.length}</span>
            </div>
          </div>
        )}

        {/* Content */}
        <div className="flex-1 overflow-y-auto px-5 py-4">
          {!connected ? (
            <div className="max-w-sm mx-auto mt-8 text-center space-y-4">
              <div className="w-16 h-16 rounded-2xl flex items-center justify-center mx-auto"
                style={{ background: "rgba(59,130,246,0.1)" }}>
                <Tags className="w-8 h-8 text-blue-400" />
              </div>
              <h2 className="text-lg font-semibold text-white">Label Studio Integration</h2>
              <p className="text-sm" style={{ color: "#484f58" }}>
                Connect to your Label Studio instance to import completed annotations
                into the human review queue.
              </p>
              <button onClick={() => setPanel("settings")}
                className="inline-flex items-center gap-2 px-4 py-2 rounded-lg text-sm font-medium"
                style={{ background: "#1d4ed8", color: "white" }}>
                <Settings className="w-4 h-4" />Configure Connection
              </button>
            </div>
          ) : projectsLoading ? (
            <div className="flex items-center justify-center py-20">
              <Loader2 className="w-8 h-8 text-blue-400 animate-spin" />
            </div>
          ) : projects.length === 0 ? (
            <div className="text-center py-16 space-y-3">
              <FolderOpen className="w-10 h-10 mx-auto opacity-30" style={{ color: "#484f58" }} />
              <p className="text-sm" style={{ color: "#484f58" }}>No projects in Label Studio</p>
              <a href={status?.host} target="_blank" rel="noreferrer"
                className="inline-flex items-center gap-2 text-sm text-blue-400 underline">
                <ExternalLink className="w-4 h-4" />Create a project
              </a>
            </div>
          ) : (
            <div className="space-y-3">
              <div className="flex items-start gap-2 px-3 py-2.5 rounded-lg mb-1"
                style={{ background: "rgba(59,130,246,0.08)", border: "1px solid rgba(59,130,246,0.15)" }}>
                <Upload className="w-4 h-4 text-blue-400 flex-shrink-0 mt-0.5" />
                <p className="text-xs" style={{ color: "rgba(147,197,253,0.8)" }}>
                  <strong>Workflow:</strong> Annotate in Label Studio → Import to queue → Human review → Training data
                </p>
              </div>
              <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
                {projects.map((p) => (
                  <ProjectCard key={p.id} project={p}
                    onImport={handleImport}
                    onViewTasks={(id) => setTasksProjectId(id)}
                    importing={importingId === p.id}
                    importMsg={importResults[p.id]}
                  />
                ))}
              </div>
            </div>
          )}
        </div>
      </div>

      {/* Right sidebar */}
      <div className="w-72 flex-shrink-0 flex flex-col overflow-y-auto"
        style={{ borderLeft: "1px solid #21262d" }}>
        <div className="flex items-center gap-2 px-4 py-3"
          style={{ borderBottom: "1px solid #21262d" }}>
          <Settings className="w-4 h-4 text-blue-400" />
          <h2 className="text-sm font-semibold text-white">Connection</h2>
        </div>
        <div className="flex-1 p-4">
          <ConnectionPanel
            status={status}
            onConnect={async (host, apiKey) => {
              await connectMutation.mutateAsync({ host, apiKey });
            }}
          />
        </div>

        {connected && (
          <div className="px-4 py-3 space-y-1" style={{ borderTop: "1px solid #21262d" }}>
            <p className="text-[10px] uppercase font-medium mb-2" style={{ color: "#484f58" }}>
              Quick Links
            </p>
            {[
              { href: "/annotation", label: "→ Review Queue" },
              { href: "/datasets", label: "→ Datasets" },
              { href: "/training", label: "→ Training" },
            ].map(({ href, label }) => (
              <a key={href} href={href} className="block text-xs py-1"
                style={{ color: "#484f58" }}>
                {label}
              </a>
            ))}
          </div>
        )}
      </div>

      {/* Tasks drawer */}
      {tasksProjectId !== null && (
        <TasksDrawer projectId={tasksProjectId} onClose={() => setTasksProjectId(null)} />
      )}
    </div>
  );
}
