"use client";

import React, { useState, useCallback, useRef } from "react";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import {
  FileText,
  Download,
  Plus,
  RefreshCw,
  Loader2,
  AlertTriangle,
  CheckCircle2,
  Clock,
  X,
  Sparkles,
  Copy,
  Check,
} from "lucide-react";
import { api } from "@/lib/api";
import { cn, formatDistanceToNow, formatBytes } from "@/lib/utils";

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

interface Report {
  id: string;
  name: string;
  format: string;
  report_type?: string;
  status: string;
  created_at: string;
  size_bytes?: number;
  download_url?: string;
}

interface GenerateRequest {
  report_type: "session" | "benchmark" | "scientific";
  format: "json" | "pdf" | "csv";
  name: string;
  include_charts?: boolean;
  include_per_image?: boolean;
}

// ---------------------------------------------------------------------------
// Ollama types
// ---------------------------------------------------------------------------

interface OllamaStatus {
  available: boolean;
  base_url: string;
  installed_models: string[];
  current_model: string;
}

interface OllamaModel {
  name: string;
  size_gb: number;
}

interface OllamaModelsResponse {
  models: OllamaModel[];
}

type NarrativeStyle = "scientific" | "summary" | "technical";
type NarrativeLanguage = "en" | "de" | "es";

interface NarrativeRequest {
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  analysis_result: Record<string, any>;
  style: NarrativeStyle;
  language: NarrativeLanguage;
  model?: string;
}

interface NarrativeResponse {
  narrative: string;
  model: string;
  style: NarrativeStyle;
  language: NarrativeLanguage;
  generation_time_ms: number;
}

// ---------------------------------------------------------------------------
// Default example analysis for the textarea
// ---------------------------------------------------------------------------

const DEFAULT_ANALYSIS = JSON.stringify(
  {
    session_id: "session_001",
    total_detections: 147,
    type_distribution: {
      CAPITATE_STALKED: 89,
      CAPITATE_SESSILE: 34,
      BULBOUS: 18,
      NON_GLANDULAR: 6,
    },
    maturity_distribution: {
      clear_pct: 22.4,
      cloudy_pct: 61.9,
      amber_pct: 15.7,
    },
    harvest_recommendation:
      "Predominantly cloudy trichomes suggest peak optical maturity for cerebral effect profile.",
    confidence_stats: { mean: 0.847, min: 0.612, max: 0.981 },
    timestamp: "2026-05-29T12:00:00Z",
  },
  null,
  2,
);

// ---------------------------------------------------------------------------
// OllamaNarrativePanel
// ---------------------------------------------------------------------------

function OllamaNarrativePanel() {
  const [style, setStyle] = useState<NarrativeStyle>("scientific");
  const [language, setLanguage] = useState<NarrativeLanguage>("en");
  const [selectedModel, setSelectedModel] = useState<string>("");
  const [analysisJson, setAnalysisJson] = useState<string>(DEFAULT_ANALYSIS);
  const [jsonError, setJsonError] = useState<string | null>(null);
  const [copied, setCopied] = useState(false);
  const narrativeRef = useRef<HTMLTextAreaElement>(null);

  // Query: Ollama status
  const {
    data: statusData,
    isLoading: statusLoading,
    refetch: refetchStatus,
    isRefetching: statusRefetching,
  } = useQuery<OllamaStatus>({
    queryKey: ["ollama-status"],
    queryFn: () => api.get("/ollama/status").then((r) => r.data as OllamaStatus),
    refetchInterval: 30_000,
    retry: 1,
  });

  const isOnline = statusData?.available === true;

  // Query: Ollama models (only when online)
  const { data: modelsData } = useQuery<OllamaModelsResponse>({
    queryKey: ["ollama-models"],
    queryFn: () => api.get("/ollama/models").then((r) => r.data as OllamaModelsResponse),
    enabled: isOnline,
    staleTime: 60_000,
  });

  const availableModels: OllamaModel[] = modelsData?.models ?? [];

  // Resolve displayed model options — fall back to current_model when list is empty
  const modelOptions: string[] =
    availableModels.length > 0
      ? availableModels.map((m) => m.name)
      : statusData?.current_model
      ? [statusData.current_model]
      : [];

  // Effective model for the request (blank = let backend decide)
  const effectiveModel = selectedModel || modelOptions[0] || undefined;

  // Mutation: generate narrative
  const narrativeMutation = useMutation<NarrativeResponse, Error, NarrativeRequest>({
    mutationFn: (body) =>
      api.post("/ollama/narrative", body).then((r) => r.data as NarrativeResponse),
  });

  // JSON validation on change
  const handleJsonChange = useCallback((val: string) => {
    setAnalysisJson(val);
    try {
      JSON.parse(val);
      setJsonError(null);
    } catch {
      setJsonError("Invalid JSON");
    }
  }, []);

  const isJsonValid = jsonError === null;

  const handleGenerate = () => {
    if (!isOnline || !isJsonValid || narrativeMutation.isPending) return;
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    const parsed = JSON.parse(analysisJson) as Record<string, any>;
    narrativeMutation.mutate({
      analysis_result: parsed,
      style,
      language,
      model: effectiveModel,
    });
  };

  const handleCopy = () => {
    const text = narrativeMutation.data?.narrative;
    if (!text) return;
    navigator.clipboard.writeText(text).then(() => {
      setCopied(true);
      setTimeout(() => setCopied(false), 2000);
    });
  };

  return (
    <div
      className="rounded-xl overflow-hidden"
      style={{ background: "#161b22", border: "1px solid #21262d" }}
    >
      {/* ── Header ── */}
      <div
        className="flex items-center justify-between px-5 py-3"
        style={{ borderBottom: "1px solid #21262d" }}
      >
        <div className="flex items-center gap-2">
          <Sparkles className="w-4 h-4" style={{ color: "#58a6ff" }} />
          <h2 className="text-sm font-semibold" style={{ color: "#e6edf3" }}>
            AI Narrative
          </h2>
        </div>

        <div className="flex items-center gap-3">
          {/* Status badge */}
          {statusLoading ? (
            <span className="flex items-center gap-1.5 text-xs" style={{ color: "#484f58" }}>
              <Loader2 className="w-3 h-3 animate-spin" />
              Checking…
            </span>
          ) : (
            <span className="flex items-center gap-1.5 text-xs font-medium">
              <span
                className="w-2 h-2 rounded-full"
                style={{ background: isOnline ? "#3fb950" : "#d29922" }}
              />
              <span style={{ color: isOnline ? "#3fb950" : "#d29922" }}>
                {isOnline ? "Online" : "Offline"}
              </span>
            </span>
          )}

          {/* Refresh status */}
          <button
            onClick={() => void refetchStatus()}
            disabled={statusRefetching}
            className="p-1 rounded transition-opacity disabled:opacity-40"
            style={{ color: "#484f58" }}
            title="Re-check Ollama status"
          >
            <RefreshCw className={cn("w-3.5 h-3.5", statusRefetching && "animate-spin")} />
          </button>
        </div>
      </div>

      <div className="p-5 space-y-5">
        {/* ── Offline info box ── */}
        {!statusLoading && !isOnline && (
          <div
            className="px-4 py-3 rounded-lg text-xs leading-relaxed"
            style={{
              background: "rgba(72,79,88,0.15)",
              border: "1px solid #21262d",
              color: "#8b949e",
            }}
          >
            Ollama is not running. Start it with:{" "}
            <code
              className="px-1 py-0.5 rounded text-[11px]"
              style={{ background: "#0d1117", color: "#e6edf3", fontFamily: "monospace" }}
            >
              ollama serve
            </code>
            {" — "}then install a model:{" "}
            <code
              className="px-1 py-0.5 rounded text-[11px]"
              style={{ background: "#0d1117", color: "#e6edf3", fontFamily: "monospace" }}
            >
              ollama pull llama3.2:3b
            </code>
          </div>
        )}

        {/* ── Config row (only when online) ── */}
        {isOnline && (
          <div className="flex flex-wrap gap-3">
            {/* Style select */}
            <div className="flex flex-col gap-1 min-w-[130px]">
              <label className="text-[11px]" style={{ color: "#484f58" }}>
                Style
              </label>
              <select
                value={style}
                onChange={(e) => setStyle(e.target.value as NarrativeStyle)}
                className="px-2.5 py-1.5 text-xs rounded-lg focus:outline-none focus:ring-1 focus:ring-blue-500/50"
                style={{
                  background: "#0d1117",
                  border: "1px solid #21262d",
                  color: "#e6edf3",
                }}
              >
                <option value="scientific">Scientific</option>
                <option value="summary">Summary</option>
                <option value="technical">Technical</option>
              </select>
            </div>

            {/* Language select */}
            <div className="flex flex-col gap-1 min-w-[90px]">
              <label className="text-[11px]" style={{ color: "#484f58" }}>
                Language
              </label>
              <select
                value={language}
                onChange={(e) => setLanguage(e.target.value as NarrativeLanguage)}
                className="px-2.5 py-1.5 text-xs rounded-lg focus:outline-none focus:ring-1 focus:ring-blue-500/50"
                style={{
                  background: "#0d1117",
                  border: "1px solid #21262d",
                  color: "#e6edf3",
                }}
              >
                <option value="en">EN</option>
                <option value="de">DE</option>
                <option value="es">ES</option>
              </select>
            </div>

            {/* Model select */}
            <div className="flex flex-col gap-1 flex-1 min-w-[160px]">
              <label className="text-[11px]" style={{ color: "#484f58" }}>
                Model
              </label>
              <select
                value={selectedModel || effectiveModel || ""}
                onChange={(e) => setSelectedModel(e.target.value)}
                className="px-2.5 py-1.5 text-xs rounded-lg focus:outline-none focus:ring-1 focus:ring-blue-500/50"
                style={{
                  background: "#0d1117",
                  border: "1px solid #21262d",
                  color: "#e6edf3",
                }}
                disabled={modelOptions.length === 0}
              >
                {modelOptions.length === 0 ? (
                  <option value="">No models found</option>
                ) : (
                  modelOptions.map((name) => {
                    const sizeGb = availableModels.find((m) => m.name === name)?.size_gb;
                    return (
                      <option key={name} value={name}>
                        {name}
                        {sizeGb !== undefined ? ` (${sizeGb.toFixed(1)} GB)` : ""}
                      </option>
                    );
                  })
                )}
              </select>
            </div>
          </div>
        )}

        {/* ── Analysis JSON textarea ── */}
        <div className="space-y-1.5">
          <div className="flex items-center justify-between">
            <label className="text-[11px]" style={{ color: "#484f58" }}>
              Analysis data (JSON)
            </label>
            {jsonError && (
              <span className="text-[10px] font-medium text-red-400">{jsonError}</span>
            )}
          </div>
          <textarea
            rows={8}
            value={analysisJson}
            onChange={(e) => handleJsonChange(e.target.value)}
            spellCheck={false}
            className="w-full px-3 py-2.5 text-xs rounded-lg resize-y focus:outline-none focus:ring-1"
            style={{
              background: "#0d1117",
              border: `1px solid ${jsonError ? "rgba(239,68,68,0.6)" : "#21262d"}`,
              color: "#e6edf3",
              fontFamily: "ui-monospace, 'Cascadia Code', 'Source Code Pro', monospace",
              lineHeight: "1.6",
              transition: "border-color 0.15s",
            }}
          />
        </div>

        {/* ── Generate button ── */}
        <button
          onClick={handleGenerate}
          disabled={!isOnline || !isJsonValid || narrativeMutation.isPending}
          className="flex items-center justify-center gap-2 w-full py-2 rounded-lg text-sm font-medium transition-colors disabled:opacity-40"
          style={{ background: "#1f6feb", color: "#e6edf3" }}
        >
          {narrativeMutation.isPending ? (
            <>
              <Loader2 className="w-4 h-4 animate-spin" />
              Generating…
            </>
          ) : (
            <>
              <Sparkles className="w-4 h-4" />
              Generate Narrative
            </>
          )}
        </button>

        {/* ── Generation error ── */}
        {narrativeMutation.isError && (
          <div
            className="flex items-start gap-2 px-3 py-2.5 rounded-lg text-xs"
            style={{
              background: "rgba(239,68,68,0.1)",
              border: "1px solid rgba(239,68,68,0.2)",
              color: "#fca5a5",
            }}
          >
            <AlertTriangle className="w-3.5 h-3.5 flex-shrink-0 mt-0.5" />
            {narrativeMutation.error.message ?? "Generation failed"}
          </div>
        )}

        {/* ── Narrative output ── */}
        {narrativeMutation.isSuccess && narrativeMutation.data && (
          <div className="space-y-2">
            <textarea
              ref={narrativeRef}
              readOnly
              value={narrativeMutation.data.narrative}
              rows={6}
              className="w-full px-4 py-3 text-sm rounded-lg resize-none focus:outline-none leading-relaxed"
              style={{
                background: "#0d1117",
                border: "1px solid #21262d",
                color: "#e6edf3",
                fontFamily:
                  "ui-sans-serif, system-ui, -apple-system, 'Segoe UI', sans-serif",
                lineHeight: "1.7",
                // auto-expand visually: use scrollHeight trick via style calc
                minHeight: "6rem",
              }}
              onInput={(e) => {
                const el = e.currentTarget;
                el.style.height = "auto";
                el.style.height = `${el.scrollHeight}px`;
              }}
            />

            {/* Meta line + copy button */}
            <div className="flex items-center justify-between">
              <p className="text-[11px]" style={{ color: "#484f58" }}>
                Generated by{" "}
                <span style={{ color: "#8b949e" }}>{narrativeMutation.data.model}</span>{" "}
                in{" "}
                <span style={{ color: "#8b949e" }}>
                  {narrativeMutation.data.generation_time_ms}ms
                </span>{" "}
                &middot;{" "}
                <span style={{ color: "#8b949e" }}>{narrativeMutation.data.style}</span>{" "}
                &middot;{" "}
                <span style={{ color: "#8b949e" }}>
                  {narrativeMutation.data.language.toUpperCase()}
                </span>
              </p>

              <button
                onClick={handleCopy}
                className="flex items-center gap-1.5 px-2.5 py-1 rounded text-[11px] font-medium transition-colors"
                style={{
                  background: "#161b22",
                  border: "1px solid #21262d",
                  color: copied ? "#3fb950" : "#8b949e",
                }}
              >
                {copied ? (
                  <>
                    <Check className="w-3 h-3" />
                    Copied
                  </>
                ) : (
                  <>
                    <Copy className="w-3 h-3" />
                    Copy
                  </>
                )}
              </button>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}

// ---------------------------------------------------------------------------
// Report card
// ---------------------------------------------------------------------------

const STATUS_COLORS: Record<string, string> = {
  completed: "bg-green-500/20 text-green-400",
  pending: "bg-yellow-500/20 text-yellow-400",
  generating: "bg-blue-500/20 text-blue-400",
  failed: "bg-red-500/20 text-red-400",
};

const FORMAT_ICON: Record<string, string> = {
  pdf: "📄",
  json: "🗂",
  csv: "📊",
};

function ReportCard({ report }: { report: Report }) {
  const handleDownload = () => {
    // Triggers file download via the backend download endpoint
    window.open(`/api/v1/reports/${report.id}/download`, "_blank");
  };

  return (
    <div
      className="flex items-center gap-4 px-4 py-3 rounded-xl"
      style={{ background: '#0d1117', border: '1px solid #21262d' }}
    >
      <div className="text-2xl">{FORMAT_ICON[report.format] ?? "📁"}</div>

      <div className="flex-1 min-w-0">
        <p className="text-sm font-medium text-white truncate">{report.name}</p>
        <div className="flex items-center gap-3 mt-0.5 text-xs" style={{ color: '#484f58' }}>
          <span className="uppercase">{report.format}</span>
          {report.size_bytes && <span>{formatBytes(report.size_bytes)}</span>}
          <span>{formatDistanceToNow(new Date(report.created_at))}</span>
          {report.report_type && <span className="capitalize">{report.report_type}</span>}
        </div>
      </div>

      <span
        className={cn(
          "text-[10px] px-2 py-0.5 rounded-full font-medium",
          STATUS_COLORS[report.status] ?? "bg-gray-500/20 text-gray-400"
        )}
      >
        {report.status === "generating" ? (
          <span className="flex items-center gap-1">
            <Loader2 className="w-3 h-3 animate-spin" />
            {report.status}
          </span>
        ) : report.status === "completed" ? (
          <span className="flex items-center gap-1">
            <CheckCircle2 className="w-3 h-3" />
            {report.status}
          </span>
        ) : report.status === "pending" ? (
          <span className="flex items-center gap-1">
            <Clock className="w-3 h-3" />
            {report.status}
          </span>
        ) : report.status}
      </span>

      {report.status === "completed" && (
        <button
          onClick={handleDownload}
          className="flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-xs font-medium transition-colors"
          style={{ background: '#161b22', border: '1px solid #21262d', color: '#8b949e' }}
        >
          <Download className="w-3.5 h-3.5" />
          Download
        </button>
      )}
    </div>
  );
}

// ---------------------------------------------------------------------------
// Generate modal
// ---------------------------------------------------------------------------

function GenerateModal({ onClose }: { onClose: () => void }) {
  const queryClient = useQueryClient();
  const [form, setForm] = useState<GenerateRequest>({
    report_type: "session",
    format: "pdf",
    name: `Report ${new Date().toLocaleDateString()}`,
    include_charts: true,
    include_per_image: false,
  });

  const generateMutation = useMutation({
    mutationFn: () => api.post("/reports/generate", form).then((r) => r.data),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["reports"] });
      onClose();
    },
  });

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
          <h2 className="text-lg font-bold text-white">Generate Report</h2>
          <button onClick={onClose} className="p-1 rounded transition-colors" style={{ color: '#484f58' }}>
            <X className="w-5 h-5" />
          </button>
        </div>

        {/* Report name */}
        <div>
          <label className="text-xs mb-1.5 block" style={{ color: '#8b949e' }}>Report name</label>
          <input
            className="w-full px-3 py-2 text-sm rounded-lg focus:outline-none"
            style={{ background: '#161b22', border: '1px solid #21262d', color: '#e6edf3' }}
            value={form.name}
            onChange={(e) => setForm({ ...form, name: e.target.value })}
          />
        </div>

        {/* Report type */}
        <div>
          <label className="text-xs mb-1.5 block" style={{ color: '#8b949e' }}>Report type</label>
          <select
            className="w-full px-3 py-2 text-sm rounded-lg focus:outline-none"
            style={{ background: '#161b22', border: '1px solid #21262d', color: '#8b949e' }}
            value={form.report_type}
            onChange={(e) => setForm({ ...form, report_type: e.target.value as GenerateRequest["report_type"] })}
          >
            <option value="session">Session Summary</option>
            <option value="benchmark">Benchmark Results</option>
            <option value="scientific">Scientific Report</option>
          </select>
        </div>

        {/* Format */}
        <div>
          <label className="text-xs mb-1.5 block" style={{ color: '#8b949e' }}>Format</label>
          <div className="flex gap-2">
            {(["pdf", "json", "csv"] as const).map((fmt) => (
              <button
                key={fmt}
                onClick={() => setForm({ ...form, format: fmt })}
                className="flex-1 py-2 rounded-lg text-sm font-medium border transition-colors"
                style={{
                  background: form.format === fmt ? 'rgba(59,130,246,0.15)' : '#161b22',
                  border: form.format === fmt ? '1px solid rgba(59,130,246,0.5)' : '1px solid #21262d',
                  color: form.format === fmt ? '#93c5fd' : '#484f58',
                }}
              >
                {FORMAT_ICON[fmt]} {fmt.toUpperCase()}
              </button>
            ))}
          </div>
        </div>

        {/* Options */}
        <div className="space-y-2">
          {[
            { key: "include_charts", label: "Embed charts (PDF only)" },
            { key: "include_per_image", label: "Include per-image breakdown" },
          ].map(({ key, label }) => (
            <label key={key} className="flex items-center gap-2 cursor-pointer">
              <input
                type="checkbox"
                checked={!!form[key as keyof GenerateRequest]}
                onChange={(e) => setForm({ ...form, [key]: e.target.checked })}
                className="accent-blue-500"
              />
              <span className="text-sm" style={{ color: '#8b949e' }}>{label}</span>
            </label>
          ))}
        </div>

        {/* Scientific caveat */}
        <div
          className="px-3 py-2.5 rounded-lg"
          style={{ background: 'rgba(234,179,8,0.1)', border: '1px solid rgba(234,179,8,0.2)' }}
        >
          <p className="text-[11px]" style={{ color: 'rgba(254,243,199,0.8)' }}>
            Scientific caveats are automatically included in all report formats.
          </p>
        </div>

        {/* Actions */}
        <div className="flex gap-3">
          <button
            onClick={onClose}
            className="flex-1 py-2 text-sm rounded-lg transition-colors"
            style={{ background: '#161b22', color: '#8b949e', border: '1px solid #21262d' }}
          >
            Cancel
          </button>
          <button
            onClick={() => generateMutation.mutate()}
            disabled={generateMutation.isPending || !form.name}
            className="flex-1 py-2 bg-blue-600 hover:bg-blue-500 disabled:opacity-50 text-white text-sm font-medium rounded-lg transition-colors"
          >
            {generateMutation.isPending ? (
              <span className="flex items-center justify-center gap-2">
                <Loader2 className="w-4 h-4 animate-spin" />
                Generating…
              </span>
            ) : "Generate"}
          </button>
        </div>

        {generateMutation.isError && (
          <p className="text-xs text-red-400">
            Failed: {(generateMutation.error as Error)?.message ?? "Unknown error"}
          </p>
        )}
      </div>
    </div>
  );
}

// ---------------------------------------------------------------------------
// Page
// ---------------------------------------------------------------------------

export default function ReportsPage() {
  const [showModal, setShowModal] = useState(false);
  const [filterFormat, setFilterFormat] = useState<string>("all");

  const { data, isLoading, isError, error, refetch } = useQuery({
    queryKey: ["reports"],
    queryFn: () => api.get("/reports").then((r) => r.data),
    refetchInterval: 5_000,
  });

  // Handle both {reports: [...]} and [...] response shapes
  const reports: Report[] = Array.isArray(data) ? data : (data?.reports ?? []);

  const filtered =
    filterFormat === "all" ? reports : reports.filter((r) => r.format === filterFormat);

  return (
    <div className="flex flex-col h-full overflow-y-auto">
      {/* Header */}
      <div
        className="flex items-center justify-between px-5 py-3"
        style={{ borderBottom: '1px solid #21262d' }}
      >
        <div className="flex items-center gap-2">
          <FileText className="w-4 h-4 text-blue-400" />
          <h1 className="text-base font-semibold text-white">Reports</h1>
        </div>
        <div className="flex items-center gap-3">
          <button
            onClick={() => refetch()}
            className="p-1.5 rounded transition-colors"
            style={{ color: '#484f58' }}
          >
            <RefreshCw className="w-4 h-4" />
          </button>
          <button
            onClick={() => setShowModal(true)}
            className="flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-sm font-medium bg-blue-600 hover:bg-blue-500 text-white transition-colors"
          >
            <Plus className="w-4 h-4" />
            New Report
          </button>
        </div>
      </div>

      <div className="flex-1 p-5 space-y-4">
        {/* Scientific caveat */}
        <div
          className="flex items-start gap-3 px-4 py-3 rounded-xl"
          style={{ background: 'rgba(234,179,8,0.08)', border: '1px solid rgba(234,179,8,0.2)' }}
        >
          <AlertTriangle className="w-4 h-4 text-yellow-400 flex-shrink-0 mt-0.5" />
          <div>
            <p className="text-xs font-semibold text-yellow-400 mb-0.5">Scientific Disclaimer</p>
            <p className="text-xs leading-relaxed" style={{ color: 'rgba(254,243,199,0.75)' }}>
              All reports include mandatory caveats: visual maturity analysis does NOT allow
              quantitative determination of THC, CBD, or other cannabinoid concentrations.
              Reports are suitable for research documentation and harvest timing guidance only.
            </p>
          </div>
        </div>

        {/* Format filters */}
        <div className="flex gap-2">
          {["all", "pdf", "json", "csv"].map((fmt) => (
            <button
              key={fmt}
              onClick={() => setFilterFormat(fmt)}
              className="px-3 py-1.5 rounded-lg text-xs font-medium transition-colors"
              style={{
                background: filterFormat === fmt ? '#161b22' : 'transparent',
                border: filterFormat === fmt ? '1px solid #30363d' : '1px solid #21262d',
                color: filterFormat === fmt ? '#e6edf3' : '#484f58',
              }}
            >
              {fmt === "all" ? "All" : fmt.toUpperCase()}
            </button>
          ))}
        </div>

        {/* Loading */}
        {isLoading && (
          <div className="flex items-center justify-center py-16">
            <Loader2 className="w-6 h-6 text-blue-400 animate-spin" />
          </div>
        )}

        {/* Error */}
        {isError && (
          <div
            className="flex items-start gap-3 px-4 py-3 rounded-lg"
            style={{ background: 'rgba(239,68,68,0.1)', border: '1px solid rgba(239,68,68,0.2)' }}
          >
            <AlertTriangle className="w-4 h-4 text-red-400 mt-0.5" />
            <p className="text-sm text-red-400">{(error as Error)?.message ?? "Failed to load reports"}</p>
          </div>
        )}

        {/* Empty state */}
        {!isLoading && !isError && filtered.length === 0 && (
          <div
            className="rounded-xl p-12 text-center"
            style={{ background: '#0d1117', border: '1px solid #21262d' }}
          >
            <FileText className="w-10 h-10 mx-auto mb-3 opacity-30" style={{ color: '#484f58' }} />
            <p className="text-sm" style={{ color: '#484f58' }}>
              {reports.length === 0
                ? 'No reports yet. Click "New Report" to generate one.'
                : `No ${filterFormat.toUpperCase()} reports found.`}
            </p>
          </div>
        )}

        {/* Report list */}
        <div className="space-y-2">
          {filtered.map((report) => (
            <ReportCard key={report.id} report={report} />
          ))}
        </div>

        {/* AI Narrative panel */}
        <OllamaNarrativePanel />
      </div>

      {showModal && <GenerateModal onClose={() => setShowModal(false)} />}
    </div>
  );
}
