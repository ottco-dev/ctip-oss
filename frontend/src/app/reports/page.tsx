"use client";

import React, { useState } from "react";
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
      </div>

      {showModal && <GenerateModal onClose={() => setShowModal(false)} />}
    </div>
  );
}
