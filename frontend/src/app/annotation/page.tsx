"use client";

import React, { useState } from "react";
import { useQuery, useMutation, useQueryClient } from "@tanstack/react-query";
import {
  Tag,
  Clock,
  CheckCircle2,
  XCircle,
  AlertTriangle,
  Loader2,
  BarChart3,
  RefreshCw,
  Brain,
  FlaskConical,
  Tags,
  Link2,
  Link2Off,
  ExternalLink,
  ArrowDownToLine,
  FolderOpen,
  List,
  Settings,
  Upload,
  AlertCircle,
  Database,
  Wand2,
  Settings2,
  ChevronDown,
} from "lucide-react";
import { api } from "@/lib/api";
import { cn, timeAgo, formatConfidence } from "@/lib/utils";

// ---------------------------------------------------------------------------
// Shared types
// ---------------------------------------------------------------------------

interface ReviewItem {
  id: string;
  image_path?: string;
  filename?: string;
  maturity_stage?: string;
  clear_fraction?: number;
  cloudy_fraction?: number;
  amber_fraction?: number;
  vlm_confidence?: number;
  confidence?: number;
  hallucination_flags?: string[];
  review_priority?: number;
  priority?: number;
  queued_at?: string;
  created_at?: string;
  vlm_backend?: string;
  status?: string;
}

interface AnnotationStats {
  total_pending?: number;
  pending_count?: number;
  total_reviewed?: number;
  reviewed_count?: number;
  throughput_per_hour?: number;
  avg_priority?: number;
  high_priority_count?: number;
}

interface AnnotationJob {
  id?: string;
  job_uuid?: string;
  job_type?: string;
  type?: string;
  status: string;
  progress?: number;
  processed_items?: number;
  total_items?: number;
}

interface LSStatus {
  host: string;
  api_key: string;
  connected: boolean;
  last_check: number;
  project_count: number;
  env_configured?: boolean;
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

interface Dataset {
  id: number;
  name: string;
  description?: string;
  num_samples?: number;
  status?: string;
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

function normalizeItem(item: ReviewItem): ReviewItem {
  return {
    ...item,
    filename: item.filename ?? item.image_path?.split("/").pop() ?? `Item ${item.id}`,
    vlm_confidence: item.vlm_confidence ?? item.confidence ?? 0,
    review_priority: item.review_priority ?? item.priority ?? 0,
    queued_at: item.queued_at ?? item.created_at ?? new Date().toISOString(),
    hallucination_flags: item.hallucination_flags ?? [],
    clear_fraction: item.clear_fraction ?? 0,
    cloudy_fraction: item.cloudy_fraction ?? 0,
    amber_fraction: item.amber_fraction ?? 0,
  };
}

// ---------------------------------------------------------------------------
// HITL sub-components
// ---------------------------------------------------------------------------

function PriorityBadge({ priority }: { priority: number }) {
  const configs = [
    { label: "Low", bg: "rgba(107,114,128,0.2)", color: "#9ca3af" },
    { label: "Med", bg: "rgba(59,130,246,0.2)", color: "#60a5fa" },
    { label: "High", bg: "rgba(234,179,8,0.2)", color: "#eab308" },
    { label: "Crit", bg: "rgba(239,68,68,0.2)", color: "#ef4444" },
  ];
  const config = configs[priority] ?? configs[0];
  return (
    <span className="text-[10px] px-1.5 py-0.5 rounded font-medium uppercase"
      style={{ background: config.bg, color: config.color }}>
      {config.label}
    </span>
  );
}

function FractionBar({ clear, cloudy, amber }: { clear: number; cloudy: number; amber: number }) {
  return (
    <div className="flex h-1.5 rounded-full overflow-hidden gap-[1px] w-full">
      <div className="bg-blue-400" style={{ width: `${clear * 100}%` }} />
      <div className="bg-gray-300" style={{ width: `${cloudy * 100}%` }} />
      <div className="bg-amber-400" style={{ width: `${amber * 100}%` }} />
    </div>
  );
}

function ReviewRow({ item, onApprove, onReject }: {
  item: ReviewItem;
  onApprove: (id: string) => void;
  onReject: (id: string) => void;
}) {
  const norm = normalizeItem(item);
  const conf = norm.vlm_confidence ?? 0;
  const confColor = conf >= 0.7 ? "#22c55e" : conf >= 0.5 ? "#eab308" : "#ef4444";

  return (
    <div className="group flex items-center gap-3 px-4 py-3 rounded-xl transition-all"
      style={{ background: '#0d1117', border: '1px solid #21262d' }}>
      <PriorityBadge priority={norm.review_priority ?? 0} />
      <div className="flex-1 min-w-0">
        <p className="text-sm truncate" style={{ color: '#8b949e' }}>{norm.filename}</p>
        <div className="flex items-center gap-3 mt-1">
          {norm.maturity_stage && (
            <span className="text-[10px] px-1.5 py-0.5 rounded font-medium capitalize"
              style={{
                background: norm.maturity_stage === "amber" ? "rgba(245,158,11,0.2)"
                  : norm.maturity_stage === "cloudy" ? "rgba(107,114,128,0.2)"
                  : norm.maturity_stage === "clear" ? "rgba(59,130,246,0.2)"
                  : "rgba(168,85,247,0.2)",
                color: norm.maturity_stage === "amber" ? "#f59e0b"
                  : norm.maturity_stage === "cloudy" ? "#9ca3af"
                  : norm.maturity_stage === "clear" ? "#60a5fa"
                  : "#a855f7",
              }}>
              {norm.maturity_stage}
            </span>
          )}
          {norm.vlm_backend && (
            <span className="text-[10px]" style={{ color: '#484f58' }}>{norm.vlm_backend}</span>
          )}
          {(norm.hallucination_flags?.length ?? 0) > 0 && (
            <div className="flex items-center gap-0.5 text-yellow-400">
              <AlertTriangle className="w-3 h-3" />
              <span className="text-[10px]">{norm.hallucination_flags!.length} flags</span>
            </div>
          )}
        </div>
      </div>
      {(norm.clear_fraction! + norm.cloudy_fraction! + norm.amber_fraction!) > 0 && (
        <div className="w-24 space-y-0.5">
          <FractionBar clear={norm.clear_fraction!} cloudy={norm.cloudy_fraction!} amber={norm.amber_fraction!} />
          <div className="flex justify-between text-[9px] font-mono" style={{ color: '#484f58' }}>
            <span>{Math.round((norm.clear_fraction ?? 0) * 100)}</span>
            <span>{Math.round((norm.cloudy_fraction ?? 0) * 100)}</span>
            <span>{Math.round((norm.amber_fraction ?? 0) * 100)}</span>
          </div>
        </div>
      )}
      <span className="text-xs font-mono w-10 text-right" style={{ color: confColor }}>
        {formatConfidence(conf)}
      </span>
      <span className="text-[10px] w-14 text-right" style={{ color: '#484f58' }}>
        {norm.queued_at ? timeAgo(new Date(norm.queued_at!).getTime() / 1000) : "—"}
      </span>
      <div className="flex items-center gap-1 opacity-0 group-hover:opacity-100 transition-opacity">
        <button onClick={() => onApprove(item.id)} className="p-1.5 rounded" style={{ color: '#484f58' }} title="Approve">
          <CheckCircle2 className="w-3.5 h-3.5 hover:text-green-400" />
        </button>
        <button onClick={() => onReject(item.id)} className="p-1.5 rounded" style={{ color: '#484f58' }} title="Reject">
          <XCircle className="w-3.5 h-3.5 hover:text-red-400" />
        </button>
      </div>
    </div>
  );
}

// ---------------------------------------------------------------------------
// VLM Config Panel types
// ---------------------------------------------------------------------------

interface VlmProvider {
  provider_id: string;
  name: string;
  tier: string;
  available: boolean;
  has_api_key: boolean;
  models: string[];
  default_model: string;
  is_active: boolean;
}

interface VlmPromptPreset {
  name: string;
  label: string;
  description: string;
  is_default: boolean;
}

// ---------------------------------------------------------------------------
// VLM Configuration Panel
// ---------------------------------------------------------------------------

interface VlmConfig {
  providerId: string | null;
  modelId: string | null;
  promptName: string;
  customSystemPrompt: string;
  customUserPrompt: string;
  ensembleMode: boolean;
  ensembleProviders: string[];
}

function VlmConfigPanel({
  config,
  onChange,
}: {
  config: VlmConfig;
  onChange: (next: VlmConfig) => void;
}) {
  const [open, setOpen] = useState(true);

  const { data: providers = [], isLoading: providersLoading } = useQuery<VlmProvider[]>({
    queryKey: ["vlm-providers"],
    queryFn: () => api.get("/vlm/providers").then((r) => r.data),
    staleTime: 60_000,
  });

  const { data: providerModels } = useQuery<{ models: string[]; default: string }>({
    queryKey: ["vlm-provider-models", config.providerId],
    queryFn: () =>
      api.get(`/vlm/providers/${config.providerId}/models`).then((r) => r.data),
    enabled: !!config.providerId,
    staleTime: 60_000,
  });

  const { data: prompts = [] } = useQuery<VlmPromptPreset[]>({
    queryKey: ["vlm-prompts"],
    queryFn: () => api.get("/vlm/providers/prompts").then((r) => r.data),
    staleTime: 300_000,
  });

  const availableModels: string[] = providerModels?.models ?? [];

  // When provider changes, reset model
  const handleProviderChange = (pid: string) => {
    onChange({ ...config, providerId: pid || null, modelId: null });
  };

  const tierColor = (tier: string) => {
    if (tier === "free") return "#4ade80";
    if (tier === "freemium") return "#60a5fa";
    return "#f59e0b";
  };

  const tierLabel = (tier: string) => tier.toUpperCase();

  return (
    <div
      className="rounded-xl overflow-hidden"
      style={{ border: "1px solid #21262d" }}
    >
      {/* Header / toggle */}
      <button
        onClick={() => setOpen((v) => !v)}
        className="w-full flex items-center justify-between px-4 py-2.5 text-left transition-colors"
        style={{ background: "#161b22" }}
      >
        <div className="flex items-center gap-2">
          <Settings2 className="w-3.5 h-3.5 text-purple-400" />
          <span className="text-xs font-semibold text-white">VLM Configuration</span>
        </div>
        <ChevronDown
          className="w-3.5 h-3.5 transition-transform"
          style={{ color: "#484f58", transform: open ? "rotate(180deg)" : "rotate(0deg)" }}
        />
      </button>

      {open && (
        <div className="px-4 py-3 space-y-4" style={{ background: "#0d1117" }}>
          {/* Provider selector */}
          <div>
            <label className="text-xs mb-1.5 block" style={{ color: "#484f58" }}>
              Provider
            </label>
            {providersLoading ? (
              <div className="flex items-center gap-2 text-xs" style={{ color: "#484f58" }}>
                <Loader2 className="w-3.5 h-3.5 animate-spin" /> Loading providers…
              </div>
            ) : (
              <select
                value={config.providerId ?? ""}
                onChange={(e) => handleProviderChange(e.target.value)}
                className="w-full px-3 py-1.5 text-xs rounded-lg focus:outline-none"
                style={{
                  background: "#161b22",
                  border: "1px solid #21262d",
                  color: config.providerId ? "#e6edf3" : "#484f58",
                }}
              >
                <option value="">Use default (vlm_backend field)</option>
                {providers.map((p) => (
                  <option key={p.provider_id} value={p.provider_id}>
                    {p.name} [{tierLabel(p.tier)}]{!p.available ? " — needs key" : ""}
                  </option>
                ))}
              </select>
            )}

            {/* Provider status badge */}
            {config.providerId && (() => {
              const prov = providers.find((p) => p.provider_id === config.providerId);
              if (!prov) return null;
              return (
                <div className="flex items-center gap-2 mt-1.5">
                  <span
                    className="text-[10px] px-1.5 py-0.5 rounded font-medium"
                    style={{
                      background: `${tierColor(prov.tier)}20`,
                      color: tierColor(prov.tier),
                    }}
                  >
                    {tierLabel(prov.tier)}
                  </span>
                  {prov.available ? (
                    <span className="text-[10px] text-green-400">Available</span>
                  ) : (
                    <span className="text-[10px] text-red-400">Needs API key</span>
                  )}
                </div>
              );
            })()}
          </div>

          {/* Model selector */}
          {config.providerId && availableModels.length > 0 && (
            <div>
              <label className="text-xs mb-1.5 block" style={{ color: "#484f58" }}>
                Model
              </label>
              <select
                value={config.modelId ?? ""}
                onChange={(e) => onChange({ ...config, modelId: e.target.value || null })}
                className="w-full px-3 py-1.5 text-xs rounded-lg focus:outline-none"
                style={{
                  background: "#161b22",
                  border: "1px solid #21262d",
                  color: config.modelId ? "#e6edf3" : "#484f58",
                }}
              >
                <option value="">
                  {providerModels?.default
                    ? `Default: ${providerModels.default}`
                    : "Default model"}
                </option>
                {availableModels.map((m) => (
                  <option key={m} value={m}>
                    {m}
                  </option>
                ))}
              </select>
            </div>
          )}

          {/* Prompt preset */}
          <div>
            <label className="text-xs mb-1.5 block" style={{ color: "#484f58" }}>
              Prompt Preset
            </label>
            <select
              value={config.promptName}
              onChange={(e) => onChange({ ...config, promptName: e.target.value })}
              className="w-full px-3 py-1.5 text-xs rounded-lg focus:outline-none"
              style={{
                background: "#161b22",
                border: "1px solid #21262d",
                color: "#e6edf3",
              }}
            >
              {prompts.length > 0 ? (
                prompts.map((p) => (
                  <option key={p.name} value={p.name}>
                    {p.label}
                  </option>
                ))
              ) : (
                <>
                  <option value="maturity_classification">
                    Maturity Classification (default)
                  </option>
                  <option value="morphology_classification">
                    Morphology Classification
                  </option>
                  <option value="trichome_detection_count">
                    Trichome Detection Count
                  </option>
                  <option value="custom">Custom…</option>
                </>
              )}
            </select>
          </div>

          {/* Custom prompt fields */}
          {config.promptName === "custom" && (
            <div className="space-y-2">
              <div>
                <label className="text-[10px] mb-1 block" style={{ color: "#484f58" }}>
                  System Prompt
                </label>
                <textarea
                  rows={3}
                  value={config.customSystemPrompt}
                  onChange={(e) =>
                    onChange({ ...config, customSystemPrompt: e.target.value })
                  }
                  placeholder="You are an expert trichome analysis AI…"
                  className="w-full px-3 py-2 text-xs rounded-lg focus:outline-none resize-y font-mono"
                  style={{
                    background: "#161b22",
                    border: "1px solid #21262d",
                    color: "#e6edf3",
                  }}
                />
              </div>
              <div>
                <label className="text-[10px] mb-1 block" style={{ color: "#484f58" }}>
                  User Prompt Template
                </label>
                <textarea
                  rows={3}
                  value={config.customUserPrompt}
                  onChange={(e) =>
                    onChange({ ...config, customUserPrompt: e.target.value })
                  }
                  placeholder="Analyse this trichome microscopy image and…"
                  className="w-full px-3 py-2 text-xs rounded-lg focus:outline-none resize-y font-mono"
                  style={{
                    background: "#161b22",
                    border: "1px solid #21262d",
                    color: "#e6edf3",
                  }}
                />
              </div>
            </div>
          )}

          {/* Ensemble toggle */}
          <div>
            <div className="flex items-center justify-between">
              <div>
                <p className="text-xs font-medium" style={{ color: "#8b949e" }}>
                  Ensemble Mode
                </p>
                <p className="text-[10px]" style={{ color: "#484f58" }}>
                  {config.ensembleMode ? "Multi-Provider" : "Single Provider"}
                </p>
              </div>
              <button
                onClick={() =>
                  onChange({
                    ...config,
                    ensembleMode: !config.ensembleMode,
                    ensembleProviders: [],
                  })
                }
                className="relative inline-flex h-5 w-9 items-center rounded-full transition-colors"
                style={{ background: config.ensembleMode ? "#7c3aed" : "#21262d" }}
                role="switch"
                aria-checked={config.ensembleMode}
              >
                <span
                  className={cn(
                    "inline-block h-3.5 w-3.5 transform rounded-full bg-white transition-transform",
                    config.ensembleMode ? "translate-x-4" : "translate-x-0.5",
                  )}
                />
              </button>
            </div>

            {/* Ensemble warning */}
            {config.ensembleMode && (
              <div
                className="flex items-start gap-2 px-2.5 py-2 rounded-lg mt-2"
                style={{
                  background: "rgba(234,179,8,0.08)",
                  border: "1px solid rgba(234,179,8,0.2)",
                }}
              >
                <AlertTriangle className="w-3 h-3 text-yellow-400 flex-shrink-0 mt-0.5" />
                <p className="text-[10px] leading-relaxed" style={{ color: "rgba(253,224,71,0.85)" }}>
                  Ensemble mode sends each image to multiple providers. May incur API costs.
                </p>
              </div>
            )}

            {/* Multi-select checkboxes for ensemble providers */}
            {config.ensembleMode && providers.length > 0 && (
              <div className="mt-2 space-y-1.5">
                <p className="text-[10px]" style={{ color: "#484f58" }}>
                  Select providers for ensemble:
                </p>
                {providers.map((p) => (
                  <label
                    key={p.provider_id}
                    className="flex items-center gap-2 cursor-pointer"
                  >
                    <input
                      type="checkbox"
                      checked={config.ensembleProviders.includes(p.provider_id)}
                      onChange={(e) => {
                        const next = e.target.checked
                          ? [...config.ensembleProviders, p.provider_id]
                          : config.ensembleProviders.filter(
                              (id) => id !== p.provider_id,
                            );
                        onChange({ ...config, ensembleProviders: next });
                      }}
                      className="rounded"
                      style={{ accentColor: "#7c3aed" }}
                    />
                    <span className="text-xs" style={{ color: p.available ? "#8b949e" : "#484f58" }}>
                      {p.name}
                    </span>
                    <span
                      className="ml-auto text-[9px] px-1 py-0.5 rounded font-medium"
                      style={{
                        background: `${tierColor(p.tier)}15`,
                        color: tierColor(p.tier),
                      }}
                    >
                      {tierLabel(p.tier)}
                    </span>
                  </label>
                ))}
              </div>
            )}
          </div>
        </div>
      )}
    </div>
  );
}

// ---------------------------------------------------------------------------
// Auto-label panel (right sidebar of HITL tab)
// ---------------------------------------------------------------------------

function AutoLabelPanel({ datasets }: { datasets: Dataset[] }) {
  const queryClient = useQueryClient();
  const [backend, setBackend] = useState("moondream");
  const [batchSize, setBatchSize] = useState(50);
  const [datasetId, setDatasetId] = useState<number | null>(null);

  // VLM Config state
  const [vlmConfig, setVlmConfig] = useState<VlmConfig>({
    providerId: null,
    modelId: null,
    promptName: "maturity_classification",
    customSystemPrompt: "",
    customUserPrompt: "",
    ensembleMode: false,
    ensembleProviders: [],
  });

  const buildPayload = () => ({
    dataset_id: String(datasetId),
    vlm_backend: backend,
    max_samples: batchSize,
    batch_size: batchSize,
    // VLM config fields
    ...(vlmConfig.providerId ? { provider_id: vlmConfig.providerId } : {}),
    ...(vlmConfig.modelId ? { model_id: vlmConfig.modelId } : {}),
    prompt_name: vlmConfig.promptName !== "custom" ? vlmConfig.promptName : undefined,
    ...(vlmConfig.promptName === "custom" && vlmConfig.customSystemPrompt
      ? { custom_system_prompt: vlmConfig.customSystemPrompt }
      : {}),
    ...(vlmConfig.promptName === "custom" && vlmConfig.customUserPrompt
      ? { custom_user_prompt: vlmConfig.customUserPrompt }
      : {}),
    ensemble_mode: vlmConfig.ensembleMode,
    ...(vlmConfig.ensembleMode && vlmConfig.ensembleProviders.length > 0
      ? { ensemble_providers: vlmConfig.ensembleProviders }
      : {}),
  });

  const startMutation = useMutation({
    mutationFn: () =>
      api.post("/annotation/auto-label", buildPayload()).then((r) => r.data),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["annotation-queue"] });
      queryClient.invalidateQueries({ queryKey: ["annotation-jobs"] });
    },
  });

  return (
    <div className="p-4 space-y-4">
      {/* Dataset selector */}
      <div>
        <label className="text-xs mb-1.5 block" style={{ color: '#484f58' }}>Target Dataset</label>
        <select
          value={datasetId ?? ""}
          onChange={(e) => setDatasetId(e.target.value ? Number(e.target.value) : null)}
          className="w-full px-3 py-1.5 text-sm rounded-lg focus:outline-none"
          style={{ background: '#0d1117', border: '1px solid #21262d', color: datasetId ? '#e6edf3' : '#484f58' }}
        >
          <option value="">Select dataset…</option>
          {datasets.map((d) => (
            <option key={d.id} value={d.id}>{d.name}</option>
          ))}
        </select>
      </div>

      {/* VLM Configuration panel */}
      <VlmConfigPanel config={vlmConfig} onChange={setVlmConfig} />

      {/* VLM Backend (legacy fallback — visible when no remote provider is chosen) */}
      {!vlmConfig.providerId && (
        <div>
          <label className="text-xs mb-1.5 block" style={{ color: '#484f58' }}>VLM Backend (local)</label>
          <div className="flex gap-1.5">
            {[
              { id: "moondream", label: "Moondream", vram: "2.1 GB" },
              { id: "florence2", label: "Florence-2", vram: "3.5 GB" },
              { id: "qwen2vl", label: "Qwen2-VL", vram: "5.5 GB" },
            ].map((m) => (
              <button key={m.id} onClick={() => setBackend(m.id)}
                className="flex-1 px-2 py-1.5 rounded-lg text-xs font-medium transition-all border text-center"
                style={{
                  background: backend === m.id ? 'rgba(168,85,247,0.2)' : 'transparent',
                  border: backend === m.id ? '1px solid rgba(168,85,247,0.4)' : '1px solid #21262d',
                  color: backend === m.id ? '#c084fc' : '#484f58',
                }}>
                <div>{m.label}</div>
                <div className="text-[9px] opacity-70">{m.vram}</div>
              </button>
            ))}
          </div>
        </div>
      )}

      {/* Batch size */}
      <div>
        <div className="flex items-center justify-between mb-1.5">
          <label className="text-xs" style={{ color: '#484f58' }}>Batch Size</label>
          <span className="text-xs font-mono" style={{ color: '#8b949e' }}>{batchSize}</span>
        </div>
        <input type="range" min={10} max={500} step={10} value={batchSize}
          onChange={(e) => setBatchSize(Number(e.target.value))}
          className="w-full h-1.5 appearance-none rounded cursor-pointer"
          style={{ background: '#21262d' }} />
      </div>

      <button onClick={() => startMutation.mutate()} disabled={!datasetId || startMutation.isPending}
        className="w-full flex items-center justify-center gap-2 py-2 rounded-lg text-sm font-medium transition-all"
        style={{
          background: !datasetId || startMutation.isPending ? 'rgba(168,85,247,0.3)' : 'rgb(147,51,234)',
          color: !datasetId || startMutation.isPending ? 'rgba(192,132,252,0.5)' : 'white',
          cursor: !datasetId || startMutation.isPending ? 'not-allowed' : 'pointer',
        }}>
        {startMutation.isPending ? (
          <><Loader2 className="w-4 h-4 animate-spin" />Labeling…</>
        ) : (
          <><Wand2 className="w-4 h-4" />Run Auto-Label</>
        )}
      </button>

      {!datasetId && (
        <p className="text-[10px] text-center" style={{ color: '#484f58' }}>Select a dataset first</p>
      )}
      {startMutation.isError && (
        <p className="text-xs text-red-400">{(startMutation.error as Error)?.message ?? "Failed to start auto-labeling"}</p>
      )}
      {startMutation.isSuccess && (
        <p className="text-xs text-green-400">Auto-labeling job started</p>
      )}

      <div className="px-3 py-2.5 rounded-lg space-y-1.5 mt-2"
        style={{ background: '#161b22', border: '1px solid #21262d' }}>
        <h4 className="text-xs font-medium" style={{ color: '#8b949e' }}>Kappa Agreement</h4>
        <p className="text-[10px] leading-relaxed" style={{ color: '#484f58' }}>
          Cohen&apos;s κ ≥ 0.80 is target for training data quality.
        </p>
        <div className="flex items-center gap-2 mt-1">
          <div className="flex-1 h-1.5 rounded-full overflow-hidden" style={{ background: '#21262d' }}>
            <div className="h-full bg-green-500 rounded-full" style={{ width: '76%' }} />
          </div>
          <span className="text-xs font-mono text-green-400">0.76</span>
        </div>
        <p className="text-[9px]" style={{ color: '#484f58' }}>VLM vs. human (estimated)</p>
      </div>
    </div>
  );
}

// ---------------------------------------------------------------------------
// HITL Stats panel
// ---------------------------------------------------------------------------

function StatsPanel() {
  const { data, isLoading } = useQuery({
    queryKey: ["annotation-stats"],
    queryFn: () => api.get("/annotation/stats").then((r) => r.data),
    refetchInterval: 10_000,
  });

  if (isLoading) return (
    <div className="flex items-center justify-center py-16">
      <Loader2 className="w-6 h-6 text-blue-400 animate-spin" />
    </div>
  );

  if (!data) return (
    <div className="text-center py-16" style={{ color: '#484f58' }}>
      <BarChart3 className="w-8 h-8 mx-auto mb-3 opacity-30" />
      <p className="text-sm">No annotation statistics available</p>
    </div>
  );

  const stats: AnnotationStats = data;
  return (
    <div className="space-y-4">
      <div className="grid grid-cols-2 gap-3">
        {[
          { label: "Pending", value: stats.total_pending ?? stats.pending_count ?? 0, color: "#eab308" },
          { label: "Reviewed", value: stats.total_reviewed ?? stats.reviewed_count ?? 0, color: "#22c55e" },
          { label: "Throughput", value: `${(stats.throughput_per_hour ?? 0).toFixed(1)}/hr`, color: "#3b82f6" },
          { label: "High Priority", value: stats.high_priority_count ?? 0, color: "#ef4444" },
        ].map(({ label, value, color }) => (
          <div key={label} className="px-4 py-3 rounded-xl"
            style={{ background: '#0d1117', border: '1px solid #21262d' }}>
            <p className="text-xs mb-1" style={{ color: '#484f58' }}>{label}</p>
            <p className="text-2xl font-bold font-mono" style={{ color }}>{value}</p>
          </div>
        ))}
      </div>
      {data.agreement !== undefined && (
        <div className="px-4 py-3 rounded-xl" style={{ background: '#0d1117', border: '1px solid #21262d' }}>
          <p className="text-xs mb-2" style={{ color: '#484f58' }}>Inter-Annotator Agreement (Cohen&apos;s κ)</p>
          <div className="flex items-center gap-3">
            <div className="flex-1 h-2 rounded-full overflow-hidden" style={{ background: '#21262d' }}>
              <div className="h-full rounded-full"
                style={{
                  width: `${Math.min(100, data.agreement * 100)}%`,
                  background: data.agreement >= 0.8 ? '#22c55e' : data.agreement >= 0.6 ? '#eab308' : '#ef4444',
                }} />
            </div>
            <span className="text-sm font-mono font-bold text-white">{data.agreement.toFixed(2)}</span>
          </div>
        </div>
      )}
    </div>
  );
}

// ---------------------------------------------------------------------------
// Label Studio sub-components
// ---------------------------------------------------------------------------

function ConnectionPanel({ status, onConnect }: {
  status: LSStatus | undefined;
  onConnect: (host: string, apiKey: string) => Promise<void>;
}) {
  const defaultHost = status?.host ?? "http://localhost:3005";
  const envConfigured = status?.env_configured ?? false;
  const [host, setHost] = useState(defaultHost);
  const [apiKey, setApiKey] = useState("");
  const [isPending, setIsPending] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [success, setSuccess] = useState(false);

  React.useEffect(() => { if (status?.host) setHost(status.host); }, [status?.host]);

  const handleConnect = async () => {
    setIsPending(true);
    setError(null);
    setSuccess(false);
    try {
      await onConnect(host, apiKey);
      setSuccess(true);
    } catch (e: unknown) {
      const err = e as { message?: string };
      setError(err.message ?? "Connection failed");
    } finally {
      setIsPending(false);
    }
  };

  const lsPort = host.match(/:(\d+)/)?.[1] ?? "3005";
  const lsUrl = `http://${host.replace(/^https?:\/\//, "").split(":")[0]}:${lsPort}`;

  return (
    <div className="space-y-4">
      {envConfigured && (
        <div className="flex items-start gap-2 px-3 py-2.5 rounded-lg"
          style={{ background: "rgba(34,197,94,0.08)", border: "1px solid rgba(34,197,94,0.2)" }}>
          <CheckCircle2 className="w-4 h-4 text-green-400 flex-shrink-0 mt-0.5" />
          <div>
            <p className="text-xs text-green-400 font-medium">API key loaded from .env</p>
            <p className="text-[10px] mt-0.5" style={{ color: "#4ade80" }}>Leave key field empty to use it.</p>
          </div>
        </div>
      )}
      <div>
        <label className="text-xs mb-1.5 block" style={{ color: "#8b949e" }}>Label Studio URL</label>
        <input type="text" value={host} onChange={(e) => setHost(e.target.value)}
          placeholder="http://localhost:3005"
          className="w-full px-3 py-2 text-sm rounded-lg focus:outline-none"
          style={{ background: "#161b22", border: "1px solid #21262d", color: "#e6edf3" }} />
      </div>
      <div>
        <label className="text-xs mb-1.5 block" style={{ color: "#8b949e" }}>
          API Key{envConfigured ? " (pre-filled from .env)" : ""}
        </label>
        <input type="password" value={apiKey} onChange={(e) => setApiKey(e.target.value)}
          placeholder={envConfigured ? "●●●●●●●● (from .env)" : "Label Studio API token"}
          className="w-full px-3 py-2 text-sm rounded-lg focus:outline-none"
          style={{ background: "#161b22", border: "1px solid #21262d", color: "#e6edf3" }} />
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
      <button onClick={handleConnect} disabled={isPending || !host}
        className="w-full flex items-center justify-center gap-2 py-2 rounded-lg text-sm font-medium transition-all"
        style={{
          background: isPending || !host ? "rgba(37,99,235,0.3)" : "#1d4ed8",
          color: isPending || !host ? "rgba(147,197,253,0.5)" : "white",
          cursor: isPending || !host ? "not-allowed" : "pointer",
        }}>
        {isPending
          ? <><Loader2 className="w-4 h-4 animate-spin" />Connecting…</>
          : <><Link2 className="w-4 h-4" />Connect{envConfigured ? " (using .env token)" : ""}</>}
      </button>
      <div className="px-3 py-2.5 rounded-lg text-xs space-y-1.5"
        style={{ background: "#0d1117", border: "1px solid #21262d" }}>
        <p className="font-medium" style={{ color: "#8b949e" }}>Label Studio läuft auf Port 3005</p>
        <p style={{ color: "#484f58" }}>
          Öffne{" "}
          <a href={lsUrl} target="_blank" rel="noreferrer" className="text-blue-400 underline">{lsUrl}</a>{" "}
          im Browser zum Annotieren
        </p>
        <p style={{ color: "#484f58" }}>
          Login: <code className="text-green-400">admin@ctip.local</code> /{" "}
          <code className="text-green-400">ctip_admin_2025</code>
        </p>
      </div>
    </div>
  );
}

function TasksDrawer({ projectId, onClose }: { projectId: number; onClose: () => void }) {
  const [page, setPage] = useState(1);
  const PAGE_SIZE = 25;
  const { data, isLoading } = useQuery({
    queryKey: ["ls-tasks", projectId, page],
    queryFn: () => api.get(`/labelstudio/tasks/${projectId}?page=${page}&page_size=${PAGE_SIZE}`).then((r) => r.data),
  });
  const tasks: LSTask[] = data?.tasks ?? [];
  const total: number = data?.total ?? 0;
  const totalPages = Math.ceil(total / PAGE_SIZE);

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center"
      style={{ background: "rgba(0,0,0,0.75)" }}
      onClick={onClose}>
      <div className="w-full max-w-xl max-h-[75vh] rounded-2xl flex flex-col overflow-hidden"
        style={{ background: "#161b22", border: "1px solid #30363d" }}
        onClick={(e) => e.stopPropagation()}>
        <div className="flex items-center justify-between px-5 py-3.5"
          style={{ borderBottom: "1px solid #21262d" }}>
          <div className="flex items-center gap-2">
            <List className="w-4 h-4 text-blue-400" />
            <h2 className="text-sm font-semibold text-white">Project #{projectId} — {total} tasks</h2>
          </div>
          <button onClick={onClose} className="text-sm px-3 py-1 rounded-lg"
            style={{ background: "#21262d", color: "#8b949e" }}>Close</button>
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
              <div key={task.id} className="flex items-center gap-3 px-3 py-2 rounded-lg"
                style={{ background: "#0d1117", border: "1px solid #21262d" }}>
                <span className="text-xs font-mono w-12 text-right" style={{ color: "#484f58" }}>#{task.id}</span>
                <span className="flex-1 text-xs truncate" style={{ color: "#8b949e" }}>
                  {task.data?.image?.split("/").pop() ?? "—"}
                </span>
                {task.is_labeled
                  ? <CheckCircle2 className="w-3.5 h-3.5 text-green-400 flex-shrink-0" />
                  : <div className="w-3.5 h-3.5 rounded-full border flex-shrink-0" style={{ borderColor: "#21262d" }} />}
                <span className="text-[10px] w-14 text-right" style={{ color: "#484f58" }}>
                  {task.total_annotations ?? 0} ann.
                </span>
              </div>
            ))
          )}
        </div>
        {totalPages > 1 && (
          <div className="flex items-center justify-between px-5 py-3" style={{ borderTop: "1px solid #21262d" }}>
            <button disabled={page <= 1} onClick={() => setPage((p) => p - 1)}
              className="text-xs px-3 py-1 rounded disabled:opacity-40"
              style={{ background: "#21262d", color: "#8b949e" }}>Prev</button>
            <span className="text-xs" style={{ color: "#484f58" }}>{page} / {totalPages}</span>
            <button disabled={page >= totalPages} onClick={() => setPage((p) => p + 1)}
              className="text-xs px-3 py-1 rounded disabled:opacity-40"
              style={{ background: "#21262d", color: "#8b949e" }}>Next</button>
          </div>
        )}
      </div>
    </div>
  );
}

function ProjectCard({ project, datasets, onImport, onViewTasks, importing, importMsg }: {
  project: LSProject;
  datasets: Dataset[];
  onImport: (id: number, datasetId: number | null) => void;
  onViewTasks: (id: number) => void;
  importing: boolean;
  importMsg?: string;
}) {
  const [selectedDataset, setSelectedDataset] = useState<number | null>(null);
  const taskCount = project.task_count ?? project.task_number ?? 0;
  const annCount = project.annotation_count ?? project.num_tasks_with_annotations ?? 0;
  const pct = taskCount > 0 ? Math.round((annCount / taskCount) * 100) : 0;

  return (
    <div className="rounded-xl p-4 space-y-3" style={{ background: "#0d1117", border: "1px solid #21262d" }}>
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

      {/* Dataset selector for import target */}
      <div>
        <label className="text-[10px] mb-1 block flex items-center gap-1" style={{ color: "#484f58" }}>
          <Database className="w-3 h-3" />Import into dataset
        </label>
        <select
          value={selectedDataset ?? ""}
          onChange={(e) => setSelectedDataset(e.target.value ? Number(e.target.value) : null)}
          className="w-full px-2.5 py-1.5 text-xs rounded-lg focus:outline-none"
          style={{ background: "#161b22", border: "1px solid #21262d", color: selectedDataset ? "#e6edf3" : "#484f58" }}>
          <option value="">Review queue (no dataset)</option>
          {datasets.map((d) => (
            <option key={d.id} value={d.id}>{d.name}</option>
          ))}
        </select>
      </div>

      {importMsg && (
        <p className="text-xs" style={{ color: importMsg.startsWith("✓") ? "#4ade80" : "#f87171" }}>
          {importMsg}
        </p>
      )}

      <div className="flex gap-2">
        <button onClick={() => onViewTasks(project.id)}
          className="flex-1 flex items-center justify-center gap-1.5 py-1.5 rounded-lg text-xs font-medium"
          style={{ background: "transparent", border: "1px solid #21262d", color: "#8b949e" }}>
          <List className="w-3.5 h-3.5" />Tasks
        </button>
        <button onClick={() => onImport(project.id, selectedDataset)}
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

export default function AnnotationPage() {
  const queryClient = useQueryClient();
  const [mainTab, setMainTab] = useState<"hitl" | "labelstudio">("hitl");
  const [hitlTab, setHitlTab] = useState<"queue" | "jobs" | "stats">("queue");
  const [lsPanel, setLsPanel] = useState<"projects" | "settings">("projects");
  const [tasksProjectId, setTasksProjectId] = useState<number | null>(null);
  const [importingId, setImportingId] = useState<number | null>(null);
  const [importResults, setImportResults] = useState<Record<number, string>>({});

  // Shared: datasets
  const { data: datasetsData = [] } = useQuery<Dataset[]>({
    queryKey: ["datasets"],
    queryFn: () => api.get("/datasets").then((r) => r.data),
    staleTime: 60_000,
  });

  // HITL data
  const { data: queueData, isLoading: queueLoading, refetch: refetchQueue } = useQuery({
    queryKey: ["annotation-queue"],
    queryFn: () => api.get("/annotation/queue").then((r) => r.data),
    refetchInterval: 10_000,
    staleTime: 5_000,
  });

  const { data: jobsData = [] } = useQuery({
    queryKey: ["annotation-jobs"],
    queryFn: () => api.get("/annotation/jobs").then((r) => r.data),
    refetchInterval: 5_000,
    enabled: hitlTab === "jobs",
  });

  const approveMutation = useMutation({
    mutationFn: (itemId: string) =>
      api.put(`/annotation/queue/${itemId}`, { status: "approved" }).then((r) => r.data),
    onSuccess: () => queryClient.invalidateQueries({ queryKey: ["annotation-queue"] }),
  });

  const rejectMutation = useMutation({
    mutationFn: (itemId: string) =>
      api.put(`/annotation/queue/${itemId}`, { status: "rejected" }).then((r) => r.data),
    onSuccess: () => queryClient.invalidateQueries({ queryKey: ["annotation-queue"] }),
  });

  const rawItems: ReviewItem[] = Array.isArray(queueData)
    ? queueData : queueData?.items ?? queueData?.queue ?? [];
  const stats: AnnotationStats = queueData?.stats ?? {};
  const pendingCount = stats.total_pending ?? stats.pending_count ?? rawItems.length;
  const sortedItems = [...rawItems].sort(
    (a, b) => ((b.review_priority ?? b.priority ?? 0) - (a.review_priority ?? a.priority ?? 0))
  );
  const jobs: AnnotationJob[] = Array.isArray(jobsData) ? jobsData : jobsData?.jobs ?? [];

  // Label Studio data
  const { data: lsStatus, refetch: refetchLsStatus } = useQuery<LSStatus>({
    queryKey: ["ls-status"],
    queryFn: () => api.get("/labelstudio/status").then((r) => r.data),
    refetchInterval: 30_000,
  });

  const { data: projectsData, isLoading: projectsLoading, refetch: refetchProjects } = useQuery({
    queryKey: ["ls-projects"],
    queryFn: () => api.get("/labelstudio/projects").then((r) => r.data),
    enabled: lsStatus?.connected === true,
    staleTime: 30_000,
  });

  const connectMutation = useMutation({
    mutationFn: ({ host, apiKey }: { host: string; apiKey: string }) =>
      api.post("/labelstudio/connect", { host, api_key: apiKey }).then((r) => r.data),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["ls-status"] });
      queryClient.invalidateQueries({ queryKey: ["ls-projects"] });
      setLsPanel("projects");
    },
  });

  const handleImport = async (projectId: number, datasetId: number | null) => {
    setImportingId(projectId);
    setImportResults((p) => ({ ...p, [projectId]: "" }));
    try {
      const payload: Record<string, unknown> = {};
      if (datasetId) payload.dataset_id = datasetId;
      const r = await api.post(`/labelstudio/import/${projectId}`, payload);
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

  const lsConnected = lsStatus?.connected ?? false;
  const projects: LSProject[] = projectsData?.projects ?? [];
  const datasets: Dataset[] = Array.isArray(datasetsData) ? datasetsData : [];

  return (
    <div className="flex h-full flex-col">
      {/* Top header with main tabs */}
      <div className="flex items-center justify-between px-5 py-3 flex-shrink-0"
        style={{ borderBottom: '1px solid #21262d' }}>
        <div className="flex items-center gap-4">
          <div className="flex items-center gap-2">
            <Tag className="w-4 h-4 text-purple-400" />
            <h1 className="text-base font-semibold text-white">Annotation</h1>
          </div>

          {/* Main tab switcher */}
          <div className="flex gap-0.5 p-0.5 rounded-lg" style={{ background: '#161b22', border: '1px solid #21262d' }}>
            <button
              onClick={() => setMainTab("hitl")}
              className="flex items-center gap-1.5 px-3 py-1 rounded text-xs font-medium transition-all"
              style={{
                background: mainTab === "hitl" ? '#0d1117' : 'transparent',
                color: mainTab === "hitl" ? '#e6edf3' : '#484f58',
              }}>
              <Brain className="w-3 h-3" />
              Review {pendingCount > 0 && <span className="ml-1 px-1 py-0.5 rounded text-[9px] font-bold"
                style={{ background: 'rgba(234,179,8,0.2)', color: '#eab308' }}>{pendingCount}</span>}
            </button>
            <button
              onClick={() => setMainTab("labelstudio")}
              className="flex items-center gap-1.5 px-3 py-1 rounded text-xs font-medium transition-all"
              style={{
                background: mainTab === "labelstudio" ? '#0d1117' : 'transparent',
                color: mainTab === "labelstudio" ? '#e6edf3' : '#484f58',
              }}>
              <Tags className="w-3 h-3" />
              Label Studio
              {lsConnected
                ? <span className="w-1.5 h-1.5 rounded-full bg-green-400 ml-1" />
                : <span className="w-1.5 h-1.5 rounded-full bg-red-500/60 ml-1" />}
            </button>
          </div>
        </div>

        {/* Tab-specific controls */}
        {mainTab === "hitl" && (
          <div className="flex items-center gap-3">
            <div className="flex gap-1 p-0.5 rounded-lg"
              style={{ background: '#161b22', border: '1px solid #21262d' }}>
              {(["queue", "jobs", "stats"] as const).map((t) => (
                <button key={t} onClick={() => setHitlTab(t)}
                  className="px-3 py-1 rounded text-xs font-medium transition-all capitalize"
                  style={{ background: hitlTab === t ? '#0d1117' : 'transparent', color: hitlTab === t ? '#e6edf3' : '#484f58' }}>
                  {t}{t === "queue" && pendingCount > 0 ? ` (${pendingCount})` : ""}
                </button>
              ))}
            </div>
            <button onClick={() => refetchQueue()} className="p-1.5 rounded" style={{ color: '#484f58' }}>
              <RefreshCw className="w-4 h-4" />
            </button>
          </div>
        )}

        {mainTab === "labelstudio" && (
          <div className="flex items-center gap-2">
            {lsConnected && (
              <>
                <a href={lsStatus?.host} target="_blank" rel="noreferrer"
                  className="flex items-center gap-1.5 text-xs px-3 py-1.5 rounded-lg"
                  style={{ border: "1px solid #21262d", color: "#8b949e" }}>
                  <ExternalLink className="w-3.5 h-3.5" />Open LS
                </a>
                <button onClick={() => refetchProjects()} className="p-2 rounded-lg" style={{ color: "#484f58" }}>
                  <RefreshCw className="w-4 h-4" />
                </button>
              </>
            )}
            <button
              onClick={() => setLsPanel((p) => p === "settings" ? "projects" : "settings")}
              className="p-2 rounded-lg transition-colors"
              style={{ background: lsPanel === "settings" ? "#21262d" : "transparent", color: lsPanel === "settings" ? "#e6edf3" : "#484f58" }}>
              <Settings className="w-4 h-4" />
            </button>
            <button onClick={() => { refetchLsStatus(); refetchProjects(); }} className="p-2 rounded-lg" style={{ color: "#484f58" }}>
              <RefreshCw className="w-4 h-4" />
            </button>
          </div>
        )}
      </div>

      {/* Stats bar (HITL only) */}
      {mainTab === "hitl" && Object.keys(stats).length > 0 && (
        <div className="flex items-center gap-6 px-5 py-2.5 flex-shrink-0"
          style={{ borderBottom: '1px solid #21262d', background: '#161b22' }}>
          {[
            { label: "Pending", value: stats.total_pending ?? stats.pending_count ?? 0, color: "#eab308" },
            { label: "Reviewed", value: stats.total_reviewed ?? stats.reviewed_count ?? 0, color: "#22c55e" },
            { label: "Throughput", value: `${(stats.throughput_per_hour ?? 0).toFixed(1)}/hr`, color: "#3b82f6" },
            { label: "High Priority", value: stats.high_priority_count ?? 0, color: "#ef4444" },
          ].map(({ label, value, color }) => (
            <div key={label} className="flex items-center gap-1.5">
              <span className="text-xs" style={{ color: '#484f58' }}>{label}:</span>
              <span className="text-xs font-bold font-mono" style={{ color }}>{value}</span>
            </div>
          ))}
        </div>
      )}

      {/* LS stats bar */}
      {mainTab === "labelstudio" && lsConnected && (
        <div className="flex items-center gap-6 px-5 py-2 flex-shrink-0"
          style={{ borderBottom: "1px solid #21262d", background: "#161b22" }}>
          <div className="flex items-center gap-1.5">
            <span className="text-xs" style={{ color: "#484f58" }}>Projects:</span>
            <span className="text-xs font-bold text-white">{lsStatus?.project_count ?? projects.length}</span>
          </div>
          <div className="flex items-center gap-1.5">
            <span className="w-1.5 h-1.5 rounded-full bg-green-400 animate-pulse" />
            <span className="text-xs" style={{ color: "#4ade80" }}>Connected — {lsStatus?.host}</span>
          </div>
        </div>
      )}

      {/* Body: two-panel layout */}
      <div className="flex flex-1 min-h-0">
        {/* ── HITL tab ── */}
        {mainTab === "hitl" && (
          <>
            <div className="flex-1 overflow-y-auto px-5 py-4">
              {hitlTab === "queue" && (
                <div className="space-y-2">
                  <div className="flex items-start gap-2 px-3 py-2.5 rounded-lg mb-4"
                    style={{ background: 'rgba(59,130,246,0.1)', border: '1px solid rgba(59,130,246,0.2)' }}>
                    <FlaskConical className="w-4 h-4 text-blue-400 flex-shrink-0 mt-0.5" />
                    <p className="text-xs" style={{ color: 'rgba(191,219,254,0.8)' }}>
                      <strong>Human-in-loop enforced:</strong> VLM auto-labels require human approval before entering the training dataset.
                    </p>
                  </div>
                  {queueLoading ? (
                    <div className="flex items-center justify-center py-16">
                      <Loader2 className="w-6 h-6 animate-spin text-blue-400" />
                    </div>
                  ) : sortedItems.length === 0 ? (
                    <div className="text-center py-16" style={{ color: '#484f58' }}>
                      <Tag className="w-10 h-10 mx-auto mb-3 opacity-30" />
                      <p className="text-sm font-medium">Queue is empty</p>
                      <p className="text-xs mt-1">Run VLM auto-labeling or import from Label Studio to populate</p>
                    </div>
                  ) : (
                    sortedItems.map((item) => (
                      <ReviewRow key={item.id} item={item}
                        onApprove={(id) => approveMutation.mutate(id)}
                        onReject={(id) => rejectMutation.mutate(id)} />
                    ))
                  )}
                </div>
              )}

              {hitlTab === "jobs" && (
                <div className="space-y-2">
                  {jobs.length === 0 ? (
                    <div className="text-center py-16" style={{ color: '#484f58' }}>
                      <Clock className="w-8 h-8 mx-auto mb-3 opacity-30" />
                      <p className="text-sm">No annotation jobs yet</p>
                    </div>
                  ) : (
                    jobs.map((job, i) => (
                      <div key={job.id ?? job.job_uuid ?? i}
                        className="flex items-center gap-3 px-4 py-3 rounded-xl"
                        style={{ background: '#0d1117', border: '1px solid #21262d' }}>
                        <div className={cn("w-2 h-2 rounded-full flex-shrink-0")}
                          style={{
                            background: job.status === "running" ? "#3b82f6"
                              : job.status === "completed" ? "#22c55e"
                              : job.status === "failed" ? "#ef4444" : "#6b7280",
                          }} />
                        <div className="flex-1 min-w-0">
                          <p className="text-sm" style={{ color: '#8b949e' }}>{job.job_type ?? job.type ?? "Annotation Job"}</p>
                          <p className="text-xs" style={{ color: '#484f58' }}>
                            {job.processed_items ?? 0} / {job.total_items ?? "?"} processed
                          </p>
                        </div>
                        {job.progress !== undefined && (
                          <div className="w-20 h-1.5 rounded-full overflow-hidden" style={{ background: '#21262d' }}>
                            <div className="h-full bg-blue-500 transition-all" style={{ width: `${job.progress}%` }} />
                          </div>
                        )}
                        <span className="text-xs font-mono" style={{ color: '#484f58' }}>{job.status}</span>
                      </div>
                    ))
                  )}
                </div>
              )}

              {hitlTab === "stats" && <StatsPanel />}
            </div>

            {/* Auto-label sidebar */}
            <div className="w-80 flex-shrink-0 overflow-y-auto" style={{ borderLeft: '1px solid #21262d' }}>
              <div className="px-4 py-3 flex items-center gap-2" style={{ borderBottom: '1px solid #21262d' }}>
                <Wand2 className="w-4 h-4 text-purple-400" />
                <h2 className="text-sm font-semibold text-white">Auto-Label</h2>
              </div>
              <AutoLabelPanel datasets={datasets} />
            </div>
          </>
        )}

        {/* ── Label Studio tab ── */}
        {mainTab === "labelstudio" && (
          <>
            <div className="flex-1 overflow-y-auto px-5 py-4">
              {!lsConnected ? (
                <div className="max-w-sm mx-auto mt-8 text-center space-y-4">
                  <div className="w-16 h-16 rounded-2xl flex items-center justify-center mx-auto"
                    style={{ background: "rgba(59,130,246,0.1)" }}>
                    <Tags className="w-8 h-8 text-blue-400" />
                  </div>
                  <h2 className="text-lg font-semibold text-white">Label Studio Integration</h2>
                  <p className="text-sm" style={{ color: "#484f58" }}>
                    Connect to your Label Studio instance to import completed annotations into the human review queue.
                  </p>
                  <button onClick={() => setLsPanel("settings")}
                    className="inline-flex items-center gap-2 px-4 py-2 rounded-lg text-sm font-medium"
                    style={{ background: "#1d4ed8", color: "white" }}>
                    <Settings className="w-4 h-4" />Configure Connection
                  </button>
                </div>
              ) : lsPanel === "settings" ? (
                <div className="max-w-sm">
                  <ConnectionPanel
                    status={lsStatus}
                    onConnect={async (host, apiKey) => {
                      await connectMutation.mutateAsync({ host, apiKey });
                    }}
                  />
                </div>
              ) : projectsLoading ? (
                <div className="flex items-center justify-center py-20">
                  <Loader2 className="w-8 h-8 text-blue-400 animate-spin" />
                </div>
              ) : projects.length === 0 ? (
                <div className="text-center py-16 space-y-3">
                  <FolderOpen className="w-10 h-10 mx-auto opacity-30" style={{ color: "#484f58" }} />
                  <p className="text-sm" style={{ color: "#484f58" }}>No projects in Label Studio</p>
                  <a href={lsStatus?.host} target="_blank" rel="noreferrer"
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
                      <strong>Workflow:</strong> Annotate in Label Studio → select target dataset → Import to queue → Human review → Training data
                    </p>
                  </div>
                  <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
                    {projects.map((p) => (
                      <ProjectCard key={p.id} project={p} datasets={datasets}
                        onImport={handleImport}
                        onViewTasks={(id) => setTasksProjectId(id)}
                        importing={importingId === p.id}
                        importMsg={importResults[p.id]} />
                    ))}
                  </div>
                </div>
              )}
            </div>

            {/* Connection sidebar */}
            <div className="w-72 flex-shrink-0 flex flex-col overflow-y-auto" style={{ borderLeft: "1px solid #21262d" }}>
              <div className="flex items-center gap-2 px-4 py-3" style={{ borderBottom: "1px solid #21262d" }}>
                {lsConnected
                  ? <Link2 className="w-4 h-4 text-green-400" />
                  : <Link2Off className="w-4 h-4 text-red-400" />}
                <h2 className="text-sm font-semibold text-white">Connection</h2>
              </div>
              <div className="flex-1 p-4">
                <ConnectionPanel
                  status={lsStatus}
                  onConnect={async (host, apiKey) => {
                    await connectMutation.mutateAsync({ host, apiKey });
                  }}
                />
              </div>
              {lsConnected && (
                <div className="px-4 py-3 space-y-1" style={{ borderTop: "1px solid #21262d" }}>
                  <p className="text-[10px] uppercase font-medium mb-2" style={{ color: "#484f58" }}>Quick Links</p>
                  {[
                    { href: "/datasets", label: "→ Datasets" },
                    { href: "/training", label: "→ Training" },
                  ].map(({ href, label }) => (
                    <a key={href} href={href} className="block text-xs py-1" style={{ color: "#484f58" }}>{label}</a>
                  ))}
                </div>
              )}
            </div>
          </>
        )}
      </div>

      {tasksProjectId !== null && (
        <TasksDrawer projectId={tasksProjectId} onClose={() => setTasksProjectId(null)} />
      )}
    </div>
  );
}
