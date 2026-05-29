"use client";

/**
 * Inference — tabbed page combining:
 *   - Workbench: single-image trichome detection with live overlays
 *   - Pipeline Builder: visual node-based pipeline editor (React Flow)
 *
 * Tab state is persisted via ?tab=workbench|pipeline query param.
 */

import React, { useCallback, useState, Suspense } from "react";
import { useMutation, useQuery } from "@tanstack/react-query";
import { Upload, Loader2, Cpu, AlertTriangle, Zap, Workflow } from "lucide-react";
import { useDropzone } from "react-dropzone";
import { useSearchParams, useRouter } from "next/navigation";
import { api } from "@/lib/api";
import { ImageViewer } from "@/components/shared/ImageViewer";
import type { AnnotationBox } from "@/components/shared/ImageViewer";
import { cn, formatConfidence, getConfidenceColor } from "@/lib/utils";
import { ModelTestBuilder } from "@/components/inference/ModelTestBuilder";
import "@xyflow/react/dist/style.css";

// ---------------------------------------------------------------------------
// API types
// ---------------------------------------------------------------------------

interface DetectionBox {
  id?: string;
  /** [x1, y1, x2, y2] — absolute pixel coords */
  bbox?: [number, number, number, number];
  x1?: number;
  y1?: number;
  x2?: number;
  y2?: number;
  confidence: number;
  calibrated_confidence?: number | null;
  class_id?: number;
  class_name?: string;
  trichome_type?: string;
}

interface DetectionResponse {
  image_id?: string;
  detections: DetectionBox[];
  num_detections?: number;
  mean_confidence?: number;
  inference_time_ms?: number;
  processing_time_ms?: number;
  model_id?: string;
  trichome_counts?: Record<string, number>;
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

function getBoxCoords(det: DetectionBox): { x1: number; y1: number; x2: number; y2: number } {
  if (det.bbox && det.bbox.length === 4) {
    return { x1: det.bbox[0], y1: det.bbox[1], x2: det.bbox[2], y2: det.bbox[3] };
  }
  return { x1: det.x1 ?? 0, y1: det.y1 ?? 0, x2: det.x2 ?? 0, y2: det.y2 ?? 0 };
}

function getLabel(det: DetectionBox): string {
  return (
    det.class_name ??
    det.trichome_type ??
    (det.class_id !== undefined ? `cls${det.class_id}` : "unknown")
  );
}

/** Convert API detections to the AnnotationBox format expected by ImageViewer */
function toAnnotationBoxes(detections: DetectionBox[]): AnnotationBox[] {
  return detections.map((det, i) => {
    const { x1, y1, x2, y2 } = getBoxCoords(det);
    return {
      id: det.id ?? String(i),
      x1,
      y1,
      x2,
      y2,
      label: getLabel(det),
      // Prefer calibrated confidence when the backend provides it
      confidence: det.calibrated_confidence ?? det.confidence,
    };
  });
}

// ---------------------------------------------------------------------------
// Results panel
// ---------------------------------------------------------------------------

function ResultsPanel({ result }: { result: DetectionResponse }) {
  const [showRaw, setShowRaw] = useState(false);
  const inferenceMs = result.inference_time_ms ?? result.processing_time_ms;

  // Build class counts: prefer backend trichome_counts if available
  const classCounts: Record<string, number> = {};
  if (result.trichome_counts && Object.keys(result.trichome_counts).length > 0) {
    Object.assign(classCounts, result.trichome_counts);
  } else {
    result.detections.forEach((d) => {
      const name = getLabel(d);
      classCounts[name] = (classCounts[name] ?? 0) + 1;
    });
  }

  const avgConf =
    result.detections.length > 0
      ? result.detections.reduce((s, d) => s + d.confidence, 0) / result.detections.length
      : 0;

  return (
    <div className="flex flex-col gap-4">
      {/* KPI cards */}
      <div className="grid grid-cols-3 gap-3">
        {[
          { label: "Detections", value: result.detections.length, textClass: "text-blue-400" },
          { label: "Avg Conf", value: formatConfidence(avgConf), textClass: getConfidenceColor(avgConf) },
          {
            label: "Inference",
            value: inferenceMs ? `${inferenceMs.toFixed(0)}ms` : "—",
            textClass: null,
          },
        ].map(({ label, value, textClass }) => (
          <div
            key={label}
            className="px-3 py-2.5 rounded-lg"
            style={{ background: "#0d1117", border: "1px solid #21262d" }}
          >
            <p className="text-[10px] uppercase tracking-wide" style={{ color: "#484f58" }}>
              {label}
            </p>
            <p
              className={cn("text-xl font-bold font-mono mt-0.5", textClass)}
              style={!textClass ? { color: "#8b949e" } : undefined}
            >
              {value}
            </p>
          </div>
        ))}
      </div>

      {/* Per-class breakdown */}
      {Object.keys(classCounts).length > 0 && (
        <div className="space-y-2">
          <h3
            className="text-xs font-medium uppercase tracking-wide"
            style={{ color: "#484f58" }}
          >
            By Class
          </h3>
          <div className="space-y-1.5">
            {Object.entries(classCounts).map(([name, count]) => (
              <div key={name} className="flex items-center gap-2">
                <span className="text-sm flex-1 capitalize" style={{ color: "#8b949e" }}>
                  {name.replace(/_/g, " ")}
                </span>
                <div className="flex items-center gap-2">
                  <div
                    className="w-20 h-1.5 rounded-full overflow-hidden"
                    style={{ background: "#21262d" }}
                  >
                    <div
                      className="h-full bg-blue-500"
                      style={{ width: `${(count / result.detections.length) * 100}%` }}
                    />
                  </div>
                  <span className="text-xs font-mono w-6 text-right" style={{ color: "#484f58" }}>
                    {count}
                  </span>
                </div>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Scientific caveat */}
      <div
        className="flex items-start gap-2 px-3 py-2.5 rounded-lg"
        style={{
          background: "rgba(234,179,8,0.1)",
          border: "1px solid rgba(234,179,8,0.2)",
        }}
      >
        <AlertTriangle className="w-3.5 h-3.5 text-yellow-400 flex-shrink-0 mt-0.5" />
        <p className="text-[11px] leading-relaxed" style={{ color: "rgba(254,243,199,0.8)" }}>
          Maturity stage reflects optical properties only. No inference about THC, CBD, or other
          cannabinoid concentrations can be made from visual appearance.
        </p>
      </div>

      {/* Raw JSON toggle */}
      <button
        onClick={() => setShowRaw((v) => !v)}
        className="text-xs text-left transition-colors"
        style={{ color: "#484f58" }}
      >
        {showRaw ? "▾" : "▸"} Raw JSON response
      </button>
      {showRaw && (
        <pre
          className="text-[10px] text-green-300/80 rounded-lg p-3 overflow-x-auto max-h-60 font-mono"
          style={{ background: "#0d1117", border: "1px solid #21262d" }}
        >
          {JSON.stringify(result, null, 2)}
        </pre>
      )}
    </div>
  );
}

// ---------------------------------------------------------------------------
// Dropzone placeholder (shown when no image is loaded)
// ---------------------------------------------------------------------------

function DropzonePlaceholder({
  getRootProps,
  getInputProps,
  isDragActive,
}: {
  getRootProps: () => Record<string, unknown>;
  getInputProps: () => Record<string, unknown>;
  isDragActive: boolean;
}) {
  return (
    <div
      {...getRootProps()}
      className={cn(
        "flex flex-col items-center justify-center gap-4 w-full max-w-lg h-72 rounded-2xl",
        "border-2 border-dashed cursor-pointer transition-all",
      )}
      style={{
        borderColor: isDragActive ? "#3b82f6" : "#21262d",
        background: isDragActive ? "rgba(59,130,246,0.1)" : "transparent",
      }}
    >
      <input {...getInputProps()} />
      <Upload className="w-10 h-10" style={{ color: "#484f58" }} />
      <div className="text-center">
        <p className="text-sm font-medium" style={{ color: "#8b949e" }}>
          {isDragActive ? "Drop image here" : "Drop a microscopy image"}
        </p>
        <p className="text-xs mt-1" style={{ color: "#484f58" }}>
          JPG · PNG · TIFF
        </p>
      </div>
    </div>
  );
}

// ---------------------------------------------------------------------------
// Workbench tab content
// ---------------------------------------------------------------------------

interface RegisteredModel {
  id: number;
  name: string;
  variant: string;
  file_path: string | null;
  metrics: Record<string, number>;
  is_active: boolean;
}

function WorkbenchTab() {
  const [imageFile, setImageFile] = useState<File | null>(null);
  const [imageUrl, setImageUrl] = useState<string | null>(null);
  const [confThreshold, setConfThreshold] = useState(0.35);
  const [modelVariant, setModelVariant] = useState("yolo11s");
  const [selectedModelId, setSelectedModelId] = useState<number | null>(null);

  const { data: registeredModels = [] } = useQuery<RegisteredModel[]>({
    queryKey: ["registered-models"],
    queryFn: () => api.get("/models").then((r) => r.data),
    staleTime: 30_000,
  });

  const detectMutation = useMutation<DetectionResponse, Error, File>({
    mutationFn: async (file: File) => {
      const formData = new FormData();
      formData.append("file", file);
      formData.append("conf_threshold", String(confThreshold));
      if (selectedModelId) {
        formData.append("model_id", String(selectedModelId));
      } else {
        formData.append("model_variant", modelVariant);
      }
      const response = await api.post("/inference/detect", formData, {
        headers: { "Content-Type": "multipart/form-data" },
      });
      return response.data;
    },
  });

  const onDrop = useCallback(
    (accepted: File[]) => {
      const file = accepted[0];
      if (!file) return;
      // Revoke previous object URL to avoid memory leak
      if (imageUrl) URL.revokeObjectURL(imageUrl);
      setImageFile(file);
      setImageUrl(URL.createObjectURL(file));
      detectMutation.reset();
    },
    [imageUrl], // eslint-disable-line react-hooks/exhaustive-deps
  );

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept: {
      "image/jpeg": [".jpg", ".jpeg"],
      "image/png": [".png"],
      "image/tiff": [".tif", ".tiff"],
    },
    maxFiles: 1,
  });

  const annotations: AnnotationBox[] = detectMutation.data
    ? toAnnotationBoxes(detectMutation.data.detections)
    : [];

  return (
    <div className="flex h-full gap-0">
      {/* ── Left: image workbench ── */}
      <div className="flex-1 flex flex-col min-w-0">
        {/* Toolbar */}
        <div
          className="flex items-center gap-3 px-5 py-3 flex-shrink-0"
          style={{ borderBottom: "1px solid #21262d" }}
        >
          <Cpu className="w-4 h-4 text-blue-400 flex-shrink-0" />
          <h1 className="text-base font-semibold text-white">Inference Workbench</h1>

          <div className="flex-1" />

          {/* Model selector — registered models first, fallback to hardcoded variants */}
          <select
            value={selectedModelId !== null ? `id:${selectedModelId}` : modelVariant}
            onChange={(e) => {
              const val = e.target.value;
              if (val.startsWith("id:")) {
                setSelectedModelId(Number(val.slice(3)));
                setModelVariant("");
              } else {
                setSelectedModelId(null);
                setModelVariant(val);
              }
            }}
            className="px-2.5 py-1.5 text-xs rounded-lg focus:outline-none focus:border-blue-500/60 max-w-xs"
            style={{ background: "#0d1117", border: "1px solid #21262d", color: "#8b949e" }}
          >
            {registeredModels.length > 0 && (
              <optgroup label="Trained models">
                {registeredModels.map((m) => (
                  <option key={m.id} value={`id:${m.id}`}>
                    {m.name} {m.metrics.map50 ? `(mAP50 ${(m.metrics.map50 * 100).toFixed(1)}%)` : ""}
                  </option>
                ))}
              </optgroup>
            )}
            <optgroup label="Base variants">
              {["yolo11n", "yolo11s", "yolo11m", "yolo11l"].map((v) => (
                <option key={v} value={v}>{v}</option>
              ))}
            </optgroup>
          </select>

          {/* Confidence threshold */}
          <div className="flex items-center gap-1.5 text-xs" style={{ color: "#484f58" }}>
            <span>Conf:</span>
            <input
              type="range"
              min={0.1}
              max={0.9}
              step={0.05}
              value={confThreshold}
              onChange={(e) => setConfThreshold(Number(e.target.value))}
              className="w-20 h-1.5 appearance-none rounded cursor-pointer"
              style={{ background: "#21262d" }}
            />
            <span className="font-mono w-8">{confThreshold.toFixed(2)}</span>
          </div>
        </div>

        {/* Image area — ImageViewer handles zoom / pan / overlay toggle */}
        <div className="flex-1 min-h-0 flex items-center justify-center" style={{ background: "#080b10" }}>
          {!imageUrl ? (
            <DropzonePlaceholder
              getRootProps={getRootProps}
              getInputProps={getInputProps}
              isDragActive={isDragActive}
            />
          ) : (
            <ImageViewer
              src={imageUrl}
              alt="Analysis target"
              annotations={annotations}
              coordsType="pixel"
              showLabels
              showConfidence
              className="w-full h-full"
            />
          )}
        </div>

        {/* Bottom action bar (visible after image is loaded) */}
        {imageFile && (
          <div
            className="flex items-center gap-3 px-5 py-3 flex-shrink-0"
            style={{ borderTop: "1px solid #21262d" }}
          >
            <button
              onClick={() => detectMutation.mutate(imageFile)}
              disabled={detectMutation.isPending}
              className={cn(
                "flex items-center gap-2 px-5 py-2 rounded-lg text-sm font-medium transition-all",
                detectMutation.isPending
                  ? "cursor-not-allowed"
                  : "bg-blue-600 hover:bg-blue-500 text-white",
              )}
              style={
                detectMutation.isPending
                  ? { background: "rgba(37,99,235,0.5)", color: "rgba(255,255,255,0.6)" }
                  : {}
              }
            >
              {detectMutation.isPending ? (
                <>
                  <Loader2 className="w-4 h-4 animate-spin" />
                  Analyzing…
                </>
              ) : (
                <>
                  <Cpu className="w-4 h-4" />
                  Run Detection
                </>
              )}
            </button>

            <button
              onClick={() => {
                if (imageUrl) URL.revokeObjectURL(imageUrl);
                setImageFile(null);
                setImageUrl(null);
                detectMutation.reset();
              }}
              className="text-sm transition-colors"
              style={{ color: "#484f58" }}
            >
              Clear
            </button>

            {/* Also allow re-drop without clearing */}
            <label
              {...getRootProps()}
              className="text-sm cursor-pointer transition-colors"
              style={{ color: "#484f58" }}
            >
              <input {...getInputProps()} />
              Change image
            </label>

            {detectMutation.data && (
              <span className="text-xs text-green-400 ml-auto">
                ✓ {detectMutation.data.detections.length} detection
                {detectMutation.data.detections.length !== 1 ? "s" : ""}
              </span>
            )}
          </div>
        )}
      </div>

      {/* ── Right panel: results ── */}
      <div
        className="w-80 flex-shrink-0 overflow-y-auto"
        style={{ borderLeft: "1px solid #21262d" }}
      >
        <div className="px-4 py-3" style={{ borderBottom: "1px solid #21262d" }}>
          <h2 className="text-sm font-semibold text-white">Results</h2>
        </div>

        <div className="p-4">
          {detectMutation.isPending && (
            <div className="flex items-center justify-center py-12">
              <Loader2 className="w-6 h-6 text-blue-400 animate-spin" />
            </div>
          )}

          {detectMutation.isError && (
            <div
              className="flex items-start gap-2 px-3 py-2.5 rounded-lg"
              style={{
                background: "rgba(239,68,68,0.1)",
                border: "1px solid rgba(239,68,68,0.2)",
              }}
            >
              <AlertTriangle className="w-4 h-4 text-red-400 mt-0.5" />
              <div>
                <p className="text-sm font-medium text-red-400">Detection failed</p>
                <p className="text-xs mt-0.5" style={{ color: "rgba(252,165,165,0.7)" }}>
                  {detectMutation.error?.message ?? "Unknown error"}
                </p>
              </div>
            </div>
          )}

          {detectMutation.data && <ResultsPanel result={detectMutation.data} />}

          {!detectMutation.isPending && !detectMutation.data && !detectMutation.isError && (
            <div className="text-center py-12" style={{ color: "#484f58" }}>
              <Cpu className="w-8 h-8 mx-auto mb-3 opacity-30" />
              <p className="text-sm">Upload an image and run detection</p>
              <p className="text-xs mt-1 opacity-60">
                Use the zoom / pan controls in the image area after loading
              </p>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}

// ---------------------------------------------------------------------------
// Tab definitions
// ---------------------------------------------------------------------------

const TABS = [
  { id: "workbench", label: "Workbench", icon: Zap },
  { id: "pipeline", label: "Pipeline Builder", icon: Workflow },
] as const;

type TabId = (typeof TABS)[number]["id"];

// ---------------------------------------------------------------------------
// Inner component that reads search params (must be inside Suspense)
// ---------------------------------------------------------------------------

function InferencePageInner() {
  const searchParams = useSearchParams();
  const router = useRouter();

  const rawTab = searchParams.get("tab");
  const activeTab: TabId =
    rawTab === "pipeline" ? "pipeline" : "workbench";

  function switchTab(id: TabId) {
    const params = new URLSearchParams(searchParams.toString());
    params.set("tab", id);
    // Strip pipeline-specific params when switching away
    if (id !== "pipeline") params.delete("test");
    router.replace(`/inference?${params.toString()}`);
  }

  return (
    <div className="flex flex-col h-full overflow-hidden">
      {/* ── Tab bar ── */}
      <div
        className="flex items-center gap-1 px-4 py-0 flex-shrink-0"
        style={{ borderBottom: "1px solid #21262d", background: "#0d1117" }}
      >
        {TABS.map(({ id, label, icon: Icon }) => (
          <button
            key={id}
            onClick={() => switchTab(id)}
            className={cn(
              "flex items-center gap-2 px-4 py-2.5 text-sm font-medium transition-colors border-b-2 -mb-px",
              activeTab === id
                ? "border-blue-500 text-white"
                : "border-transparent text-[#484f58] hover:text-[#8b949e]",
            )}
          >
            <Icon className="w-3.5 h-3.5" />
            {label}
          </button>
        ))}
      </div>

      {/* ── Tab content ── */}
      <div className="flex-1 min-h-0 overflow-hidden">
        {activeTab === "workbench" && <WorkbenchTab />}
        {activeTab === "pipeline" && (
          <div className="h-full w-full overflow-hidden">
            <ModelTestBuilder />
          </div>
        )}
      </div>
    </div>
  );
}

// ---------------------------------------------------------------------------
// Page export — wraps inner component in Suspense for useSearchParams
// ---------------------------------------------------------------------------

export default function InferencePage() {
  return (
    <Suspense
      fallback={
        <div className="flex h-full items-center justify-center" style={{ color: "#484f58" }}>
          <Loader2 className="w-5 h-5 animate-spin mr-2" />
          <span className="text-sm">Loading…</span>
        </div>
      }
    >
      <InferencePageInner />
    </Suspense>
  );
}
