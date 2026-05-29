"use client";

/**
 * ModelTestBuilder — visual node-based pipeline editor extracted from model-tests page.
 *
 * Node types:
 *   ImageInput       — file upload zone, emits image data
 *   ModelNode        — registry model selector + conf/iou sliders
 *   FilterNode       — confidence threshold + class filter
 *   DetectionOutput  — annotated image with bounding boxes
 *   StatsOutput      — class distribution bar chart
 *
 * Persistence: graph JSON → POST /model-tests → UUID → ?test=<uuid>
 * Loading:     ?test=<uuid> → GET /model-tests/<uuid> → restore graph
 */

import React, {
  useCallback,
  useEffect,
  useRef,
  useState,
  Suspense,
  DragEvent,
} from "react";
import {
  ReactFlow,
  ReactFlowProvider,
  Background,
  Controls,
  MiniMap,
  addEdge,
  useNodesState,
  useEdgesState,
  useReactFlow,
  type Node,
  type Edge,
  type Connection,
  type NodeTypes,
  BackgroundVariant,
  Panel,
} from "@xyflow/react";
import { useQuery, useMutation } from "@tanstack/react-query";
import {
  Image,
  Brain,
  Filter,
  BarChart2,
  Eye,
  Play,
  Save,
  Share2,
  Loader2,
  Trash2,
  Check,
  ChevronDown,
  ChevronUp,
  X,
  Layers,
} from "lucide-react";
import { useSearchParams, useRouter } from "next/navigation";
import { api } from "@/lib/api";
import { cn } from "@/lib/utils";

// ---------------------------------------------------------------------------
// API types
// ---------------------------------------------------------------------------

interface RegisteredModel {
  id: number;
  name: string;
  variant: string | null;
  metrics_json?: string;
  file_path?: string;
}

interface TestMeta {
  uuid: string;
  name: string;
  description: string;
  created_at: number;
  updated_at: number;
}

// ---------------------------------------------------------------------------
// Node data types
// ---------------------------------------------------------------------------

interface ImageInputData {
  label: string;
  imageFile?: File;
  imageUrl?: string;
  imageWidth?: number;
  imageHeight?: number;
}

interface ModelNodeData {
  label: string;
  modelId?: number;
  modelVariant: string;
  confThreshold: number;
  iouThreshold: number;
  useTiled: boolean;
}

interface FilterNodeData {
  label: string;
  minConf: number;
  allowedClasses: string[];
}

interface DetectionOutputData {
  label: string;
  detections?: DetectionBox[];
  imageUrl?: string;
  imageWidth?: number;
  imageHeight?: number;
}

interface StatsOutputData {
  label: string;
  detections?: DetectionBox[];
}

interface DetectionBox {
  x1: number;
  y1: number;
  x2: number;
  y2: number;
  confidence: number;
  class_id: number;
  class_name: string;
}

// ---------------------------------------------------------------------------
// Shared node chrome
// ---------------------------------------------------------------------------

function NodeShell({
  title,
  icon: Icon,
  color,
  children,
  onDelete,
}: {
  title: string;
  icon: React.ElementType;
  color: string;
  children: React.ReactNode;
  onDelete?: () => void;
}) {
  return (
    <div
      className="rounded-xl border border-border bg-surface shadow-lg min-w-[220px] max-w-[280px] overflow-hidden"
      style={{ boxShadow: `0 0 0 2px ${color}22` }}
    >
      <div
        className="flex items-center gap-2 px-3 py-2 text-xs font-semibold"
        style={{ background: `${color}18`, borderBottom: `1px solid ${color}33` }}
      >
        <Icon className="w-3.5 h-3.5" style={{ color }} />
        <span className="truncate" style={{ color }}>
          {title}
        </span>
        {onDelete && (
          <button
            onClick={onDelete}
            className="ml-auto text-text-muted hover:text-red-400 transition-colors"
          >
            <X className="w-3 h-3" />
          </button>
        )}
      </div>
      <div className="px-3 py-2.5 space-y-2">{children}</div>
    </div>
  );
}

// ---------------------------------------------------------------------------
// Custom node: ImageInput
// ---------------------------------------------------------------------------

function ImageInputNode({ id, data }: { id: string; data: ImageInputData }) {
  const { updateNodeData, deleteElements } = useReactFlow();
  const inputRef = useRef<HTMLInputElement>(null);

  function handleFile(file: File) {
    const url = URL.createObjectURL(file);
    const img = new window.Image();
    img.onload = () =>
      updateNodeData(id, {
        imageFile: file,
        imageUrl: url,
        imageWidth: img.naturalWidth,
        imageHeight: img.naturalHeight,
      });
    img.src = url;
  }

  return (
    <NodeShell
      title="Image Input"
      icon={Image}
      color="#60a5fa"
      onDelete={() => deleteElements({ nodes: [{ id }] })}
    >
      <div
        onClick={() => inputRef.current?.click()}
        className="cursor-pointer rounded-lg border border-dashed border-blue-400/40 hover:border-blue-400/80 transition-colors text-center p-2"
      >
        {data.imageUrl ? (
          <img
            src={data.imageUrl}
            alt="input"
            className="max-h-24 mx-auto rounded object-contain"
          />
        ) : (
          <div className="py-3 text-xs text-text-muted">
            <Image className="w-6 h-6 mx-auto mb-1 opacity-40" />
            Click to upload
          </div>
        )}
      </div>
      {data.imageWidth && (
        <p className="text-[10px] text-text-muted text-center">
          {data.imageWidth}×{data.imageHeight}px
        </p>
      )}
      <input
        ref={inputRef}
        type="file"
        accept="image/*"
        className="hidden"
        onChange={(e) => e.target.files?.[0] && handleFile(e.target.files[0])}
      />
      {/* output handle */}
      <div className="absolute right-[-8px] top-1/2 -translate-y-1/2 w-3 h-3 rounded-full bg-blue-400 border-2 border-surface" />
    </NodeShell>
  );
}

// ---------------------------------------------------------------------------
// Custom node: ModelNode
// ---------------------------------------------------------------------------

function ModelNodeComp({ id, data }: { id: string; data: ModelNodeData }) {
  const { updateNodeData, deleteElements } = useReactFlow();
  const { data: models } = useQuery<RegisteredModel[]>({
    queryKey: ["models"],
    queryFn: () => api.get("/models").then((r) => r.data),
    staleTime: 30_000,
  });

  return (
    <NodeShell
      title="Model"
      icon={Brain}
      color="#a78bfa"
      onDelete={() => deleteElements({ nodes: [{ id }] })}
    >
      {/* input handle */}
      <div className="absolute left-[-8px] top-1/2 -translate-y-1/2 w-3 h-3 rounded-full bg-purple-400 border-2 border-surface" />
      {/* output handle */}
      <div className="absolute right-[-8px] top-1/2 -translate-y-1/2 w-3 h-3 rounded-full bg-purple-400 border-2 border-surface" />

      <label className="block text-[10px] text-text-muted mb-0.5">Model</label>
      <select
        className="w-full bg-panel border border-border rounded text-xs px-2 py-1 text-text-primary"
        value={data.modelId ?? ""}
        onChange={(e) => {
          const val = e.target.value;
          if (val === "") {
            updateNodeData(id, { modelId: undefined });
          } else if (val.startsWith("base:")) {
            updateNodeData(id, { modelId: undefined, modelVariant: val.slice(5) });
          } else {
            const m = models?.find((m) => m.id === Number(val));
            updateNodeData(id, {
              modelId: Number(val),
              modelVariant: m?.variant ?? "yolo11s",
            });
          }
        }}
      >
        <option value="">— base variant —</option>
        {["yolo11n", "yolo11s", "yolo11m"].map((v) => (
          <option key={v} value={`base:${v}`}>
            {v}
          </option>
        ))}
        {models && models.length > 0 && (
          <optgroup label="Trained models">
            {models.map((m) => {
              let mAP = "";
              try {
                const metrics = JSON.parse(m.metrics_json ?? "{}");
                if (metrics.best_map50)
                  mAP = ` (mAP50=${(metrics.best_map50 * 100).toFixed(1)}%)`;
              } catch {}
              return (
                <option key={m.id} value={m.id}>
                  {m.name}{mAP}
                </option>
              );
            })}
          </optgroup>
        )}
      </select>

      <label className="block text-[10px] text-text-muted mt-1.5 mb-0.5">
        Conf {data.confThreshold.toFixed(2)}
      </label>
      <input
        type="range"
        min="0.1"
        max="0.9"
        step="0.05"
        value={data.confThreshold}
        onChange={(e) =>
          updateNodeData(id, { confThreshold: parseFloat(e.target.value) })
        }
        className="w-full accent-purple-400"
      />

      <label className="block text-[10px] text-text-muted mt-1 mb-0.5">
        IoU {data.iouThreshold.toFixed(2)}
      </label>
      <input
        type="range"
        min="0.1"
        max="0.9"
        step="0.05"
        value={data.iouThreshold}
        onChange={(e) =>
          updateNodeData(id, { iouThreshold: parseFloat(e.target.value) })
        }
        className="w-full accent-purple-400"
      />

      <label className="flex items-center gap-1.5 text-[10px] text-text-muted mt-1 cursor-pointer">
        <input
          type="checkbox"
          checked={data.useTiled}
          onChange={(e) => updateNodeData(id, { useTiled: e.target.checked })}
          className="accent-purple-400"
        />
        Tiled inference (4K)
      </label>
    </NodeShell>
  );
}

// ---------------------------------------------------------------------------
// Custom node: FilterNode
// ---------------------------------------------------------------------------

function FilterNodeComp({ id, data }: { id: string; data: FilterNodeData }) {
  const { updateNodeData, deleteElements } = useReactFlow();
  const [classInput, setClassInput] = useState("");

  return (
    <NodeShell
      title="Filter"
      icon={Filter}
      color="#34d399"
      onDelete={() => deleteElements({ nodes: [{ id }] })}
    >
      <div className="absolute left-[-8px] top-1/2 -translate-y-1/2 w-3 h-3 rounded-full bg-emerald-400 border-2 border-surface" />
      <div className="absolute right-[-8px] top-1/2 -translate-y-1/2 w-3 h-3 rounded-full bg-emerald-400 border-2 border-surface" />

      <label className="block text-[10px] text-text-muted mb-0.5">
        Min confidence {data.minConf.toFixed(2)}
      </label>
      <input
        type="range"
        min="0"
        max="1"
        step="0.05"
        value={data.minConf}
        onChange={(e) =>
          updateNodeData(id, { minConf: parseFloat(e.target.value) })
        }
        className="w-full accent-emerald-400"
      />

      <label className="block text-[10px] text-text-muted mt-1.5 mb-0.5">
        Class whitelist (empty = all)
      </label>
      <div className="flex gap-1">
        <input
          value={classInput}
          onChange={(e) => setClassInput(e.target.value)}
          onKeyDown={(e) => {
            if (e.key === "Enter" && classInput.trim()) {
              updateNodeData(id, {
                allowedClasses: [...data.allowedClasses, classInput.trim()],
              });
              setClassInput("");
            }
          }}
          placeholder="class name…"
          className="flex-1 bg-panel border border-border rounded text-[10px] px-1.5 py-0.5 text-text-primary"
        />
      </div>
      <div className="flex flex-wrap gap-1 mt-1">
        {data.allowedClasses.map((cls) => (
          <span
            key={cls}
            className="flex items-center gap-0.5 text-[9px] bg-emerald-400/15 text-emerald-300 px-1.5 py-0.5 rounded-full"
          >
            {cls}
            <button
              onClick={() =>
                updateNodeData(id, {
                  allowedClasses: data.allowedClasses.filter((c) => c !== cls),
                })
              }
            >
              <X className="w-2 h-2" />
            </button>
          </span>
        ))}
      </div>
    </NodeShell>
  );
}

// ---------------------------------------------------------------------------
// Custom node: DetectionOutput
// ---------------------------------------------------------------------------

function DetectionOutputNode({
  id,
  data,
}: {
  id: string;
  data: DetectionOutputData;
}) {
  const { deleteElements } = useReactFlow();
  const canvasRef = useRef<HTMLCanvasElement>(null);

  useEffect(() => {
    if (!data.imageUrl || !data.detections || !canvasRef.current) return;
    const canvas = canvasRef.current;
    const ctx = canvas.getContext("2d")!;
    const img = new window.Image();
    img.onload = () => {
      canvas.width = img.naturalWidth;
      canvas.height = img.naturalHeight;
      ctx.drawImage(img, 0, 0);
      const scale = canvas.width / (data.imageWidth ?? canvas.width);
      ctx.lineWidth = Math.max(1, 2 / scale);
      ctx.font = `${Math.max(10, 12 / scale)}px monospace`;
      for (const det of data.detections!) {
        const hue = (det.class_id * 47 + 120) % 360;
        const color = `hsl(${hue},80%,60%)`;
        ctx.strokeStyle = color;
        ctx.strokeRect(det.x1, det.y1, det.x2 - det.x1, det.y2 - det.y1);
        ctx.fillStyle = color + "cc";
        ctx.fillRect(det.x1, det.y1 - 14 / scale, (det.class_name.length + 6) * 6 / scale, 14 / scale);
        ctx.fillStyle = "#000";
        ctx.fillText(
          `${det.class_name} ${(det.confidence * 100).toFixed(0)}%`,
          det.x1 + 2,
          det.y1 - 3 / scale
        );
      }
    };
    img.src = data.imageUrl;
  }, [data.detections, data.imageUrl, data.imageWidth, data.imageHeight]);

  return (
    <NodeShell
      title="Detection Output"
      icon={Eye}
      color="#f472b6"
      onDelete={() => deleteElements({ nodes: [{ id }] })}
    >
      <div className="absolute left-[-8px] top-1/2 -translate-y-1/2 w-3 h-3 rounded-full bg-pink-400 border-2 border-surface" />

      {data.imageUrl && data.detections ? (
        <>
          <canvas
            ref={canvasRef}
            className="w-full max-h-40 rounded object-contain border border-border"
          />
          <p className="text-[10px] text-text-muted text-center">
            {data.detections.length} detections
          </p>
        </>
      ) : (
        <div className="py-4 text-center text-[10px] text-text-muted">
          <Eye className="w-5 h-5 mx-auto mb-1 opacity-30" />
          Connect model output
        </div>
      )}
    </NodeShell>
  );
}

// ---------------------------------------------------------------------------
// Custom node: StatsOutput
// ---------------------------------------------------------------------------

function StatsOutputNode({ id, data }: { id: string; data: StatsOutputData }) {
  const { deleteElements } = useReactFlow();

  const counts: Record<string, number> = {};
  const confSums: Record<string, number> = {};
  for (const d of data.detections ?? []) {
    counts[d.class_name] = (counts[d.class_name] ?? 0) + 1;
    confSums[d.class_name] = (confSums[d.class_name] ?? 0) + d.confidence;
  }
  const classes = Object.keys(counts);
  const total = Object.values(counts).reduce((a, b) => a + b, 0);

  return (
    <NodeShell
      title="Stats Output"
      icon={BarChart2}
      color="#fb923c"
      onDelete={() => deleteElements({ nodes: [{ id }] })}
    >
      <div className="absolute left-[-8px] top-1/2 -translate-y-1/2 w-3 h-3 rounded-full bg-orange-400 border-2 border-surface" />

      {classes.length === 0 ? (
        <div className="py-4 text-center text-[10px] text-text-muted">
          <BarChart2 className="w-5 h-5 mx-auto mb-1 opacity-30" />
          Connect model output
        </div>
      ) : (
        <div className="space-y-1.5">
          <p className="text-[10px] text-text-muted">
            Total: <span className="text-text-primary font-medium">{total}</span>
          </p>
          {classes.map((cls) => {
            const hue = (classes.indexOf(cls) * 47 + 200) % 360;
            const pct = ((counts[cls] / total) * 100).toFixed(0);
            const avgConf = ((confSums[cls] / counts[cls]) * 100).toFixed(0);
            return (
              <div key={cls}>
                <div className="flex justify-between text-[10px] mb-0.5">
                  <span className="text-text-secondary truncate max-w-[110px]">
                    {cls}
                  </span>
                  <span className="text-text-muted">
                    {counts[cls]} · {avgConf}%
                  </span>
                </div>
                <div className="h-1.5 bg-panel rounded-full overflow-hidden">
                  <div
                    className="h-full rounded-full"
                    style={{
                      width: `${pct}%`,
                      background: `hsl(${hue},70%,55%)`,
                    }}
                  />
                </div>
              </div>
            );
          })}
        </div>
      )}
    </NodeShell>
  );
}

// ---------------------------------------------------------------------------
// Node type registry
// ---------------------------------------------------------------------------

const NODE_TYPES: NodeTypes = {
  imageInput: ImageInputNode as any,
  model: ModelNodeComp as any,
  filter: FilterNodeComp as any,
  detectionOutput: DetectionOutputNode as any,
  statsOutput: StatsOutputNode as any,
};

// ---------------------------------------------------------------------------
// Palette item definitions
// ---------------------------------------------------------------------------

const PALETTE_ITEMS = [
  {
    type: "imageInput",
    label: "Image Input",
    icon: Image,
    color: "#60a5fa",
    defaultData: { label: "Image Input" },
  },
  {
    type: "model",
    label: "Model",
    icon: Brain,
    color: "#a78bfa",
    defaultData: {
      label: "Model",
      modelVariant: "yolo11s",
      confThreshold: 0.35,
      iouThreshold: 0.45,
      useTiled: false,
    },
  },
  {
    type: "filter",
    label: "Filter",
    icon: Filter,
    color: "#34d399",
    defaultData: { label: "Filter", minConf: 0.3, allowedClasses: [] },
  },
  {
    type: "detectionOutput",
    label: "Detection Output",
    icon: Eye,
    color: "#f472b6",
    defaultData: { label: "Detection Output" },
  },
  {
    type: "statsOutput",
    label: "Stats Output",
    icon: BarChart2,
    color: "#fb923c",
    defaultData: { label: "Stats Output" },
  },
];

// ---------------------------------------------------------------------------
// Main canvas component (must be inside ReactFlowProvider)
// ---------------------------------------------------------------------------

let _nodeCounter = 0;

function ModelTestCanvas() {
  const reactFlowWrapper = useRef<HTMLDivElement>(null);
  const { screenToFlowPosition, getNodes, getEdges, setNodes, setEdges } =
    useReactFlow();
  const [nodes, setNodesState, onNodesChange] = useNodesState<Node>([]);
  const [edges, setEdgesState, onEdgesChange] = useEdgesState<Edge>([]);

  const [testName, setTestName] = useState("Untitled test");
  const [savedUuid, setSavedUuid] = useState<string | null>(null);
  const [shareTooltip, setShareTooltip] = useState(false);
  const [running, setRunning] = useState(false);
  const [runError, setRunError] = useState<string | null>(null);
  const [showList, setShowList] = useState(false);

  const searchParams = useSearchParams();
  const router = useRouter();

  // ── Load from URL param ──────────────────────────────────────────────────

  useEffect(() => {
    const testId = searchParams.get("test");
    if (!testId) return;
    api.get(`/model-tests/${testId}`).then((r) => {
      const { name, graph } = r.data;
      setTestName(name);
      setSavedUuid(testId);
      if (graph?.nodes) setNodesState(graph.nodes);
      if (graph?.edges) setEdgesState(graph.edges);
    });
  }, [searchParams]);

  // ── Drag-from-palette drop handler ──────────────────────────────────────

  const onDrop = useCallback(
    (e: DragEvent<HTMLDivElement>) => {
      e.preventDefault();
      const raw = e.dataTransfer.getData("application/reactflow");
      if (!raw) return;
      const { type, defaultData } = JSON.parse(raw);
      const pos = screenToFlowPosition({ x: e.clientX, y: e.clientY });
      const id = `${type}_${++_nodeCounter}`;
      setNodesState((nds) => [
        ...nds,
        { id, type, position: pos, data: { ...defaultData } },
      ]);
    },
    [screenToFlowPosition, setNodesState]
  );

  const onDragOver = (e: DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    e.dataTransfer.dropEffect = "move";
  };

  // ── Edge connection ──────────────────────────────────────────────────────

  const onConnect = useCallback(
    (params: Connection) =>
      setEdgesState((eds) =>
        addEdge({ ...params, animated: true, style: { stroke: "#6b7280" } }, eds)
      ),
    [setEdgesState]
  );

  // ── Run pipeline ─────────────────────────────────────────────────────────

  async function runPipeline() {
    setRunError(null);
    setRunning(true);

    try {
      const allNodes = getNodes();
      const allEdges = getEdges();

      // Build adjacency: nodeId → list of target nodeIds
      const downstream: Record<string, string[]> = {};
      const upstream: Record<string, string[]> = {};
      for (const e of allEdges) {
        if (!e.source || !e.target) continue;
        (downstream[e.source] ??= []).push(e.target);
        (upstream[e.target] ??= []).push(e.source);
      }

      // Topological sort (Kahn's)
      const inDegree: Record<string, number> = {};
      for (const n of allNodes) inDegree[n.id] = (upstream[n.id] ?? []).length;
      const queue = allNodes.filter((n) => inDegree[n.id] === 0).map((n) => n.id);
      const order: string[] = [];
      while (queue.length) {
        const cur = queue.shift()!;
        order.push(cur);
        for (const next of downstream[cur] ?? []) {
          inDegree[next]--;
          if (inDegree[next] === 0) queue.push(next);
        }
      }

      // State map: nodeId → { imageUrl, imageFile, imageWidth, imageHeight, detections }
      type NodeState = {
        imageUrl?: string;
        imageFile?: File;
        imageWidth?: number;
        imageHeight?: number;
        detections?: DetectionBox[];
      };
      const state: Record<string, NodeState> = {};

      for (const nodeId of order) {
        const node = allNodes.find((n) => n.id === nodeId)!;
        const upIds = upstream[nodeId] ?? [];
        const upState = upIds.map((id) => state[id] ?? {});

        if (node.type === "imageInput") {
          const d = node.data as unknown as ImageInputData;
          state[nodeId] = {
            imageUrl: d.imageUrl,
            imageFile: d.imageFile,
            imageWidth: d.imageWidth,
            imageHeight: d.imageHeight,
          };
        }

        else if (node.type === "model") {
          const d = node.data as unknown as ModelNodeData;
          const src = upState[0];
          if (!src?.imageFile) continue;

          const form = new FormData();
          form.append("file", src.imageFile);
          form.append("conf_threshold", String(d.confThreshold));
          form.append("iou_threshold", String(d.iouThreshold));
          form.append("model_variant", d.modelVariant);
          if (d.modelId != null) form.append("model_id", String(d.modelId));
          form.append("use_tiled", String(d.useTiled));

          const resp = await api.post("/inference/detect", form, {
            headers: { "Content-Type": "multipart/form-data" },
          });
          state[nodeId] = {
            ...src,
            detections: resp.data.detections,
          };
        }

        else if (node.type === "filter") {
          const d = node.data as unknown as FilterNodeData;
          const src = upState[0];
          if (!src) continue;
          const filtered = (src.detections ?? []).filter(
            (det) =>
              det.confidence >= d.minConf &&
              (d.allowedClasses.length === 0 ||
                d.allowedClasses.includes(det.class_name))
          );
          state[nodeId] = { ...src, detections: filtered };
        }

        else if (node.type === "detectionOutput" || node.type === "statsOutput") {
          const src = upState[0];
          if (src) state[nodeId] = src;
        }
      }

      // Push results back into node data
      setNodesState((nds) =>
        nds.map((n) => {
          const s = state[n.id];
          if (!s) return n;
          if (n.type === "detectionOutput") {
            return {
              ...n,
              data: {
                ...n.data,
                detections: s.detections,
                imageUrl: s.imageUrl,
                imageWidth: s.imageWidth,
                imageHeight: s.imageHeight,
              },
            };
          }
          if (n.type === "statsOutput") {
            return { ...n, data: { ...n.data, detections: s.detections } };
          }
          return n;
        })
      );
    } catch (err: any) {
      setRunError(err?.response?.data?.detail ?? err?.message ?? "Run failed");
    } finally {
      setRunning(false);
    }
  }

  // ── Save graph ────────────────────────────────────────────────────────────

  const saveMutation = useMutation({
    mutationFn: async () => {
      const graph = { nodes: getNodes(), edges: getEdges() };
      if (savedUuid) {
        await api.put(`/model-tests/${savedUuid}`, { name: testName, description: "", graph });
        return savedUuid;
      }
      const r = await api.post("/model-tests", { name: testName, description: "", graph });
      return r.data.uuid as string;
    },
    onSuccess: (uuid) => {
      setSavedUuid(uuid);
      router.replace(`?test=${uuid}`);
    },
  });

  // ── Share (copy URL) ──────────────────────────────────────────────────────

  async function shareTest() {
    if (!savedUuid) {
      const uuid = await saveMutation.mutateAsync();
      await navigator.clipboard.writeText(`${window.location.origin}/inference?tab=pipeline&test=${uuid}`);
    } else {
      await navigator.clipboard.writeText(`${window.location.origin}/inference?tab=pipeline&test=${savedUuid}`);
    }
    setShareTooltip(true);
    setTimeout(() => setShareTooltip(false), 2000);
  }

  // ── Saved tests list ─────────────────────────────────────────────────────

  const { data: savedTests, refetch: refetchTests } = useQuery<TestMeta[]>({
    queryKey: ["model-tests-list"],
    queryFn: () => api.get("/model-tests").then((r) => r.data),
    enabled: showList,
  });

  function loadTest(uuid: string) {
    router.push(`?tab=pipeline&test=${uuid}`);
    setShowList(false);
  }

  async function deleteTest(uuid: string, e: React.MouseEvent) {
    e.stopPropagation();
    await api.delete(`/model-tests/${uuid}`);
    refetchTests();
    if (uuid === savedUuid) {
      setSavedUuid(null);
      setNodesState([]);
      setEdgesState([]);
      router.replace("/inference?tab=pipeline");
    }
  }

  // ── Clear canvas ─────────────────────────────────────────────────────────

  function clearCanvas() {
    setNodesState([]);
    setEdgesState([]);
    setSavedUuid(null);
    setTestName("Untitled test");
    router.replace("/inference?tab=pipeline");
  }

  // ── Render ────────────────────────────────────────────────────────────────

  return (
    <div className="flex h-full w-full overflow-hidden bg-background">
      {/* ── Left palette ──────────────────────────────────────────────────── */}
      <div className="w-44 shrink-0 border-r border-border bg-surface flex flex-col gap-1 p-2 overflow-y-auto">
        <p className="text-[10px] font-semibold text-text-muted uppercase tracking-wider px-1 mb-1">
          Nodes
        </p>
        {PALETTE_ITEMS.map((item) => {
          const Icon = item.icon;
          return (
            <div
              key={item.type}
              draggable
              onDragStart={(e) => {
                e.dataTransfer.setData(
                  "application/reactflow",
                  JSON.stringify({ type: item.type, defaultData: item.defaultData })
                );
                e.dataTransfer.effectAllowed = "move";
              }}
              className="flex items-center gap-2 px-2 py-2 rounded-lg border border-border hover:border-text-muted cursor-grab active:cursor-grabbing transition-colors select-none"
              style={{ borderLeftColor: item.color, borderLeftWidth: 3 }}
            >
              <Icon className="w-3.5 h-3.5 shrink-0" style={{ color: item.color }} />
              <span className="text-xs text-text-secondary truncate">{item.label}</span>
            </div>
          );
        })}

        <div className="mt-auto pt-3 border-t border-border">
          <button
            onClick={() => { setShowList((v) => !v); refetchTests(); }}
            className="w-full flex items-center gap-1.5 px-2 py-1.5 text-xs text-text-muted hover:text-text-primary rounded transition-colors"
          >
            <Layers className="w-3.5 h-3.5" />
            Saved tests
            {showList ? <ChevronUp className="w-3 h-3 ml-auto" /> : <ChevronDown className="w-3 h-3 ml-auto" />}
          </button>

          {showList && (
            <div className="mt-1 space-y-0.5 max-h-48 overflow-y-auto">
              {savedTests?.length === 0 && (
                <p className="text-[10px] text-text-muted px-1 py-1">No saved tests</p>
              )}
              {savedTests?.map((t) => (
                <div
                  key={t.uuid}
                  onClick={() => loadTest(t.uuid)}
                  className={cn(
                    "flex items-center gap-1 px-2 py-1.5 rounded cursor-pointer hover:bg-panel text-[11px]",
                    t.uuid === savedUuid ? "text-accent" : "text-text-secondary"
                  )}
                >
                  <span className="flex-1 truncate">{t.name}</span>
                  <button
                    onClick={(e) => deleteTest(t.uuid, e)}
                    className="text-text-muted hover:text-red-400 shrink-0"
                  >
                    <Trash2 className="w-3 h-3" />
                  </button>
                </div>
              ))}
            </div>
          )}
        </div>
      </div>

      {/* ── Main area ─────────────────────────────────────────────────────── */}
      <div className="flex-1 flex flex-col min-w-0">
        {/* Top bar */}
        <div className="flex items-center gap-2 px-4 py-2 border-b border-border bg-surface shrink-0">
          <input
            value={testName}
            onChange={(e) => setTestName(e.target.value)}
            className="text-sm font-medium bg-transparent border-none outline-none text-text-primary w-52 focus:bg-panel focus:px-2 rounded transition-colors"
          />
          {savedUuid && (
            <span className="text-[10px] text-text-muted font-mono bg-panel px-1.5 py-0.5 rounded truncate max-w-[120px]">
              {savedUuid.slice(0, 8)}…
            </span>
          )}

          <div className="ml-auto flex items-center gap-2">
            {runError && (
              <span className="text-[11px] text-red-400 max-w-xs truncate">
                {runError}
              </span>
            )}

            <button
              onClick={clearCanvas}
              className="flex items-center gap-1.5 px-2.5 py-1.5 rounded-md text-xs text-text-muted hover:text-text-primary hover:bg-panel transition-colors"
            >
              <Trash2 className="w-3.5 h-3.5" />
              Clear
            </button>

            <button
              onClick={() => saveMutation.mutate()}
              disabled={saveMutation.isPending}
              className="flex items-center gap-1.5 px-2.5 py-1.5 rounded-md text-xs bg-panel hover:bg-border text-text-primary transition-colors border border-border"
            >
              {saveMutation.isPending ? (
                <Loader2 className="w-3.5 h-3.5 animate-spin" />
              ) : (
                <Save className="w-3.5 h-3.5" />
              )}
              Save
            </button>

            <div className="relative">
              <button
                onClick={shareTest}
                className="flex items-center gap-1.5 px-2.5 py-1.5 rounded-md text-xs bg-panel hover:bg-border text-text-primary transition-colors border border-border"
              >
                {shareTooltip ? (
                  <Check className="w-3.5 h-3.5 text-emerald-400" />
                ) : (
                  <Share2 className="w-3.5 h-3.5" />
                )}
                {shareTooltip ? "Copied!" : "Share"}
              </button>
            </div>

            <button
              onClick={runPipeline}
              disabled={running}
              className={cn(
                "flex items-center gap-1.5 px-3 py-1.5 rounded-md text-xs font-medium transition-colors",
                running
                  ? "bg-accent/40 text-accent/60 cursor-not-allowed"
                  : "bg-accent hover:bg-accent/80 text-background"
              )}
            >
              {running ? (
                <Loader2 className="w-3.5 h-3.5 animate-spin" />
              ) : (
                <Play className="w-3.5 h-3.5" />
              )}
              Run
            </button>
          </div>
        </div>

        {/* React Flow canvas */}
        <div
          ref={reactFlowWrapper}
          className="flex-1 min-h-0"
          onDrop={onDrop}
          onDragOver={onDragOver}
        >
          <ReactFlow
            nodes={nodes}
            edges={edges}
            onNodesChange={onNodesChange}
            onEdgesChange={onEdgesChange}
            onConnect={onConnect}
            nodeTypes={NODE_TYPES}
            fitView
            proOptions={{ hideAttribution: true }}
            deleteKeyCode={["Backspace", "Delete"]}
          >
            <Background
              variant={BackgroundVariant.Dots}
              gap={20}
              size={1}
              color="#374151"
            />
            <Controls
              className="!bg-surface !border-border [&_button]:!bg-surface [&_button]:!border-border [&_button]:!text-text-secondary"
            />
            <MiniMap
              nodeColor={(n) => {
                const colors: Record<string, string> = {
                  imageInput: "#60a5fa",
                  model: "#a78bfa",
                  filter: "#34d399",
                  detectionOutput: "#f472b6",
                  statsOutput: "#fb923c",
                };
                return colors[n.type ?? ""] ?? "#6b7280";
              }}
              maskColor="rgba(0,0,0,0.5)"
              className="!bg-surface !border-border"
            />
            <Panel position="bottom-center">
              {nodes.length === 0 && (
                <p className="text-xs text-text-muted pointer-events-none select-none">
                  Drag nodes from the palette to start building your pipeline
                </p>
              )}
            </Panel>
          </ReactFlow>
        </div>
      </div>
    </div>
  );
}

// ---------------------------------------------------------------------------
// Exported component — wraps ModelTestCanvas in ReactFlowProvider + Suspense
// ---------------------------------------------------------------------------

export function ModelTestBuilder() {
  return (
    <Suspense fallback={<div className="flex h-full items-center justify-center text-text-muted text-sm">Loading…</div>}>
      <ReactFlowProvider>
        <div className="h-full w-full">
          <ModelTestCanvas />
        </div>
      </ReactFlowProvider>
    </Suspense>
  );
}
