"use client";

import React, { useCallback, useMemo, useState, useEffect } from "react";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import {
  Film,
  Upload,
  Loader2,
  Play,
  RefreshCw,
  AlertTriangle,
  Download,
  Trash2,
  Star,
  BarChart2,
} from "lucide-react";
import { useDropzone } from "react-dropzone";
import { api } from "@/lib/api";
import { cn, timeAgo } from "@/lib/utils";

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

interface VideoRecord {
  id: string;
  filename: string;
  file_size_bytes: number;
  duration_s?: number;
  fps?: number;
  status: string;
  created_at: number;
}

interface VideoUploadResponse {
  video_id: string;
  filename: string;
  file_size_bytes: number;
}

interface AnalyzeResponse {
  job_id: string;
  video_id: string;
  status: string;
}

interface JobStatus {
  job_id: string;
  status: string;
  progress?: number;
  result?: VideoAnalysisResult;
  error?: string;
}

/**
 * FrameMeta — adapted from backend FrameResult.
 * quality_score is sharpness normalised to [0,1] (÷500).
 */
interface FrameMeta {
  frame_idx: number;        // ← backend: frame_index
  timestamp_s: number;
  quality_score: number;    // ← normalised sharpness [0,1]
  focus_score: number;      // same as quality_score
  is_selected: boolean;     // ← backend: selected
  is_duplicate: boolean;    // not in backend; always false
  file_path: string | null; // ← backend: path
}

/**
 * VideoAnalysisResult — matches backend VideoAnalysisResult schema exactly.
 */
interface VideoAnalysisResult {
  video_id: string;
  total_frames: number;
  analyzed_frames: number;
  selected_frames: number;  // count integer, not array
  best_sharpness: number;
  mean_sharpness: number;
  frames: {
    frame_index: number;
    timestamp_s: number;
    sharpness: number;
    exposure_ok: boolean;
    path: string;
    selected: boolean;
  }[];
  duration_s: number;
  processing_time_s: number;
}

/** Convert backend FrameResult[] → FrameMeta[] for chart and list rendering. */
function toFrameMeta(frames: VideoAnalysisResult['frames']): FrameMeta[] {
  return frames.map((f) => ({
    frame_idx: f.frame_index,
    timestamp_s: f.timestamp_s,
    quality_score: Math.min(1.0, f.sharpness / 500),
    focus_score: Math.min(1.0, f.sharpness / 500),
    is_selected: f.selected,
    is_duplicate: false,
    file_path: f.path,
  }));
}

/** Thumbnail URL for a frame via the backend serving endpoint. */
function frameThumbnailUrl(videoId: string, frameIndex: number): string {
  return `/api/v1/video/thumbnail/${videoId}/${frameIndex}`;
}

// ---------------------------------------------------------------------------
// Frame quality timeline — SVG bar chart showing quality score per frame
// ---------------------------------------------------------------------------

/**
 * FrameQualityTimeline renders a compact quality-score bar chart over all
 * analysed frames. Bars are coloured:
 *   green  — selected (above quality gate, not duplicate)
 *   amber  — above gate but marked duplicate / not selected
 *   red    — below quality gate
 * A dashed horizontal line marks the quality gate threshold.
 */
function FrameQualityTimeline({
  frames,
  qualityGate,
}: {
  frames: FrameMeta[];
  qualityGate: number;
}) {
  const WIDTH = 600;
  const HEIGHT = 80;
  const PAD_L = 28;
  const PAD_R = 8;
  const PAD_TOP = 6;
  const PAD_BOT = 16;

  const chartW = WIDTH - PAD_L - PAD_R;
  const chartH = HEIGHT - PAD_TOP - PAD_BOT;

  const barW = Math.max(1, chartW / frames.length - 0.5);

  const gateY = PAD_TOP + chartH * (1 - qualityGate);

  // Y-axis ticks
  const ticks = [0, 0.25, 0.5, 0.75, 1.0];

  return (
    <svg
      viewBox={`0 0 ${WIDTH} ${HEIGHT}`}
      className="w-full"
      style={{ height: HEIGHT, display: "block" }}
      preserveAspectRatio="none"
    >
      {/* Y gridlines + labels */}
      {ticks.map((t) => {
        const y = PAD_TOP + chartH * (1 - t);
        return (
          <g key={t}>
            <line
              x1={PAD_L}
              y1={y}
              x2={WIDTH - PAD_R}
              y2={y}
              stroke="#21262d"
              strokeWidth={0.5}
            />
            <text
              x={PAD_L - 3}
              y={y + 3}
              textAnchor="end"
              fontSize={7}
              fill="#484f58"
            >
              {Math.round(t * 100)}
            </text>
          </g>
        );
      })}

      {/* Bars */}
      {frames.map((f, i) => {
        const x = PAD_L + (i / frames.length) * chartW;
        const barH = f.quality_score * chartH;
        const y = PAD_TOP + chartH - barH;

        const color = f.is_selected
          ? "#22c55e"
          : f.is_duplicate
          ? "#f59e0b"
          : f.quality_score >= qualityGate
          ? "#3b82f6"
          : "#374151";

        return (
          <rect
            key={f.frame_idx}
            x={x}
            y={y}
            width={Math.max(barW, 0.5)}
            height={Math.max(barH, 0.5)}
            fill={color}
            opacity={0.85}
          />
        );
      })}

      {/* Quality gate line */}
      <line
        x1={PAD_L}
        y1={gateY}
        x2={WIDTH - PAD_R}
        y2={gateY}
        stroke="#f59e0b"
        strokeWidth={1}
        strokeDasharray="4 3"
      />

      {/* X-axis label */}
      <text
        x={PAD_L + chartW / 2}
        y={HEIGHT - 2}
        textAnchor="middle"
        fontSize={7}
        fill="#484f58"
      >
        Frame index →
      </text>

      {/* Legend */}
      {[
        { color: "#22c55e", label: "selected" },
        { color: "#3b82f6", label: "above gate" },
        { color: "#f59e0b", label: "duplicate" },
        { color: "#374151", label: "rejected" },
      ].map((item, i) => (
        <g key={item.label} transform={`translate(${PAD_L + i * 80}, ${HEIGHT - 2})`}>
          <rect x={0} y={-6} width={6} height={6} fill={item.color} rx={1} />
          <text x={9} y={0} fontSize={6.5} fill="#484f58">
            {item.label}
          </text>
        </g>
      ))}
    </svg>
  );
}

// ---------------------------------------------------------------------------
// AnalysisResults — full results panel with KPIs, timeline, and frame list
// ---------------------------------------------------------------------------

function AnalysisResults({
  result,
  qualityGate,
  onClear,
}: {
  result: VideoAnalysisResult;
  qualityGate: number;
  onClear: () => void;
}) {
  // Convert backend frames to UI-friendly FrameMeta
  const allFrames = useMemo(() => toFrameMeta(result.frames ?? []), [result.frames]);
  const selected = useMemo(() => allFrames.filter((f) => f.is_selected), [allFrames]);

  const selectionRate = result.total_frames > 0
    ? (selected.length / result.total_frames) * 100
    : 0;

  const avgQuality = selected.length > 0
    ? selected.reduce((s, f) => s + f.quality_score, 0) / selected.length
    : 0;

  return (
    <div className="space-y-4">
      {/* KPIs */}
      <div className="grid grid-cols-5 gap-2">
        {[
          { label: "Total Frames", value: result.total_frames.toLocaleString(), color: "#e6edf3" },
          { label: "Selected", value: selected.length, color: "#22c55e" },
          { label: "Selection Rate", value: `${selectionRate.toFixed(1)}%`, color: "#3b82f6" },
          { label: "Avg Quality", value: `${(avgQuality * 100).toFixed(0)}%`, color: avgQuality >= qualityGate ? "#22c55e" : "#f59e0b" },
          { label: "Process Time", value: `${result.processing_time_s?.toFixed(1) ?? "—"}s`, color: "#484f58" },
        ].map(({ label, value, color }) => (
          <div
            key={label}
            className="rounded-xl p-3"
            style={{ background: "#0d1117", border: "1px solid #21262d" }}
          >
            <p
              className="text-[10px] uppercase tracking-wide mb-1"
              style={{ color: "#484f58" }}
            >
              {label}
            </p>
            <p className="text-lg font-bold font-mono" style={{ color }}>
              {value}
            </p>
          </div>
        ))}
      </div>

      {/* Quality timeline */}
      {allFrames.length > 0 && (
        <div
          className="rounded-xl p-4 space-y-2"
          style={{ background: "#0d1117", border: "1px solid #21262d" }}
        >
          <div className="flex items-center gap-2 mb-2">
            <BarChart2 className="w-3.5 h-3.5" style={{ color: "#484f58" }} />
            <span className="text-xs font-medium text-white">
              Frame Quality Timeline
            </span>
            <span className="text-[10px] ml-auto" style={{ color: "#484f58" }}>
              {allFrames.length} frames analysed · gate = {(qualityGate * 100).toFixed(0)}%
            </span>
          </div>
          <FrameQualityTimeline frames={allFrames} qualityGate={qualityGate} />
        </div>
      )}

      {/* Selected frames list */}
      {selected.length > 0 && (
        <div
          className="rounded-xl p-4 space-y-3"
          style={{ background: "#0d1117", border: "1px solid #21262d" }}
        >
          <div className="flex items-center gap-2">
            <Star className="w-3.5 h-3.5 text-yellow-400" />
            <span className="text-xs font-semibold text-white">
              Best Frames ({selected.length})
            </span>
          </div>
          <div className="overflow-x-auto">
            <div className="flex gap-2 pb-1" style={{ minWidth: "max-content" }}>
              {selected.slice(0, 30).map((frame) => {
                const q = frame.quality_score;
                const barColor = q >= 0.7 ? "#22c55e" : q >= 0.4 ? "#f59e0b" : "#ef4444";
                return (
                  <div
                    key={frame.frame_idx}
                    className="flex-shrink-0 w-[72px] rounded-lg overflow-hidden"
                    style={{ border: "1px solid #21262d" }}
                  >
                    {/* Frame thumbnail — served by GET /api/v1/video/thumbnail/{video_id}/{frame_index} */}
                    <img
                      src={frameThumbnailUrl(result.video_id, frame.frame_idx)}
                      alt={`Frame ${frame.frame_idx} @ ${frame.timestamp_s.toFixed(2)}s`}
                      className="w-full object-cover"
                      style={{ height: 48, background: "#161b22" }}
                      loading="lazy"
                      onError={(e) => {
                        // Fall back to placeholder if frame not saved (no output_dir set)
                        const target = e.currentTarget;
                        target.style.display = "none";
                        const sib = target.nextElementSibling as HTMLElement | null;
                        if (sib) sib.style.display = "flex";
                      }}
                    />
                    {/* Fallback placeholder shown only when thumbnail fetch fails */}
                    <div
                      className="w-full items-center justify-center hidden"
                      style={{ height: 48, background: "#161b22" }}
                    >
                      <span className="text-[9px] font-mono" style={{ color: "#484f58" }}>
                        f {frame.frame_idx}
                      </span>
                    </div>

                    {/* Quality bar */}
                    <div style={{ background: "#21262d", height: 2 }}>
                      <div
                        style={{
                          height: 2,
                          width: `${q * 100}%`,
                          background: barColor,
                        }}
                      />
                    </div>

                    {/* Metadata */}
                    <div className="px-1.5 py-1">
                      <p className="text-[9px] font-mono" style={{ color: barColor }}>
                        {Math.round(q * 100)}%
                      </p>
                      <p className="text-[8px] font-mono" style={{ color: "#484f58" }}>
                        {frame.timestamp_s.toFixed(2)}s
                      </p>
                    </div>
                  </div>
                );
              })}
              {selected.length > 30 && (
                <div
                  className="flex-shrink-0 w-[72px] rounded-lg flex items-center justify-center"
                  style={{ border: "1px dashed #21262d", color: "#484f58" }}
                >
                  <span className="text-[10px]">+{selected.length - 30}</span>
                </div>
              )}
            </div>
          </div>
        </div>
      )}

      {/* Actions */}
      <div className="flex items-center gap-3">
        <button
          onClick={onClear}
          className="text-xs transition-colors"
          style={{ color: "#484f58" }}
        >
          Clear results
        </button>
      </div>
    </div>
  );
}

// ---------------------------------------------------------------------------
// Upload + analyze flow
// ---------------------------------------------------------------------------

function UploadZone({
  onAnalysisComplete,
}: {
  onAnalysisComplete: () => void;
}) {
  const [videoFile, setVideoFile] = useState<File | null>(null);
  const [videoId, setVideoId] = useState<string | null>(null);
  const [jobId, setJobId] = useState<string | null>(null);
  const [maxFrames, setMaxFrames] = useState(100);
  const [qualityGate, setQualityGate] = useState(0.4);
  const [result, setResult] = useState<VideoAnalysisResult | null>(null);
  const [pollError, setPollError] = useState<string | null>(null);
  const queryClient = useQueryClient();

  // Upload mutation
  const uploadMutation = useMutation<VideoUploadResponse, Error, File>({
    mutationFn: async (file: File) => {
      const formData = new FormData();
      formData.append("file", file);
      const r = await api.post("/video/upload", formData, {
        headers: { "Content-Type": "multipart/form-data" },
      });
      return r.data;
    },
    onSuccess: (data) => {
      setVideoId(data.video_id);
    },
  });

  // Analyze mutation
  const analyzeMutation = useMutation<AnalyzeResponse, Error, { video_id: string }>({
    mutationFn: async (payload) => {
      const r = await api.post("/video/analyze", {
        video_id: payload.video_id,
        max_frames: maxFrames,
        quality_gate: qualityGate,
      });
      return r.data;
    },
    onSuccess: (data) => {
      setJobId(data.job_id);
      setPollError(null);
    },
  });

  // Poll job status
  const { data: jobStatus } = useQuery<JobStatus>({
    queryKey: ["video-job", jobId],
    queryFn: () => api.get(`/video/jobs/${jobId}`).then((r) => r.data),
    enabled: !!jobId && !result,
    refetchInterval: (query) => {
      const status = query.state.data?.status;
      if (status === "completed" || status === "failed") return false;
      return 2000;
    },
  });

  useEffect(() => {
    if (!jobStatus) return;
    if (jobStatus.status === "completed" && jobStatus.result) {
      setResult(jobStatus.result);
      setJobId(null);
      queryClient.invalidateQueries({ queryKey: ["videos"] });
      onAnalysisComplete();
    } else if (jobStatus.status === "failed") {
      setPollError(jobStatus.error ?? "Job failed");
      setJobId(null);
    }
  }, [jobStatus, queryClient, onAnalysisComplete]);

  const onDrop = useCallback((accepted: File[]) => {
    const file = accepted[0];
    if (!file) return;
    setVideoFile(file);
    setVideoId(null);
    setJobId(null);
    setResult(null);
    setPollError(null);
    uploadMutation.reset();
    analyzeMutation.reset();
  }, []); // eslint-disable-line react-hooks/exhaustive-deps

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept: {
      "video/mp4": [".mp4"],
      "video/avi": [".avi"],
      "video/quicktime": [".mov"],
      "video/x-matroska": [".mkv"],
    },
    maxFiles: 1,
    disabled: uploadMutation.isPending || !!videoId,
  });

  const isProcessing = uploadMutation.isPending || analyzeMutation.isPending || !!jobId;
  const jobProgress = jobStatus?.progress ?? 0;

  return (
    <div className="space-y-4">
      {/* Drop zone */}
      {!videoFile ? (
        <div
          {...getRootProps()}
          className="flex flex-col items-center justify-center gap-4 h-52 rounded-2xl border-2 border-dashed cursor-pointer transition-all"
          style={{
            borderColor: isDragActive ? '#3b82f6' : '#21262d',
            background: isDragActive ? 'rgba(59,130,246,0.1)' : 'transparent',
          }}
        >
          <input {...getInputProps()} />
          <Film className="w-10 h-10" style={{ color: '#484f58' }} />
          <div className="text-center">
            <p className="text-sm font-medium" style={{ color: '#8b949e' }}>
              {isDragActive ? "Drop video here" : "Drop a microscopy video"}
            </p>
            <p className="text-xs mt-1" style={{ color: '#484f58' }}>MP4, AVI, MOV, MKV</p>
          </div>
        </div>
      ) : (
        <div
          className="flex items-center gap-3 px-4 py-3 rounded-xl"
          style={{ background: '#0d1117', border: '1px solid #21262d' }}
        >
          <Film className="w-5 h-5 text-blue-400" />
          <div className="flex-1 min-w-0">
            <p className="text-sm truncate" style={{ color: '#8b949e' }}>{videoFile.name}</p>
            <p className="text-xs" style={{ color: '#484f58' }}>
              {(videoFile.size / 1024 / 1024).toFixed(1)} MB
              {videoId && <span className="text-green-400 ml-2">✓ Uploaded (ID: {videoId.slice(0, 8)}…)</span>}
            </p>
          </div>
          {!isProcessing && !result && (
            <button
              onClick={() => {
                setVideoFile(null);
                setVideoId(null);
                setJobId(null);
                setResult(null);
                uploadMutation.reset();
                analyzeMutation.reset();
              }}
              className="text-xs hover:text-red-400 transition-colors"
              style={{ color: '#484f58' }}
            >
              Remove
            </button>
          )}
        </div>
      )}

      {/* Settings */}
      {videoFile && !result && (
        <div className="flex items-center gap-4 flex-wrap">
          <div className="flex items-center gap-2 text-xs" style={{ color: '#484f58' }}>
            <span>Max frames:</span>
            <input
              type="number"
              value={maxFrames}
              onChange={(e) => setMaxFrames(Number(e.target.value))}
              min={10}
              max={500}
              className="w-16 px-2 py-1 rounded text-xs focus:outline-none"
              style={{ background: '#0d1117', border: '1px solid #21262d', color: '#8b949e' }}
            />
          </div>
          <div className="flex items-center gap-2 text-xs" style={{ color: '#484f58' }}>
            <span>Quality gate:</span>
            <input
              type="range"
              min={0.1}
              max={0.9}
              step={0.05}
              value={qualityGate}
              onChange={(e) => setQualityGate(Number(e.target.value))}
              className="w-24 h-1.5 appearance-none rounded cursor-pointer"
              style={{ background: '#21262d' }}
            />
            <span className="font-mono">{qualityGate.toFixed(2)}</span>
          </div>

          {/* Action buttons */}
          {!videoId && !uploadMutation.isPending && (
            <button
              onClick={() => uploadMutation.mutate(videoFile)}
              className="flex items-center gap-2 px-4 py-2 rounded-lg text-sm font-medium bg-blue-600 hover:bg-blue-500 text-white transition-all"
            >
              <Upload className="w-4 h-4" />
              Upload
            </button>
          )}

          {uploadMutation.isPending && (
            <div className="flex items-center gap-2 text-sm" style={{ color: '#8b949e' }}>
              <Loader2 className="w-4 h-4 animate-spin text-blue-400" />
              Uploading…
            </div>
          )}

          {videoId && !analyzeMutation.isPending && !jobId && (
            <button
              onClick={() => analyzeMutation.mutate({ video_id: videoId })}
              className="flex items-center gap-2 px-4 py-2 rounded-lg text-sm font-medium bg-blue-600 hover:bg-blue-500 text-white transition-all"
            >
              <Play className="w-4 h-4" />
              Analyze Video
            </button>
          )}

          {(analyzeMutation.isPending || jobId) && (
            <div className="flex items-center gap-2 text-sm" style={{ color: '#8b949e' }}>
              <Loader2 className="w-4 h-4 animate-spin text-blue-400" />
              {jobId ? `Analyzing… ${jobProgress}%` : "Starting analysis…"}
            </div>
          )}
        </div>
      )}

      {/* Error states */}
      {uploadMutation.isError && (
        <div
          className="flex items-start gap-2 px-3 py-2.5 rounded-lg"
          style={{ background: 'rgba(239,68,68,0.1)', border: '1px solid rgba(239,68,68,0.2)' }}
        >
          <AlertTriangle className="w-4 h-4 text-red-400 mt-0.5" />
          <div>
            <p className="text-sm font-medium text-red-400">Upload failed</p>
            <p className="text-xs mt-0.5" style={{ color: 'rgba(252,165,165,0.7)' }}>
              {uploadMutation.error?.message}
            </p>
          </div>
        </div>
      )}

      {pollError && (
        <div
          className="flex items-start gap-2 px-3 py-2.5 rounded-lg"
          style={{ background: 'rgba(239,68,68,0.1)', border: '1px solid rgba(239,68,68,0.2)' }}
        >
          <AlertTriangle className="w-4 h-4 text-red-400 mt-0.5" />
          <div>
            <p className="text-sm font-medium text-red-400">Analysis failed</p>
            <p className="text-xs mt-0.5" style={{ color: 'rgba(252,165,165,0.7)' }}>{pollError}</p>
          </div>
        </div>
      )}

      {/* Results */}
      {result && (
        <AnalysisResults
          result={result}
          qualityGate={qualityGate}
          onClear={() => {
            setResult(null);
            setVideoFile(null);
            setVideoId(null);
          }}
        />
      )}
    </div>
  );
}

// ---------------------------------------------------------------------------
// Video list
// ---------------------------------------------------------------------------

function VideoList({ onDelete }: { onDelete: () => void }) {
  const queryClient = useQueryClient();

  const { data, isLoading } = useQuery<{ videos: VideoRecord[] }>({
    queryKey: ["videos"],
    queryFn: () => api.get("/video/videos").then((r) => r.data),
    refetchInterval: 10_000,
  });

  const deleteMutation = useMutation({
    mutationFn: (id: string) => api.delete(`/video/videos/${id}`).then((r) => r.data),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["videos"] });
      onDelete();
    },
  });

  const videos = data?.videos ?? (Array.isArray(data) ? data as VideoRecord[] : []);

  if (isLoading) {
    return (
      <div className="flex items-center justify-center py-8">
        <Loader2 className="w-5 h-5 text-blue-400 animate-spin" />
      </div>
    );
  }

  if (videos.length === 0) {
    return (
      <div className="text-center py-10" style={{ color: '#484f58' }}>
        <Film className="w-8 h-8 mx-auto mb-2 opacity-30" />
        <p className="text-sm">No videos uploaded yet</p>
      </div>
    );
  }

  return (
    <div className="space-y-2">
      {videos.map((video) => (
        <div
          key={video.id}
          className="flex items-center gap-3 px-4 py-3 rounded-xl"
          style={{ background: '#0d1117', border: '1px solid #21262d' }}
        >
          <Film className="w-4 h-4 text-blue-400 flex-shrink-0" />
          <div className="flex-1 min-w-0">
            <p className="text-sm truncate" style={{ color: '#8b949e' }}>{video.filename}</p>
            <div className="flex items-center gap-3 mt-0.5">
              <span className="text-[10px] font-mono" style={{ color: '#484f58' }}>
                {(video.file_size_bytes / 1024 / 1024).toFixed(1)} MB
              </span>
              {video.duration_s && (
                <span className="text-[10px] font-mono" style={{ color: '#484f58' }}>
                  {video.duration_s.toFixed(1)}s
                </span>
              )}
              <span className="text-[10px]" style={{ color: '#484f58' }}>
                {timeAgo(video.created_at)}
              </span>
            </div>
          </div>
          <div className="flex items-center gap-2">
            <span
              className={cn(
                "text-[10px] px-1.5 py-0.5 rounded font-medium",
                video.status === "completed"
                  ? "bg-green-500/20 text-green-400"
                  : video.status === "processing"
                  ? "bg-blue-500/20 text-blue-400"
                  : video.status === "failed"
                  ? "bg-red-500/20 text-red-400"
                  : "bg-gray-500/20 text-gray-400"
              )}
            >
              {video.status}
            </span>
            <button
              onClick={() => deleteMutation.mutate(video.id)}
              disabled={deleteMutation.isPending}
              className="p-1 rounded hover:text-red-400 transition-colors disabled:opacity-50"
              style={{ color: '#484f58' }}
            >
              <Trash2 className="w-3.5 h-3.5" />
            </button>
          </div>
        </div>
      ))}
    </div>
  );
}

// ---------------------------------------------------------------------------
// Main video analysis page
// ---------------------------------------------------------------------------

export default function VideoPage() {
  const queryClient = useQueryClient();
  const [tab, setTab] = useState<"upload" | "library">("upload");

  return (
    <div className="flex flex-col h-full overflow-y-auto">
      {/* Header */}
      <div
        className="flex items-center justify-between px-5 py-3"
        style={{ borderBottom: '1px solid #21262d' }}
      >
        <div className="flex items-center gap-2">
          <Film className="w-4 h-4 text-blue-400" />
          <h1 className="text-base font-semibold text-white">Video Analysis</h1>
        </div>
        <div className="flex items-center gap-3">
          <div
            className="flex gap-1 p-0.5 rounded-lg"
            style={{ background: '#161b22', border: '1px solid #21262d' }}
          >
            {(["upload", "library"] as const).map((t) => (
              <button
                key={t}
                onClick={() => setTab(t)}
                className="px-3 py-1 rounded text-xs font-medium capitalize transition-all"
                style={{
                  background: tab === t ? '#0d1117' : 'transparent',
                  color: tab === t ? '#e6edf3' : '#484f58',
                }}
              >
                {t}
              </button>
            ))}
          </div>
          <button
            onClick={() => queryClient.invalidateQueries({ queryKey: ["videos"] })}
            className="p-1.5 rounded transition-colors"
            style={{ color: '#484f58' }}
          >
            <RefreshCw className="w-4 h-4" />
          </button>
        </div>
      </div>

      <div className="flex-1 p-5">
        {tab === "upload" && (
          <UploadZone onAnalysisComplete={() => setTab("library")} />
        )}

        {tab === "library" && (
          <VideoList onDelete={() => {}} />
        )}
      </div>
    </div>
  );
}
