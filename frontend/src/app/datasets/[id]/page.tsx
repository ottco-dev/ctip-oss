"use client";

/**
 * Dataset Detail Page — image grid with filter, multi-select, and zoom/pan viewer.
 *
 * Uses the shared <ImageViewer> component for the lightbox so users get
 * full zoom-to-cursor, pan, and annotation overlay support (annotations
 * will be wired once the sample-annotation endpoint is available).
 */

import React, { useState } from "react";
import { useParams, useRouter } from "next/navigation";
import { useQuery } from "@tanstack/react-query";
import { X, ChevronLeft, ChevronRight } from "lucide-react";
import { api } from "@/lib/api";
import { ImageViewer } from "@/components/shared/ImageViewer";

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

interface Sample {
  id: string;
  path: string;
  filename: string;
  quality_score?: number;
  annotation_count?: number;
  reviewed?: boolean;
  split?: string;
  metadata?: Record<string, unknown>;
}

interface Dataset {
  id: string;
  name: string;
  path: string;
  num_samples: number;
  created_at: string;
}

// ---------------------------------------------------------------------------
// Lightbox (wraps shared ImageViewer)
// ---------------------------------------------------------------------------

function SampleLightbox({
  sample,
  datasetId,
  allSamples,
  onClose,
}: {
  sample: Sample;
  datasetId: string;
  allSamples: Sample[];
  onClose: () => void;
}) {
  const [current, setCurrent] = useState(sample);

  const idx = allSamples.findIndex((s) => s.id === current.id);
  const hasPrev = idx > 0;
  const hasNext = idx < allSamples.length - 1;

  const navigate = (dir: -1 | 1) => {
    const next = allSamples[idx + dir];
    if (next) setCurrent(next);
  };

  // Build the image URL via the proxy so no auth headers are needed in <img>
  const imageUrl = `/api/v1/datasets/${datasetId}/image/${encodeURIComponent(current.filename)}`;

  // Keyboard navigation
  React.useEffect(() => {
    function onKey(e: KeyboardEvent) {
      if (e.key === "Escape") onClose();
      if (e.key === "ArrowLeft" && hasPrev) navigate(-1);
      if (e.key === "ArrowRight" && hasNext) navigate(1);
    }
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [hasPrev, hasNext, idx]); // eslint-disable-line react-hooks/exhaustive-deps

  return (
    <div className="fixed inset-0 z-50 flex flex-col" style={{ background: "rgba(0,0,0,0.95)" }}>
      {/* Top bar */}
      <div
        className="flex items-center gap-3 px-4 py-2.5 flex-shrink-0"
        style={{ borderBottom: "1px solid #21262d" }}
      >
        <span className="text-sm font-medium text-white truncate flex-1">{current.filename}</span>
        <div className="flex items-center gap-3 text-xs" style={{ color: "#484f58" }}>
          {current.quality_score !== undefined && (
            <span>Focus: <span className="text-white font-mono">{current.quality_score.toFixed(0)}</span></span>
          )}
          {current.annotation_count !== undefined && (
            <span>Annotations: <span className="text-white font-mono">{current.annotation_count}</span></span>
          )}
          <span
            className={`px-2 py-0.5 rounded-full text-[11px] ${
              current.split === "train"
                ? "bg-blue-500/20 text-blue-300"
                : current.split === "val"
                ? "bg-purple-500/20 text-purple-300"
                : current.split === "test"
                ? "bg-orange-500/20 text-orange-300"
                : "bg-zinc-800 text-zinc-400"
            }`}
          >
            {current.split ?? "unassigned"}
          </span>
          {current.reviewed && (
            <span className="px-2 py-0.5 rounded-full text-[11px] bg-green-500/20 text-green-300">
              ✓ reviewed
            </span>
          )}
        </div>
        <button
          onClick={onClose}
          className="p-1 rounded transition-colors hover:bg-white/10"
          style={{ color: "#8b949e" }}
          title="Close (Esc)"
        >
          <X className="w-4 h-4" />
        </button>
      </div>

      {/* Image area */}
      <div className="flex-1 min-h-0 relative">
        <ImageViewer
          src={imageUrl}
          alt={current.filename}
          className="w-full h-full"
          showLabels
          showConfidence
        />

        {/* Prev / Next navigation */}
        {hasPrev && (
          <button
            onClick={() => navigate(-1)}
            className="absolute left-3 top-1/2 -translate-y-1/2 p-2 rounded-full transition-colors"
            style={{ background: "rgba(0,0,0,0.6)", color: "#8b949e" }}
            title="Previous (←)"
          >
            <ChevronLeft className="w-5 h-5" />
          </button>
        )}
        {hasNext && (
          <button
            onClick={() => navigate(1)}
            className="absolute right-3 top-1/2 -translate-y-1/2 p-2 rounded-full transition-colors"
            style={{ background: "rgba(0,0,0,0.6)", color: "#8b949e" }}
            title="Next (→)"
          >
            <ChevronRight className="w-5 h-5" />
          </button>
        )}
      </div>

      {/* Bottom strip: position indicator */}
      <div
        className="flex items-center justify-center px-4 py-2 flex-shrink-0 text-xs"
        style={{ borderTop: "1px solid #21262d", color: "#484f58" }}
      >
        {idx + 1} / {allSamples.length}
      </div>
    </div>
  );
}

// ---------------------------------------------------------------------------
// Sample thumbnail card
// ---------------------------------------------------------------------------

function SampleThumbnail({
  sample,
  datasetId,
  onClick,
  selected,
  onSelect,
}: {
  sample: Sample;
  datasetId: string;
  onClick: () => void;
  selected: boolean;
  onSelect: (id: string, checked: boolean) => void;
}) {
  const imageUrl = `/api/v1/datasets/${datasetId}/image/${encodeURIComponent(sample.filename)}`;

  return (
    <div
      className={`relative group rounded-lg overflow-hidden cursor-pointer transition-all ${
        selected
          ? "ring-2 ring-blue-500 ring-offset-1 ring-offset-zinc-950"
          : "hover:ring-1 hover:ring-zinc-600 ring-offset-zinc-950"
      }`}
      style={{ background: "#0d1117", border: "1px solid #21262d" }}
      onClick={onClick}
    >
      {/* eslint-disable-next-line @next/next/no-img-element */}
      <img
        src={imageUrl}
        alt={sample.filename}
        className="w-full aspect-square object-cover"
        loading="lazy"
      />

      {/* Hover overlay */}
      <div className="absolute inset-0 bg-black/0 group-hover:bg-black/20 transition-colors" />

      {/* Select checkbox */}
      <div
        className="absolute top-1.5 left-1.5"
        onClick={(e) => {
          e.stopPropagation();
          onSelect(sample.id, !selected);
        }}
      >
        <input
          type="checkbox"
          checked={selected}
          onChange={(e) => onSelect(sample.id, e.target.checked)}
          className="w-4 h-4 accent-blue-500"
        />
      </div>

      {/* Review badge */}
      {sample.reviewed && (
        <div
          className="absolute top-1.5 right-1.5 text-white text-[10px] px-1.5 py-0.5 rounded-full font-medium"
          style={{ background: "rgba(34,197,94,0.8)" }}
        >
          ✓
        </div>
      )}

      {/* Footer */}
      <div className="px-2 py-1.5" style={{ borderTop: "1px solid #21262d" }}>
        <p className="text-xs truncate" style={{ color: "#8b949e" }}>
          {sample.filename}
        </p>
        {sample.quality_score !== undefined && (
          <div className="flex items-center gap-1 mt-1">
            <div
              className="h-1 flex-1 rounded-full overflow-hidden"
              style={{ background: "#21262d" }}
            >
              <div
                className="h-full rounded-full"
                style={{
                  width: `${Math.min(100, sample.quality_score)}%`,
                  background:
                    sample.quality_score >= 70
                      ? "#22c55e"
                      : sample.quality_score >= 40
                      ? "#f59e0b"
                      : "#ef4444",
                }}
              />
            </div>
            <span className="text-[10px] font-mono" style={{ color: "#484f58" }}>
              {sample.quality_score.toFixed(0)}
            </span>
          </div>
        )}
      </div>
    </div>
  );
}

// ---------------------------------------------------------------------------
// Page
// ---------------------------------------------------------------------------

export default function DatasetDetailPage() {
  const { id } = useParams<{ id: string }>();
  const router = useRouter();

  const [search, setSearch] = useState("");
  const [selectedSplit, setSelectedSplit] = useState<string>("all");
  const [selectedIds, setSelectedIds] = useState<Set<string>>(new Set());
  const [viewerSample, setViewerSample] = useState<Sample | null>(null);
  const [columns, setColumns] = useState(4);

  const { data: dataset } = useQuery({
    queryKey: ["dataset", id],
    queryFn: () => api.get(`/datasets/${id}`).then((r) => r.data as Dataset),
    enabled: !!id,
  });

  const { data: samplesData, isLoading } = useQuery({
    queryKey: ["samples", id],
    queryFn: () => api.get(`/datasets/${id}/samples?limit=500`).then((r) => r.data),
    enabled: !!id,
  });

  const samples: Sample[] = samplesData?.samples ?? [];

  const filtered = samples.filter((s) => {
    if (search && !s.filename.toLowerCase().includes(search.toLowerCase())) return false;
    if (selectedSplit !== "all" && s.split !== selectedSplit) return false;
    return true;
  });

  function handleSelect(sampleId: string, checked: boolean) {
    setSelectedIds((prev) => {
      const next = new Set(prev);
      if (checked) next.add(sampleId);
      else next.delete(sampleId);
      return next;
    });
  }

  function handleSelectAll() {
    if (selectedIds.size === filtered.length) {
      setSelectedIds(new Set());
    } else {
      setSelectedIds(new Set(filtered.map((s) => s.id)));
    }
  }

  return (
    <div className="h-full flex flex-col">
      {/* Header */}
      <div
        className="px-6 py-4 flex items-center gap-4 flex-shrink-0"
        style={{ borderBottom: "1px solid #21262d" }}
      >
        <button
          onClick={() => router.push("/datasets")}
          className="text-sm transition-colors"
          style={{ color: "#484f58" }}
        >
          ← Datasets
        </button>
        <div className="flex-1 min-w-0">
          <h1 className="text-xl font-bold text-white truncate">
            {dataset?.name ?? "Loading…"}
          </h1>
          <p className="text-xs mt-0.5" style={{ color: "#484f58" }}>
            {dataset?.num_samples ?? 0} images
          </p>
        </div>

        {/* Column density selector */}
        <div className="flex items-center gap-1">
          {[3, 4, 5, 6].map((n) => (
            <button
              key={n}
              onClick={() => setColumns(n)}
              className="w-7 h-7 rounded text-xs font-mono transition-colors"
              style={
                columns === n
                  ? { background: "#21262d", color: "#e6edf3" }
                  : { background: "transparent", color: "#484f58" }
              }
            >
              {n}
            </button>
          ))}
        </div>
      </div>

      {/* Toolbar */}
      <div
        className="px-6 py-2.5 flex items-center gap-3 flex-shrink-0"
        style={{ borderBottom: "1px solid #21262d" }}
      >
        <input
          type="text"
          placeholder="Search by filename…"
          value={search}
          onChange={(e) => setSearch(e.target.value)}
          className="px-3 py-1.5 text-sm rounded-lg focus:outline-none w-56"
          style={{
            background: "#0d1117",
            border: "1px solid #21262d",
            color: "#e6edf3",
          }}
        />

        {/* Split filter pills */}
        <div className="flex gap-1">
          {["all", "train", "val", "test"].map((split) => (
            <button
              key={split}
              onClick={() => setSelectedSplit(split)}
              className="px-2.5 py-1 rounded text-xs font-medium transition-colors"
              style={
                selectedSplit === split
                  ? { background: "#21262d", color: "#e6edf3" }
                  : { background: "transparent", color: "#484f58" }
              }
            >
              {split}
            </button>
          ))}
        </div>

        <div className="ml-auto text-xs" style={{ color: "#484f58" }}>
          {filtered.length} / {samples.length}
          {selectedIds.size > 0 && (
            <span className="ml-2 text-blue-400">{selectedIds.size} selected</span>
          )}
        </div>

        {filtered.length > 0 && (
          <button
            onClick={handleSelectAll}
            className="text-xs transition-colors px-2 py-1 rounded"
            style={{ color: "#484f58", border: "1px solid #21262d" }}
          >
            {selectedIds.size === filtered.length ? "Deselect all" : "Select all"}
          </button>
        )}
      </div>

      {/* Grid */}
      <div className="flex-1 overflow-auto p-6">
        {isLoading && (
          <div className="text-center py-12 text-sm" style={{ color: "#484f58" }}>
            Loading images…
          </div>
        )}

        {!isLoading && filtered.length === 0 && (
          <div className="text-center py-12 text-sm" style={{ color: "#484f58" }}>
            {search || selectedSplit !== "all"
              ? "No images match the current filters."
              : "No images in this dataset."}
          </div>
        )}

        <div
          className="grid gap-3"
          style={{ gridTemplateColumns: `repeat(${columns}, minmax(0, 1fr))` }}
        >
          {filtered.map((sample) => (
            <SampleThumbnail
              key={sample.id}
              sample={sample}
              datasetId={id}
              selected={selectedIds.has(sample.id)}
              onSelect={handleSelect}
              onClick={() => setViewerSample(sample)}
            />
          ))}
        </div>
      </div>

      {/* Lightbox with shared ImageViewer */}
      {viewerSample && (
        <SampleLightbox
          sample={viewerSample}
          datasetId={id}
          allSamples={filtered}
          onClose={() => setViewerSample(null)}
        />
      )}
    </div>
  );
}
