"use client";

import React, { useCallback, useMemo, useState } from "react";
import { useQuery, useMutation, useQueryClient } from "@tanstack/react-query";
import {
  Database,
  Plus,
  RefreshCw,
  LayoutGrid,
  List,
  ChevronRight,
  X,
  Loader2,
} from "lucide-react";
import Link from "next/link";
import { api } from "@/lib/api";
import { cn, timeAgo, formatBytes } from "@/lib/utils";
import type { DatasetSummary, SampleSummary } from "@/lib/types";
import { ImageGrid, SelectionToolbar } from "@/components/datasets/ImageGrid";
import { FilterPanel, DEFAULT_FILTERS } from "@/components/datasets/FilterPanel";
import type { DatasetFilters } from "@/components/datasets/FilterPanel";
import { UploadZone } from "@/components/datasets/UploadZone";

// ---------------------------------------------------------------------------
// Create dataset modal
// ---------------------------------------------------------------------------

function CreateDatasetModal({
  onClose,
  onCreate,
}: {
  onClose: () => void;
  onCreate: (name: string, description: string) => void;
}) {
  const [name, setName] = useState("");
  const [description, setDescription] = useState("");

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/60 backdrop-blur-sm">
      <div className="w-full max-w-md bg-[var(--color-panel)] border border-[var(--color-border)] rounded-xl shadow-xl p-6">
        <div className="flex items-center justify-between mb-5">
          <h2 className="text-lg font-semibold text-white">New Dataset</h2>
          <button onClick={onClose} className="text-[var(--color-text-muted)] hover:text-white">
            <X className="w-5 h-5" />
          </button>
        </div>

        <div className="space-y-4">
          <div>
            <label className="block text-sm font-medium text-[var(--color-text-secondary)] mb-1.5">
              Name <span className="text-red-400">*</span>
            </label>
            <input
              type="text"
              value={name}
              onChange={(e) => setName(e.target.value)}
              placeholder="e.g. strain_og_kush_batch1"
              className="w-full px-3 py-2 bg-[var(--color-surface)] border border-[var(--color-border)]
                rounded-lg text-sm text-[var(--color-text-secondary)] placeholder:text-[var(--color-text-muted)]
                focus:outline-none focus:border-blue-500/60 focus:ring-1 focus:ring-blue-500/20"
            />
          </div>

          <div>
            <label className="block text-sm font-medium text-[var(--color-text-secondary)] mb-1.5">
              Description
            </label>
            <textarea
              value={description}
              onChange={(e) => setDescription(e.target.value)}
              rows={3}
              placeholder="Strain, microscope, magnification, conditions…"
              className="w-full px-3 py-2 bg-[var(--color-surface)] border border-[var(--color-border)]
                rounded-lg text-sm text-[var(--color-text-secondary)] placeholder:text-[var(--color-text-muted)]
                focus:outline-none focus:border-blue-500/60 focus:ring-1 focus:ring-blue-500/20 resize-none"
            />
          </div>

          <div className="flex gap-3 pt-2">
            <button
              onClick={onClose}
              className="flex-1 py-2 rounded-lg text-sm border border-[var(--color-border)] text-[var(--color-text-muted)]
                hover:text-[var(--color-text-secondary)] hover:border-[var(--color-border-hover)] transition-colors"
            >
              Cancel
            </button>
            <button
              onClick={() => {
                if (name.trim()) {
                  onCreate(name.trim(), description.trim());
                  onClose();
                }
              }}
              disabled={!name.trim()}
              className="flex-1 py-2 rounded-lg text-sm font-medium bg-blue-600 hover:bg-blue-500 text-white
                disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
            >
              Create Dataset
            </button>
          </div>
        </div>
      </div>
    </div>
  );
}

// ---------------------------------------------------------------------------
// Dataset card (list view)
// ---------------------------------------------------------------------------

function DatasetCard({ dataset }: { dataset: DatasetSummary }) {
  return (
    <Link
      href={`/datasets/${dataset.id}`}
      className="group flex items-center gap-4 px-4 py-3 rounded-xl bg-[var(--color-surface)]
        border border-[var(--color-border)] hover:border-[var(--color-border-hover)] transition-all"
    >
      <div className="w-10 h-10 rounded-lg bg-blue-500/10 flex items-center justify-center flex-shrink-0">
        <Database className="w-5 h-5 text-blue-400" />
      </div>

      <div className="flex-1 min-w-0">
        <div className="flex items-center gap-2">
          <h3 className="text-sm font-medium text-white truncate">{dataset.name}</h3>
          {dataset.status && (
            <span
              className={cn(
                "text-[10px] px-1.5 py-0.5 rounded font-medium uppercase tracking-wide",
                dataset.status === "ready"
                  ? "bg-green-500/20 text-green-400"
                  : "bg-yellow-500/20 text-yellow-400"
              )}
            >
              {dataset.status}
            </span>
          )}
        </div>
        <div className="flex items-center gap-3 mt-0.5 text-xs text-[var(--color-text-muted)]">
          <span>{dataset.num_samples.toLocaleString()} images</span>
          {dataset.created_at && <span>Created {timeAgo(dataset.created_at)}</span>}
        </div>
      </div>

      {/* Class distribution mini-bar */}
      {dataset.class_names && dataset.class_names.length > 0 && (
        <div className="hidden sm:flex items-center gap-1 text-xs text-[var(--color-text-muted)]">
          <span>{dataset.class_names.length} classes</span>
        </div>
      )}

      <ChevronRight className="w-4 h-4 text-[var(--color-text-muted)] group-hover:text-[var(--color-text-secondary)] transition-colors" />
    </Link>
  );
}

// ---------------------------------------------------------------------------
// Main Datasets page
// ---------------------------------------------------------------------------

export default function DatasetsPage() {
  const queryClient = useQueryClient();

  const [selectedDatasetId, setSelectedDatasetId] = useState<number | null>(null);
  const [viewMode, setViewMode] = useState<"list" | "grid">("list");
  const [showCreateModal, setShowCreateModal] = useState(false);
  const [showUpload, setShowUpload] = useState(false);
  const [filters, setFilters] = useState<DatasetFilters>(DEFAULT_FILTERS);
  const [selectedSampleIds, setSelectedSampleIds] = useState<Set<number>>(new Set());
  const [columnCount, setColumnCount] = useState(5);

  // ---------------------------------------------------------------------------
  // Queries
  // ---------------------------------------------------------------------------

  const {
    data: datasets = [],
    isLoading: datasetsLoading,
    refetch: refetchDatasets,
  } = useQuery<DatasetSummary[]>({
    queryKey: ["datasets"],
    queryFn: () => api.get("/datasets").then((r) => r.data),
    staleTime: 30_000,
  });

  const {
    data: samples = [],
    isLoading: samplesLoading,
    refetch: refetchSamples,
  } = useQuery<SampleSummary[]>({
    queryKey: ["dataset-samples", selectedDatasetId, filters.split],
    queryFn: () =>
      selectedDatasetId
        ? api
            .get(`/datasets/${selectedDatasetId}/samples`, {
              params: {
                split: filters.split !== "all" ? filters.split : undefined,
                limit: 2000,
              },
            })
            .then((r) => r.data)
        : Promise.resolve([]),
    enabled: selectedDatasetId !== null,
    staleTime: 60_000,
  });

  // ---------------------------------------------------------------------------
  // Mutations
  // ---------------------------------------------------------------------------

  const createDatasetMutation = useMutation({
    mutationFn: ({ name, description }: { name: string; description: string }) =>
      api.post("/datasets", { name, description }).then((r) => r.data),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["datasets"] });
    },
  });

  // ---------------------------------------------------------------------------
  // Filtered samples (client-side)
  // ---------------------------------------------------------------------------

  const filteredSamples = useMemo(() => {
    return samples.filter((s) => {
      if (filters.search && !s.filename.toLowerCase().includes(filters.search.toLowerCase()))
        return false;
      if (filters.reviewed === "reviewed" && !s.reviewed) return false;
      if (filters.reviewed === "unreviewed" && s.reviewed) return false;
      const q = s.quality_score ?? 1;
      if (q < filters.minQuality || q > filters.maxQuality) return false;
      if (s.num_annotations < filters.minAnnotations) return false;
      return true;
    });
  }, [samples, filters]);

  // ---------------------------------------------------------------------------
  // Selection handlers
  // ---------------------------------------------------------------------------

  const handleSelect = useCallback(
    (id: number, multiSelect: boolean) => {
      setSelectedSampleIds((prev) => {
        const next = new Set(prev);
        if (multiSelect) {
          if (next.has(id)) next.delete(id);
          else next.add(id);
        } else {
          if (next.has(id) && next.size === 1) next.clear();
          else {
            next.clear();
            next.add(id);
          }
        }
        return next;
      });
    },
    []
  );

  const handleSelectAll = useCallback(() => {
    setSelectedSampleIds(new Set(filteredSamples.map((s) => s.id)));
  }, [filteredSamples]);

  const handleDeselectAll = useCallback(() => {
    setSelectedSampleIds(new Set());
  }, []);

  const selectedDataset = datasets.find((d) => d.id === selectedDatasetId);

  // ---------------------------------------------------------------------------
  // Render
  // ---------------------------------------------------------------------------

  return (
    <div className="flex h-full">
      {/* Left panel: dataset list */}
      <div className="w-64 flex-shrink-0 border-r border-[var(--color-border)] flex flex-col">
        {/* Header */}
        <div className="flex items-center justify-between px-4 py-3 border-b border-[var(--color-border)]">
          <div className="flex items-center gap-2">
            <Database className="w-4 h-4 text-blue-400" />
            <h2 className="text-sm font-semibold text-white">Datasets</h2>
          </div>
          <button
            onClick={() => setShowCreateModal(true)}
            className="p-1 rounded hover:bg-[var(--color-surface)] text-[var(--color-text-muted)] hover:text-white transition-colors"
            title="Create dataset"
          >
            <Plus className="w-4 h-4" />
          </button>
        </div>

        {/* List */}
        <div className="flex-1 overflow-y-auto py-2 px-2 space-y-0.5">
          {datasetsLoading ? (
            <div className="flex items-center justify-center py-8">
              <Loader2 className="w-5 h-5 text-[var(--color-text-muted)] animate-spin" />
            </div>
          ) : datasets.length === 0 ? (
            <div className="text-center py-8 px-3">
              <p className="text-sm text-[var(--color-text-muted)]">No datasets yet</p>
              <button
                onClick={() => setShowCreateModal(true)}
                className="mt-2 text-xs text-blue-400 hover:text-blue-300"
              >
                Create your first dataset →
              </button>
            </div>
          ) : (
            datasets.map((dataset) => (
              <button
                key={dataset.id}
                onClick={() => {
                  setSelectedDatasetId(dataset.id);
                  setSelectedSampleIds(new Set());
                }}
                className={cn(
                  "w-full text-left px-3 py-2 rounded-lg transition-all",
                  "text-sm flex items-center gap-2.5 group",
                  selectedDatasetId === dataset.id
                    ? "bg-blue-600/20 text-white border border-blue-500/30"
                    : "text-[var(--color-text-muted)] hover:text-[var(--color-text-secondary)] hover:bg-[var(--color-surface)]"
                )}
              >
                <Database className="w-3.5 h-3.5 flex-shrink-0" />
                <div className="min-w-0 flex-1">
                  <p className="truncate font-medium">{dataset.name}</p>
                  <p className="text-[10px] text-[var(--color-text-muted)]">
                    {dataset.num_samples.toLocaleString()} images
                  </p>
                </div>
              </button>
            ))
          )}
        </div>

        {/* Footer */}
        <div className="border-t border-[var(--color-border)] px-4 py-3">
          <button
            onClick={() => refetchDatasets()}
            className="flex items-center gap-1.5 text-xs text-[var(--color-text-muted)] hover:text-[var(--color-text-secondary)] transition-colors"
          >
            <RefreshCw className="w-3 h-3" />
            Refresh
          </button>
        </div>
      </div>

      {/* Right panel: sample browser */}
      <div className="flex-1 flex flex-col min-w-0">
        {selectedDatasetId ? (
          <>
            {/* Dataset toolbar */}
            <div className="flex items-center justify-between px-5 py-3 border-b border-[var(--color-border)]">
              <div>
                <h1 className="text-base font-semibold text-white">
                  {selectedDataset?.name ?? "Dataset"}
                </h1>
                <p className="text-xs text-[var(--color-text-muted)]">
                  {samples.length.toLocaleString()} images
                  {filteredSamples.length !== samples.length &&
                    ` · ${filteredSamples.length} matching filters`}
                </p>
              </div>

              <div className="flex items-center gap-2">
                {/* Column count */}
                <div className="flex items-center gap-1 text-xs text-[var(--color-text-muted)]">
                  <span>Cols:</span>
                  {[3, 4, 5, 6, 8].map((n) => (
                    <button
                      key={n}
                      onClick={() => setColumnCount(n)}
                      className={cn(
                        "px-1.5 py-0.5 rounded transition-colors",
                        columnCount === n
                          ? "text-white bg-blue-600"
                          : "hover:text-[var(--color-text-secondary)]"
                      )}
                    >
                      {n}
                    </button>
                  ))}
                </div>

                {/* Upload toggle */}
                <button
                  onClick={() => setShowUpload((v) => !v)}
                  className={cn(
                    "flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-xs font-medium transition-all border",
                    showUpload
                      ? "bg-blue-600/20 border-blue-500/40 text-blue-300"
                      : "border-[var(--color-border)] text-[var(--color-text-muted)] hover:text-white hover:border-[var(--color-border-hover)]"
                  )}
                >
                  <Plus className="w-3.5 h-3.5" />
                  Upload
                </button>

                {/* View toggle */}
                <div className="flex border border-[var(--color-border)] rounded-lg overflow-hidden">
                  <button
                    onClick={() => setViewMode("grid")}
                    className={cn(
                      "p-1.5 transition-colors",
                      viewMode === "grid"
                        ? "bg-blue-600 text-white"
                        : "text-[var(--color-text-muted)] hover:text-white"
                    )}
                  >
                    <LayoutGrid className="w-4 h-4" />
                  </button>
                  <button
                    onClick={() => setViewMode("list")}
                    className={cn(
                      "p-1.5 transition-colors",
                      viewMode === "list"
                        ? "bg-blue-600 text-white"
                        : "text-[var(--color-text-muted)] hover:text-white"
                    )}
                  >
                    <List className="w-4 h-4" />
                  </button>
                </div>

                <button
                  onClick={() => refetchSamples()}
                  className="p-1.5 rounded text-[var(--color-text-muted)] hover:text-white hover:bg-[var(--color-surface)] transition-colors"
                >
                  <RefreshCw className="w-4 h-4" />
                </button>
              </div>
            </div>

            {/* Upload zone (collapsible) */}
            {showUpload && (
              <div className="px-5 py-4 border-b border-[var(--color-border)] bg-[var(--color-surface)]">
                <UploadZone
                  datasetId={selectedDatasetId}
                  onUploadComplete={(count) => {
                    queryClient.invalidateQueries({
                      queryKey: ["dataset-samples", selectedDatasetId],
                    });
                    queryClient.invalidateQueries({ queryKey: ["datasets"] });
                  }}
                />
              </div>
            )}

            {/* Filters */}
            <div className="px-5 py-3 border-b border-[var(--color-border)]">
              <FilterPanel
                filters={filters}
                onChange={setFilters}
                totalCount={samples.length}
                filteredCount={filteredSamples.length}
              />
            </div>

            {/* Selection toolbar */}
            {selectedSampleIds.size > 0 && (
              <div className="px-5 py-2 border-b border-[var(--color-border)]">
                <SelectionToolbar
                  count={selectedSampleIds.size}
                  total={filteredSamples.length}
                  onSelectAll={handleSelectAll}
                  onDeselectAll={handleDeselectAll}
                />
              </div>
            )}

            {/* Image grid / list */}
            <div className="flex-1 overflow-hidden px-5 py-4">
              {samplesLoading ? (
                <div className="flex items-center justify-center py-20">
                  <Loader2 className="w-6 h-6 text-[var(--color-text-muted)] animate-spin" />
                </div>
              ) : viewMode === "grid" ? (
                <ImageGrid
                  samples={filteredSamples}
                  selectedIds={selectedSampleIds}
                  onSelect={handleSelect}
                  onOpen={(s) => {
                    // TODO: open ImageViewer
                    console.log("Open:", s.filename);
                  }}
                  columnCount={columnCount}
                />
              ) : (
                /* List view */
                <div className="space-y-1 overflow-y-auto max-h-[calc(100vh-280px)] pr-1">
                  {filteredSamples.length === 0 ? (
                    <div className="text-center py-12 text-[var(--color-text-muted)]">
                      No samples match the current filters
                    </div>
                  ) : (
                    filteredSamples.map((sample) => (
                      <div
                        key={sample.id}
                        onClick={() => handleSelect(sample.id, false)}
                        className={cn(
                          "flex items-center gap-3 px-3 py-2 rounded-lg cursor-pointer transition-all border",
                          selectedSampleIds.has(sample.id)
                            ? "border-blue-500/40 bg-blue-500/5"
                            : "border-transparent hover:border-[var(--color-border)] hover:bg-[var(--color-surface)]"
                        )}
                      >
                        <span className="text-xs font-mono text-[var(--color-text-muted)] w-6">
                          {sample.split.charAt(0).toUpperCase()}
                        </span>
                        <span className="flex-1 text-sm text-[var(--color-text-secondary)] truncate">
                          {sample.filename}
                        </span>
                        <span className="text-xs text-[var(--color-text-muted)]">
                          {sample.num_annotations} ann.
                        </span>
                        {sample.quality_score !== undefined && (
                          <span className="text-xs font-mono text-green-400">
                            {Math.round((sample.quality_score ?? 0) * 100)}%
                          </span>
                        )}
                      </div>
                    ))
                  )}
                </div>
              )}
            </div>
          </>
        ) : (
          /* No dataset selected */
          <div className="flex-1 flex flex-col items-center justify-center gap-4 text-[var(--color-text-muted)]">
            <Database className="w-12 h-12 opacity-30" />
            <div className="text-center">
              <p className="text-base font-medium">Select a dataset</p>
              <p className="text-sm mt-1">or create a new one to get started</p>
            </div>
            <button
              onClick={() => setShowCreateModal(true)}
              className="flex items-center gap-2 px-4 py-2 rounded-lg bg-blue-600 hover:bg-blue-500 text-white text-sm font-medium transition-colors"
            >
              <Plus className="w-4 h-4" />
              Create Dataset
            </button>
          </div>
        )}
      </div>

      {/* Create modal */}
      {showCreateModal && (
        <CreateDatasetModal
          onClose={() => setShowCreateModal(false)}
          onCreate={(name, description) =>
            createDatasetMutation.mutate({ name, description })
          }
        />
      )}
    </div>
  );
}
