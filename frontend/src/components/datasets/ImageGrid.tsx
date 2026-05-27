"use client";

import React, { useCallback, useMemo, useRef, useState, useEffect } from "react";
import { FixedSizeGrid } from "react-window";
import type { GridChildComponentProps } from "react-window";
import { CheckCircle2, AlertCircle, Eye, Tag } from "lucide-react";
import { cn, formatConfidence, getMaturityColor, getMaturityLabel } from "@/lib/utils";
import type { SampleSummary } from "@/lib/types";

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

interface ImageGridProps {
  samples: SampleSummary[];
  selectedIds: Set<number>;
  onSelect: (id: number, multiSelect: boolean) => void;
  onOpen: (sample: SampleSummary) => void;
  columnCount?: number;
  itemPadding?: number;
}

interface CellData {
  samples: SampleSummary[];
  columnCount: number;
  itemPadding: number;
  selectedIds: Set<number>;
  onSelect: (id: number, multiSelect: boolean) => void;
  onOpen: (sample: SampleSummary) => void;
}

// ---------------------------------------------------------------------------
// Image card (rendered inside virtualized grid)
// ---------------------------------------------------------------------------

function SampleCard({
  sample,
  isSelected,
  onSelect,
  onOpen,
}: {
  sample: SampleSummary;
  isSelected: boolean;
  onSelect: (id: number, multiSelect: boolean) => void;
  onOpen: (sample: SampleSummary) => void;
}) {
  const [imageError, setImageError] = useState(false);
  const [imageLoaded, setImageLoaded] = useState(false);

  const qualityColor = useMemo(() => {
    const q = sample.quality_score ?? 0;
    if (q >= 0.75) return "text-green-400";
    if (q >= 0.50) return "text-yellow-400";
    return "text-red-400";
  }, [sample.quality_score]);

  const handleClick = useCallback(
    (e: React.MouseEvent) => {
      e.stopPropagation();
      onSelect(sample.id, e.ctrlKey || e.metaKey || e.shiftKey);
    },
    [sample.id, onSelect]
  );

  const handleDoubleClick = useCallback(
    (e: React.MouseEvent) => {
      e.stopPropagation();
      onOpen(sample);
    },
    [sample, onOpen]
  );

  // Build image URL — served by FastAPI /datasets/{id}/samples static mount
  const imageUrl = `/api/v1/datasets/${sample.dataset_id}/image/${encodeURIComponent(
    sample.filename
  )}`;

  return (
    <div
      className={cn(
        "group relative rounded-lg overflow-hidden cursor-pointer",
        "border transition-all duration-150 select-none",
        isSelected
          ? "border-blue-500 ring-2 ring-blue-500/40 bg-blue-500/5"
          : "border-[var(--color-border)] hover:border-[var(--color-border-hover)] bg-[var(--color-surface)]"
      )}
      onClick={handleClick}
      onDoubleClick={handleDoubleClick}
      role="button"
      tabIndex={0}
      aria-selected={isSelected}
      aria-label={`Sample ${sample.filename}`}
      onKeyDown={(e) => {
        if (e.key === "Enter" || e.key === " ") handleClick(e as unknown as React.MouseEvent);
      }}
    >
      {/* Image area */}
      <div className="relative w-full aspect-square bg-[var(--color-panel)]">
        {!imageError ? (
          <>
            {!imageLoaded && (
              <div className="absolute inset-0 flex items-center justify-center">
                <div className="w-6 h-6 rounded-full border-2 border-[var(--color-border)] border-t-blue-400 animate-spin" />
              </div>
            )}
            <img
              src={imageUrl}
              alt={sample.filename}
              className={cn(
                "w-full h-full object-cover transition-opacity duration-200",
                imageLoaded ? "opacity-100" : "opacity-0"
              )}
              loading="lazy"
              onLoad={() => setImageLoaded(true)}
              onError={() => setImageError(true)}
            />
          </>
        ) : (
          <div className="absolute inset-0 flex flex-col items-center justify-center gap-1 text-[var(--color-text-muted)]">
            <AlertCircle className="w-6 h-6" />
            <span className="text-[10px]">No preview</span>
          </div>
        )}

        {/* Selection overlay */}
        {isSelected && (
          <div className="absolute top-1.5 left-1.5">
            <CheckCircle2 className="w-5 h-5 text-blue-400 drop-shadow" />
          </div>
        )}

        {/* Reviewed badge */}
        {sample.reviewed && (
          <div className="absolute top-1.5 right-1.5">
            <div className="w-2 h-2 rounded-full bg-green-400 shadow-[0_0_6px_rgba(74,222,128,0.6)]" />
          </div>
        )}

        {/* Hover actions */}
        <div className="absolute inset-0 bg-black/60 opacity-0 group-hover:opacity-100 transition-opacity flex items-center justify-center gap-2">
          <button
            className="flex items-center gap-1 px-2 py-1 rounded bg-blue-600 hover:bg-blue-500 text-white text-xs font-medium"
            onClick={(e) => {
              e.stopPropagation();
              onOpen(sample);
            }}
          >
            <Eye className="w-3 h-3" />
            View
          </button>
        </div>
      </div>

      {/* Footer info */}
      <div className="px-2 py-1.5 space-y-0.5">
        <p
          className="text-[11px] text-[var(--color-text-secondary)] truncate"
          title={sample.filename}
        >
          {sample.filename}
        </p>

        <div className="flex items-center justify-between">
          {/* Quality score */}
          {sample.quality_score !== undefined && (
            <span className={cn("text-[10px] font-mono", qualityColor)}>
              Q: {formatConfidence(sample.quality_score)}
            </span>
          )}

          {/* Annotation count */}
          {sample.num_annotations > 0 && (
            <div className="flex items-center gap-0.5 text-[10px] text-[var(--color-text-muted)]">
              <Tag className="w-2.5 h-2.5" />
              <span>{sample.num_annotations}</span>
            </div>
          )}

          {/* Split badge */}
          <span
            className={cn(
              "text-[9px] px-1 rounded font-medium uppercase tracking-wide",
              sample.split === "train"
                ? "bg-blue-500/20 text-blue-400"
                : sample.split === "val"
                ? "bg-purple-500/20 text-purple-400"
                : "bg-orange-500/20 text-orange-400"
            )}
          >
            {sample.split}
          </span>
        </div>
      </div>
    </div>
  );
}

// ---------------------------------------------------------------------------
// Virtualized grid cell renderer
// ---------------------------------------------------------------------------

function GridCell({
  columnIndex,
  rowIndex,
  style,
  data,
}: GridChildComponentProps<CellData>) {
  const { samples, columnCount, itemPadding, selectedIds, onSelect, onOpen } = data;
  const index = rowIndex * columnCount + columnIndex;

  if (index >= samples.length) {
    return <div style={style} />;
  }

  const sample = samples[index];

  return (
    <div
      style={{
        ...style,
        paddingLeft: columnIndex === 0 ? 0 : itemPadding / 2,
        paddingRight: columnIndex === columnCount - 1 ? 0 : itemPadding / 2,
        paddingTop: rowIndex === 0 ? 0 : itemPadding / 2,
        paddingBottom: itemPadding / 2,
      }}
    >
      <SampleCard
        sample={sample}
        isSelected={selectedIds.has(sample.id)}
        onSelect={onSelect}
        onOpen={onOpen}
      />
    </div>
  );
}

// ---------------------------------------------------------------------------
// Main ImageGrid component
// ---------------------------------------------------------------------------

export function ImageGrid({
  samples,
  selectedIds,
  onSelect,
  onOpen,
  columnCount = 5,
  itemPadding = 8,
}: ImageGridProps) {
  const containerRef = useRef<HTMLDivElement>(null);
  const [containerWidth, setContainerWidth] = useState(0);
  const [containerHeight, setContainerHeight] = useState(600);

  // Responsive container measurement
  useEffect(() => {
    if (!containerRef.current) return;

    const observer = new ResizeObserver((entries) => {
      for (const entry of entries) {
        setContainerWidth(entry.contentRect.width);
        // Grid height: 60% of viewport height, min 400px
        setContainerHeight(Math.max(400, window.innerHeight * 0.62));
      }
    });

    observer.observe(containerRef.current);
    return () => observer.disconnect();
  }, []);

  const cellSize = useMemo(() => {
    if (containerWidth === 0) return 200;
    return Math.floor((containerWidth - itemPadding * (columnCount - 1)) / columnCount);
  }, [containerWidth, columnCount, itemPadding]);

  // Card total height = image (square) + footer (~46px)
  const cellHeight = cellSize + 46;

  const rowCount = Math.ceil(samples.length / columnCount);

  const itemData = useMemo<CellData>(
    () => ({
      samples,
      columnCount,
      itemPadding,
      selectedIds,
      onSelect,
      onOpen,
    }),
    [samples, columnCount, itemPadding, selectedIds, onSelect, onOpen]
  );

  if (samples.length === 0) {
    return (
      <div className="flex flex-col items-center justify-center py-20 text-[var(--color-text-muted)]">
        <p className="text-lg font-medium mb-1">No images found</p>
        <p className="text-sm">Upload images or adjust your filters</p>
      </div>
    );
  }

  return (
    <div ref={containerRef} className="w-full">
      {containerWidth > 0 && (
        <FixedSizeGrid
          columnCount={columnCount}
          columnWidth={cellSize}
          rowCount={rowCount}
          rowHeight={cellHeight}
          width={containerWidth}
          height={containerHeight}
          itemData={itemData}
          overscanRowCount={3}
          style={{ outline: "none" }}
        >
          {GridCell}
        </FixedSizeGrid>
      )}
    </div>
  );
}

// ---------------------------------------------------------------------------
// Selection toolbar (shown when items are selected)
// ---------------------------------------------------------------------------

export function SelectionToolbar({
  count,
  total,
  onSelectAll,
  onDeselectAll,
  onDeleteSelected,
  onMoveToSplit,
}: {
  count: number;
  total: number;
  onSelectAll: () => void;
  onDeselectAll: () => void;
  onDeleteSelected?: () => void;
  onMoveToSplit?: (split: "train" | "val" | "test") => void;
}) {
  if (count === 0) return null;

  return (
    <div className="flex items-center gap-3 px-4 py-2 bg-blue-500/10 border border-blue-500/30 rounded-lg text-sm">
      <CheckCircle2 className="w-4 h-4 text-blue-400" />
      <span className="text-[var(--color-text-secondary)]">
        <span className="text-white font-medium">{count}</span>
        {" of "}
        <span className="text-white font-medium">{total}</span>
        {" selected"}
      </span>

      <div className="flex-1" />

      <button
        onClick={onSelectAll}
        className="text-blue-400 hover:text-blue-300 transition-colors"
      >
        Select all
      </button>

      <button
        onClick={onDeselectAll}
        className="text-[var(--color-text-muted)] hover:text-[var(--color-text-secondary)] transition-colors"
      >
        Deselect
      </button>

      {onMoveToSplit && (
        <div className="flex items-center gap-1">
          <span className="text-[var(--color-text-muted)]">Move to:</span>
          {(["train", "val", "test"] as const).map((split) => (
            <button
              key={split}
              onClick={() => onMoveToSplit(split)}
              className="px-2 py-0.5 rounded text-xs font-medium bg-[var(--color-panel)] hover:bg-[var(--color-surface)] border border-[var(--color-border)] text-[var(--color-text-secondary)] hover:text-white transition-colors uppercase"
            >
              {split}
            </button>
          ))}
        </div>
      )}

      {onDeleteSelected && (
        <button
          onClick={onDeleteSelected}
          className="text-red-400 hover:text-red-300 transition-colors"
        >
          Delete
        </button>
      )}
    </div>
  );
}
