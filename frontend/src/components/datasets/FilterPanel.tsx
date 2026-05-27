"use client";

import React, { useCallback } from "react";
import { SlidersHorizontal, X, Search } from "lucide-react";
import { cn } from "@/lib/utils";

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

export interface DatasetFilters {
  split: "all" | "train" | "val" | "test";
  minQuality: number;
  maxQuality: number;
  reviewed: "all" | "reviewed" | "unreviewed";
  search: string;
  minAnnotations: number;
}

export const DEFAULT_FILTERS: DatasetFilters = {
  split: "all",
  minQuality: 0,
  maxQuality: 1,
  reviewed: "all",
  search: "",
  minAnnotations: 0,
};

interface FilterPanelProps {
  filters: DatasetFilters;
  onChange: (filters: DatasetFilters) => void;
  totalCount: number;
  filteredCount: number;
}

// ---------------------------------------------------------------------------
// Pill toggle button
// ---------------------------------------------------------------------------

function PillToggle<T extends string>({
  options,
  value,
  onChange,
  label,
}: {
  options: { label: string; value: T }[];
  value: T;
  onChange: (value: T) => void;
  label: string;
}) {
  return (
    <div className="space-y-1.5">
      <label className="text-xs font-medium text-[var(--color-text-muted)] uppercase tracking-wider">
        {label}
      </label>
      <div className="flex gap-1 flex-wrap">
        {options.map((opt) => (
          <button
            key={opt.value}
            onClick={() => onChange(opt.value)}
            className={cn(
              "px-2.5 py-1 rounded text-xs font-medium transition-all",
              value === opt.value
                ? "bg-blue-600 text-white"
                : "bg-[var(--color-panel)] text-[var(--color-text-muted)] hover:text-[var(--color-text-secondary)] border border-[var(--color-border)]"
            )}
          >
            {opt.label}
          </button>
        ))}
      </div>
    </div>
  );
}

// ---------------------------------------------------------------------------
// Range slider
// ---------------------------------------------------------------------------

function QualitySlider({
  min,
  max,
  onChange,
}: {
  min: number;
  max: number;
  onChange: (min: number, max: number) => void;
}) {
  return (
    <div className="space-y-1.5">
      <div className="flex items-center justify-between">
        <label className="text-xs font-medium text-[var(--color-text-muted)] uppercase tracking-wider">
          Quality Range
        </label>
        <span className="text-xs font-mono text-[var(--color-text-secondary)]">
          {Math.round(min * 100)}% – {Math.round(max * 100)}%
        </span>
      </div>
      <div className="space-y-1">
        <input
          type="range"
          min={0}
          max={100}
          step={5}
          value={Math.round(min * 100)}
          onChange={(e) => onChange(Number(e.target.value) / 100, max)}
          className="w-full h-1.5 appearance-none bg-[var(--color-border)] rounded-full
            [&::-webkit-slider-thumb]:appearance-none [&::-webkit-slider-thumb]:w-3.5
            [&::-webkit-slider-thumb]:h-3.5 [&::-webkit-slider-thumb]:rounded-full
            [&::-webkit-slider-thumb]:bg-blue-500 [&::-webkit-slider-thumb]:cursor-pointer"
        />
        <input
          type="range"
          min={0}
          max={100}
          step={5}
          value={Math.round(max * 100)}
          onChange={(e) => onChange(min, Number(e.target.value) / 100)}
          className="w-full h-1.5 appearance-none bg-[var(--color-border)] rounded-full
            [&::-webkit-slider-thumb]:appearance-none [&::-webkit-slider-thumb]:w-3.5
            [&::-webkit-slider-thumb]:h-3.5 [&::-webkit-slider-thumb]:rounded-full
            [&::-webkit-slider-thumb]:bg-blue-500 [&::-webkit-slider-thumb]:cursor-pointer"
        />
      </div>
    </div>
  );
}

// ---------------------------------------------------------------------------
// FilterPanel
// ---------------------------------------------------------------------------

export function FilterPanel({
  filters,
  onChange,
  totalCount,
  filteredCount,
}: FilterPanelProps) {
  const update = useCallback(
    (patch: Partial<DatasetFilters>) => onChange({ ...filters, ...patch }),
    [filters, onChange]
  );

  const hasActiveFilters =
    filters.split !== "all" ||
    filters.reviewed !== "all" ||
    filters.minQuality > 0 ||
    filters.maxQuality < 1 ||
    filters.search !== "" ||
    filters.minAnnotations > 0;

  return (
    <div className="flex flex-col gap-4 w-full">
      {/* Top row: search + reset */}
      <div className="flex items-center gap-2">
        {/* Search */}
        <div className="relative flex-1">
          <Search className="absolute left-2.5 top-1/2 -translate-y-1/2 w-3.5 h-3.5 text-[var(--color-text-muted)]" />
          <input
            type="text"
            placeholder="Filter by filename…"
            value={filters.search}
            onChange={(e) => update({ search: e.target.value })}
            className="w-full pl-8 pr-3 py-1.5 text-sm bg-[var(--color-surface)] border border-[var(--color-border)]
              rounded-lg text-[var(--color-text-secondary)] placeholder:text-[var(--color-text-muted)]
              focus:outline-none focus:border-blue-500/60 focus:ring-1 focus:ring-blue-500/20"
          />
          {filters.search && (
            <button
              onClick={() => update({ search: "" })}
              className="absolute right-2.5 top-1/2 -translate-y-1/2 text-[var(--color-text-muted)] hover:text-white"
            >
              <X className="w-3 h-3" />
            </button>
          )}
        </div>

        {/* Filter icon + count */}
        <div className="flex items-center gap-1.5 text-xs text-[var(--color-text-muted)]">
          <SlidersHorizontal className="w-3.5 h-3.5" />
          <span>
            <span className="text-white font-medium">{filteredCount}</span>
            {" / "}
            {totalCount}
          </span>
        </div>

        {/* Reset */}
        {hasActiveFilters && (
          <button
            onClick={() => onChange(DEFAULT_FILTERS)}
            className="flex items-center gap-1 px-2 py-1.5 rounded text-xs text-red-400 hover:text-red-300
              bg-red-500/10 hover:bg-red-500/15 border border-red-500/20 transition-colors"
          >
            <X className="w-3 h-3" />
            Reset
          </button>
        )}
      </div>

      {/* Filter controls row */}
      <div className="flex flex-wrap gap-6">
        {/* Split */}
        <PillToggle
          label="Split"
          value={filters.split}
          onChange={(v) => update({ split: v })}
          options={[
            { label: "All", value: "all" },
            { label: "Train", value: "train" },
            { label: "Val", value: "val" },
            { label: "Test", value: "test" },
          ]}
        />

        {/* Review status */}
        <PillToggle
          label="Review"
          value={filters.reviewed}
          onChange={(v) => update({ reviewed: v })}
          options={[
            { label: "All", value: "all" },
            { label: "Reviewed", value: "reviewed" },
            { label: "Pending", value: "unreviewed" },
          ]}
        />

        {/* Min annotations */}
        <div className="space-y-1.5">
          <label className="text-xs font-medium text-[var(--color-text-muted)] uppercase tracking-wider">
            Min Annotations
          </label>
          <div className="flex gap-1">
            {[0, 1, 5, 10].map((v) => (
              <button
                key={v}
                onClick={() => update({ minAnnotations: v })}
                className={cn(
                  "px-2.5 py-1 rounded text-xs font-medium transition-all",
                  filters.minAnnotations === v
                    ? "bg-blue-600 text-white"
                    : "bg-[var(--color-panel)] text-[var(--color-text-muted)] hover:text-[var(--color-text-secondary)] border border-[var(--color-border)]"
                )}
              >
                {v === 0 ? "Any" : `≥ ${v}`}
              </button>
            ))}
          </div>
        </div>

        {/* Quality range */}
        <div className="min-w-[160px]">
          <QualitySlider
            min={filters.minQuality}
            max={filters.maxQuality}
            onChange={(min, max) => update({ minQuality: min, maxQuality: max })}
          />
        </div>
      </div>
    </div>
  );
}
