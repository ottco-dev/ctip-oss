"use client";

import React from "react";

interface ProgressBarProps {
  value: number;           // 0–100
  max?: number;            // default 100
  color?: string;          // CSS color
  height?: number;         // px, default 6
  label?: string;
  showValue?: boolean;
  animated?: boolean;
  className?: string;
}

/**
 * ProgressBar — used for GPU VRAM, training progress, annotation completion.
 */
export function ProgressBar({
  value,
  max = 100,
  color = "#3b82f6",
  height = 6,
  label,
  showValue = false,
  animated = false,
  className = "",
}: ProgressBarProps) {
  const pct = Math.min(100, Math.max(0, (value / max) * 100));

  // Color shifts: green → yellow → red based on fill %
  const dynamicColor =
    color === "auto"
      ? pct < 60 ? "#22c55e" : pct < 85 ? "#eab308" : "#ef4444"
      : color;

  return (
    <div className={`w-full ${className}`}>
      {(label || showValue) && (
        <div className="flex justify-between text-xs mb-1" style={{ color: "#8b949e" }}>
          {label && <span>{label}</span>}
          {showValue && <span>{pct.toFixed(0)}%</span>}
        </div>
      )}
      <div
        className="w-full overflow-hidden rounded-full"
        style={{ height: `${height}px`, background: "#21262d" }}
      >
        <div
          className={`h-full rounded-full transition-all duration-500 ${animated ? "animate-pulse" : ""}`}
          style={{ width: `${pct}%`, background: dynamicColor }}
        />
      </div>
    </div>
  );
}
