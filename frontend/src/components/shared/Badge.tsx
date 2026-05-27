"use client";

import React from "react";

type BadgeVariant = "success" | "error" | "warning" | "info" | "muted" | "purple";

interface BadgeProps {
  children: React.ReactNode;
  variant?: BadgeVariant;
  className?: string;
  dot?: boolean;
}

const VARIANT_STYLES: Record<BadgeVariant, { bg: string; color: string }> = {
  success: { bg: "rgba(34,197,94,0.12)", color: "#4ade80" },
  error: { bg: "rgba(239,68,68,0.12)", color: "#f87171" },
  warning: { bg: "rgba(234,179,8,0.12)", color: "#fbbf24" },
  info: { bg: "rgba(59,130,246,0.12)", color: "#60a5fa" },
  muted: { bg: "rgba(107,114,128,0.12)", color: "#9ca3af" },
  purple: { bg: "rgba(168,85,247,0.12)", color: "#c084fc" },
};

/**
 * Badge — colored label chip used for status, confidence, and classification labels.
 */
export function Badge({ children, variant = "muted", className = "", dot = false }: BadgeProps) {
  const style = VARIANT_STYLES[variant];
  return (
    <span
      className={`inline-flex items-center gap-1.5 text-[11px] px-2 py-0.5 rounded-full font-medium ${className}`}
      style={{ background: style.bg, color: style.color }}
    >
      {dot && (
        <span
          className="w-1.5 h-1.5 rounded-full flex-shrink-0"
          style={{ background: style.color }}
        />
      )}
      {children}
    </span>
  );
}

/** Confidence badge — color depends on value. */
export function ConfidenceBadge({ value }: { value: number }) {
  const variant: BadgeVariant =
    value >= 0.75 ? "success" : value >= 0.50 ? "warning" : "error";
  return <Badge variant={variant}>{(value * 100).toFixed(0)}%</Badge>;
}
