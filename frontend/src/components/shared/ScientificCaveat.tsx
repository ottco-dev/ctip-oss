"use client";

import React from "react";
import { AlertTriangle } from "lucide-react";

interface ScientificCaveatProps {
  message?: string;
  className?: string;
}

/**
 * ScientificCaveat — yellow warning banner for scientifically uncertain claims.
 *
 * Used wherever maturity analysis results are displayed to clarify that
 * optical observations do NOT imply cannabinoid content, potency, or harvest
 * readiness with biochemical certainty.
 */
export function ScientificCaveat({
  message = "Maturity analysis is based on optical observation only. Trichome color and morphology indicate developmental stage, not cannabinoid concentration, potency, or any specific compound level. Harvest timing decisions require additional agronomic judgment.",
  className = "",
}: ScientificCaveatProps) {
  return (
    <div
      className={`flex gap-3 p-3 rounded-xl text-sm ${className}`}
      style={{
        background: "rgba(234,179,8,0.08)",
        border: "1px solid rgba(234,179,8,0.2)",
        color: "#ca8a04",
      }}
    >
      <AlertTriangle className="w-4 h-4 mt-0.5 flex-shrink-0" style={{ color: "#eab308" }} />
      <p className="leading-relaxed">{message}</p>
    </div>
  );
}
