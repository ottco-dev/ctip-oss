"use client";

import React from "react";
import { useGpuStatus } from "@/hooks/useGpuStatus";
import { ProgressBar } from "@/components/shared/ProgressBar";

/**
 * StatusBar — fixed bottom bar showing GPU%, VRAM, and queue depth in realtime.
 * Powered by WebSocket /ws/system (2s interval).
 */
export function StatusBar() {
  const { gpu, cpuRam, wsConnected: connected } = useGpuStatus();

  const vramUsedGb = gpu?.vram_used_gb ?? 0;
  const vramTotalGb = gpu?.vram_total_gb ?? 8;
  const gpuUtil = gpu?.gpu_utilization_pct ?? 0;
  const cpuPct = (cpuRam as { cpu_percent?: number } | null)?.cpu_percent ?? 0;
  const ramPct = (cpuRam as { ram_percent?: number } | null)?.ram_percent ?? 0;

  return (
    <div
      className="h-8 flex items-center px-4 gap-6 text-[11px] flex-shrink-0"
      style={{
        background: "#0d1117",
        borderTop: "1px solid #21262d",
      }}
    >
      {/* GPU */}
      <div className="flex items-center gap-2">
        <span style={{ color: "#484f58" }}>GPU</span>
        <div className="w-16">
          <ProgressBar value={gpuUtil} color="auto" height={4} />
        </div>
        <span style={{ color: "#8b949e" }}>{gpuUtil.toFixed(0)}%</span>
      </div>

      {/* VRAM */}
      <div className="flex items-center gap-2">
        <span style={{ color: "#484f58" }}>VRAM</span>
        <div className="w-20">
          <ProgressBar value={vramUsedGb} max={vramTotalGb} color="auto" height={4} />
        </div>
        <span style={{ color: "#8b949e" }}>
          {vramUsedGb.toFixed(1)}/{vramTotalGb.toFixed(0)} GB
        </span>
      </div>

      {/* CPU */}
      <div className="flex items-center gap-2">
        <span style={{ color: "#484f58" }}>CPU</span>
        <span style={{ color: "#8b949e" }}>{cpuPct.toFixed(0)}%</span>
      </div>

      {/* RAM */}
      <div className="flex items-center gap-2">
        <span style={{ color: "#484f58" }}>RAM</span>
        <span style={{ color: "#8b949e" }}>{ramPct.toFixed(0)}%</span>
      </div>

      <div className="flex-1" />

      {/* WS connection dot */}
      <div className="flex items-center gap-1.5">
        <span
          className="w-1.5 h-1.5 rounded-full"
          style={{ background: connected ? "#22c55e" : "#ef4444" }}
        />
        <span style={{ color: "#484f58" }}>
          {connected ? "Connected" : "Disconnected"}
        </span>
      </div>
    </div>
  );
}
