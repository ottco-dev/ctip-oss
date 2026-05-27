"use client";

import React from "react";
import { Wifi, WifiOff, Loader2 } from "lucide-react";

interface WebSocketStatusProps {
  connected: boolean;
  reconnectAttempts?: number;
  label?: string;
  className?: string;
}

/**
 * WebSocketStatus — compact connection indicator chip.
 * Shows: Live (green) | Reconnecting (orange) | Disconnected (red)
 */
export function WebSocketStatus({
  connected,
  reconnectAttempts = 0,
  label = "Live",
  className = "",
}: WebSocketStatusProps) {
  if (connected) {
    return (
      <span
        className={`inline-flex items-center gap-1.5 text-[11px] px-2 py-1 rounded-full font-medium ${className}`}
        style={{ background: "rgba(34,197,94,0.12)", color: "#4ade80" }}
      >
        <span className="w-1.5 h-1.5 rounded-full bg-green-400 animate-pulse" />
        {label}
      </span>
    );
  }

  if (reconnectAttempts > 0) {
    return (
      <span
        className={`inline-flex items-center gap-1.5 text-[11px] px-2 py-1 rounded-full font-medium ${className}`}
        style={{ background: "rgba(234,179,8,0.12)", color: "#fbbf24" }}
      >
        <Loader2 className="w-2.5 h-2.5 animate-spin" />
        Reconnecting ({reconnectAttempts})
      </span>
    );
  }

  return (
    <span
      className={`inline-flex items-center gap-1.5 text-[11px] px-2 py-1 rounded-full font-medium ${className}`}
      style={{ background: "rgba(107,114,128,0.12)", color: "#9ca3af" }}
    >
      <WifiOff className="w-2.5 h-2.5" />
      Connecting…
    </span>
  );
}
